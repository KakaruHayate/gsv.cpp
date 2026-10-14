"""端到端驱动: 文本 + 参考音频 -> wav。

    python -m tools.frontend.tts --text "大家好" --ref tests/audio/ref_zh_32k.wav --out out.wav

分四步 (与 repo TTS.py 同构):
    1. server.prepRef  参考音频三链 -> prompt codes / ge / mel2   (可缓存, 换句子不必重算)
    2. server.bert     BERT ids -> [1024, P] (按 word2ph 重复)     (非 zh 跳过)
    3. server.ar       phones+bert+prompt -> 语义 token
    4. server.synth    语义 token -> 条件段 -> DiT -> vocoder -> wav
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.frontend import Frontend          # noqa: E402
from frontend.gsv_client import GsvClient       # noqa: E402


def write_wav(path, samples, sr=32000):
    """float32 [-1,1] -> 16-bit PCM wav。"""
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def normalize(x, target_peak=0.708):
    """去直流 + 峰值归一到 target_peak。

    必须先去掉直流分量: 输出里可能有极小的 DC 偏置 (vocoder 的 tanh 端点), 而峰值归一化
    会拿这个偏置当峰值, 把整段压成"静音"(实测过)。去 DC 后峰值才是真正的波形幅度。
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    x = x - float(x.mean())
    pk = float(np.abs(x).max())
    return x if pk <= 1e-9 else (x * (target_peak / pk)).astype(np.float32)


class Tts:
    """一次连接里做参考缓存 + 反复合成。"""

    def __init__(self, client, language="zh", steps=4, cfg_rate=0.0, seed=0,
                 early_stop=400, max_steps=1500):
        self.c = client
        self.fe = Frontend(language=language)
        self.language = language
        self.steps = steps
        self.cfg_rate = cfg_rate
        self.seed = seed
        self.early_stop = early_stop
        self.max_steps = max_steps
        self.ref = None          # (codes, ge, mel2, melT)

    def set_reference(self, wav_path):
        data, t = self.c.request("prepRef", {"wavPath": wav_path})
        codes = np.frombuffer(t["codes"], dtype="<i4").astype(np.int32)
        ge = np.frombuffer(t["ge"], dtype="<f4").astype(np.float32)
        mel2 = np.frombuffer(t["mel2"], dtype="<f4").astype(np.float32)
        melT = int(data["melT"])
        assert mel2.size == 100 * melT, (mel2.size, melT)
        self.ref = (codes, ge, mel2, melT)
        return self.ref

    def synthesize(self, text, language=None, steps=None, verbose=True):
        if self.ref is None:
            raise RuntimeError("先调 set_reference()")
        codes, ge, mel2, melT = self.ref
        lang = language or self.language
        t0 = time.time()
        phones, word2ph, bert_ids = self.fe(text, language=lang)
        if verbose:
            print("  phones=%d word2ph=%d bert_ids=%d (%.0f ms)"
                  % (len(phones), len(word2ph), len(bert_ids), (time.time() - t0) * 1000))

        ph32 = np.asarray(phones, dtype="<i4")

        # BERT 特征 (只有 zh 需要; 其余语言 repo 给全零)
        bert = b""
        if bert_ids:
            t1 = time.time()
            data, t = self.c.request(
                "bert",
                {"idsLen": len(bert_ids), "word2phLen": len(word2ph)},
                tensors={"ids": np.asarray(bert_ids, dtype="<i4").tobytes(),
                         "word2ph": np.asarray(word2ph, dtype="<i4").tobytes()},
            )
            bert = t["feat"]
            n = int(data["phones"])
            assert len(bert) == 1024 * n * 4, (len(bert), n)
            if verbose:
                print("  bert feat [1024, %d] (%.0f ms)" % (n, (time.time() - t1) * 1000))

        # AR -> 语义 token
        t2 = time.time()
        req = {"maxSteps": self.max_steps, "earlyStop": self.early_stop, "seed": self.seed}
        tens = {"phones": ph32.tobytes(), "prompt": codes.tobytes()}
        if bert:
            tens["bert"] = bert
        data, t = self.c.request("ar", req, tensors=tens)
        sem = np.frombuffer(t["sem"], dtype="<i4").astype(np.int32)
        if verbose:
            print("  sem=%d (%.0f ms)" % (sem.size, (time.time() - t2) * 1000))

        # 条件段 + DiT + vocoder
        t3 = time.time()
        data, t = self.c.request(
            "synth",
            {"melT": melT, "steps": steps or self.steps, "cfg": self.cfg_rate, "seed": self.seed},
            tensors={
                "sem": sem.tobytes(),
                "phones": ph32.tobytes(),
                "prompt": codes.tobytes(),
                "ge": ge.tobytes(),
                "mel2": mel2.tobytes(),
            },
        )
        wav = np.frombuffer(t["wav"], dtype="<f4").astype(np.float32)
        sr = int(data.get("sampleRate", 32000))
        if verbose:
            print("  wav=%d (%.2f s @%d) (%.0f ms)"
                  % (wav.size, wav.size / sr, sr, (time.time() - t3) * 1000))
        return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=[],
                    help="要合成的句子 (可重复; 不填则用内置示例)")
    ap.add_argument("--ref", default=None, help="参考音频 wav")
    ap.add_argument("--out", default=None, help="单句输出路径; 多句时按序号加后缀")
    ap.add_argument("--outdir", default="tests/e2e_out", help="多句输出目录")
    ap.add_argument("--language", default="zh", help="zh/en/ja/ko/yue/auto")
    ap.add_argument("--steps", type=int, default=4, help="DiT 采样步数 (v5turbo=4)")
    ap.add_argument("--device", default="", help="ggml 后端: 空=CPU, vulkan=GPU")
    ap.add_argument("--voc-device", default="", help="ORT EP: 空=自动, cpu/dml/webgpu")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--early-stop", type=int, default=400,
                    help="AR 生成上限 (25Hz 语义 token 数); repo 用 hz*max_sec")
    ap.add_argument("--no-norm", action="store_true", help="不做峰值归一化")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 参考音频必须 3~10 s (repo TTS.py::_set_prompt_semantic 会直接 OSError 拒掉其余长度),
    # 所以默认用 4.38 s 的短素材, 而不是 14.3 s 的长素材。
    ref = args.ref or os.path.join(root, "tests", "audio", "ref_zh_short_16k.wav")
    exe = os.path.join(root, "tests", "relcpu", "gsv_ggml_cli.exe")

    texts = args.text or [
        "大家好，这是一段用来测试的语音。",
        "今天天气不错，我们出去走走吧。",
    ]

    print("[tts] 启动 server (模型加载约需 30 s)...")
    c = GsvClient(autostart=True, exe=exe, device=args.device, voc_device=args.voc_device,
                  verbose=not args.quiet)
    try:
        t = Tts(c, language=args.language, seed=args.seed, early_stop=args.early_stop)
        print("[tts] 参考音频: %s" % ref)
        t0 = time.time()
        codes, ge, mel2, melT = t.set_reference(ref)
        print("  prompt codes=%d  ge[512]  mel2[100,%d]  (%.0f ms)"
              % (codes.size, melT, (time.time() - t0) * 1000))

        multi = len(texts) > 1
        if multi:
            os.makedirs(args.outdir, exist_ok=True)
        for i, txt in enumerate(texts):
            print("[tts] (%d/%d) %s" % (i + 1, len(texts), txt))
            t0 = time.time()
            wav, sr = t.synthesize(txt, verbose=not args.quiet)
            if not args.no_norm:
                wav = normalize(wav)
            rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
            print("  -> %.2f s, peak %.3f, rms %.4f (%.1f dBFS)  [总 %.1f s]"
                  % (wav.size / sr, float(np.abs(wav).max()), rms,
                     20 * np.log10(max(rms, 1e-9)), time.time() - t0))
            if multi:
                out = os.path.join(args.outdir, "%02d.wav" % i)
            else:
                out = args.out or os.path.join(root, "tests", "e2e_out.wav")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            write_wav(out, wav, sr)
            print("  写出 %s" % out)
    finally:
        c.close()


if __name__ == "__main__":
    main()
