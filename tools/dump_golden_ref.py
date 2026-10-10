# 参考音频预处理 golden: wav -> torchaudio 重采样(16k/32k) -> mel_fn_v4 -> norm_spec
# 落盘 float32 raw:
#   ref.{tag}.wav32k / ref.{tag}.wav16k  重采样后的样本 (C++ 端 wav 读取/重采样的对照)
#   ref.{tag}.mel / ref.{tag}.mel_norm   mel [100, T] / norm_spec 后 (torch 布局 m*T+t)
import argparse, os, sys
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, os.pardir, "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

from module.mel_processing import mel_spectrogram_torch  # noqa: E402

spec_min, spec_max = -12, 2


def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def mel_fn_v4(x):
    return mel_spectrogram_torch(
        x, n_fft=1280, num_mels=100, sampling_rate=32000,
        hop_size=320, win_size=1280, fmin=0, fmax=None, center=False,
    )


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-24s %s" % (name, tuple(arr.shape)))


def load_audio(path):
    """避免 torchcodec (环境缺 DLL): soundfile 优先, 回退 stdlib wave (PCM16 解码与 torchaudio 一致)"""
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=True)      # [n, C]
        return torch.from_numpy(a.T.copy()), sr                     # [C, n]
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        assert sw == 2, "stdlib 回退只支持 PCM16 (sampwidth=%d)" % sw
        a = np.frombuffer(raw, dtype="<i2").astype("float32").reshape(-1, ch).T / 32768.0
        return torch.from_numpy(a.copy()), sr


def run(out, tag, wav_path):
    a, sr = load_audio(wav_path)                       # [C, n]
    a = a.mean(0, keepdim=True)                        # mono
    a32 = a if sr == 32000 else torchaudio.transforms.Resample(sr, 32000)(a)
    a16 = a if sr == 16000 else torchaudio.transforms.Resample(sr, 16000)(a)
    dump(out, "ref." + tag + ".wav32k", a32[0])
    dump(out, "ref." + tag + ".wav16k", a16[0])
    with torch.no_grad():
        m = mel_fn_v4(a32)                             # [1, n] -> [1, 100, T]
    dump(out, "ref." + tag + ".mel", m[0])
    dump(out, "ref." + tag + ".mel_norm", norm_spec(m[0]))
    print("[%s] src_sr=%d n=%d -> 32k n=%d / 16k n=%d; mel T=%d (%.2fs)"
          % (tag, sr, a.shape[-1], a32.shape[-1], a16.shape[-1], m.shape[-1], m.shape[-1] / 100.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--audio", default="tests/audio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    for tag, fn in [("zh", "ref_zh_32k.wav"), ("en", "ref_en_16k.wav")]:
        p = os.path.join(args.audio, fn)
        if not os.path.exists(p):
            print("missing", p, "-> skip")
            continue
        run(args.out, tag, p)
    print("ref golden done ->", args.out)


if __name__ == "__main__":
    main()
