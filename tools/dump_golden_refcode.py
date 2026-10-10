# 参考音频 -> prompt semantic tokens golden（v5 的 _set_prompt_semantic 链）
#   1) wav --(match_librosa 重采样, TTS.py::resample 的等价复刻)--> 16k
#   2) 追加 int(32000*0.3)=9600 个零 (configs.sampling_rate*0.3)
#   3) CNHubert -> last_hidden_state [1,T,768] -> transpose -> [1,768,T]
#   4) ssl_proj = Conv1d(768,768,k=2,stride=2)（从 s2Gv5turbo.pth 取权重）
#   5) RVQ 编码: argmin(x² - 2x·Eᵀ + E²)（core_vq.EuclideanCodebook.quantize 等价形式）
# 落盘 float32 raw: ref.{tag}.wav16k_res(重采样后) / wav16k_fed(含零尾) /
#                   hubert_ssl [768,T] / sslproj_out [768,T/2] / codes int32 [T/2]
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, os.pardir, "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

import resampy  # noqa: E402

FILTER_CACHE = {}


def load_audio(path):
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(a.T.copy()), sr
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        assert sw == 2
        a = np.frombuffer(raw, dtype="<i2").astype("float32").reshape(-1, ch).T / 32768.0
        return torch.from_numpy(a.copy()), sr


def resample_match_librosa(x, sr0, sr1):
    """TTS.py::resample(match_librosa=True) 的等价复刻 (resampy kaiser_best 多相插值)"""
    if int(sr0) == int(sr1):
        return x.clone()
    ratio = float(sr1) / float(sr0)
    scale = min(1.0, ratio)
    n_orig = x.shape[-1]
    n_out = int(n_orig * ratio)
    key = "kb"
    if key not in FILTER_CACHE:
        win, precision, rolloff = resampy.filters.get_filter("kaiser_best")
        win = torch.from_numpy(win.astype(np.float32))
        delta = torch.diff(win, append=win[-1:])
        FILTER_CACHE[key] = (win, delta, int(precision))
    interp_win, interp_delta, num_table = FILTER_CACHE[key]
    if ratio < 1.0:
        interp_win = interp_win * ratio
        interp_delta = interp_delta * ratio
    index_step = int(scale * num_table)
    nwin = interp_win.numel()
    t_all = torch.arange(n_out, dtype=torch.float32) / ratio
    out = torch.zeros((x.shape[0], n_out), dtype=x.dtype)
    chunk = 4096
    max_taps = (nwin + index_step - 1) // index_step
    for begin in range(0, n_out, chunk):
        t = t_all[begin:begin + chunk]
        n = torch.floor(t).long()
        frac = scale * (t - n.to(t.dtype))
        index_frac = frac * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ii = torch.arange(max_taps).view(1, -1)
        li = offset.view(-1, 1) + ii * index_step
        lmax = torch.minimum(n + 1, (nwin - offset) // index_step).view(-1, 1)
        lm = ii < lmax
        li_safe = li.clamp(0, nwin - 1)
        lw = interp_win[li_safe] + eta.view(-1, 1) * interp_delta[li_safe]
        left_idx = (n.view(-1, 1) - ii).clamp(0, n_orig - 1)
        left = (lw * lm).to(x.dtype)
        lv = x[:, left_idx]
        frac_r = scale - frac
        index_frac = frac_r * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ri = offset.view(-1, 1) + ii * index_step
        rmax = torch.minimum(n_orig - n - 1, (nwin - offset) // index_step).view(-1, 1)
        rm = ii < rmax
        ri_safe = ri.clamp(0, nwin - 1)
        rw = interp_win[ri_safe] + eta.view(-1, 1) * interp_delta[ri_safe]
        right_idx = (n.view(-1, 1) + ii + 1).clamp(0, n_orig - 1)
        right = (rw * rm).to(x.dtype)
        rv = x[:, right_idx]
        out[:, begin:begin + chunk] = (lv * left).sum(-1) + (rv * right).sum(-1)
    return out


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-26s %s" % (name, tuple(arr.shape)))


def run(out, tag, wav_path, ckpt, hubert_dir, sr_config=32000):
    from feature_extractor.cnhubert import CNHubert
    a, sr = load_audio(wav_path)
    a = a.mean(0, keepdim=True)
    wav16k = resample_match_librosa(a, sr, 16000)
    n16 = wav16k.shape[-1]
    if n16 > 160000 or n16 < 48000:
        print("[%s] 警告: 参考音频 %.1fs 不在 3~10s 内 (repo 会报错)" % (tag, n16 / 16000))
    zero = torch.zeros(int(sr_config * 0.3), dtype=torch.float32)
    fed = torch.cat([wav16k[0], zero])
    dump(out, "ref." + tag + ".wav16k_res", wav16k[0])
    dump(out, "ref." + tag + ".wav16k_fed", fed)

    hub = CNHubert(hubert_dir).eval().float()      # checkpoint 是 fp16, 参考统一用 fp32
    with torch.no_grad():
        ssl = hub.model(fed.unsqueeze(0))["last_hidden_state"].transpose(1, 2)      # [1,768,T]
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    w_ssl = sd["ssl_proj.weight"].float()
    b_ssl = sd["ssl_proj.bias"].float()
    emb = sd["quantizer.vq.layers.0._codebook.embed"].float()                       # [1024,768]
    with torch.no_grad():
        z = torch.nn.functional.conv1d(ssl, w_ssl, b_ssl, stride=2)                 # [1,768,T/2]
        x = z.transpose(1, 2).reshape(-1, 768)                                      # [N,768]
        dist = x.pow(2).sum(1, keepdim=True) - 2 * x @ emb.t() + emb.pow(2).sum(1, keepdim=True).t()
        codes = dist.argmin(dim=-1).to(torch.int32)                                 # [N]
    dump(out, "ref." + tag + ".hubert_ssl", ssl[0])
    dump(out, "ref." + tag + ".sslproj_out", z[0])
    codes.numpy().tofile(os.path.join(out, "ref." + tag + ".codes.bin"))
    print("[%s] n16=%d (%.2fs) T_ssl=%d T_code=%d codes[:8]=%s"
          % (tag, n16, n16 / 16000, ssl.shape[-1], codes.numel(), codes[:8].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--audio", default="tests/audio")
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--hubert", default="models/chinese-hubert-base")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    for tag, fn in [("zhs", "ref_zh_short_16k.wav"), ("en", "ref_en_16k.wav"),
                    ("zh32", "ref_zh_32k.wav")]:
        p = os.path.join(args.audio, fn)
        if not os.path.exists(p):
            print("missing", p, "-> skip")
            continue
        run(args.out, tag, p, args.ckpt, args.hubert)
    print("refcode golden done ->", args.out)


if __name__ == "__main__":
    main()
