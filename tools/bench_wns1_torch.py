# wns1 baseline: torch CPU / CUDA (fp32), same module + inputs as the golden dump
import argparse, importlib.util, os, time
import numpy as np
import torch

_h = importlib.util.spec_from_file_location(
    "dwg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_golden_wns1.py"))
dwg = importlib.util.module_from_spec(_h)
_h.loader.exec_module(dwg)


# 他们的 Encoder.forward 里 torch.arange(T) 不带 device, CUDA 下会 device mismatch;
# 这里用等价实现覆盖 (不改他们的 dump 脚本)
def _enc_fwd(self, x, x_lengths, g):
    g = g.detach()
    T = x.size(2)
    x_mask = (torch.arange(T, device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1))
    x_mask = x_mask.unsqueeze(1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    return self.proj(x) * x_mask, x_mask


dwg.Encoder.forward = _enc_fwd


def build(ckpt, T=120, LEN=100):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    raw = {k[5:]: v for k, v in sd.items() if k.startswith("wns1.")}
    msd = {}
    for k, v in raw.items():
        if k.endswith(".weight_g"):
            b = k[: -len("_g")]
            msd[b] = dwg.wn_mat(v, raw[b + "_v"])
        elif k.endswith(".weight_v"):
            continue
        else:
            msd[k] = v
    m = dwg.Encoder(512, 512, 512, 5, 8, 512).eval()
    missing, unexpected = m.load_state_dict(msd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 512, T, generator=g)
    ge = torch.randn(1, 512, 1, generator=g)
    xlen = torch.LongTensor([LEN])
    return m, x, ge, xlen


def timeit(fn, n, warm, dev):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return sum(ts) / len(ts), ts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--len", type=int, default=100)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--dest", default="cuda,cpu")
    args = ap.parse_args()

    m, x, ge, xlen = build(args.ckpt, args.T, args.len)
    for dest in args.dest.split(","):
        for dt in (torch.float32, torch.float16):
            if dest == "cpu" and dt == torch.float16:
                continue
            if dest == "cuda" and not torch.cuda.is_available():
                print("cuda unavailable"); continue
            mm = m.to(dest).to(dt)
            xx = x.to(dest, dtype=dt); gg = ge.to(dest, dtype=dt); ll = xlen.to(dest)
            avg, mn = timeit(lambda: mm(xx, ll, gg), args.iters, args.warmup, dest)
            fp = "fp16" if dt == torch.float16 else "fp32"
            print(f"torch {dest:4s} {fp}: avg {avg:8.3f} ms  min {mn:8.3f} ms  (T={args.T} len={args.len})")


if __name__ == "__main__":
    main()
