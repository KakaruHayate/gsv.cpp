# ref_enc baseline: torch CPU / CUDA (fp32), 与 golden dump 同模块同输入 (T=200, len=180)
# 用法: python tools/bench_refenc_torch.py [--T 200] [--len 180] [--iters 30] [--dest cuda,cpu]
import argparse, importlib.util, os, time
import numpy as np
import torch

_h = importlib.util.spec_from_file_location(
    "dgr", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_golden_refenc.py"))
dgr = importlib.util.module_from_spec(_h)
_h.loader.exec_module(dgr)


def build(ckpt, T, LEN):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    m = dgr.MelStyleEncoder()
    msd = {k[len("ref_enc."):]: v for k, v in sd.items() if k.startswith("ref_enc.")}
    m.load_state_dict(msd, strict=True)
    m = m.eval()
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 704, T, generator=g)
    mask = torch.zeros(1, 1, T)
    mask[:, :, :LEN] = 1.0
    return m, x, mask


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
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--len", type=int, default=180)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--dest", default="cuda,cpu")
    args = ap.parse_args()

    m, x, mask = build(args.ckpt, args.T, args.len)
    for dest in args.dest.split(","):
        if dest == "cuda" and not torch.cuda.is_available():
            print("cuda unavailable"); continue
        for dt in (torch.float32, torch.float16):
            if dest == "cpu" and dt == torch.float16:
                continue
            mm = m.to(dest).to(dt)
            xx = x.to(dest, dtype=dt); kk = mask.to(dest, dtype=dt)
            avg, mn = timeit(lambda: mm(xx * kk, kk), args.iters, args.warmup, dest)
            fp = "fp16" if dt == torch.float16 else "fp32"
            print(f"torch {dest:4s} {fp}: avg {avg:8.3f} ms  min {mn:8.3f} ms  (T={args.T} len={args.len})")


if __name__ == "__main__":
    main()
