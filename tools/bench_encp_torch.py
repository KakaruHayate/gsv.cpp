# enc_p baseline: torch CPU / CUDA (fp32)，与 golden dump 同模块同输入
# 用法: python tools/bench_encp_torch.py [--T 120] [--ntext 50] [--iters 30] [--dest cuda,cpu]
import argparse, importlib.util, os, time
import numpy as np
import torch

_h = importlib.util.spec_from_file_location(
    "dge", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump_golden_encp.py"))
dge = importlib.util.module_from_spec(_h)
_h.loader.exec_module(dge)


def build(ckpt, T, NT):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    msd = {}
    for k, v in sd.items():
        if not k.startswith("enc_p."):
            continue
        kk = k[len("enc_p."):]
        kk = kk.replace("norm_layers_1.", "norm1.").replace("norm_layers_2.", "norm2.")
        msd[kk] = v
    m = dge.TextEncoder().eval()
    m.load_state_dict(msd, strict=True)
    g = torch.Generator().manual_seed(42)
    y = torch.randn(1, 768, T, generator=g)
    text = torch.randint(0, 732, (1, NT), generator=g)
    ge = torch.randn(1, 512, 1, generator=g)
    return m, y, text, ge


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
    ap.add_argument("--ntext", type=int, default=50)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--dest", default="cuda,cpu")
    args = ap.parse_args()

    m, y, text, ge = build(args.ckpt, args.T, args.ntext)
    for dest in args.dest.split(","):
        if dest == "cuda" and not torch.cuda.is_available():
            print("cuda unavailable"); continue
        for dt in (torch.float32, torch.float16):
            if dest == "cpu" and dt == torch.float16:
                continue
            mm = m.to(dest).to(dt)
            yy = y.to(dest, dtype=dt); tt = text.to(dest); gg = ge.to(dest, dtype=dt)
            avg, mn = timeit(lambda: mm(yy, tt, gg), args.iters, args.warmup, dest)
            fp = "fp16" if dt == torch.float16 else "fp32"
            print(f"torch {dest:4s} {fp}: avg {avg:8.3f} ms  min {mn:8.3f} ms  (T={args.T} ntext={args.ntext})")


if __name__ == "__main__":
    main()
