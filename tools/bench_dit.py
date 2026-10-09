# DiT 生产尺寸基准 (torch): 与 tests/test_dit.cpp 的 GSV_DIT_BENCH 同 workload
#   T=1000 (prompt 500), x=0.5 常量, t=0.25, 单步 estimator 前向 (v5turbo 4 步形态)
#   prepare_static_cache 一次, 计 N 次前向的 avg/min
# 用法: python tools/bench_dit.py --device cpu --n 10
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, os.pardir, "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

from tools_dit_common import build_dit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="cpu 线程数 (0 = 不设置)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)

    dit = build_dit(args.ckpt).eval().to(dev)
    if args.device.startswith("cuda"):
        dit = dit.half()          # torch 推理惯例: CUDA 跑 f16 (checkpoint 即 fp16)

    T, P = 1000, 500
    prompt = (0.1 * (np.arange(100 * T) % 17 - 8)).astype(np.float32).reshape(1, 100, T)
    mu = (0.1 * (np.arange(512 * T) % 13 - 6)).astype(np.float32).reshape(1, 512, T)
    dt = dit.time_embed.time_mlp[0].weight.dtype
    prompt = torch.from_numpy(prompt).to(dev, dt)
    mu = torch.from_numpy(mu).to(dev, dt)
    x = torch.full((1, 100, T), 0.5, device=dev, dtype=dt)
    t = torch.tensor([0.25], device=dev, dtype=dt)
    x_lens = torch.LongTensor([T]).to(dev)

    with torch.inference_mode():
        cache = dit.prepare_static_cache(prompt, x_lens, mu)
        for _ in range(2):   # warmup
            dit(x, prompt, x_lens, t, text0=mu, infer=True, static_cache=cache)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            dit(x, prompt, x_lens, t, text0=mu, infer=True, static_cache=cache)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    print("torch DiT step (%s, T=%d): avg %.1f ms min %.1f ms n=%d" %
          (args.device, T, sum(ts) / len(ts), ts[0], len(ts)))


if __name__ == "__main__":
    main()
