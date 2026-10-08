# HuBERT baseline: torch CPU / CUDA, fp32 & fp16 (same 16k-sample input as golden)
import os, sys, time, argparse
import numpy as np, torch
from transformers import HubertModel

def bench(model, x, iters, warmup, dev):
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_values=x)
        if dev == 'cuda': torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(input_values=x)
            if dev == 'cuda': torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return sum(ts)/len(ts), ts[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/chinese-hubert-base')
    ap.add_argument('--input', default='tests/golden/hubert.input_norm.bin')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=6)
    ap.add_argument('--dest', default='cuda,cpu')
    args = ap.parse_args()

    wav = torch.from_numpy(np.fromfile(args.input, dtype=np.float32))
    model = HubertModel.from_pretrained(args.ckpt, local_files_only=True).eval()

    for dest in args.dest.split(','):
        for dt in (torch.float32, torch.float16):
            if dest == 'cpu' and dt == torch.float16:
                continue          # cpu fp16 无意义
            if dest == 'cuda' and not torch.cuda.is_available():
                print('cuda unavailable'); continue
            m = model.to(dest).to(dt) if dest == 'cuda' else model.to(dest)
            m = m.to(dt)
            x = wav.unsqueeze(0).to(dest, dtype=dt)
            avg, mn = bench(m, x, args.iters, args.warmup, dest)
            fp16 = 'fp16' if dt == torch.float16 else 'fp32'
            print(f'torch {dest:4s} {fp16}: avg {avg:8.2f} ms  min {mn:8.2f} ms  (n={args.iters})')
            m.to(torch.float32)

if __name__ == '__main__':
    main()
