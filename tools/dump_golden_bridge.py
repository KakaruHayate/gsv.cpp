# bridge golden dump: Conv1d(192->512, k=1) + LeakyReLU(0.01)
import argparse, os
import numpy as np
import torch
from torch import nn

def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %s: %s" % (name, tuple(t.shape)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--T", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    conv = nn.Conv1d(192, 512, 1).eval()
    conv.weight.data.copy_(sd["bridge.0.weight"])
    conv.bias.data.copy_(sd["bridge.0.bias"])
    act = nn.LeakyReLU()
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 192, args.T, generator=g)
    with torch.no_grad():
        y = act(conv(x))
    dump(args.out, "bridge.input", x)
    dump(args.out, "bridge.out", y)
    print("bridge golden done ->", args.out)

if __name__ == "__main__":
    main()
