# 条件链 torch 基准 (v5turbo decode_encp 链): CPU / CUDA
# 用法: python tools/bench_chain_torch.py [--ncodes 40] [--ntext 50] [--iters 30] [--dest cuda,cpu]
import argparse, importlib.util, os, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(ckpt, nc, nt):
    dge = _load("tools/dump_golden_encp.py", "dge")
    dgw = _load("tools/dump_golden_wns1.py", "dgw")

    # wns1 Encoder.forward 里 torch.arange(T) 不带 device, CUDA 会 mismatch;
    # 等价实现覆盖 (不改他们的 dump 脚本)
    def _enc_fwd(self, x, x_lengths, g):
        g = g.detach()
        T = x.size(2)
        x_mask = (torch.arange(T, device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1))
        x_mask = x_mask.unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        return self.proj(x) * x_mask, x_mask
    dgw.Encoder.forward = _enc_fwd
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)

    msd = {}
    for k, v in sd.items():
        if not k.startswith("enc_p."):
            continue
        kk = k[len("enc_p."):].replace("norm_layers_1.", "norm1.").replace("norm_layers_2.", "norm2.")
        msd[kk] = v
    encp = dge.TextEncoder().eval()
    encp.load_state_dict(msd, strict=True)

    raw = {k[5:]: v for k, v in sd.items() if k.startswith("wns1.")}
    wsd = {}
    for k, v in raw.items():
        if k.endswith(".weight_g"):
            b = k[:-len("_g")]
            wsd[b] = dgw.wn_mat(v, raw[b + "_v"])
        elif k.endswith(".weight_v"):
            continue
        else:
            wsd[k] = v
    wns1 = dgw.Encoder(512, 512, 512, 5, 8, 512).eval()
    wns1.load_state_dict(wsd, strict=False)

    gw = torch.Generator().manual_seed(42)
    codes = torch.randint(0, 1024, (1, 1, nc), generator=gw)
    text = torch.randint(0, 732, (1, nt), generator=gw)
    ge = torch.randn(1, 512, 1, generator=gw)
    return (sd["quantizer.vq.layers.0._codebook.embed"].float(),
            encp, sd["bridge.0.weight"].float(), sd["bridge.0.bias"].float(), wns1,
            codes, text, ge)


def run_chain(mods, codes, text, ge):
    cb, encp, cbw, cbb, wns1 = mods
    quant = torch.nn.functional.embedding(codes[:, 0, :], cb).transpose(1, 2)
    quant = torch.nn.functional.interpolate(quant, scale_factor=2, mode="nearest")
    x, m, logs, ymask = encp(quant, text, ge)
    br = torch.nn.functional.leaky_relu(torch.nn.functional.conv1d(x, cbw, cbb), 0.01)
    up = torch.nn.functional.interpolate(br, scale_factor=2, mode="nearest")
    fea, _ = wns1(up, torch.LongTensor([up.size(2)]).to(up.device), ge)
    return fea


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
    ap.add_argument("--ncodes", type=int, default=40)
    ap.add_argument("--ntext", type=int, default=50)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--dest", default="cuda,cpu")
    args = ap.parse_args()

    cb, encp, cbw, cbb, wns1, codes, text, ge = build(args.ckpt, args.ncodes, args.ntext)
    mods = (cb, encp, cbw, cbb, wns1)
    for dest in args.dest.split(","):
        if dest == "cuda" and not torch.cuda.is_available():
            print("cuda unavailable"); continue
        for dt in (torch.float32, torch.float16):
            if dest == "cpu" and dt == torch.float16:
                continue
            mm = tuple(m.to(dest).to(dt) if isinstance(m, torch.Tensor) else m.to(dest).to(dt)
                       for m in mods)
            cc = codes.to(dest); tt = text.to(dest); gg = ge.to(dest, dtype=dt)
            avg, mn = timeit(lambda: run_chain(mm, cc, tt, gg), args.iters, args.warmup, dest)
            fp = "fp16" if dt == torch.float16 else "fp32"
            print(f"torch {dest:4s} {fp}: avg {avg:8.3f} ms  min {mn:8.3f} ms  (Tc={args.ncodes})")


if __name__ == "__main__":
    main()
