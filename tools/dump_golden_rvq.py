# RVQ golden: codes -> quantizer.decode -> (25hz) x2 nearest
#   codes  : [T] int32 (取自 AR golden 的真实 token 流 + 合成用例)
#   quant  : [T, 768] (torch [768,T] 转置后 dump, 与 ggml 展平顺序 [ne0=768,ne1=T] 一致)
#   up     : [2T, 768] (×2 nearest 之后)
import argparse, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("GSV_REPO", os.path.normpath(os.path.join(ROOT, os.pardir, "repo")))
CKPT = os.path.join(ROOT, "models", "s2Gv5turbo.pth")


def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print(f"  dumped {name}: shape={t.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--n", type=int, default=0, help="额外用 AR golden 的前 n 个 token 做用例")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    cb = sd["quantizer.vq.layers.0._codebook.embed"]     # [1024, 768]
    print(f"codebook: {tuple(cb.shape)} dtype={cb.dtype}")

    cases = []
    # 1) 合成: 全 0 / 递增覆盖 / 随机
    cases.append(("zeros", np.zeros(8, dtype=np.int32)))
    cases.append(("ramp", (np.arange(16, dtype=np.int32) * 64) % 1024))
    rng = np.random.RandomState(1234)
    cases.append(("rand", rng.randint(0, 1024, size=13).astype(np.int32)))
    # 2) 真实: AR greedy golden 的 token 流 (前 n 个)
    if args.n > 0:
        tok = np.fromfile(os.path.join(args.out, "batch.greedy100.seq0.tokens.bin"), dtype=np.float32)
        tok = tok[: args.n].astype(np.int32)
        tok = np.clip(tok, 0, 1023)
        cases.append(("ar%d" % args.n, tok))

    for i, (name, codes) in enumerate(cases):
        c = torch.from_numpy(np.asarray(codes, dtype=np.int64)).unsqueeze(0).unsqueeze(0)  # [1,1,T]
        quant = torch.nn.functional.embedding(c[:, 0, :], cb)      # [1, T, 768]
        quant = quant.transpose(1, 2).contiguous()                 # [1, 768, T]
        up = torch.nn.functional.interpolate(quant, scale_factor=2, mode="nearest")
        dump(args.out, f"rvq.{i}.codes", torch.from_numpy(np.asarray(codes, dtype=np.float32)))
        dump(args.out, f"rvq.{i}.quant", quant[0].transpose(0, 1).contiguous())   # [T, 768]
        dump(args.out, f"rvq.{i}.up",    up[0].transpose(0, 1).contiguous())      # [2T, 768]
        print(f"  case{i} '{name}': T={len(codes)} -> quant {tuple(quant.shape[2:])}, up {tuple(up.shape[2:])}")
    dump(args.out, "rvq.n_cases", torch.tensor([len(cases)], dtype=torch.float32))
    print("RVQ golden done ->", args.out)


if __name__ == "__main__":
    main()
