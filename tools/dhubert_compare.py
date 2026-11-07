#!/usr/bin/env python3
"""逐元素比较两份 [D, T] f32 特征 (our gsv_dhubert vs torch 参考)。

    python3 tools/dhubert_compare.py ours.f32 ref.f32 [--dim 768] [--tol 1e-2] [--corr 0.999]

两份文件都是 [D, T] f32 行主序 (idx = d + t*D), T 由文件长度 / D 推出。
"""
import argparse
import sys

import numpy as np


def load(path, dim, axis):
    x = np.fromfile(path, dtype=np.float32)
    if x.size % dim:
        raise SystemExit(f"{path}: {x.size} floats not divisible by dim={dim}")
    # axis=t: 帧主序 (T, D) 文件, 即 ggml [D, T] 的字节序 -> (D, T)
    # axis=c: 通道主序 (D, T) 文件, 即 ggml [T, D] 的字节序 -> (D, T)
    return x.reshape(-1, dim).T.copy() if axis == "t" else x.reshape(dim, -1).copy()


def stats(a, b):
    d = a - b
    absd = np.abs(d)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.sqrt((ac * ac).sum() * (bc * bc).sum())
    corr = float((ac * bc).sum() / denom) if denom > 0 else 0.0
    corr = max(-1.0, min(1.0, corr))
    # 每帧余弦相似度 (帧向量 = 一列)
    na = np.linalg.norm(a, axis=0) + 1e-12
    nb = np.linalg.norm(b, axis=0) + 1e-12
    cos = (a * b).sum(axis=0) / (na * nb)
    return {
        "max_abs": float(absd.max()),
        "mean_abs": float(absd.mean()),
        "rms": float(np.sqrt((d * d).mean())),
        "corr": corr,
        "rel_max": float(absd.max() / (np.abs(b).max() + 1e-12)),
        "ref_max": float(np.abs(b).max()),
        "out_max": float(np.abs(a).max()),
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("ref")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--axis", choices=["t", "c"], default="t",
                    help="t = 帧主序 (主输出/attn dump), c = 通道主序 (front/tcn dump)")
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--corr", type=float, default=0.999)
    a = ap.parse_args()

    x = load(a.ours, a.dim, a.axis)
    y = load(a.ref, a.dim, a.axis)
    if x.shape != y.shape:
        raise SystemExit(f"shape mismatch: ours {x.T.shape} vs ref {y.T.shape}")
    s = stats(x, y)
    print(f"shape            : {x.T.shape}  (D={x.shape[0]}, T={x.shape[1]})")
    print(f"max abs diff     : {s['max_abs']:.6e}")
    print(f"mean abs diff    : {s['mean_abs']:.6e}")
    print(f"rms diff         : {s['rms']:.6e}")
    print(f"corr             : {s['corr']:.9f}")
    print(f"rel max / |ref|  : {s['rel_max']:.3e}  (|ref|max={s['ref_max']:.4f}, |ours|max={s['out_max']:.4f})")
    print(f"per-frame cosine : min {s['cos_min']:.9f}  mean {s['cos_mean']:.9f}")
    ok = s["max_abs"] < a.tol and s["corr"] > a.corr
    print("verdict          : " + ("PASS" if ok else "FAIL") +
          f"  (need max_abs < {a.tol:g} && corr > {a.corr:g})")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
