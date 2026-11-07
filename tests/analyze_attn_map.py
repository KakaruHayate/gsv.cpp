"""DiT attention-map structure analysis.

Reads the raw [T_k, T_q] float32 matrices written by gsv_dit's GSV_ATTN_DUMP path
(file name: attn_<pos|neg>_t<ms>_l<layer>_h<head>.f32, idx = k + T_k*q) and looks for
*structure* in the sparsity: quadrant split (reference vs generated frames), diagonal
band, block-level concentration, per-layer/head/step variation, cross-step stability.

    python tests/analyze_attn_map.py <dump_dir> [out_dir]
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NAME = re.compile(r"attn_(pos|neg)_t(\d+)_l(\d+)_h(\d+)\.f32$")


def load(dirpath):
    meta = {}
    mp = os.path.join(dirpath, "attn_meta.txt")
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8"):
            for kv in line.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    meta[k] = v
    T = int(meta.get("T", "1000"))
    mats = {}
    for fn in sorted(os.listdir(dirpath)):
        m = NAME.search(fn)
        if not m:
            continue
        br, t, li, h = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        a = np.fromfile(os.path.join(dirpath, fn), dtype=np.float32)
        if a.size != T * T:
            print("skip %s: %d floats != T^2" % (fn, a.size))
            continue
        mats[(br, t, li, h)] = a.reshape(T, T)      # [k, q]
    return T, int(meta.get("prompt_T", meta.get("P", "500"))), mats


def block_sum(a, B):
    """[k, q] -> [ceil(T/B), ceil(T/B)] sums of a BxB block grid."""
    T = a.shape[0]
    nb = (T + B - 1) // B
    pad = nb * B - T
    b = np.pad(a, ((0, pad), (0, pad))) if pad else a
    return b.reshape(nb, B, nb, B).sum(axis=(1, 3))


def topk_blocks(a, B, frac):
    """Per query block: how many of the 16 key blocks are needed for `frac` of the mass."""
    bs = block_sum(a, B)                       # [kb, qb]
    nb = bs.shape[1]
    k = max(1, int(round(frac * nb)))
    need = []
    for qb in range(nb):
        col = bs[:, qb]
        order = np.argsort(-col)
        c = np.cumsum(col[order])
        tot = c[-1]
        if tot <= 0:
            need.append(nb)
            continue
        need.append(int(np.searchsorted(c, frac * tot) + 1))
    return np.array(need), bs


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else d
    os.makedirs(out, exist_ok=True)
    T, P, mats = load(d)
    G = T - P
    print("T=%d  P(prompt)=%d  gen=%d  matrices=%d" % (T, P, G, len(mats)))
    if not mats:
        return 1

    keys = sorted(mats)
    report = []

    def say(s):
        report.append(s)

    # ---------- 1. 基本统计: 行归一 / 象限质量占比 ----------
    say("== 1. 行归一与象限质量占比 (q: 查询, k: 键) ==")
    say("分支 层 头  t     | 行和[min,max]   gen行->ref键  gen行->gen键  ref行->ref键  "
        "| gen×gen 对角带 (w=16/32/64/128)")
    for key in keys:
        br, t, li, h = key
        a = mats[key]
        rs = a.sum(axis=1)
        gen_ref = a[:P, P:].sum(axis=0) / np.maximum(a[:, P:].sum(axis=0), 1e-30)
        gen_gen = 1.0 - gen_ref
        ref_ref = a[:P, :P].sum(axis=0) / np.maximum(a[:, :P].sum(axis=0), 1e-30)
        sub = a[P:, P:]                                   # gen x gen
        diag = []
        for w in (16, 32, 64, 128):
            ii = np.arange(G)
            lo = np.maximum(0, ii - w)
            hi = np.minimum(G, ii + w + 1)
            m = 0.0
            for q in range(G):
                m += sub[lo[q]:hi[q], q].sum()
            diag.append(m / max(sub.sum(), 1e-30))
        say("%s l%02d h%02d t=%.2f | [%.4f,%.4f]  %.3f           %.3f          %.3f"
            "         | %.3f %.3f %.3f %.3f"
            % (br, li, h, t / 1000.0, rs.min(), rs.max(), gen_ref.mean(), gen_gen.mean(),
               ref_ref.mean(), diag[0], diag[1], diag[2], diag[3]))

    # ---------- 2. 块级集中度: 达到 90%/95% 质量需要多少个块 ----------
    say("")
    say("== 2. 块级集中度 (B=32 块 => 参考段16块/生成段16块; 每行需要多少个块才到 90%/95%) ==")
    for key in keys:
        br, t, li, h = key
        a = mats[key]
        q90, bs = topk_blocks(a, 32, 0.90)
        q95, _ = topk_blocks(a, 32, 0.95)
        say("%s l%02d h%02d t=%.2f | 90%%: 均值 %.2f 块 (max %d) | 95%%: 均值 %.2f 块 | 全矩阵 %.1f%% 的块占 99%% 质量"
            % (br, li, h, t / 1000.0, q90.mean(), q90.max(), q95.mean(),
               100.0 * margh_mass(bs, 0.99)))

    # ---------- 3. 参考段键剖面 (哪些参考帧被用) ----------
    say("")
    say("== 3. 参考段键剖面 (gen 查询在 k<P 上的质量, 归一) ==")
    for key in keys:
        br, t, li, h = key
        if br != "pos" or t != 0:
            continue
        a = mats[key]
        prof = a[:P, P:].sum(axis=1)
        prof = prof / max(prof.sum(), 1e-30)
        q = np.array_split(prof, 10)
        say("l%02d h%02d 十个等分段的参考段质量: %s" % (li, h, " ".join("%.3f" % x.sum() for x in q)))

    # ---------- 4. 跨步/跨层/跨头稳定性 (块级 top-k 集合的重合度) ----------
    say("")
    say("== 4. 选择稳定性: 块级(16x16) top-25% 块的 Jaccard 重合度 ==")
    def sel_mask(a, B=64, frac=0.25):
        bs = block_sum(a, B)
        thr = np.quantile(bs, 1.0 - frac)
        return bs >= thr
    for li in sorted({k[2] for k in keys}):
        for h in sorted({k[3] for k in keys}):
            m0 = mats.get(("pos", 0, li, h))
            m5 = mats.get(("pos", 500, li, h))
            if m0 is None or m5 is None:
                continue
            s0, s5 = sel_mask(m0), sel_mask(m5)
            j = np.logical_and(s0, s5).sum() / max(np.logical_or(s0, s5).sum(), 1)
            say("l%02d h%02d  pos t=0 vs t=0.5 : Jaccard %.3f" % (li, h, j))
    for h in sorted({k[3] for k in keys}):
        ms = [mats[("pos", 0, li, h)] for li in sorted({k[2] for k in keys}) if ("pos", 0, li, h) in mats]
        if len(ms) >= 2:
            s = [sel_mask(m) for m in ms]
            j = np.logical_and(s[0], s[-1]).sum() / max(np.logical_or(s[0], s[-1]).sum(), 1)
            say("层 l%02d vs l%02d (h%02d, pos t=0): Jaccard %.3f" % (
                sorted({k[2] for k in keys})[0], sorted({k[2] for k in keys})[-1], h, j))

    # ---------- 5. 图 ----------
    # 5a 全图 log 热力 (pos t=0, 每层/头一行)
    sel = [k for k in keys if k[0] == "pos" and k[1] == 0]
    sel.sort(key=lambda k: (k[2], k[3]))
    if sel:
        fig, axes = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 3.4))
        if len(sel) == 1:
            axes = [axes]
        for ax, k in zip(axes, sel):
            a = mats[k]
            small = a.reshape(100, T // 100, 100, T // 100).mean(axis=(1, 3)) if T % 100 == 0 else a
            im = ax.imshow(small, origin="lower", cmap="magma", norm=LogNorm(vmin=1e-6, vmax=1),
                           extent=[0, T, 0, T])
            ax.axhline(P, color="cyan", lw=0.6)
            ax.axvline(P, color="cyan", lw=0.6)
            ax.set_title("l%02d h%02d pos t=0" % (k[2], k[3]), fontsize=8)
            ax.set_xlabel("query frame", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel("key frame", fontsize=7)
        fig.colorbar(im, ax=axes, shrink=0.8, label="attention prob (log)")
        fig.savefig(os.path.join(out, "map_full_log.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5b 块级(32)质量图, gen 查询 vs 键块
    sel = [k for k in keys if k[0] == "pos" and k[1] == 0]
    sel.sort(key=lambda k: (k[2], k[3]))
    if sel:
        fig, axes = plt.subplots(1, len(sel), figsize=(3.1 * len(sel), 3.4))
        if len(sel) == 1:
            axes = [axes]
        B = 32
        for ax, k in zip(axes, sel):
            bs = block_sum(mats[k], B)
            keep = bs[: P // B, P // B:]                       # ref keys x gen queries
            im = ax.imshow(keep.T, origin="lower", cmap="viridis", aspect="auto")
            ax.set_title("gen query -> ref block mass\nl%02d h%02d" % (k[2], k[3]), fontsize=8)
            ax.set_xlabel("ref block (32 frames)", fontsize=7)
            ax.set_ylabel("gen query block", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=axes, shrink=0.8, label="block mass")
        fig.savefig(os.path.join(out, "map_gen2ref_blocks.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5c 对角带: gen×gen 每行质量 vs |k-q|
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for k in [k for k in keys if k[0] == "pos" and k[1] == 0][:4]:
        a = mats[k]
        sub = a[P:, P:]
        prof = np.zeros(201)
        for dlt in range(0, 201):
            ii = np.arange(0, G - dlt)
            v = sub[ii + dlt, ii].sum()
            prof[dlt] = v
        prof = prof / prof.sum()
        axes[0].plot(np.arange(201), np.cumsum(prof), label="l%02d h%02d" % (k[2], k[3]), lw=1.2)
        axes[1].plot(np.arange(1, 201), prof[1:], label="l%02d h%02d" % (k[2], k[3]), lw=1.0)
    axes[0].set_xlabel("|k-q| (frames)")
    axes[0].set_ylabel("cumulative mass in gen×gen")
    axes[1].set_xlabel("|k-q| (frames)")
    axes[1].set_ylabel("mass at offset")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("pos t=0, generated-frame band structure", fontsize=9)
    fig.savefig(os.path.join(out, "band_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5d pos vs neg 对照
    pairs = [(k, ("neg", k[1], k[2], k[3])) for k in keys if k[0] == "pos"]
    pairs = [p for p in pairs if p[1] in mats]
    if pairs:
        n = min(len(pairs), 4)
        fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
        for i, (kpos, kneg) in enumerate(pairs[:n]):
            for row, k in enumerate((kpos, kneg)):
                a = mats[k]
                small = a.reshape(100, T // 100, 100, T // 100).mean(axis=(1, 3)) if T % 100 == 0 else a
                ax = axes[row, i]
                ax.imshow(small, origin="lower", cmap="magma", norm=LogNorm(vmin=1e-6, vmax=1),
                          extent=[0, T, 0, T])
                ax.axhline(P, color="cyan", lw=0.6)
                ax.axvline(P, color="cyan", lw=0.6)
                ax.set_title("%s l%02d h%02d t=%.1f" % (k[0], k[2], k[3], k[1] / 1000.0), fontsize=8)
                ax.tick_params(labelsize=6)
        fig.suptitle("cond (top) vs uncond (bottom)", fontsize=9)
        fig.savefig(os.path.join(out, "map_pos_vs_neg.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    txt = "\n".join(report)
    open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def margh_mass(bs, frac):
    """Fraction of blocks that hold `frac` of the total mass (whole matrix)."""
    flat = np.sort(bs.ravel())[::-1]
    c = np.cumsum(flat) / max(flat.sum(), 1e-30)
    return (np.searchsorted(c, frac) + 1) / flat.size


if __name__ == "__main__":
    sys.exit(main())
