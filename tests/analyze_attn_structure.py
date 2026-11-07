"""Second pass: is the reference-segment attention *structured* (a few blocks) or *uniform*
(redundant)?  And are the generation-segment rows local (diagonal band) or driven by sinks?

    python tests/analyze_attn_structure.py <dump_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attn_map import load, block_sum          # noqa: E402


def main():
    d = sys.argv[1]
    T, P, mats = load(d)
    G = T - P
    print("T=%d P=%d gen=%d matrices=%d\n" % (T, P, G, len(mats)))

    print("== A. 参考段注意力是不是'几乎均匀'? ==")
    print("   每行 ref 剖面(500 维) 与 全体平均剖面的 相对 L1 距离; 以及剖面熵 (归一化到 [0,1], 1=完全均匀)")
    print("   %-22s %10s %10s %10s" % ("matrix", "relL1(row,mean)", "entropy", "std/mean"))
    for key in sorted(mats):
        br, t, li, h = key
        if br != "pos" or t != 0:
            continue
        a = mats[key]
        R = a[:P, P:]                      # [k<P, q>=P]
        prof = R.T                         # 每行一个 ref 剖面
        m = prof.mean(axis=0)
        rel = np.abs(prof - m).sum(axis=1) / np.maximum(np.abs(prof).sum(axis=1), 1e-30)
        p = prof / np.maximum(prof.sum(axis=1, keepdims=True), 1e-30)
        ent = -(p * np.log(np.maximum(p, 1e-30))).sum(axis=1) / np.log(P)
        print("   l%02d h%02d t=0          %10.3f %10.3f %10.3f"
              % (li, h, rel.mean(), ent.mean(), (prof.std(axis=0) / np.maximum(m, 1e-30)).mean()))

    print("\n== B. 生成段: 每行的质量分布 (是否局部? 有没有'注意力汇') ==")
    print("   %-16s %8s %8s %8s %8s | %s" % ("matrix", "|k-q|<=16", "<=64", "<=128",
                                             "首3帧", "全局 top-3 key (占总质量)"))
    for key in sorted(mats):
        br, t, li, h = key
        if br != "pos" or t != 0:
            continue
        a = mats[key]
        S = a[P:, P:]                      # gen x gen
        rows = S.sum(axis=1)
        near = {}
        for w in (16, 64, 128):
            tot = 0.0
            for q in range(G):
                lo, hi = max(0, q - w), min(G, q + w + 1)
                tot += S[lo:hi, q].sum()
            near[w] = tot / S.sum()
        first = 0.0
        for q in range(G):
            first += S[max(0, q - 3):q + 1, q].sum()
        colmass = a.sum(axis=1)
        top = np.argsort(-colmass)[:3]
        print("   l%02d h%02d t=0     %8.3f %8.3f %8.3f %8.3f | %s"
              % (li, h, near[16], near[64], near[128], first / S.sum(),
                 ", ".join("k=%d(%.2f%%)" % (i, 100.0 * colmass[i] / colmass.sum()) for i in top)))

    print("\n== C. 块级(32)质量分布: 每行最大块占比 / 达到 90% 需要几块 ==")
    print("   %-16s %10s %10s %10s" % ("matrix", "max block", "blocks@90%", "blocks@99%"))
    for key in sorted(mats):
        br, t, li, h = key
        if br != "pos" or t != 0:
            continue
        a = mats[key]
        bs = block_sum(a, 32)              # [31, 31]
        nb = bs.shape[0]
        mx, need90, need99 = [], [], []
        for q in range(nb):
            col = bs[:, q]
            tot = col.sum()
            if tot <= 0:
                continue
            mx.append(col.max() / tot)
            c = np.cumsum(np.sort(col)[::-1]) / tot
            need90.append(int(np.searchsorted(c, 0.90) + 1))
            need99.append(int(np.searchsorted(c, 0.99) + 1))
        print("   l%02d h%02d t=0     %10.3f %10.1f %10.1f"
              % (li, h, np.mean(mx), np.mean(need90), np.mean(need99)))

    print("\n== D. 跨步稳定性 (整矩阵相关, t=0 vs t=0.5, pos) ==")
    for li in sorted({k[2] for k in mats}):
        for h in sorted({k[3] for k in mats}):
            a0 = mats.get(("pos", 0, li, h))
            a5 = mats.get(("pos", 500, li, h))
            if a0 is None or a5 is None:
                continue
            x = a0.ravel()
            y = a5.ravel()
            c = np.corrcoef(x, y)[0, 1]
            # 参考段剖面的稳定性 (生成行)
            R0 = a0[:P, P:].sum(axis=1)
            R5 = a5[:P, P:].sum(axis=1)
            cR = np.corrcoef(R0, R5)[0, 1]
            print("   l%02d h%02d : corr(full)=%.3f  corr(ref profile)=%.3f" % (li, h, c, cR))


if __name__ == "__main__":
    main()
