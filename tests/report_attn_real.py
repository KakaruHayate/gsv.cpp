"""Aggregate the cross-input real-render attention scan (12 renders) and compare with the fixture.

    python tests/report_attn_real.py <realinputs_root>
"""
import os
import sys
from collections import defaultdict

import numpy as np

NLAYER, NHEAD = 22, 16


def load(p):
    hdr = open(p, encoding="utf-8").readline().strip().split(",")
    d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=object)
    out = {}
    for i, k in enumerate(hdr):
        out[k] = d[:, i].astype(str) if k == "branch" else d[:, i].astype(float)
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"J:\GPT-SoVITS\_attn\realinputs"
    renders = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "attn_summary.csv")
        if os.path.exists(p) and os.path.getsize(p) > 100:
            renders.append((name, load(p)))
    print("renders with dumps: %d\n" % len(renders))

    # ---- 每种 (T,P) 形状的规模 ----
    shapes = defaultdict(int)
    for name, d in renders:
        for T, P in zip(np.unique(d["T"]), np.unique(d["P"])):
            pass
        for T, P in set(zip(d["T"].astype(int), d["P"].astype(int))):
            shapes[(T, P)] += 1
    print("== 1. 真实渲染里的 (T, P) 形状 ==")
    print("   %s" % "  ".join("T%d/P%d(x%d)" % (t, p, c) for (t, p), c in sorted(shapes.items())))
    pt = [p / t for (t, p) in shapes]
    print("   P/T 比例: %.2f–%.2f (均值 %.2f)  <- 前缀 KV 能省的行数比例" %
          (min(pt), max(pt), float(np.mean(pt))))

    # ---- 全量统计 (pos, t=0, 每层全头均值) ----
    def collect(branch="pos", t=0.0, voice=None):
        rows = []
        for name, d in renders:
            if voice and not name.startswith(voice):
                continue
            m = (d["branch"] == branch) & (np.abs(d["t"] - t) < 1e-6)
            if not m.any():
                continue
            T = int(d["T"][m][0])
            nb = int(-(-T // 32))
            rows.append((name, T, int(d["P"][m][0]), d["layer"][m].astype(int),
                         d["head"][m].astype(int), d["b90"][m] / nb, d["ent_ref"][m],
                         d["band16"][m], d["gen_ref"][m], d["maxblk32"][m]))
        return rows

    rows = collect()
    print("\n== 2. 头型/稀疏度: 12 次真实渲染合并 (pos, t=0; b90 已按各自 nb 归一) ==")
    allb90 = np.concatenate([r[5] for r in rows])
    allent = np.concatenate([r[6] for r in rows])
    allband = np.concatenate([r[7] for r in rows])
    allgr = np.concatenate([r[8] for r in rows])
    print("   b90/nb      中位 %.1f%%   均值 %.1f%%   -> 平均可丢 %.1f%%" %
          (100 * np.median(allb90), 100 * allb90.mean(), 100 * (1 - allb90.mean())))
    print("   ent_ref     中位 %.3f   均值 %.3f   (1 = 参考段完全均匀)" %
          (np.median(allent), allent.mean()))
    print("   band16      中位 %.3f   均值 %.3f" % (np.median(allband), allband.mean()))
    print("   gen_ref     中位 %.3f   均值 %.3f   (生成查询落在参考键上的质量)" %
          (np.median(allgr), allgr.mean()))
    print("   均匀头(ent>0.93 且 b90/nb>0.75): %.0f%%  稀疏头(b90/nb<=0.25): %.0f%%" %
          (100 * np.mean((allent > 0.93) & (allb90 > 0.75)), 100 * np.mean(allb90 <= 0.25)))

    # ---- 分层剖面 (跨渲染 mean ± std) ----
    print("\n== 3. 分层剖面 (跨 12 个真实输入的 层均值 ± 跨输入标准差) ==")
    print("   层   b90/nb         ent_ref          band16")
    per_layer = defaultdict(list)
    for name, T, P, li, h, b90, ent, band, gr, mx in rows:
        for x, y, z, l in zip(b90, ent, band, li):
            per_layer[l].append((x, y, z))
    ramp = []
    for l in range(NLAYER):
        a = np.array(per_layer[l])
        ramp.append(a.mean(axis=0))
        print("   %2d   %5.1f%% ±%4.1f   %.3f ±%.3f   %.3f ±%.3f" %
              (l, 100 * a[:, 0].mean(), 100 * a[:, 0].std(),
               a[:, 1].mean(), a[:, 1].std(), a[:, 2].mean(), a[:, 2].std()))
    ramp = np.array(ramp)
    early, late = ramp[:13], ramp[13:]
    print("   浅层 0-12: b90 %.1f%% / ent %.3f   深层 13-21: b90 %.1f%% / ent %.3f" %
          (100 * early[:, 0].mean(), early[:, 1].mean(), 100 * late[:, 0].mean(), late[:, 1].mean()))
    print("   单调性: 层0→层21 b90 是否单调下降? %s   相关系数 %.3f" %
          (bool(np.all(np.diff(ramp[:, 0]) < 0.05)), float(np.corrcoef(np.arange(NLAYER), ramp[:, 0])[0, 1])))

    # ---- 声音/分支的一致性 ----
    print("\n== 4. 一致性检查 ==")
    for br in ("pos", "neg"):
        for t in (0.0, 0.5):
            r = collect(br, t)
            if not r:
                continue
            b = np.concatenate([x[5] for x in r])
            e = np.concatenate([x[6] for x in r])
            print("   %s t=%.1f : b90/nb %.1f%%  ent_ref %.3f  (n=%d 头×渲染)" %
                  (br, t, 100 * b.mean(), e.mean(), b.size))
    for v in ("v5dev_", "default_"):
        r = collect("pos", 0.0, v)
        if r:
            b = np.concatenate([x[5] for x in r])
            e = np.concatenate([x[6] for x in r])
            print("   %-8s: b90/nb %.1f%%  ent_ref %.3f" % (v.rstrip("_"), 100 * b.mean(), e.mean()))

    # ---- 与 fixture 对照 ----
    fx = os.path.join(os.path.dirname(root), "scan", "s32c13_big", "attn_summary.csv")
    if os.path.exists(fx):
        d = load(fx)
        m = (d["branch"] == "pos") & (np.abs(d["t"]) < 1e-6)
        nb = int(-(-1000 // 32))
        print("\n== 5. 与 fixture (s32c13_big, T=1000/P=500) 对照 ==")
        print("   fixture : b90/nb %.1f%%  ent_ref %.3f  band16 %.3f" %
              (100 * (d["b90"][m] / nb).mean(), d["ent_ref"][m].mean(), d["band16"][m].mean()))
        print("   真实输入: b90/nb %.1f%%  ent_ref %.3f  band16 %.3f" %
              (100 * allb90.mean(), allent.mean(), allband.mean()))


if __name__ == "__main__":
    main()
