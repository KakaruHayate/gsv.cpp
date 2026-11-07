"""Aggregate the full attention scan (all 22 layers x 16 heads) into a report + figures.

    python tests/report_attn_scan.py "J:\\GPT-SoVITS\\_attn\\scan" [out_dir]

Reads <root>/<family>/attn_summary.csv (written by gsv_dit GSV_ATTN_MODE=summary) and
<root>/<family>/attn_refprof.csv.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NLAYER, NHEAD = 22, 16
NB32 = 31          # ceil(1000/32)


def load(fam_dir):
    p = os.path.join(fam_dir, "attn_summary.csv")
    hdr = open(p, encoding="utf-8").readline().strip().split(",")
    d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=object)
    out = {}
    for i, k in enumerate(hdr):
        col = d[:, i]
        if k in ("branch",):
            out[k] = col.astype(str)
        else:
            out[k] = col.astype(float)
    return out


def grid(d, key, branch="pos", t=0.0):
    g = np.full((NLAYER, NHEAD), np.nan)
    m = (d["branch"] == branch) & (np.abs(d["t"] - t) < 1e-6)
    for li, h, v in zip(d["layer"][m].astype(int), d["head"][m].astype(int), d[key][m]):
        if 0 <= li < NLAYER and 0 <= h < NHEAD:
            g[li, h] = v
    return g


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"J:\GPT-SoVITS\_attn\scan"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "figs")
    os.makedirs(out, exist_ok=True)
    fams = [f for f in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, f))
            and os.path.exists(os.path.join(root, f, "attn_summary.csv"))]
    rep = []

    def say(s=""):
        rep.append(s)
        print(s)

    say("families: " + ", ".join(fams))
    data = {f: load(os.path.join(root, f)) for f in fams}

    # ---------- 1. 每族的全局统计 ----------
    say("\n== 1. 每族汇总 (pos, t=0; 块粒度 B=32, nb = ceil(T/32); 比例 = b90/nb) ==")
    say("%-20s %5s %4s %4s %8s %9s %9s %8s %8s"
        % ("family", "T", "P", "nb", "b90/nb", "<=25%", "<=50%", "ent_ref", "band16"))
    for f in fams:
        d = data[f]
        nb = int(-(-int(d["T"][0]) // 32))
        g = grid(d, "b90")
        v = g[~np.isnan(g)] / nb
        er = grid(d, "ent_ref")
        b16 = grid(d, "band16")
        say("%-20s %5d %4d %4d %8.1f %8.0f%% %8.0f%% %8.3f %8.3f"
            % (f, d["T"][0], d["P"][0], nb, 100 * np.median(v),
               100.0 * (v <= 0.25).mean(), 100.0 * (v <= 0.5).mean(),
               np.nanmean(er), np.nanmean(b16)))

    # ---------- 2. 可稀疏化的上界 ----------
    say("\n== 2. 如果按块 top-k 丢块, 最多能丢多少 (b90/nb 作为需保留比例) ==")
    say("   family                平均可丢比例  最优头可丢比例  (a=19.4% 时 DiT 上限加速)")
    for f in fams:
        d = data[f]
        nb = int(-(-int(d["T"][0]) // 32))
        v = grid(d, "b90")
        v = v[~np.isnan(v)] / nb
        drop = 1.0 - v
        s_mean, s_best = np.nanmean(drop), np.nanmax(drop)
        say("   %-22s %9.1f%% %11.1f%%   %.2fx (均值) / %.2fx (最优)"
            % (f, 100 * s_mean, 100 * s_best,
               1.0 / (1.0 - s_mean * 0.194), 1.0 / (1.0 - s_best * 0.194)))

    # ---------- 3. 头型分布 ----------
    say("\n== 3. 头型分布 (pos t=0, 全部 352 个头) ==")
    d = data["s32c13_big"]
    b90 = grid(d, "b90")
    er = grid(d, "ent_ref")
    b16 = grid(d, "band16")
    gr = grid(d, "gen_ref")
    mx = grid(d, "maxblk32")
    nb0 = int(-(-int(d["T"][0]) // 32))
    say("   均匀头 (ent_ref>0.93 且 b90/nb>0.75):    %d / 352" % np.sum((er > 0.93) & (b90 > 0.75 * nb0)))
    say("   强局部头 (band16>0.35):                 %d / 352" % np.sum(b16 > 0.35))
    say("   集中头 (maxblk32>0.25):                 %d / 352" % np.sum(mx > 0.25))
    say("   块稀疏头 (b90/nb<=0.25, 可丢 >=75%%):     %d / 352" % np.sum(b90 <= 0.25 * nb0))
    say("   参考占比 gen_ref>0.7 的头:              %d / 352" % np.sum(gr > 0.7))
    say("   ent_ref 范围 [%.3f, %.3f], 中位 %.3f" % (np.nanmin(er), np.nanmax(er), np.nanmedian(er)))

    # ---------- 4. t 依赖与 cond/uncond ----------
    say("\n== 4. 随 t 的变化 (s32c13_big, 全头均值) / cond vs uncond ==")
    say("   t        b90      ent_ref   band16   maxblk32   | neg: b90   ent_ref")
    for t in (0.0, 0.25, 0.5, 0.75):
        row = []
        for br in ("pos", "neg"):
            g = lambda k: grid(d, k, br, t)      # noqa: E731
            row.append((np.nanmean(g("b90")), np.nanmean(g("ent_ref")),
                        np.nanmean(g("band16")), np.nanmean(g("maxblk32"))))
        say("   %.2f    %6.1f   %7.3f   %7.3f  %8.3f   | %6.1f   %7.3f"
            % (t, row[0][0], row[0][1], row[0][2], row[0][3], row[1][0], row[1][1]))

    # ---------- 5. 参考块剖面的一致性 ----------
    say("\n== 5. 参考段 16 桶剖面 (gen 查询) 的层间一致性 ==")
    p = os.path.join(root, "s32c13_big", "attn_refprof.csv")
    if os.path.exists(p):
        hdr = open(p, encoding="utf-8").readline().strip().split(",")
        d2 = np.loadtxt(p, delimiter=",", skiprows=1, dtype=object)
        br = d2[:, 0].astype(str)
        tt = d2[:, 1].astype(float)
        li = d2[:, 2].astype(int)
        prof = d2[:, 4:20].astype(float)
        m = (br == "pos") & (np.abs(tt - 0.0) < 1e-6)
        P0 = prof[m]
        L0 = li[m]
        # 与层 0 的余弦相似 / 各桶的标准差
        mean_prof = P0.mean(axis=0)
        say("   平均剖面 (16 桶): " + " ".join("%.3f" % x for x in mean_prof))
        say("   各桶跨头标准差:   " + " ".join("%.3f" % x for x in P0.std(axis=0)))
        cos = [float(np.dot(a, mean_prof) / (np.linalg.norm(a) * np.linalg.norm(mean_prof) + 1e-12))
               for a in P0]
        say("   与平均剖面的余弦: 中位 %.3f, 最小 %.3f" % (np.median(cos), np.min(cos)))

    # ---------- 图 ----------
    for key, title, fname, vmin, vmax in [
        ("b90", "blocks needed for 90% mass (of 31, B=32)", "scan_b90.png", 4, 30),
        ("ent_ref", "entropy over reference keys (1 = uniform)", "scan_entref.png", 0.6, 1.0),
        ("band16", "gen-quadrant mass within |k-q|<=16", "scan_band16.png", 0.0, 0.6),
        ("maxblk32", "largest single block share", "scan_maxblk.png", 0.0, 0.5),
    ]:
        g = grid(data["s32c13_big"], key)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        im = ax.imshow(g, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(title + "  (s32c13_big, pos, t=0)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=140)
        plt.close(fig)

    # 散点: 均匀性 vs 局部性, 点大小 = b90
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    sc = ax.scatter(er.ravel(), b16.ravel(), c=b90.ravel(), s=28, cmap="plasma")
    ax.set_xlabel("entropy over reference keys (1 = uniform)")
    ax.set_ylabel("mass within |k-q|<=16 (gen quadrant)")
    ax.set_title("head typology (352 heads, t=0, colour = b90)", fontsize=9)
    fig.colorbar(sc, ax=ax, label="blocks@90%")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scan_typology.png"), dpi=140)
    plt.close(fig)

    txt = "\n".join(rep)
    open(os.path.join(out, "scan_report.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    main()
