#!/usr/bin/env python3
# Compare AR decode probability traces (GSV_AR_PROB_DUMP output) between quant variants.
# 无 numpy 依赖 (box 上没装), 用 array 模块。
# 用法: ar_prob_compare.py <ref.bin> <cand.bin> [...]
# 每步 vocab=1025 floats; 输出: argmax 一致率 / top5 集合重合 / max|dp| / mean|dp| / KL(ref||cand)
import math
import struct
import sys

V = 1025


def load(p):
    with open(p, "rb") as f:
        raw = f.read()
    assert len(raw) % (4 * V) == 0, (p, len(raw))
    steps = len(raw) // (4 * V)
    vals = struct.unpack("<%df" % (len(raw) // 4), raw)
    return [vals[i * V:(i + 1) * V] for i in range(steps)]


def compare(ref, cand, name):
    n = min(len(ref), len(cand))
    am = 0
    top5 = 0
    maxd = 0.0
    sumd = 0.0
    kl = 0.0
    worst = None
    for i in range(n):
        a, b = ref[i], cand[i]
        ia = max(range(V), key=a.__getitem__)
        ib = max(range(V), key=b.__getitem__)
        if ia == ib:
            am += 1
        elif worst is None:
            worst = (i, ia, a[ia], ib, b[ib])
        sa = set(sorted(range(V), key=a.__getitem__, reverse=True)[:5])
        sb = set(sorted(range(V), key=b.__getitem__, reverse=True)[:5])
        if sa == sb:
            top5 += 1
        for x, y in zip(a, b):
            d = abs(x - y)
            if d > maxd:
                maxd = d
            sumd += d
            if x > 1e-9 and y > 1e-12:
                kl += x * math.log(x / y)
    print("%-14s steps=%d argmax=%5.2f%% top5_set=%5.2f%% max|dp|=%.2e mean|dp|=%.2e KL=%.5f%s" % (
        name, n, 100.0 * am / n, 100.0 * top5 / n, maxd, sumd / (n * V), kl / n,
        "" if worst is None else "  first_diff@%d ref=%d(%.4f) cand=%d(%.4f)" % worst))


def main():
    ref = load(sys.argv[1])
    print("ref %s: %d steps" % (sys.argv[1], len(ref)))
    for p in sys.argv[2:]:
        compare(ref, load(p), p.split("/")[-1])


if __name__ == "__main__":
    main()
