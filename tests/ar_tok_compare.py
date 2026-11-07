#!/usr/bin/env python3
# Compare free-running AR semantic token sequences (bench_ar --fixture --tok-out output).
# 报告: 长度, 首次分歧位置, 位置一致率 (前缀 + 全序列), 最长公共前缀。
# 用法: ar_tok_compare.py <ref.i32> <cand.i32> [...]
import struct
import sys


def load(p):
    with open(p, "rb") as f:
        raw = f.read()
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


def main():
    ref = load(sys.argv[1])
    print("ref %s: %d tokens" % (sys.argv[1], len(ref)))
    for p in sys.argv[2:]:
        c = load(p)
        n = min(len(ref), len(c))
        pref = 0
        while pref < n and ref[pref] == c[pref]:
            pref += 1
        same = sum(1 for i in range(n) if ref[i] == c[i])
        print("%-20s len=%d  common_prefix=%d/%d  positional_match=%d/%d (%.1f%%)" % (
            p.split("/")[-1], len(c), pref, n, same, n, 100.0 * same / max(1, n)))


if __name__ == "__main__":
    main()
