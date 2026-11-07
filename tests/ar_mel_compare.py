#!/usr/bin/env python3
# Compare AR/synthesis renders across quant variants: waveform correlation + log-mel correlation.
# 无 numpy 依赖 (纯 python, 512 点 radix-2 FFT) —— box 与本地都没装 numpy。
# 输入是 probe_v2sovits 写出的 ours_engine_wav.f32 (raw float32, mono, 32 kHz)。
# 用法: ar_mel_compare.py <ref.f32> <cand.f32> [...]
import cmath
import math
import struct
import sys

SR = 32000
NFFT = 512
HOP = 256
NMEL = 40
FMIN, FMAX = 80.0, 7600.0


def read_f32(p):
    with open(p, "rb") as f:
        raw = f.read()
    return struct.unpack("<%df" % (len(raw) // 4), raw)


def fft(a):
    n = len(a)
    j = 0
    a = list(a)
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    ln = 2
    while ln <= n:
        ang = -2 * math.pi / ln
        wl = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, ln):
            w = 1 + 0j
            for k in range(i, i + ln // 2):
                u = a[k]
                v = a[k + ln // 2] * w
                a[k] = u + v
                a[k + ln // 2] = u - v
                w *= wl
        ln <<= 1
    return a


def mel_fb():
    def hz2mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    m0, m1 = hz2mel(FMIN), hz2mel(FMAX)
    pts = [mel2hz(m0 + (m1 - m0) * i / (NMEL + 1)) for i in range(NMEL + 2)]
    bins = [int(math.floor((NFFT + 1) * p / SR)) for p in pts]
    fb = []
    for m in range(NMEL):
        lo, ce, hi = bins[m], bins[m + 1], bins[m + 2]
        row = [0.0] * (NFFT // 2 + 1)
        for k in range(lo, ce):
            if ce > lo:
                row[k] = (k - lo) / (ce - lo)
        for k in range(ce, hi):
            if hi > ce:
                row[k] = (hi - k) / (hi - ce)
        fb.append(row)
    return fb


def logmel(x):
    fb = mel_fb()
    win = [0.5 - 0.5 * math.cos(2 * math.pi * i / NFFT) for i in range(NFFT)]
    frames = []
    for s in range(0, max(0, len(x) - NFFT), HOP):
        seg = [x[s + i] * win[i] for i in range(NFFT)]
        sp = fft([complex(v, 0.0) for v in seg])
        pw = [abs(sp[k]) for k in range(NFFT // 2 + 1)]
        frames.append([math.log(max(sum(fb[m][k] * pw[k] for k in range(NFFT // 2 + 1)), 1e-8)) for m in range(NMEL)])
    return frames


def pearson(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def main():
    ref = read_f32(sys.argv[1])
    refmel = None
    print("ref %s: %d samples (%.2fs)" % (sys.argv[1], len(ref), len(ref) / SR))
    for p in sys.argv[2:]:
        c = read_f32(p)
        rw = pearson(ref, c)
        if refmel is None:
            refmel = logmel(ref)
        cm = logmel(c)
        nf = min(len(refmel), len(cm))
        flat_a = [v for f in refmel[:nf] for v in f]
        flat_b = [v for f in cm[:nf] for v in f]
        rm = pearson(flat_a, flat_b)
        mad = sum(abs(x - y) for x, y in zip(flat_a, flat_b)) / max(1, len(flat_a))
        print("%-22s len=%d (%.2fs) wav_r=%.5f  logmel_r=%.5f  logmel_MAE=%.4f  frames=%d" % (
            p.split("/")[-1], len(c), len(c) / SR, rw, rm, mad, nf))


if __name__ == "__main__":
    main()
