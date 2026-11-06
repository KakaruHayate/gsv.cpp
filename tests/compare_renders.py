"""Compare the two renders the user saved 37 s apart with identical settings.

If the second one echoes the reference or repeats itself, that shows up as (a) much longer
duration than the text justifies, (b) the reference waveform correlating inside the output,
(c) the output correlating with itself at some lag.
"""
import os
import sys
import wave

import numpy as np

A = r'J:\こんにちは音声合成シ_20261106_205550.wav'
B = r'J:\こんにちは音声合成シ_20261106_205627.wav'
JOBS = [('first  (20:55:50)', A), ('second (20:56:27)', B)]


def rd(p):
    w = wave.open(p, 'rb')
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
    if w.getnchannels() > 1:
        x = x.reshape(-1, w.getnchannels()).mean(axis=1)
    return x, w.getframerate()


def flat(x, sr):
    """Energy and zero-crossing overview in 1 s blocks, to see where things go wrong."""
    n = sr
    out = []
    for i in range(0, len(x) - n + 1, n):
        seg = x[i:i + n]
        out.append(np.sqrt((seg ** 2).mean()))
    return out


def selflag(x, sr, min_s=0.4, max_s=8.0):
    """Best normalised autocorrelation peak (excluding zero lag) -> repeating period."""
    x = x - x.mean()
    n = len(x)
    best = (0.0, 0.0)
    for lag in range(int(min_s * sr), min(int(max_s * sr), n // 2), max(1, sr // 200)):
        a = x[:n - lag]
        b = x[lag:]
        d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
        v = float((a * b).sum() / d)
        if v > best[0]:
            best = (v, lag / sr)
    return best


def xcorr(ref, out, hop=240):
    r = ref - ref.mean()
    rn = np.sqrt((r * r).sum()) + 1e-9
    L = len(r)
    best = (0.0, 0.0)
    for s in range(0, max(len(out) - L, 1), hop):
        seg = out[s:s + L]
        seg = seg - seg.mean()
        v = float((r * seg).sum() / (rn * (np.sqrt((seg * seg).sum()) + 1e-9)))
        if v > best[0]:
            best = (v, s / 48000.0)
    return best


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else None
    print('%-18s %8s %10s %10s %10s %8s' % ('file', 'sec', 'min', 'max', 'self-corr', 'lag'))
    data = {}
    for label, p in JOBS:
        if not os.path.exists(p):
            print('%-18s MISSING %s' % (label, p))
            continue
        x, sr = rd(p)
        data[label] = (x, sr)
        c, lag = selflag(x, sr)
        print('%-18s %8.2f %10.0f %10.0f %10.3f %7.2fs' % (label, len(x) / sr, x.min(), x.max(), c, lag))

    if len(data) == 2:
        (la, (xa, sra)), (lb, (xb, srb)) = list(data.items())
        n = min(len(xa), len(xb))
        cc = float(np.corrcoef(xa[:n], xb[:n])[0, 1])
        print('\nfirst vs second (first %d s): corr=%.4f' % (n // sra, cc))
        ea, eb = flat(xa, sra), flat(xb, srb)
        print('\nper-second RMS (first 20 s):')
        print('  first : ' + ' '.join('%.0f' % v for v in ea[:20]))
        print('  second: ' + ' '.join('%.0f' % v for v in eb[:20]))

    if ref_path and os.path.exists(ref_path):
        r, srr = rd(ref_path)
        print('\nreference %s: %.2fs' % (os.path.basename(ref_path), len(r) / srr))
        for label, (x, sr) in data.items():
            v, at = xcorr(r, x)
            print('  %-18s max corr vs reference = %.3f at %.1fs' % (label, v, at))
    return 0


if __name__ == '__main__':
    sys.exit(main())
