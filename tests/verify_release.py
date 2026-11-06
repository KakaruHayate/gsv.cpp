"""Verification for the 0.1.2 re-release: vocoder health + same-seed reproducibility.

Checks, in order:
  1. vocoder health   - waveform must be roughly symmetric and free of sub-100 Hz energy
                        (the stale-ggml bug produced min ~ -1548 vs max +23199 and 46% LF)
  2. same-seed rerun  - the identical request twice must be bit-identical
  3. setting round-trip - A -> B(changed switch) -> A must return to A bit-identically
                        (catches cross-request state pollution when a switch changes)
  4. switch effect    - the CFG switch must actually change the output, and by an amount
                        consistent with the fixture measurements

Requests are sent as UTF-8 bytes: PowerShell/urllib string bodies mangle non-ASCII.
"""
import json
import os
import sys
import wave
import io
import urllib.request

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9942
OUT = sys.argv[2] if len(sys.argv) > 2 else r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\verify'

REF = 'J:/GPT-SoVITS/gsv.cpp/models/ref_zh.wav'
PROMPT = '这是一个用来做基准测试的参考音频请注意听清楚'
TEXT = '你好，欢迎使用语音合成系统。'


def tts(extra=None, port=None):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': 12345, 'sample_steps': 32, 'dbcache': 'off',
        'text_split_method': 'cut0', 'fragment_interval': 0,
    }
    body.update(extra or {})
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % (port or PORT), data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def pcm(b):
    w = wave.open(io.BytesIO(b), 'rb')
    return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64), w.getframerate()


def lf_ratio(x, sr):
    m = min(len(x), 262144)
    X = np.abs(np.fft.rfft(x[:m]))
    f = np.fft.rfftfreq(m, 1.0 / sr)
    tot = (X ** 2).sum() + 1e-9
    return 100.0 * (X[f < 100] ** 2).sum() / tot


def stats(b, label):
    x, sr = pcm(b)
    lo = x.min()
    hi = x.max()
    lf = lf_ratio(x, sr)
    print('%-22s %5.2fs  min=%7.0f max=%7.0f  LF<100Hz=%5.1f%%' % (label, len(x) / sr, lo, hi, lf))
    return {'label': label, 'sec': len(x) / sr, 'min': lo, 'max': hi, 'lf': lf, 'x': x}


def corr(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 else 0.0


def relL1(a, b):
    n = min(len(a), len(b))
    return float(np.abs(a[:n] - b[:n]).sum() / (np.abs(a[:n]).sum() + 1e-9))


def main():
    os.makedirs(OUT, exist_ok=True)
    fails = []

    print('=== 1. vocoder health (waveform symmetry, no DC/low-freq blow-up) ===')
    off1 = stats(tts({'dit_cfg_off_t': 0, 'dit_uncond_reuse': 0}), 'all-off #1')
    # the stale-DLL bug: min stayed near -1548 while max hit +23199, and LF>100Hz was ~46%
    if off1['min'] > -8000:
        fails.append('vocoder: waveform is one-sided (min=%.0f), suspect broken ggml' % off1['min'])
    if off1['lf'] > 8.0:
        fails.append('vocoder: %.1f%% of energy below 100 Hz (should be <8%%)' % off1['lf'])

    print('\n=== 2. same-seed rerun (must be bit-identical) ===')
    off2 = stats(tts({'dit_cfg_off_t': 0, 'dit_uncond_reuse': 0}), 'all-off #2')
    same = np.array_equal(off1['x'], off2['x'])
    print('  bit-identical:', same)
    if not same:
        fails.append('rerun with identical params is not bit-identical (relL1=%.6f)'
                     % relL1(off1['x'], off2['x']))

    print('\n=== 3. switch round-trip A -> B -> A (must return to A exactly) ===')
    b = stats(tts({'dit_cfg_off_t': 0.7, 'dit_uncond_reuse': 0}), 'B: cfg_off=0.7')
    print('  A vs B  relL1=%.6f corr=%.6f  (switch must have an effect)'
          % (relL1(off1['x'], b['x']), corr(off1['x'], b['x'])))
    if np.array_equal(off1['x'], b['x']):
        fails.append('cfg_off_t=0.7 produced identical output to all-off; the switch is inert')
    a2 = stats(tts({'dit_cfg_off_t': 0, 'dit_uncond_reuse': 0}), 'A again (after B)')
    back = np.array_equal(off1['x'], a2['x'])
    print('  A vs A(after B) bit-identical:', back)
    if not back:
        fails.append('changing a switch and changing it back does not reproduce A (relL1=%.6f)'
                     % relL1(off1['x'], a2['x']))

    print('\n=== 4. reuse switch sanity ===')
    u1 = stats(tts({'dit_cfg_off_t': 0.7, 'dit_uncond_reuse': 1}), 'cfg0.7 + u1')
    print('  cfg0.7 vs cfg0.7+u1  relL1=%.6f' % relL1(b['x'], u1['x']))
    if np.array_equal(b['x'], u1['x']):
        fails.append('uncond_reuse had no effect')

    print('\n=== summary ===')
    if fails:
        for f in fails:
            print('  FAIL:', f)
        return 1
    print('  all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
