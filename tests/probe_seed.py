"""Probe whether the seed (3047) or skip_first_step causes the reported degradation.

The user reported: after typing seed 3047, generation repeats the prompt text and loses the
timbre. Two candidate causes: AR seed sensitivity, or skip_first_step (which skips the first
DiT network call entirely). Neither is a frontend construction bug per se: the GUI sends
seed as an int and skip_first_step as 0/1 over a plain UTF-8 JSON body.

Reported per run: output duration, AR token count if available, mel-level spectrum stats and
the low-frequency share. Repetition shows up as an abnormally long output for the same text.
"""
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9952

TEXT = ('所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，'
        '你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！'
        '那就是他们已经亮出了石头剪刀布了，你再出招儿')

# English VCTK reference with its transcript: the pair behaves consistently in this engine.
REF = 'H:/downloads/LJ037-0171.wav'
PROMPT = '.e Commission to conclude that five moths may have been fired.'


def tts(seed, skip, steps=32, dbcache='auto'):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'en',
        'seed': seed, 'sample_steps': steps, 'dbcache': dbcache,
        'skip_first_step': skip,
        'text_split_method': 'cut4', 'fragment_interval': 0.3,
        'dit_uncond_reuse': 0,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def info(b):
    w = wave.open(io.BytesIO(b), 'rb')
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
    sr = w.getframerate()
    dur = len(x) / sr
    m = min(len(x), 262144)
    X = np.abs(np.fft.rfft(x[:m]))
    f = np.fft.rfftfreq(m, 1.0 / sr)
    tot = (X ** 2).sum() + 1e-9
    lf = 100.0 * (X[f < 100] ** 2).sum() / tot
    return dur, x, lf


def main():
    print('text=%d chars, ref=%s' % (len(TEXT), os.path.basename(REF)))
    print('%-8s %-6s %8s %10s %10s %8s' % ('seed', 'skip', 'sec', 'min', 'max', 'LF%'))
    rows = []
    for skip in (0, 1):
        for seed in (3047, 1, 42):
            try:
                dur, x, lf = info(tts(seed, skip))
                print('%-8s %-6d %8.2f %10.0f %10.0f %8.1f'
                      % (seed, skip, dur, x.min(), x.max(), lf))
                rows.append((seed, skip, dur))
            except Exception as e:
                print('%-8s %-6d  ERR %s' % (seed, skip, str(e)[:70]))
    # 同一 seed 再跑一次确认确定性
    print('\nrepeatability (seed=3047, skip=0):')
    d1, x1, _ = info(tts(3047, 0))
    d2, x2, _ = info(tts(3047, 0))
    n = min(len(x1), len(x2))
    print('  run1=%.2fs run2=%.2fs  bit-identical=%s'
          % (d1, d2, np.array_equal(x1[:n], x2[:n])))

    if rows:
        durs = [r[2] for r in rows]
        print('\nduration spread: min=%.2fs max=%.2fs ratio=%.2fx'
              % (min(durs), max(durs), max(durs) / max(min(durs), 1e-9)))
        s3047 = [r[2] for r in rows if r[0] == 3047]
        others = [r[2] for r in rows if r[0] != 3047]
        if s3047 and others:
            print('seed 3047 avg=%.2fs, other seeds avg=%.2fs'
                  % (np.mean(s3047), np.mean(others)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
