"""Is the degradation a bad AR seed? Measure repetition directly on the AR token stream.

The AR (t2s) output is dumped as sem.i32 by the engine (GSV_ENGINE_DUMP). Autoregressive
models can lock into a loop for particular seeds: the token stream then repeats an n-gram
many times, which is heard as "the prompt text repeating" and drags the timbre with it.

For each seed we report the token count and how much of the stream is a repeated n-gram.
The reference is fixed, so the only variable is the seed.
"""
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9953
DUMP = sys.argv[2] if len(sys.argv) > 2 else r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\seedsem\dump'

TEXT = ('所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，'
        '你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！'
        '那就是他们已经亮出了石头剪刀布了，你再出招儿')
REF = 'H:/downloads/LJ037-0171.wav'
PROMPT = '.e Commission to conclude that five moths may have been fired.'
SEEDS = [3047, 1, 42, 2, 3, 7, 12345]


def tts(seed):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'en',
        'seed': seed, 'sample_steps': 32, 'dbcache': 'off',
        'text_split_method': 'cut0', 'fragment_interval': 0,
        'dit_cfg_off_t': 0.0, 'dit_uncond_reuse': 0, 'skip_first_step': 0,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def repetition(sem, n=8):
    """Largest multiplicity of any n-gram, and the share of tokens inside repeated n-grams."""
    if len(sem) < n * 2:
        return 1, 0.0
    counts = {}
    for i in range(len(sem) - n + 1):
        k = tuple(sem[i:i + n].tolist())
        counts[k] = counts.get(k, 0) + 1
    top = max(counts.values()) if counts else 1
    dup_tokens = sum((c - 1) * n for c in counts.values() if c > 1)
    return top, dup_tokens / max(len(sem), 1)


def main():
    print('text=%d chars, ref=%s' % (len(TEXT), os.path.basename(REF)))
    sem_path = os.path.join(DUMP, 'sem.i32')
    print('%-8s %8s %10s %14s %8s' % ('seed', 'AR tokens', 'max ngram', 'dup share', 'sec'))
    rows = []
    for seed in SEEDS:
        if os.path.exists(sem_path):
            os.remove(sem_path)
        try:
            wav = tts(seed)
        except Exception as e:
            print('%-8d  ERR %s' % (seed, str(e)[:60]))
            continue
        w = wave.open(io.BytesIO(wav), 'rb')
        sec = w.getnframes() / w.getframerate()
        if not os.path.exists(sem_path):
            print('%-8d %8s %10s %14s %8.2f  (no dump)' % (seed, '?', '?', '?', sec))
            continue
        sem = np.fromfile(sem_path, dtype='<i4')
        top, dup = repetition(sem)
        rows.append((seed, len(sem), top, dup, sec))
        print('%-8d %8d %10d %13.1f%% %8.2f' % (seed, len(sem), top, dup * 100, sec))

    if len(rows) >= 2:
        tops = [r[2] for r in rows]
        print('\nmax-ngram multiplicity: min=%d max=%d' % (min(tops), max(tops)))
        worst = max(rows, key=lambda r: r[2])
        print('worst seed: %d (repeat x%d, %.1f%% of tokens in repeats)'
              % (worst[0], worst[2], worst[3] * 100))
    return 0


if __name__ == '__main__':
    sys.exit(main())
