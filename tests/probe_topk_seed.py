"""Does the user's sampler setting (top_k=5) make some seeds loop?

The GUI sends top_k=5, top_p=1.0, temperature=1.0, repetition_penalty=1.35 while the engine
default is top_k=15. A very peaked sampler raises the chance that an autoregressive model
falls into a loop, which is heard as "the prompt repeating" and drags the timbre along.

Sweeps top_k x seed on the reported text with a matched Chinese reference and the user's
text_split_method=cut4, and reports AR token repetition plus output length.
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
ZH_REF = r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\seedsem\ref_zh_matched.wav'
ZH_REF_TEXT = '你好，欢迎使用语音合成系统。'
USER_TEXT = ('所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，'
             '你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！'
             '那就是他们已经亮出了石头剪刀布了，你再出招儿')


def tts(seed, top_k):
    body = {
        'text': USER_TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': ZH_REF.replace('\\', '/'), 'prompt_text': ZH_REF_TEXT,
        'prompt_lang': 'zh', 'seed': seed, 'sample_steps': 32, 'dbcache': 'off',
        'top_k': top_k, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
        'text_split_method': 'cut4', 'fragment_interval': 0.3,
        'dit_cfg_off_t': 0.0, 'dit_uncond_reuse': 0, 'skip_first_step': 0,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def repetition(sem, n=8):
    if len(sem) < n * 2:
        return 1, 0.0
    counts = {}
    for i in range(len(sem) - n + 1):
        k = tuple(sem[i:i + n].tolist())
        counts[k] = counts.get(k, 0) + 1
    top = max(counts.values()) if counts else 1
    dup = sum((c - 1) * n for c in counts.values() if c > 1)
    return top, dup / max(len(sem), 1)


def main():
    sem_path = os.path.join(DUMP, 'sem.i32')
    if not os.path.exists(ZH_REF):
        print('missing reference:', ZH_REF)
        return 1
    print('%-6s %-6s %9s %10s %9s %7s' % ('top_k', 'seed', 'AR tok', 'max ngram', 'dup', 'sec'))
    worst = []
    for top_k in (5, 15):
        for seed in (3047, 1, 42):
            if os.path.exists(sem_path):
                os.remove(sem_path)
            wav = tts(seed, top_k)
            sec = wave.open(io.BytesIO(wav), 'rb').getnframes() / 48000.0
            if not os.path.exists(sem_path):
                print('%-6d %-6d %9s %10s %9s %7.2f  (no dump)' % (top_k, seed, '?', '?', '?', sec))
                continue
            sem = np.fromfile(sem_path, dtype='<i4')
            top, dup = repetition(sem)
            print('%-6d %-6d %9d %10d %8.1f%% %7.2f' % (top_k, seed, len(sem), top, dup * 100, sec))
            worst.append((top_k, seed, top, dup, sec))
    if worst:
        w = max(worst, key=lambda r: r[2])
        print('\nworst: top_k=%d seed=%d  max-ngram x%d  dup=%.1f%%  %.2fs'
              % (w[0], w[1], w[2], w[3] * 100, w[4]))
        loops = [r for r in worst if r[2] >= 3]
        print('runs with a repeated 8-gram >=3x: %d/%d' % (len(loops), len(worst)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
