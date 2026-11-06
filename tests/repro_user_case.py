"""Reproduce the reported case: matched Chinese reference, long Chinese text, cut4, seed sweep.

Step 1 builds a reference whose audio really says the prompt text (bootstrap from the English
VCTK clip), so the AR prompt conditioning is sound -- that is the user's situation.
Step 2 synthesises the reported text with the user's settings (text_split_method=cut4) and
looks for AR repetition per seed.
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
OUT = r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\seedsem'

EN_REF = 'H:/downloads/LJ037-0171.wav'
EN_PROMPT = '.e Commission to conclude that five moths may have been fired.'
ZH_REF_TEXT = '你好，欢迎使用语音合成系统。'
USER_TEXT = ('所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，'
             '你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！'
             '那就是他们已经亮出了石头剪刀布了，你再出招儿')
SEEDS = [3047, 1, 42, 2, 7]


def tts(body):
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def save_wav(b, path):
    w = wave.open(path, 'wb')
    src = wave.open(io.BytesIO(b), 'rb')
    w.setnchannels(src.getnchannels())
    w.setsampwidth(src.getsampwidth())
    w.setframerate(src.getframerate())
    w.writeframes(src.readframes(src.getnframes()))
    w.close()
    return src.getnframes() / src.getframerate()


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
    os.makedirs(OUT, exist_ok=True)
    zh_ref = os.path.join(OUT, 'ref_zh_matched.wav')

    print('=== step 1: build a matched Chinese reference ===')
    b = tts({'text': ZH_REF_TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
             'ref_audio_path': EN_REF, 'prompt_text': EN_PROMPT, 'prompt_lang': 'en',
             'seed': 12345, 'sample_steps': 32, 'dbcache': 'off',
             'text_split_method': 'cut0', 'fragment_interval': 0})
    dur = save_wav(b, zh_ref)
    print('  ref audio %.2fs for %d chars -> %.2f s/char' % (dur, len(ZH_REF_TEXT), dur / len(ZH_REF_TEXT)))

    print('\n=== step 2: user text, cut4, seed sweep ===')
    sem_path = os.path.join(DUMP, 'sem.i32')
    print('%-8s %8s %10s %10s %8s' % ('seed', 'AR tokens', 'max ngram', 'dup share', 'sec'))
    for seed in SEEDS:
        if os.path.exists(sem_path):
            os.remove(sem_path)
        wav = tts({'text': USER_TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
                   'ref_audio_path': zh_ref.replace('\\', '/'), 'prompt_text': ZH_REF_TEXT,
                   'prompt_lang': 'zh', 'seed': seed, 'sample_steps': 32, 'dbcache': 'off',
                   'text_split_method': 'cut4', 'fragment_interval': 0.3,
                   'dit_cfg_off_t': 0.0, 'dit_uncond_reuse': 0, 'skip_first_step': 0})
        sec = wave.open(io.BytesIO(wav), 'rb').getnframes() / 48000.0
        if not os.path.exists(sem_path):
            print('%-8d %8s %10s %10s %8.2f (no dump)' % (seed, '?', '?', '?', sec))
            continue
        sem = np.fromfile(sem_path, dtype='<i4')
        top, dup = repetition(sem)
        print('%-8d %8d %10d %9.1f%% %8.2f' % (seed, len(sem), top, dup * 100, sec))

    print('\n(87 chars: a healthy run is roughly 13-17 s; ~47 s means it repeated)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
