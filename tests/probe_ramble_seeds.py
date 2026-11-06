"""Find the seeds that make this request ramble, and look at what the AR actually produced.

The two saved renders (18.10 s vs 48.98 s, identical settings, seed=-1 = random) differ only by
the random draw, so some seeds must degenerate. This sweeps fixed seeds with the user's exact
text/reference/settings against a server running with GSV_ENGINE_DUMP so the AR token stream
(sem.i32) can be inspected: token count, EOS position and repetition.
"""
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9971
DUMP = sys.argv[2] if len(sys.argv) > 2 else r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\ramsem\dump'
REF = 'C:/Users/kakar/AppData/Local/Temp/gsv_refs/39795769_gsv_ref_xtj552en.wav'
PROMPT = '开心的事，不开心的事，都可以告诉我哦！'
TEXT = ('こんにちは、音声合成システムへようこそ。これはテスト用のサンプルです。'
        '数字の12345や、句読点（読点、句点、疑問符、感嘆符）が含まれています。'
        '今日の天気はどうですか？')
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def tts(seed):
    body = {
        'text': TEXT, 'text_lang': 'ja', 'voice': 'default',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': seed, 'sample_steps': 32, 'dbcache': 'auto',
        'top_k': 5, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
        'text_split_method': 'cut1', 'fragment_interval': 0.3,
        'dit_uncond_reuse': 0, 'skip_first_step': 1,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def rep(sem, n=8):
    if len(sem) < n * 2:
        return 1
    c = {}
    for i in range(len(sem) - n + 1):
        k = tuple(sem[i:i + n].tolist())
        c[k] = c.get(k, 0) + 1
    return max(c.values())


def main():
    sem_path = os.path.join(DUMP, 'sem.i32')
    print('text %d chars, reference as uploaded by the GUI' % len(TEXT))
    print('%-6s %8s %9s %7s %9s' % ('seed', 'sec', 'AR tok', 'ngram', 'flags'))
    good, bad = [], []
    for seed in SEEDS:
        if os.path.exists(sem_path):
            os.remove(sem_path)
        try:
            b = tts(seed)
        except Exception as e:
            print('%-6d  ERR %s' % (seed, str(e)[:56]))
            continue
        w = wave.open(io.BytesIO(b), 'rb')
        sec = w.getnframes() / w.getframerate()
        tok = -1
        ng = -1
        if os.path.exists(sem_path):
            sem = np.fromfile(sem_path, dtype='<i4')
            tok = len(sem)
            ng = rep(sem)
        flag = 'RAMBLE' if sec > 30 else ''
        print('%-6d %8.2f %9d %7d %9s' % (seed, sec, tok, ng, flag))
        (bad if sec > 30 else good).append((seed, sec, tok))
    print('\ngood: %d, rambling: %d' % (len(good), len(bad)))
    if good:
        print('  good  tok/char = %.1f' % (np.mean([g[2] for g in good]) / len(TEXT)))
    if bad:
        print('  ramble tok/char = %.1f' % (np.mean([g[2] for g in bad]) / len(TEXT)))
        print('  rambling seeds:', [b[0] for b in bad])
    return 0


if __name__ == '__main__':
    sys.exit(main())
