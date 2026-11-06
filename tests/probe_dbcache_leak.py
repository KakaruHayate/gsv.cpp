"""Is the per-request degradation a cache-dit state leak?

Symptom reported: the FIRST render after (re)starting is fine, the SECOND is garbage, every time,
for different texts and references. That is the signature of state carried over between requests
rather than a bad random seed.

All earlier verification in this session forced dbcache='off'; the GUI uses 'auto', which for
32 steps resolves to cache-dit F8. This runs the identical request twice with a FIXED seed and
compares, once with cache-dit on and once with it off:

  - cache-dit on : run2 must equal run1 exactly
  - cache-dit off: run2 must equal run1 exactly

A mismatch only in the "on" case pins the leak on the cache-dit path.
"""
import hashlib
import io
import json
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9971
REF = 'C:/Users/kakar/AppData/Local/Temp/gsv_refs/39795769_gsv_ref_xtj552en.wav'
PROMPT = '开心的事，不开心的事，都可以告诉我哦！'
TEXT = ('こんにちは、音声合成システムへようこそ。これはテスト用のサンプルです。'
        '数字の12345や、句読点（読点、句点、疑問符、感嘆符）が含まれています。'
        '今日の天気はどうですか？')


def tts(dbcache, seed, text=TEXT):
    body = {
        'text': text, 'text_lang': 'ja', 'voice': 'default',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': seed, 'sample_steps': 32, 'dbcache': dbcache,
        'top_k': 5, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
        'text_split_method': 'cut0', 'fragment_interval': 0.0,
        'dit_cfg_off_t': 0.5, 'dit_uncond_reuse': 0, 'skip_first_step': 1,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def pcm(b):
    w = wave.open(io.BytesIO(b), 'rb')
    return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64), w.getframerate()


def cmp(a, b):
    xa, sr = pcm(a)
    xb, _ = pcm(b)
    n = min(len(xa), len(xb))
    if n < 2:
        return -1.0, 0.0
    cc = float(np.corrcoef(xa[:n], xb[:n])[0, 1])
    rl = float(np.abs(xa[:n] - xb[:n]).sum() / (np.abs(xa[:n]).sum() + 1e-9))
    return cc, rl


def main():
    for dbc in ('off', 'auto'):
        print('=== dbcache=%s, seed=12345, cut0 ===' % dbc)
        r1 = tts(dbc, 12345)
        r2 = tts(dbc, 12345)
        d1 = len(pcm(r1)[0]) / pcm(r1)[1]
        d2 = len(pcm(r2)[0]) / pcm(r2)[1]
        cc, rl = cmp(r1, r2)
        same = hashlib.sha256(r1).hexdigest() == hashlib.sha256(r2).hexdigest()
        print('  run1 %.2fs  run2 %.2fs  bit-identical=%s  corr=%.4f  relL1=%.4f'
              % (d1, d2, same, cc, rl))
        if not same:
            # 第三次，看是否收敛到 run2 的状态
            r3 = tts(dbc, 12345)
            d3 = len(pcm(r3)[0]) / pcm(r3)[1]
            cc23, rl23 = cmp(r2, r3)
            print('  run3 %.2fs  run2 vs run3 corr=%.4f relL1=%.4f' % (d3, cc23, rl23))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
