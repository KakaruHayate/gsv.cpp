"""Which part of the render path carries state between two identical requests?

Runs the same fixed-seed request twice and compares bit-exactly, across the two axes that
differ between this session's earlier verification (cut0 + dbcache off) and the GUI's settings
(cut1 = "凑四句一切" + dbcache auto).
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


def tts(split, dbcache, seed=12345):
    body = {
        'text': TEXT, 'text_lang': 'ja', 'voice': 'default',
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': seed, 'sample_steps': 32, 'dbcache': dbcache,
        'top_k': 5, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
        'text_split_method': split, 'fragment_interval': 0.3,
        'dit_uncond_reuse': 0, 'skip_first_step': 1,
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


def line(tag, r1, r2):
    x1, sr = pcm(r1)
    x2, _ = pcm(r2)
    n = min(len(x1), len(x2))
    same = hashlib.sha256(r1).hexdigest() == hashlib.sha256(r2).hexdigest()
    cc = float(np.corrcoef(x1[:n], x2[:n])[0, 1]) if n > 1 else -1.0
    print('  %-22s run1 %6.2fs  run2 %6.2fs  identical=%-5s corr=%.4f'
          % (tag, len(x1) / sr, len(x2) / sr, same, cc))
    return same, cc


def main():
    for split in ('cut0', 'cut1'):
        for dbc in ('off', 'auto'):
            tag = '%s + dbcache=%s' % (split, dbc)
            r1 = tts(split, dbc)
            r2 = tts(split, dbc)
            same, cc = line(tag, r1, r2)
            if not same:
                r3 = tts(split, dbc)
                line(tag + ' (run3)', r2, r3)
    return 0


if __name__ == '__main__':
    sys.exit(main())
