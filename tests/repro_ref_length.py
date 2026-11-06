"""With the user's own reference: is the length blow-up caused by fragmentation, by the
reference being much longer than its transcript, or by the seed?

The uploaded reference is 4.63 s of audio for an 11-character prompt ("好奇怪，心跳变得好快"),
i.e. 0.42 s/char -- very slow speech. The target text is 65 characters, so a model that copies
the reference's pace would already need ~27 s, and with text_split_method=cut4 the text becomes
several fragments, each re-primed with that same 4.63 s reference.

Varies: fragment (cut0 vs cut4) x reference length (4.63 s, 3.5 s, 2.5 s). Seed and every other
setting stay fixed, so any difference is attributable to those two factors.
"""
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9880
REFDIR = r'C:\Users\kakar\AppData\Local\Temp\gsv_refs'
BASE = os.path.join(REFDIR, '38062252_gsv_ref_effp8uke.wav')
PROMPT = '好奇怪，心跳变得好快'
TEXT = ('你好，欢迎使用语音合成系统。这是一个测试样例，包含数字12345，以及标点符号：'
        '逗号、句号、问号、和感叹号！请问今天的天气怎么样？')


def trim(src, dst, seconds):
    w = wave.open(src, 'rb')
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    n = int(seconds * sr)
    frames = w.readframes(min(n, w.getnframes()))
    w.close()
    o = wave.open(dst, 'wb')
    o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(sr)
    o.writeframes(frames)
    o.close()


def tts(ref, split, seed=3047):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': 'default',
        'ref_audio_path': ref.replace('\\', '/'), 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': seed, 'sample_steps': 32, 'dbcache': 'auto',
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


def sec_of(b):
    w = wave.open(io.BytesIO(b), 'rb')
    return w.getnframes() / w.getframerate()


def main():
    refs = [('4.63s (as uploaded)', BASE)]
    for s in (3.5, 2.5):
        p = os.path.join(REFDIR, 'probe_trim_%.1fs.wav' % s)
        trim(BASE, p, s)
        refs.append(('%.1fs' % s, p))

    print('target text: %d chars; prompt: %d chars' % (len(TEXT), len(PROMPT)))
    print('%-20s %-6s %8s' % ('reference', 'split', 'sec'))
    for label, path in refs:
        for split in ('cut0', 'cut4'):
            try:
                print('%-20s %-6s %8.2f' % (label, split, sec_of(tts(path, split))))
            except Exception as e:
                print('%-20s %-6s  ERR %s' % (label, split, str(e)[:50]))
    print('\n(expected for 65 chars at a normal 0.2-0.3 s/char pace: 13-20 s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
