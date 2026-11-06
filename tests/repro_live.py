"""Reproduce the live report against the running GAG/gsv_server instance.

Uses the exact reference the GUI uploaded (24 kHz mono 4.63 s, 222364 bytes), the reported text,
and the GUI's settings, then varies only what can explain "same request, only the seed changed":

  - voice: "default" (what the GUI's voice box shows) vs "v5dev" (what the model-version box says)
  - top_k: 5 (the GUI default) vs 15 (the engine default)
  - seed: 3047 (reported bad) vs -1 (reported fine)

Reported: output length, plus how much of it repeats (long output for a short text is the
audible symptom the user described).
"""
import io
import json
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9880
REF = 'C:/Users/kakar/AppData/Local/Temp/gsv_refs/38062252_gsv_ref_effp8uke.wav'
PROMPT = '好奇怪，心跳变得好快'
TEXT = ('你好，欢迎使用语音合成系统。这是一个测试样例，包含数字12345，以及标点符号：'
        '逗号、句号、问号、和感叹号！请问今天的天气怎么样？')


def tts(voice, seed, top_k, steps=32):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': voice,
        'ref_audio_path': REF, 'prompt_text': PROMPT, 'prompt_lang': 'zh',
        'seed': seed, 'sample_steps': steps, 'dbcache': 'auto',
        'top_k': top_k, 'top_p': 1.0, 'temperature': 1.0, 'repetition_penalty': 1.35,
        'text_split_method': 'cut4', 'fragment_interval': 0.3,
        'dit_cfg_off_t': 0.7, 'dit_uncond_reuse': 0, 'skip_first_step': 1,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=2400) as r:
        return r.read()


def measure(b):
    w = wave.open(io.BytesIO(b), 'rb')
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
    sr = w.getframerate()
    return len(x) / sr, x, sr


def selfsim(x, sr, tpl_sec=0.8, hop=480):
    """How many times the opening of the clip recurs later on (crude repetition detector)."""
    L = int(tpl_sec * sr)
    if len(x) < L * 3:
        return 0
    tpl = x[:L] - x[:L].mean()
    tn = np.sqrt((tpl * tpl).sum()) + 1e-9
    hits = 0
    last = -1e9
    for s in range(L, max(len(x) - L, 1), hop):
        seg = x[s:s + L]
        seg = seg - seg.mean()
        v = float((tpl * seg).sum() / (tn * (np.sqrt((seg * seg).sum()) + 1e-9)))
        if v > 0.6 and s - last > L:
            hits += 1
            last = s
    return hits


def main():
    print('text %d chars, reference 4.63 s (as uploaded by the GUI)' % len(TEXT))
    print('%-9s %-7s %-6s %8s %8s %7s' % ('voice', 'seed', 'top_k', 'sec', 'repeats', 'min/max'))
    cases = [
        ('default', 3047, 5),
        ('default', -1, 5),
        ('v5dev', 3047, 5),
        ('v5dev', 3047, 15),
    ]
    for voice, seed, top_k in cases:
        try:
            sec, x, sr = measure(tts(voice, seed, top_k))
            print('%-9s %-7d %-6d %8.2f %8d %7.0f/%.0f'
                  % (voice, seed, top_k, sec, selfsim(x, sr), x.min(), x.max()))
        except Exception as e:
            print('%-9s %-7d %-6d  ERR %s' % (voice, seed, top_k, str(e)[:60]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
