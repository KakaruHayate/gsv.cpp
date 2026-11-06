"""Does every fragment start with the same lead-in (the prompt's content re-spoken)?

Takes the first N seconds of the fragmented output as a template and correlates it across the
whole output. If each fragment opens by re-speaking the reference prompt, the template shows up
again once per fragment, which is exactly what the user hears as "every sentence starts by
repeating the reference audio".

Also runs the same scan on a single-shot (cut0) render as a control: there the template should
appear at most once.
"""
import io
import json
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9953
ZH_REF = r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\seedsem\ref_zh_matched.wav'
ZH_REF_TEXT = '你好，欢迎使用语音合成系统。'
USER_TEXT = ('所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，'
             '你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！'
             '那就是他们已经亮出了石头剪刀布了，你再出招儿')


def tts(split, seed=3047):
    body = {
        'text': USER_TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': ZH_REF.replace('\\', '/'), 'prompt_text': ZH_REF_TEXT,
        'prompt_lang': 'zh', 'seed': seed, 'sample_steps': 32, 'dbcache': 'off',
        'text_split_method': split, 'fragment_interval': 0.3,
        'dit_cfg_off_t': 0.0, 'dit_uncond_reuse': 0, 'skip_first_step': 0,
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


def self_similarity(out, sr, tpl_sec=1.2, hop=240):
    L = int(tpl_sec * sr)
    tpl = out[:L]
    tpl = tpl - tpl.mean()
    tn = np.sqrt((tpl * tpl).sum()) + 1e-9
    sc = []
    for s in range(0, max(len(out) - L, 1), hop):
        seg = out[s:s + L]
        seg = seg - seg.mean()
        sc.append((s, float((tpl * seg).sum() / (tn * (np.sqrt((seg * seg).sum()) + 1e-9)))))
    peaks, last = [], -1e9
    for i in range(1, len(sc) - 1):
        s, v = sc[i]
        if v >= 0.6 and v >= sc[i - 1][1] and v >= sc[i + 1][1] and s - last > L // 2:
            peaks.append((s, v)); last = s
    return peaks


def main():
    print('template = first 1.2 s of the output; peaks are places where it recurs')
    print('%-8s %8s %8s %s' % ('split', 'sec', 'peaks', 'peak positions (s)'))
    for split in ('cut0', 'cut4'):
        out, sr = pcm(tts(split))
        pk = self_similarity(out, sr)
        print('%-8s %8.2f %8d %s' % (split, len(out) / sr, len(pk),
                                     [('%.1f' % (s / sr)) + ('@%.2f' % v) for s, v in pk][:10]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
