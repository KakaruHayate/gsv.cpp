"""Is the per-sentence degradation caused by the fragmented synthesis path re-priming each
fragment from the reference prompt?

The user describes: every sentence starts by repeating the reference audio's content, then the
quality collapses and the timbre is gone. That is what a prompt echo per fragment looks like:
with text_split_method=cut4 the text is cut into fragments (split_text splits on punctuation
with a >=8 char minimum) and each fragment runs its own AR pass primed with the reference codes.

Prediction: if each fragment echoes ~one prompt length, total duration grows by roughly
(n_fragments - 1) x prompt_duration compared with a single-shot (cut0) render of the same text.

Also varies the prompt audio/text length ratio, since a reference much longer than its
transcript is the other classic cause of prompt echo.
"""
import io
import json
import os
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


def tts(text, split, prompt_text, seed=3047):
    body = {
        'text': text, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': ZH_REF.replace('\\', '/'), 'prompt_text': prompt_text,
        'prompt_lang': 'zh', 'seed': seed, 'sample_steps': 32, 'dbcache': 'off',
        'text_split_method': split, 'fragment_interval': 0.3,
        'dit_uncond_reuse': 0, 'skip_first_step': 0,
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


def ref_dur():
    w = wave.open(ZH_REF, 'rb')
    return w.getnframes() / w.getframerate()


def fragments(text, method):
    """Mirror src/svc_core.h split_text: punctuation split, merge until >=8 chars."""
    if method == 'cut0':
        return [text]
    punct = set('，。！？；：、,.!?;:')
    raw, cur = [], ''
    for ch in text:
        cur += ch
        if ch in punct:
            raw.append(cur); cur = ''
    if cur:
        raw.append(cur)
    out, acc = [], ''
    for s in raw:
        acc += s
        if len(acc) >= 8:
            out.append(acc); acc = ''
    if acc:
        if out:
            out[-1] += acc
        else:
            out.append(acc)
    return out


def main():
    rd = ref_dur()
    frags = fragments(USER_TEXT, 'cut4')
    print('reference %.2fs, prompt text %d chars' % (rd, len(ZH_REF_TEXT)))
    print('cut4 fragments: %d -> %s' % (len(frags), [len(f) for f in frags]))
    print()

    print('%-22s %10s %10s %12s' % ('case', 'sec', 'vs cut0', 'expected extra'))
    base = None
    for split in ('cut0', 'cut1', 'cut4', 'cut5'):
        s = sec_of(tts(USER_TEXT, split, ZH_REF_TEXT))
        if base is None:
            base = s
        n = len(fragments(USER_TEXT, split))
        extra = (n - 1) * rd
        print('%-22s %10.2f %10s %11.2fs' % (split + ' (%d frag)' % n, s,
                                             '-' if split == 'cut0' else '%+.2fs' % (s - base),
                                             extra))
    print('\n如果每段都回放一次 prompt，cut4 会比 cut0 多出约 %.1fs' % ((len(frags) - 1) * rd))

    print('\n=== prompt 音频/文本 失配 ===')
    for pt, label in ((ZH_REF_TEXT, 'matched (14 chars)'),
                      ('你好', 'text too short (2 chars)'),
                      (ZH_REF_TEXT + ZH_REF_TEXT + ZH_REF_TEXT, 'text too long (42 chars)')):
        s = sec_of(tts(USER_TEXT, 'cut0', pt))
        print('%-26s %10.2f' % (label, s))
    return 0


if __name__ == '__main__':
    sys.exit(main())
