"""Is the reference audio literally embedded at each fragment boundary?

Correlates the reference waveform against the generated output with a sliding window. A prompt
echo shows up as a sharp, high normalised cross-correlation peak, and the number of peaks
should match the fragment count for the split methods.

Run against the same text/reference/seed as the fragment-length probe.
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


def pcm_file(p):
    w = wave.open(p, 'rb')
    return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64), w.getframerate()


def resample(x, a, b):
    if a == b:
        return x
    n = int(round(len(x) * b / a))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def scan(ref, out, hop=240):
    """Normalised cross-correlation of ref over all lags, coarse hop then local refine."""
    r = ref - ref.mean()
    rn = np.sqrt((r * r).sum()) + 1e-9
    L = len(r)
    scores = []
    for s in range(0, max(len(out) - L, 1), hop):
        seg = out[s:s + L]
        seg = seg - seg.mean()
        scores.append((s, float((r * seg).sum() / (rn * (np.sqrt((seg * seg).sum()) + 1e-9)))))
    return scores


def peaks(scores, thresh=0.35):
    """Local maxima above threshold, min 0.5 s apart."""
    out = []
    for i in range(1, len(scores) - 1):
        s, v = scores[i]
        if v >= thresh and v >= scores[i - 1][1] and v >= scores[i + 1][1]:
            if not out or s - out[-1][0] > 24000:
                out.append((s, v))
    return out


def main():
    ref, sr_ref = pcm_file(ZH_REF)
    print('reference: %.2fs @ %d Hz' % (len(ref) / sr_ref, sr_ref))
    print('%-10s %8s %10s %s' % ('split', 'sec', 'max corr', 'peaks>=0.35'))
    for split in ('cut0', 'cut4'):
        b = tts(split)
        out, sr_out = pcm(b)
        r = resample(ref, sr_ref, sr_out)
        sc = scan(r, out)
        mx = max((v for _, v in sc), default=0.0)
        pk = peaks(sc)
        print('%-10s %8.2f %10.3f %d %s' % (split, len(out) / sr_out, mx, len(pk),
                                            [('%.1fs' % (s / sr_out)) for s, _ in pk][:8]))
    print('\n参考音频被原样拼入 -> 会出现 >=0.35 的相关峰; 峰数应约等于片段数')
    return 0


if __name__ == '__main__':
    sys.exit(main())
