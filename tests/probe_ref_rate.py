"""Does the reference audio's sample rate change how faithfully the engine conditions on it?

GAG converts non-wav references with soundfile and keeps the source rate, so an mp3 typically
arrives as 44.1 kHz mono PCM16. This checks the same audio content at 16/22.05/32/44.1/48 kHz:
the outputs should be nearly identical (the engine resamples internally). A large divergence,
or an output that no longer follows the text length, points at rate handling.

Reported per rate: output duration (should track the text, not the rate), mel-free waveform
stats, and the pairwise distance to the 32 kHz run as a baseline.
"""
import io
import json
import sys
import urllib.request
import wave

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9952
OUT = sys.argv[2] if len(sys.argv) > 2 else r'J:\GPT-SoVITS\gsv.cpp\tests\relcpu\rateprobe'

SRC = 'H:/downloads/LJ037-0171.wav'
PROMPT = '.e Commission to conclude that five moths may have been fired.'
TEXT = '你好，欢迎使用语音合成系统。'


def read_wav(path):
    w = wave.open(path, 'rb')
    sr = w.getframerate()
    ch = w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(path, x, sr):
    w = wave.open(path, 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(np.clip(np.round(x), -32768, 32767).astype('<i2').tobytes())
    w.close()


def resample(x, sr_in, sr_out):
    n = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x))
    t_out = np.linspace(0, len(x) - 1, n)
    return np.interp(t_out, t_in, x)


def tts(ref_path):
    body = {
        'text': TEXT, 'text_lang': 'zh', 'voice': 'v5dev',
        'ref_audio_path': ref_path, 'prompt_text': PROMPT, 'prompt_lang': 'en',
        'seed': 12345, 'sample_steps': 32, 'dbcache': 'off',
        'text_split_method': 'cut0', 'fragment_interval': 0,
        'dit_uncond_reuse': 0, 'skip_first_step': 0,
    }
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:%d/tts' % PORT, data=data,
                                 headers={'Content-Type': 'application/json; charset=utf-8'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()


def wav_info(b):
    w = wave.open(io.BytesIO(b), 'rb')
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float64)
    return len(x) / w.getframerate(), x


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    x0, sr0 = read_wav(SRC)
    print('source: %s  %d Hz %.2fs' % (SRC, sr0, len(x0) / sr0))
    outs = {}
    print('\n%-8s %8s %10s %10s' % ('rate', 'sec', 'min', 'max'))
    for sr in (16000, 22050, 32000, 44100, 48000):
        p = os.path.join(OUT, 'ref_%d.wav' % sr)
        write_wav(p, resample(x0, sr0, sr) if sr != sr0 else x0, sr)
        dur, y = wav_info(tts(p.replace('\\', '/')))
        outs[sr] = y
        print('%-8d %8.2f %10.0f %10.0f' % (sr, dur, y.min(), y.max()))

    base = outs[32000]
    print('\ndistance to the 32 kHz reference run (same audio content):')
    for sr, y in sorted(outs.items()):
        n = min(len(y), len(base))
        if n < 2:
            continue
        a, b = y[:n], base[:n]
        rl = np.abs(a - b).sum() / (np.abs(b).sum() + 1e-9)
        cc = np.corrcoef(a, b)[0, 1]
        flag = '  <-- diverges' if rl > 0.3 else ''
        print('  %-6d relL1=%.4f corr=%.5f%s' % (sr, rl, cc, flag))
    return 0


if __name__ == '__main__':
    sys.exit(main())
