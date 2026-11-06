#!/usr/bin/env python3
"""Measure the three DiT acceleration directions against a live gsv_server.

Every request is sent as UTF-8 bytes (PowerShell's Invoke-WebRequest mangles non-ASCII
bodies, which silently produced bogus results in earlier sessions). Quality is measured on
the DiT mel output (`mel_out`), because vocoded waveform correlation is dominated by phase
and is not a usable fidelity proxy.

Usage:
  python tests/bench_dit_alg.py --port 9934 --sidecar 9931 --models models --out DIR [--group d1]
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
import urllib.request

import numpy as np

MEL = 100

REFS = {
    # a female English reader + a Chinese prompt; both are 16k/22k mono wavs on disk
    "en": r"H:/downloads/LJ037-0171.wav",
    "zh": r"J:/GPT-SoVITS/gsv.cpp/models/ref_test.wav",
}

# sentences of different lengths (Chinese, so the mel length is easy to reason about)
TEXTS = {
    "s1": "你好。",
    "s2": "你好，欢迎使用语音合成系统。",
    "s4": "今天天气不错，我们一起出去走走吧，顺便买点东西回来。",
    "s8": "在这个安静的下午，我坐在窗边，看着外面来来往往的人群，忽然想起很多年以前的那些事情。",
}


def post(port, payload, timeout=1800):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:%d/tts" % port, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        hdr = dict(r.headers)
    return body, (time.perf_counter() - t0) * 1000.0, hdr


def load_mel(path):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(MEL, -1) if a.size % MEL == 0 else None


def metrics(base, cur):
    """Mel-domain distance. relL1 is scale-free; corr is Pearson on the flattened mel."""
    if base is None or cur is None:
        return None
    n = min(base.shape[1], cur.shape[1])
    a, b = base[:, :n], cur[:, :n]
    d = np.abs(a - b)
    rel = d.sum() / (np.abs(a).sum() + 1e-9)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    # spectral convergence on the mel band, in dB (perceptual-ish)
    num = np.sqrt(((a - b) ** 2).sum(axis=0))
    den = np.sqrt((a ** 2).sum(axis=0)) + 1e-9
    sc = float(np.mean(20 * np.log10(np.maximum(num / den, 1e-9))))
    return {"relL1": float(rel), "corr": corr, "sc_db": sc, "frames": int(n)}


class Runner:
    def __init__(self, port, dump, out, voice, ref, prompt_text, prompt_lang, repeats):
        self.port, self.dump, self.out = port, dump, out
        self.voice, self.ref = voice, ref
        self.prompt_text, self.prompt_lang = prompt_text, prompt_lang
        self.repeats = repeats
        self.rows = []

    def run(self, label, text_key, extra=None, want_mel=True):
        text = TEXTS[text_key]
        payload = {
            "text": text, "text_lang": "zh", "voice": self.voice,
            "ref_audio_path": self.ref, "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang, "seed": 12345,
            # one continuous utterance: the default cut5 splits on punctuation and the mel
            # dump would then only hold the last fragment
            "text_split_method": "cut0", "fragment_interval": 0,
        }
        payload.update(extra or {})
        times, dit_ms, wav = [], [], None
        mel_path = os.path.join(self.dump, "mel_out.f32")
        mel = None
        for _ in range(self.repeats):
            if os.path.exists(mel_path):
                os.remove(mel_path)
            body, ms, _ = post(self.port, payload)
            wav = body
            times.append(ms)
            if want_mel and os.path.exists(mel_path):
                dst = os.path.join(self.out, "mel_%s.f32" % label)
                shutil.copyfile(mel_path, dst)
                mel = load_mel(dst)
            # per-stage timing is echoed into the sidecar log only; use wall time
        row = {"label": label, "text": text_key, "params": extra or {},
               "ms_median": statistics.median(times), "ms_all": times,
               "wav_bytes": len(wav or b""),
               "frames": int(mel.shape[1]) if mel is not None else None,
               "mel_file": os.path.join(self.out, "mel_%s.f32" % label) if mel is not None else None}
        if wav:
            with open(os.path.join(self.out, "wav_%s.wav" % label), "wb") as f:
                f.write(wav)
        self.rows.append(row)
        return row, mel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="v5dev")
    ap.add_argument("--ref", default=REFS["zh"])
    ap.add_argument("--prompt-text", default=None)
    ap.add_argument("--prompt-lang", default="zh")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--group", default="all")
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    prompt_text = a.prompt_text if a.prompt_text is not None else TEXTS["s2"]
    r = Runner(a.port, a.dump, a.out, a.voice, a.ref, prompt_text, a.prompt_lang, a.repeats)

    def base_params():
        return {"sample_steps": a.steps, "dbcache": "off",
                "dit_window": 0, "dit_cfg_off_t": 0, "dit_uncond_reuse": 0}

    results = {"meta": {"voice": a.voice, "ref": a.ref, "prompt_text": prompt_text,
                        "steps": a.steps, "repeats": a.repeats}, "groups": {}}

    # ---------- direction 1: DiTFastAttn window ----------
    if a.group in ("all", "d1"):
        g = []
        for tk in ("s2", "s4", "s8"):
            b, bm = r.run("d1_base_%s" % tk, tk, base_params())
            for w in (0, 64, 128, 192, 256, 384, 512):
                p = base_params(); p["dit_window"] = w
                row, m = r.run("d1_%s_w%d" % (tk, w), tk, p)
                met = metrics(bm, m)
                g.append({"text": tk, "window": w, "ms": row["ms_median"],
                          "frames": row["frames"], "base_ms": b["ms_median"],
                          "speedup": (b["ms_median"] / row["ms_median"]) if row["ms_median"] else None,
                          **(met or {})})
                print("d1 %s w=%-4d frames=%-4s relL1=%s corr=%s"
                      % (tk, w, row["frames"], met and round(met["relL1"], 5),
                         met and round(met["corr"], 6)), flush=True)
        results["groups"]["d1"] = g

    # ---------- direction 3: CFG truncation + uncond reuse ----------
    if a.group in ("all", "d3"):
        g = []
        for tk in ("s2", "s4"):
            b, bm = r.run("d3_base_%s" % tk, tk, base_params())
            for off, ur in ((0.95, 0), (0.9, 0), (0.8, 0), (0.7, 0), (0.6, 0), (0.5, 0),
                            (0.9, 1), (0.7, 1), (0.7, 2), (0.6, 1)):
                p = base_params(); p["dit_cfg_off_t"] = off; p["dit_uncond_reuse"] = ur
                row, m = r.run("d3_%s_o%s_u%d" % (tk, str(off).replace('.', ''), ur), tk, p)
                met = metrics(bm, m)
                g.append({"text": tk, "cfg_off_t": off, "uncond_reuse": ur,
                          "ms": row["ms_median"], "base_ms": b["ms_median"],
                          "speedup": (b["ms_median"] / row["ms_median"]) if row["ms_median"] else None,
                          **(met or {})})
                print("d3 %s off=%.2f ur=%d relL1=%s corr=%s speed=%.2fx"
                      % (tk, off, ur, met and round(met["relL1"], 5),
                         met and round(met["corr"], 6),
                         b["ms_median"] / row["ms_median"]), flush=True)
        results["groups"]["d3"] = g

    # ---------- direction 2: reference isolation (precompute feasibility) ----------
    if a.group in ("all", "d2"):
        # GSV_DIT_REF_ISO is a process-wide env var, so this group must be run in its own
        # server process; the runner only records what the server gives it.
        g = []
        for tk in ("s2", "s4", "s8"):
            row, m = r.run("d2_%s" % tk, tk, base_params())
            g.append({"text": tk, "ms": row["ms_median"], "frames": row["frames"]})
            print("d2 %s iso=%s frames=%s ms=%.0f"
                  % (tk, os.environ.get("GSV_DIT_REF_ISO", "0"), row["frames"], row["ms_median"]),
                  flush=True)
        results["groups"]["d2"] = g

    with open(os.path.join(a.out, "results_%s.json" % a.group), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("wrote", os.path.join(a.out, "results_%s.json" % a.group))


if __name__ == "__main__":
    sys.exit(main())
