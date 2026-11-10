"""Dump the AR token stream for a few texts and look for repetition / n-gram matches directly."""
import collections
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

REPO = r"J:\GPT-SoVITS\gsv.cpp"
DUMP = os.path.join(REPO, "tests", "_lookup_dump")
PORT = 9975
SC_PORT = 9977
REF = r"H:\downloads\LJ037-0171.wav"
PROMPT = "on to conclude that five shots may have been fired."

TEXTS = {
    "repeat": "哈哈哈哈哈哈哈哈哈哈，慢慢慢，好好好，对对对。",
    "long": ("所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，"
             "你知道每个人的手上剩下的筹码是多少。"),
    "aaaa": "啊啊啊啊啊啊啊啊啊啊啊啊啊啊。",
}


def wait(url, timeout=300):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def tts(body):
    req = urllib.request.Request("http://127.0.0.1:%d/tts" % PORT,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def stats(sem, prompt):
    hist = prompt + sem
    print("  sem tokens (%d): %s" % (len(sem), sem[:40]))
    # consecutive duplicates
    runs = 1
    best = 1
    for i in range(1, len(sem)):
        runs = runs + 1 if sem[i] == sem[i - 1] else 1
        best = max(best, runs)
    print("  longest run of identical tokens: %d" % best)
    for n in (2, 3, 4):
        c = collections.Counter(tuple(hist[i:i + n]) for i in range(len(hist) - n + 1))
        top = c.most_common(3)
        print("  n=%d distinct=%d/%d top=%s" % (n, len(c), len(hist) - n + 1,
                                                [(list(k), v) for k, v in top]))


def main():
    os.makedirs(DUMP, exist_ok=True)
    env = os.environ.copy()
    env["GSV_ENGINE_DUMP"] = DUMP
    env["GSV_G2PW_MODEL"] = os.path.join(REPO, "models", "G2PWModel")
    env["GSV_BERT_DIR"] = os.path.join(REPO, "models", "chinese-roberta-wwm-ext-large")
    env["NLTK_DATA"] = os.path.join(REPO, ".nltk_data")
    env["GSV_FAST_LANGDETECT"] = os.path.join(REPO, "models", "frontend", "fast_langdetect")
    py = sys.executable

    sc = subprocess.Popen([py, os.path.join(REPO, "tools", "text_sidecar.py"), "--port", str(SC_PORT)],
                          cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait("http://127.0.0.1:%d/health" % SC_PORT):
        print("sidecar failed")
        return 1
    srv = subprocess.Popen([os.path.join(REPO, "tests", "relcpu", "gsv_server.exe"),
                            "--models", os.path.join(REPO, "models"), "--device", "vulkan",
                            "--port", str(PORT), "--sidecar", "127.0.0.1:%d" % SC_PORT,
                            "--allow-path", "*"],
                           cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait("http://127.0.0.1:%d/health" % PORT):
        print("server failed")
        sc.kill()
        return 1
    try:
        for name, text in TEXTS.items():
            for f in ("sem.i32", "prompt_codes.i32"):
                p = os.path.join(DUMP, f)
                if os.path.exists(p):
                    os.remove(p)
            t0 = time.perf_counter()
            wav = tts({"text": text, "text_lang": "all_zh", "ref_audio_path": REF,
                       "prompt_text": PROMPT, "prompt_lang": "en", "voice": "turbo-q8",
                       "sample_steps": 4, "seed": 1, "speed_factor": 1.0, "text_split_method": "cut0"})
            print("\n== %s  (%d chars, audio %.2fs, wall %.2fs)"
                  % (name, len(text), (len(wav) - 44) / 2 / 48000, time.perf_counter() - t0))
            sem_p = os.path.join(DUMP, "sem.i32")
            if not os.path.exists(sem_p):
                print("  no dump")
                continue
            sem = np.fromfile(sem_p, dtype="<i4").tolist()
            pc = os.path.join(DUMP, "prompt_codes.i32")
            prompt = np.fromfile(pc, dtype="<i4").tolist() if os.path.exists(pc) else []
            print("  prompt_codes (%d): %s" % (len(prompt), prompt[:20]))
            stats(sem, prompt)
    finally:
        for p in (srv, sc):
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
