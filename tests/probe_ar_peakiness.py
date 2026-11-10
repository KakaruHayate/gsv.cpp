"""How peaked is the AR token distribution? Determines the acceptance ceiling of any draft.

GSV_AR_PROB_DUMP writes the post-sampler probability vector ([steps][vocab], seq 0) per request.
For a *hard* draft (n-gram, lookahead, greedy self-draft) the rejection-sampling acceptance is
exactly p(draft token), so mean p(top1) is the ceiling. Also reports how much mass the top_k=15
candidates hold and the entropy, i.e. how much room a *sampled* draft (self-speculative) has.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

REPO = r"J:\GPT-SoVITS\gsv.cpp"
PORT = 9991
SC_PORT = 9993
REF = r"H:\downloads\LJ037-0171.wav"
PROMPT = "on to conclude that five shots may have been fired."
TEXT = ("所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，"
        "你知道每个人的手上剩下的筹码是多少。")

CASES = [
    ("top_k=15 temp=1.0 (default)", {}),
    ("top_k=15 temp=0.7", {"temperature": 0.7}),
    ("top_k=5  temp=1.0", {"top_k": 5}),
]


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


def main():
    env = os.environ.copy()
    env["GSV_G2PW_MODEL"] = os.path.join(REPO, "models", "G2PWModel")
    env["GSV_BERT_DIR"] = os.path.join(REPO, "models", "chinese-roberta-wwm-ext-large")
    env["NLTK_DATA"] = os.path.join(REPO, ".nltk_data")
    env["GSV_FAST_LANGDETECT"] = os.path.join(REPO, "models", "frontend", "fast_langdetect")
    dump = os.path.join(REPO, "tests", "_prob_dump.f32")
    env["GSV_AR_PROB_DUMP"] = dump
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
        print("%-30s %6s %10s %10s %10s" % ("case", "steps", "mean p1", "min p1", "top5 mass"))
        for name, extra in CASES:
            if os.path.exists(dump):
                os.remove(dump)
            body = {"text": TEXT, "text_lang": "all_zh", "ref_audio_path": REF,
                    "prompt_text": PROMPT, "prompt_lang": "en", "voice": "turbo-q8",
                    "sample_steps": 4, "seed": 1, "text_split_method": "cut0"}
            body.update(extra)
            req = urllib.request.Request("http://127.0.0.1:%d/tts" % PORT,
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
            if not os.path.exists(dump):
                print("%-30s  (no dump)" % name)
                continue
            p = np.fromfile(dump, dtype="<f4")
            # vocab size from the model: 1025
            V = 1025
            if p.size % V:
                print("%-30s  unexpected dump size %d" % (name, p.size))
                continue
            P = p.reshape(-1, V)
            p1 = P.max(axis=1)
            top5 = np.sort(P, axis=1)[:, -5:].sum(axis=1)
            print("%-30s %6d %10.3f %10.3f %10.3f"
                  % (name, P.shape[0], p1.mean(), p1.min(), top5.mean()))
    finally:
        for p_ in (srv, sc):
            try:
                p_.terminate()
                p_.wait(timeout=10)
            except Exception:
                p_.kill()
        if os.path.exists(dump):
            os.remove(dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
