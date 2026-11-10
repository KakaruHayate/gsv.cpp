"""Measure prompt-lookup (n-gram) acceptance on real AR token streams.

Runs the 0.1.3 server with GSV_ENGINE_DUMP so the engine writes sem.i32 (generated tokens) and
prompt_codes.i32 (reference-audio tokens). Then simulates the lookup decoder: at each step take
the last n generated tokens, find the most recent earlier occurrence in history
(prompt + generated), and use the following K tokens as the draft. Acceptance is measured as the
greedy match length (upper bound: sampling mode can only do worse).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np

REPO = r"J:\GPT-SoVITS\gsv.cpp"
DUMP = os.path.join(REPO, "tests", "_lookup_dump")
PORT = 9971
SC_PORT = 9973
REF = r"H:\downloads\LJ037-0171.wav"
PROMPT = "on to conclude that five shots may have been fired."

TEXTS = {
    "short": "你好，欢迎使用语音合成系统。",
    "repeat": "哈哈哈哈哈哈哈哈哈哈，慢慢慢，好好好，对对对。",
    "long": ("所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，"
             "你知道每个人的手上剩下的筹码是多少。"),
    "mixed": "Hello，你好！Welcome to 这个多语种混合测试。",
}


def wait(url, timeout=240):
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


def lookup_drafts(hist, sem, n, K):
    """Simulate: for every generated position, propose a draft from the most recent n-gram match."""
    accepted, proposed, steps_with_draft = [], [], 0
    i = 0
    while i < len(sem):
        if i == 0:
            i += 1
            continue
        ctx = hist[: len(hist) - len(sem) + i]          # everything before position i
        draft = []
        if len(ctx) >= n:
            key = ctx[-n:]
            for j in range(len(ctx) - n - 1, -1, -1):  # most recent earlier occurrence
                if ctx[j : j + n] == key:
                    draft = ctx[j + n : j + n + K]
                    break
        if not draft:
            i += 1
            continue
        steps_with_draft += 1
        match = 0
        while match < len(draft) and i + match < len(sem) and draft[match] == sem[i + match]:
            match += 1
        accepted.append(match)
        proposed.append(len(draft))
        i += max(1, match)
    return accepted, proposed, steps_with_draft


def main():
    os.makedirs(DUMP, exist_ok=True)
    env = os.environ.copy()
    env["GSV_ENGINE_DUMP"] = DUMP
    env["GSV_G2PW_MODEL"] = os.path.join(REPO, "models", "G2PWModel")
    env["GSV_BERT_DIR"] = os.path.join(REPO, "models", "chinese-roberta-wwm-ext-large")
    env["NLTK_DATA"] = os.path.join(REPO, ".nltk_data")
    env["GSV_FAST_LANGDETECT"] = os.path.join(REPO, "models", "frontend", "fast_langdetect")

    py = sys.executable
    sc = subprocess.Popen([py, os.path.join(REPO, "tools", "text_sidecar.py"),
                           "--port", str(SC_PORT)],
                          cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait("http://127.0.0.1:%d/health" % SC_PORT):
        print("sidecar failed")
        sc.kill()
        return 1

    srv = subprocess.Popen([os.path.join(REPO, "tests", "relcpu", "gsv_server.exe"),
                            "--models", os.path.join(REPO, "models"),
                            "--device", "vulkan", "--port", str(PORT),
                            "--sidecar", "127.0.0.1:%d" % SC_PORT, "--allow-path", "*"],
                           cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait("http://127.0.0.1:%d/health" % PORT):
        print("server failed")
        srv.kill(), sc.kill()
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
                       "sample_steps": 4, "seed": 1, "speed_factor": 1.0,
                       "text_split_method": "cut0"})
            wall = time.perf_counter() - t0
            sem_p = os.path.join(DUMP, "sem.i32")
            if not os.path.exists(sem_p):
                print("%-8s no dump" % name)
                continue
            sem = np.fromfile(sem_p, dtype="<i4").tolist()
            pc_p = os.path.join(DUMP, "prompt_codes.i32")
            prompt = np.fromfile(pc_p, dtype="<i4").tolist() if os.path.exists(pc_p) else []
            hist = prompt + sem
            audio = (len(wav) - 44) / 2 / 48000
            print("\n== %s: %d chars, %d sem tokens, prompt_codes=%d, audio %.2fs, wall %.2fs"
                  % (name, len(text), len(sem), len(prompt), audio, wall))
            for n in (2, 3):
                for K in (3, 5, 8):
                    acc, prop, swd = lookup_drafts(hist, sem, n, K)
                    if not acc:
                        print("   n=%d K=%d  no drafts" % (n, K))
                        continue
                    total_acc = sum(acc)
                    mean = total_acc / len(acc)
                    print("   n=%d K=%d  drafts=%d  mean_accept=%.2f  tokens_hit=%d/%d (%.1f%%)"
                          % (n, K, swd, mean, total_acc, len(sem), 100.0 * total_acc / len(sem)))
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
