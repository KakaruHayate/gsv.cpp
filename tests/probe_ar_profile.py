"""Where does AR decode time go? Runs the 0.1.3 server with GSV_AR_PROFILE=1."""
import json
import os
import subprocess
import sys
import time
import urllib.request

REPO = r"J:\GPT-SoVITS\gsv.cpp"
PORT = 9981
SC_PORT = 9983
REF = r"H:\downloads\LJ037-0171.wav"
PROMPT = "on to conclude that five shots may have been fired."
TEXT = ("所以你拿到四数据，相当于是开了天眼，你知道每个人账户地址是多少，"
        "你知道每个人的手上剩下的筹码是多少。你知道每个人爆仓的点位是多少！")


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
    env["GSV_AR_PROFILE"] = "1"
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
    log = open(os.path.join(REPO, "tests", "_ar_profile.log"), "wb")
    srv = subprocess.Popen([os.path.join(REPO, "tests", "relcpu", "gsv_server.exe"),
                            "--models", os.path.join(REPO, "models"), "--device", "vulkan",
                            "--port", str(PORT), "--sidecar", "127.0.0.1:%d" % SC_PORT,
                            "--allow-path", "*", "--verbose"],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not wait("http://127.0.0.1:%d/health" % PORT):
        print("server failed")
        sc.kill()
        return 1
    try:
        # warm up (shader compile) then measure
        for run in range(2):
            body = {"text": TEXT, "text_lang": "all_zh", "ref_audio_path": REF,
                    "prompt_text": PROMPT, "prompt_lang": "en", "voice": "turbo-q8",
                    "sample_steps": 4, "seed": 1, "speed_factor": 1.0, "text_split_method": "cut0"}
            req = urllib.request.Request("http://127.0.0.1:%d/tts" % PORT,
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=600) as r:
                wav = r.read()
            print("run %d: wall %.2fs, audio %.2fs" % (run, time.perf_counter() - t0,
                                                       (len(wav) - 44) / 2 / 48000))
    finally:
        for p in (srv, sc):
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                p.kill()
        log.close()

    print("\n=== AR profile lines ===")
    with open(os.path.join(REPO, "tests", "_ar_profile.log"), "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "prof:decode" in line or "ar_generate" in line or "fused ops" in line:
                print(line.rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
