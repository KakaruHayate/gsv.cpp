#!/usr/bin/env python3
"""Assemble the Windows one-click package (extract and run).

    python scripts/package_windows.py --out dist/GSV.CPP-win64

Layout produced:

    GSV.CPP-win64/
      Start GAG.bat            # runtime\\python\\python.exe gsv_api_gui.py
      gsv_api_gui.py  GAG_tools/ translations/ resources_rc.py
      GAG_config.json          # presets pointing at runtime/
      README.txt
      runtime/
        gsv/                   # gsv_server.exe + ggml DLLs
        python/                # embeddable CPython + site-packages (GAG + frontend deps)
        frontend/              # tools/text_sidecar.py + frontend code (GPT_SoVITS/text)
        models/                # selected GGUF + resampy + frontend/{g2pw,bert_tokenizer} + nltk_data
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSV = HERE                                               # gsv.cpp checkout
GAG = os.path.abspath(os.path.join(GSV, "..", "GAG"))    # GAG fork checkout
PY_EMBED_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"

TURBO = ["gsv-ar-q8.gguf", "gsv-cond-f16.gguf", "gsv-dit-f16all.gguf", "gsv-refcode.gguf",
         "gsv-bert-f16.gguf", "gsv-hubert-f16.gguf", "gsv-vocoder-f32.gguf",
         "resampy_kaiser_best.bin"]
V5DEV = ["gsv-v5dev-cond-f16.gguf", "gsv-v5dev-dit-f16all.gguf", "gsv-v5dev-refcode.gguf"]

GAG_REQS = ["PyQt5", "pyqtdarktheme", "psutil", "requests", "sounddevice", "soundfile", "pyqtgraph"]
FRONTEND_ZH = ["numpy>=1.26", "onnxruntime>=1.20", "pypinyin", "jieba_fast", "cn2an", "OpenCC", "nltk"]
# best effort; failures are reported but do not abort the build
FRONTEND_EXTRA = ["g2p_en", "inflect", "wordsegment", "jieba", "fast-langdetect>=0.3.1",
                  "split-lang", "pyopenjtalk>=0.4.1", "ToJyutping", "jamo"]


def log(msg):
    print("[pkg] " + msg, flush=True)


def copytree(src, dst, ignore=None):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def ensure_python(runtime_python, cache_dir):
    py = os.path.join(runtime_python, "python.exe")
    if os.path.exists(py):
        return
    os.makedirs(runtime_python, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "python-3.10.11-embed-amd64.zip")
    if not os.path.exists(zpath):
        log("downloading %s" % PY_EMBED_URL)
        urllib.request.urlretrieve(PY_EMBED_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(runtime_python)
    pth = os.path.join(runtime_python, "python310._pth")
    with open(pth, "w", encoding="utf-8") as f:
        f.write("python310.zip\n.\nLib\\site-packages\nimport site\n")
    log("embeddable python ready")


def pip_install(target, packages, extra_args=()):
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location",
           "--target", target, "--prefer-binary"] + list(extra_args) + list(packages)
    log("pip install %d package(s)" % len(packages))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("pip failed:\n" + (r.stdout or "")[-1500:] + (r.stderr or "")[-1500:])
    return r.returncode == 0



CRT_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "concrt140.dll"]

def copy_crt(*targets):
    src_dirs = [os.path.join(GSV, "tests", "relcpu"), os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32")]
    for name in CRT_DLLS:
        for d in src_dirs:
            src = os.path.join(d, name)
            if os.path.exists(src):
                for t in targets:
                    shutil.copy2(src, os.path.join(t, name))
                break
    log("VC runtime DLLs copied")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(GSV, "dist", "GSV.CPP-win64"))
    ap.add_argument("--server-zip", default=os.path.join(GSV, "..", "_hf_upload", "server",
                                                         "gsv.cpp-windows-x64-vulkan.zip"))
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--reuse-models", action="store_true", help="keep an existing runtime/models")
    ap.add_argument("--skip-pip", action="store_true")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    log("output: " + out)
    if not os.path.isdir(GAG):
        raise SystemExit("GAG checkout not found: " + GAG)
    os.makedirs(out, exist_ok=True)

    # 1. GAG sources
    for name in ["gsv_api_gui.py", "GAG_tools", "translations", "resources_rc.py", "requirements.txt", "LICENSE"]:
        src = os.path.join(GAG, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out, name)
        if os.path.isdir(src):
            copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    log("GAG sources copied")

    runtime = os.path.join(out, "runtime")
    # 2. engine
    gsv_dir = os.path.join(runtime, "gsv")
    if os.path.isdir(gsv_dir):
        shutil.rmtree(gsv_dir)
    os.makedirs(gsv_dir)
    with zipfile.ZipFile(args.server_zip) as z:
        z.extractall(gsv_dir)
    log("engine extracted from " + args.server_zip)

    # 3. frontend code
    fe = os.path.join(runtime, "frontend")
    if os.path.isdir(fe):
        shutil.rmtree(fe)
    os.makedirs(fe)
    shutil.copy2(os.path.join(GSV, "tools", "text_sidecar.py"), os.path.join(fe, "text_sidecar.py"))
    copytree(os.path.join(GSV, "frontend"), os.path.join(fe, "frontend"),
             ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "G2PWModel"))
    log("frontend code copied")

    # 4. models
    models = os.path.join(runtime, "models")
    os.makedirs(models, exist_ok=True)
    if args.reuse_models and os.path.exists(os.path.join(models, "gsv-ar-q8.gguf")):
        log("models reused")
    elif not args.no_models:
        names = TURBO + V5DEV
        for n in names:
            src = os.path.join(GSV, "models", n)
            if not os.path.exists(src):
                log("missing model: " + n)
                continue
            shutil.copy2(src, os.path.join(models, n))
        # frontend data
        data = os.path.join(models, "frontend")
        os.makedirs(os.path.join(data, "g2pw"), exist_ok=True)
        os.makedirs(os.path.join(data, "bert_tokenizer"), exist_ok=True)
        for f in os.listdir(os.path.join(GSV, "models", "G2PWModel")):
            s = os.path.join(GSV, "models", "G2PWModel", f)
            if os.path.isfile(s) and not f.endswith(".pyc"):
                shutil.copy2(s, os.path.join(data, "g2pw", f))
        bt = os.path.join(GSV, "..", "repo", "GPT_SoVITS", "pretrained_models",
                          "chinese-roberta-wwm-ext-large")
        for f in ["vocab.txt", "config.json", "tokenizer_config.json"]:
            shutil.copy2(os.path.join(bt, f), os.path.join(data, "bert_tokenizer", f))
        nltk_src = os.path.join(GSV, ".nltk_data")
        if os.path.isdir(nltk_src):
            copytree(nltk_src, os.path.join(models, "nltk_data"))
        log("models copied")

    # 5. embeddable python + deps
    py_dir = os.path.join(runtime, "python")
    ensure_python(py_dir, os.path.join(GSV, "dist", "cache"))
    if not args.skip_pip:
        site = os.path.join(py_dir, "Lib", "site-packages")
        os.makedirs(site, exist_ok=True)
        pip_install(site, GAG_REQS + FRONTEND_ZH)
        pip_install(site, FRONTEND_EXTRA)

    copy_crt(py_dir, gsv_dir)
    # 6. launcher, config, readme
    bat = os.path.join(out, "Start GAG.bat")
    with open(bat, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("@echo off\r\nchcp 65001 >nul\r\ncd /d \"%~dp0\"\r\n")
        f.write("start \"\" \"runtime\\python\\python.exe\" gsv_api_gui.py\r\n")
    cfg = os.path.join(out, "GAG_config.json")
    import json
    cfg_data = {
        "api_url": "http://127.0.0.1:9880",
        "gsv_server_path": "runtime/gsv/gsv_server.exe",
        "models_dir": "runtime/models",
        "device": "vulkan",
        "sidecar_python": "runtime/python/python.exe",
        "sidecar_data_dir": "runtime/models/frontend",
        "allow_dirs": "runtime/models",
        "autostart_api": True,
        "language": "auto",
    }
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as f:
        f.write("GSV.CPP one-click package\n\n"
                "1. Extract this folder anywhere (ASCII path recommended).\n"
                "2. Double-click 'Start GAG.bat'.\n"
                "3. The API tab starts gsv_server + the text sidecar automatically.\n\n"
                "Models live in runtime\\models, the engine in runtime\\gsv.\n"
                "If Vulkan is unavailable, set Device to 'cpu' in the API tab.\n")
    log("done: " + out)


if __name__ == "__main__":
    main()
