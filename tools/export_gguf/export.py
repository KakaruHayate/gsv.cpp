#!/usr/bin/env python
"""Unified GGUF exporter for the GPT-SoVITS V5 components.

Kinds
-----
  ar        s1*.ckpt        -> AR (Text2Semantic) GGUF
  cond      s2Gv5*.pth      -> condition segment (enc_p / ref_enc / wns1 / bridge / rvq) GGUF
  dit       s2Gv5*.pth      -> DiT/CFM estimator GGUF
  refcode   s2Gv5*.pth      -> reference semantic-token chain (ssl_proj + codebook) GGUF
  vocoder   vocoder.pth     -> F32 HiFi-GAN vocoder GGUF (sub-pixel layout, never quantised)
  bert      chinese-roberta -> BERT GGUF (optional: download the prebuilt file instead)
  hubert    chinese-hubert  -> CN-HuBERT GGUF (optional: download the prebuilt file instead)

Everything the engine consumes is GGUF; ONNX is not used anywhere anymore.

All converters live next to this file so the export tooling is self-contained.

Usage
-----
  python tools/export_gguf/export.py --kind ar --ckpt models/s1v3.ckpt \
      --out build/gsv-ar-f16.gguf --f16 --self-test

Requires: torch, numpy, gguf (pip install gguf); transformers for bert/hubert.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

KINDS = {
    "ar":      ("convert_ar.py",            295, True),
    "cond":    ("convert_cond.py",          298, True),
    "dit":     ("convert_dit.py",           365, True),
    "refcode": ("convert_refcode.py",         4, True),
    "vocoder": ("export_vocoder_gguf.py",  None, False),
    "bert":    ("convert_bert.py",         None, False),
    "hubert":  ("convert_hubert.py",       None, False),
    "dhubert": ("convert_distilled_hubert.py", 102, False),
    # v2-family SoVITS (VITS decoder). Not wired into the engine yet: the converter fixes the
    # GGUF contract for src/gsv_v2sovits.cpp, which is still to be written.
    "v2sovits": ("convert_v2sovits.py",    None, False),
}


def build_cmd(kind, a):
    script, _, _ = KINDS[kind]
    cmd = [sys.executable, os.path.join(HERE, script), "--ckpt", a.ckpt, "--out", a.out]
    if kind == "ar":
        # 显式的组级 spec 优先；否则 --q8 / --f16 落到推荐的分组
        if a.spec:
            cmd += ["--spec", a.spec]
        elif a.q8:
            cmd += ["--spec", "attn=q8_0,ffn=q8_0"]
        elif a.f16:
            # emb 不能压: gsv_ar 在 host 侧直读 text_emb / audio_emb (logits 与 embed 表),
            # 非 F32 会直接加载失败。所以 F16 档只压 attn/ffn/predict。
            cmd += ["--spec", "attn=f16,ffn=f16,predict=f16"]
    elif kind == "v2sovits":
        if a.f16:
            cmd += ["--f16"]
        if a.version:
            cmd += ["--version", a.version]
    elif kind in ("cond", "dit") and a.spec:
        cmd += ["--spec", a.spec]
    return cmd


def self_test(kind, path):
    if not os.path.exists(path):
        raise SystemExit("[self-test] missing output: %s" % path)

    try:
        from gguf import GGUFReader
    except ImportError:
        raise SystemExit("[self-test] pip install gguf (llama.cpp ecosystem)")

    r = GGUFReader(path)
    n = len(r.tensors)
    exp = KINDS[kind][1]
    if exp is not None and n != exp:
        raise SystemExit("[self-test] %s: %d tensors, expected %d" % (kind, n, exp))
    names = [t.name for t in r.tensors]
    if len(set(names)) != len(names):
        raise SystemExit("[self-test] duplicate tensor names")
    if kind == "refcode" and not all(x.startswith("refcode.") for x in names):
        raise SystemExit("[self-test] refcode: unexpected tensor names")
    return "%d tensors ok (e.g. %s)" % (n, names[0] if names else "-")


def main():
    ap = argparse.ArgumentParser(description="GGUF exporter for GPT-SoVITS V5 components")
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--ckpt", required=True, help="input checkpoint (.ckpt / .pth / HF dir)")
    ap.add_argument("--out", required=True, help="output .gguf")
    ap.add_argument("--spec", default="", help="per-group dtypes, e.g. attn=f16,ffn=f16")
    ap.add_argument("--f16", action="store_true", help="ar: store weights as F16 (norm/bias/emb stay F32)")
    ap.add_argument("--q8", action="store_true", help="ar: store attn/ffn as Q8_0")
    ap.add_argument("--version", default=None,
                    help="v2sovits: v2 | v2pro | v2proplus (sample rate / widths; default v2)")
    ap.add_argument("--self-test", action="store_true", help="re-open the artifact and validate it")
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        raise SystemExit("checkpoint not found: %s" % a.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    cmd = build_cmd(a.kind, a)
    print("[export] %s\n[export] %s" % (a.kind, " ".join(cmd)), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        raise SystemExit("[export] converter failed (%d)" % rc)

    print("[export] wrote %s (%.1f MiB)" % (a.out, os.path.getsize(a.out) / 1048576.0))
    if a.self_test:
        print("[self-test] " + self_test(a.kind, a.out))


if __name__ == "__main__":
    main()

