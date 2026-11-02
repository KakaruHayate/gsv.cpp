#!/usr/bin/env python
"""Unified GGUF/ONNX exporter for the GPT-SoVITS V5 components.

Kinds
-----
  ar        s1*.ckpt        -> AR (Text2Semantic) GGUF
  cond      s2Gv5*.pth      -> condition segment (enc_p / ref_enc / wns1 / bridge / rvq) GGUF
  dit       s2Gv5*.pth      -> DiT/CFM estimator GGUF
  refcode   s2Gv5*.pth      -> reference semantic-token chain (ssl_proj + codebook) GGUF
  vocoder   vocoder.pth     -> fp32 ONNX (strictly not quantised)

HuBERT / BERT are intentionally **out of scope**: they are plain HuggingFace models
and are exported by the runtime project (gsv.cpp), not by GPT-SoVITS.

This file is a thin dispatcher over the component converters in ../ (same repo) so
that upstream users have one entry point, plus a --self-test that re-opens the
artifact and validates tensor count / names.

Usage
-----
  python tools/export_gguf/export.py --kind ar --ckpt models/s1v3.ckpt \
      --out build/gsv-ar-f16.gguf --f16 --self-test

Requires: torch, numpy, gguf (pip install gguf); onnx for the vocoder export.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(TOOLS)

KINDS = {
    "ar":      ("convert_ar.py",            295, True),
    "cond":    ("convert_cond.py",          298, True),
    "dit":     ("convert_dit.py",           365, True),
    "refcode": ("convert_refcode.py",         4, True),
    "vocoder": ("export_vocoder_onnx.py",  None, False),
}


def build_cmd(kind, a):
    script, _, _ = KINDS[kind]
    cmd = [sys.executable, os.path.join(TOOLS, script), "--ckpt", a.ckpt, "--out", a.out]
    if kind == "ar":
        # 显式的组级 spec 优先；否则 --q8 / --f16 落到推荐的分组
        if a.spec:
            cmd += ["--spec", a.spec]
        elif a.q8:
            cmd += ["--spec", "attn=q8_0,ffn=q8_0"]
        elif a.f16:
            cmd += ["--spec", "attn=f16,ffn=f16,predict=f16,emb=f16"]
    elif kind in ("cond", "dit") and a.spec:
        cmd += ["--spec", a.spec]
    if kind == "vocoder":
        if a.opset:
            cmd += ["--opset", str(a.opset)]
        if not a.bench:
            cmd.append("--skip-bench")
    return cmd


def self_test(kind, path):
    if not os.path.exists(path):
        raise SystemExit("[self-test] missing output: %s" % path)
    if kind == "vocoder":
        size = os.path.getsize(path)
        if size < 1_000_000:
            raise SystemExit("[self-test] vocoder onnx too small: %d bytes" % size)
        return "onnx ok (%d bytes)" % size

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
    ap = argparse.ArgumentParser(description="GGUF/ONNX exporter for GPT-SoVITS V5 components")
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--ckpt", required=True, help="input checkpoint (.ckpt / .pth)")
    ap.add_argument("--out", required=True, help="output .gguf / .onnx")
    ap.add_argument("--spec", default="", help="per-group dtypes, e.g. attn=f16,ffn=f16")
    ap.add_argument("--f16", action="store_true", help="ar: store weights as F16 (norm/bias/emb stay F32)")
    ap.add_argument("--q8", action="store_true", help="ar: store attn/ffn as Q8_0")
    ap.add_argument("--opset", type=int, default=17, help="vocoder: onnx opset")
    ap.add_argument("--bench", action="store_true", help="vocoder: also run the torch-vs-onnx benchmark")
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
