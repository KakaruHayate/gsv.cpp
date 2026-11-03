#!/usr/bin/env python3
"""Refresh frontend/GPT_SoVITS/text from a GPT-SoVITS checkout and re-apply our two edits.

    python frontend/sync_frontend.py --repo /path/to/GPT-SoVITS
    python frontend/sync_frontend.py --repo /path/to/GPT-SoVITS --check

The vendored tree keeps the upstream layout so `git diff` against a checkout is readable.
Edits applied after copying (both are asserted, so an upstream refactor fails loudly):
  1. text/g2pw/onnx_api.py: use gsv_bert_tokenizer instead of transformers' AutoTokenizer
  2. text/chinese2.py:      G2PWModel directory honours GSV_G2PW_MODEL
"""
import argparse
import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "GPT_SoVITS", "text")
SKIP_DIRS = {"__pycache__", "G2PWModel"}
SKIP_EXT = {".pyc"}

TOKENIZER_IMPORT = "from transformers.models.auto.tokenization_auto import AutoTokenizer"
TOKENIZER_IMPORT_NEW = '''try:  # gsv.cpp: tokenizer without transformers/torch (frontend/gsv_bert_tokenizer.py)
    from gsv_bert_tokenizer import BertWordPieceTokenizer
except Exception:  # upstream checkout without our overlay: keep the original behaviour
    BertWordPieceTokenizer = None'''

TOKENIZER_INIT = "        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)"
TOKENIZER_INIT_NEW = '''        if BertWordPieceTokenizer is not None:
            vocab = os.path.join(self.model_source, "vocab.txt")
            if os.path.exists(vocab):
                self.tokenizer = BertWordPieceTokenizer(vocab)
            else:
                from transformers.models.auto.tokenization_auto import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)
        else:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_source)'''

CHINESE2_OLD = '        model_dir="GPT_SoVITS/text/G2PWModel",'
CHINESE2_NEW = '        model_dir=os.environ.get("GSV_G2PW_MODEL", "GPT_SoVITS/text/G2PWModel"),'


def copy_tree(src, dst):
    copied = changed = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for name in files:
            if os.path.splitext(name)[1] in SKIP_EXT:
                continue
            s, d = os.path.join(root, name), os.path.join(out, name)
            copied += 1
            if os.path.exists(d) and filecmp.cmp(s, d, shallow=False):
                continue
            changed += 1
            shutil.copy2(s, d)
    return copied, changed


def patch_file(path, pairs):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for old, new in pairs:
        if new in text:
            continue
        if old not in text:
            raise SystemExit("sync_frontend: anchor not found in %s:\n  %s" % (path, old.strip()))
        text = text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a GPT-SoVITS checkout")
    ap.add_argument("--check", action="store_true", help="only report differences")
    args = ap.parse_args()

    src = os.path.join(args.repo, "GPT_SoVITS", "text")
    if not os.path.isdir(src):
        raise SystemExit("not found: %s" % src)

    if args.check:
        drift = []
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            rel = os.path.relpath(root, src)
            for name in files:
                if os.path.splitext(name)[1] in SKIP_EXT:
                    continue
                s = os.path.join(root, name)
                d = os.path.join(DST, rel, name) if rel != "." else os.path.join(DST, name)
                if not os.path.exists(d) or not filecmp.cmp(s, d, shallow=False):
                    drift.append(os.path.join(rel, name))
        print("drift: %d file(s)" % len(drift))
        for f in sorted(drift)[:40]:
            print("  ", f)
        return 1 if drift else 0

    copied, changed = copy_tree(src, DST)
    patch_file(os.path.join(DST, "g2pw", "onnx_api.py"),
               [(TOKENIZER_IMPORT, TOKENIZER_IMPORT_NEW),
                (TOKENIZER_INIT, TOKENIZER_INIT_NEW)])
    patch_file(os.path.join(DST, "chinese2.py"), [(CHINESE2_OLD, CHINESE2_NEW)])
    print("synced %d file(s), %d changed; patches applied" % (copied, changed))
    print("review with: git diff -- frontend/GPT_SoVITS/text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
