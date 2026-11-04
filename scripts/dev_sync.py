#!/usr/bin/env python3
"""Fast dev sync: copy changed files from the built package into a test install.

    python scripts/dev_sync.py --from dist/GSV.CPP-win64 --to <install dir>

Only files whose size or mtime differ are copied, so iterating on the GUI/frontend/server takes
seconds instead of rebuilding and re-zipping the 3 GB package. Models are skipped by default
(--with-models to include them). The zip is only rebuilt once testing is finished.
"""
import argparse
import filecmp
import os
import shutil
import sys

SKIP_DIRS = {"__pycache__"}
SKIP_TOP = {"runtime/models"}          # 2.6 GB, unchanged during UI/engine fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--with-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gag", help="also copy GUI sources from a GAG checkout (avoids a stale "
                                  "dist/ build when only Python code changed)")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isdir(src):
        raise SystemExit("source not found: " + src)
    if not os.path.isdir(dst):
        raise SystemExit("target not found: " + dst)

    copied = skipped = 0
    if args.gag:
        gag = os.path.abspath(args.gag)
        for name in ("gsv_api_gui.py", "launch_gag.py", "GAG_tools", "translations",
                     "resources_rc.py"):
            s_path = os.path.join(gag, name)
            if not os.path.exists(s_path):
                continue
            if os.path.isdir(s_path):
                for root, dirs, files in os.walk(s_path):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    rel = os.path.relpath(root, gag)
                    out = os.path.join(dst, rel)
                    for f in files:
                        if f.endswith((".pyc", ".pyo")):
                            continue
                        sp, dp = os.path.join(root, f), os.path.join(out, f)
                        if os.path.exists(dp) and filecmp.cmp(sp, dp, shallow=False):
                            skipped += 1
                            continue
                        if args.dry_run:
                            print("would copy", os.path.join(rel, f))
                            copied += 1
                            continue
                        os.makedirs(out, exist_ok=True)
                        shutil.copy2(sp, dp)
                        copied += 1
            else:
                shutil.copy2(s_path, os.path.join(dst, name))
                copied += 1
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        rel_norm = rel.replace("\\", "/")
        if any(rel_norm == s or rel_norm.startswith(s + "/") for s in SKIP_TOP) and not args.with_models:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out = dst if rel == "." else os.path.join(dst, rel)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            s_path = os.path.join(root, name)
            d_path = os.path.join(out, name)
            if os.path.exists(d_path) and filecmp.cmp(s_path, d_path, shallow=False):
                skipped += 1
                continue
            if args.dry_run:
                print("would copy", os.path.join(rel, name))
                copied += 1
                continue
            os.makedirs(out, exist_ok=True)
            try:
                shutil.copy2(s_path, d_path)
                copied += 1
            except PermissionError:
                print("locked, close the GUI/backend first:", d_path)
    print("copied %d file(s), %d unchanged" % (copied, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
