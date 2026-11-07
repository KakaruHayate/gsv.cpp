#!/usr/bin/env python
"""Compare / time the polyphone backend of the text frontend.

    python frontend/check_g2ps.py --backend g2ps --repeat 20
    python frontend/check_g2ps.py --backend g2pw --repeat 20     # needs g2pW.onnx

Emits one JSON document so two runs can be diffed:

    python frontend/check_g2ps.py --backend g2ps > /tmp/a.json
    python frontend/check_g2ps.py --backend g2pw > /tmp/b.json
    python - <<'EOF'
    import json
    a, b = (json.load(open(p)) for p in ("/tmp/a.json", "/tmp/b.json"))
    for x, y in zip(a["cases"], b["cases"]):
        print("SAME" if x == y else "DIFF", x["text"])
    EOF

Exits non-zero if any case raises (e.g. the sparse-preset None crash).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    "人工智能的发展正在深刻改变我们的生活方式，语音合成技术已经可以生成非常自然的人声。",
    "第一，下一季度的产品路线图",
    "关于第一点",
    "第1，第2",
    "我去银行取钱",
    "他的行踪很隐蔽",
    "这是重量级选手",
    "重新开始",
    "你好，世界",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="g2ps", choices=["g2ps", "g2pw"])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()

    os.environ["GSV_G2P_BACKEND"] = args.backend
    os.environ.setdefault("version", "v2")
    # chinese2 resolves the G2PW model dir relative to cwd at import time
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "GPT_SoVITS"))

    t0 = time.perf_counter()
    from text.cleaner import clean_text  # noqa: E402

    import_time = time.perf_counter() - t0

    out = {"backend": args.backend, "import_s": round(import_time, 3), "cases": [], "timing": []}
    for text in CASES:
        case = {"text": text}
        try:
            phones, word2ph, norm_text = clean_text(text, args.lang, "v2")
            case.update(
                {
                    "norm_text": norm_text,
                    "phones_len": len(phones),
                    "word2ph": list(word2ph),
                    "phones": [int(p) for p in phones],
                }
            )
        except Exception as exc:  # keep going, report per case
            case["error"] = f"{type(exc).__name__}: {exc}"
        out["cases"].append(case)

    texts = [c["text"] for c in CASES if "error" not in c]
    for text in texts:  # warm the per-sentence caches once
        clean_text(text, args.lang, "v2")
    t0 = time.perf_counter()
    chars = 0
    for _ in range(args.repeat):
        for text in texts:
            clean_text(text, args.lang, "v2")
            chars += len(text)
    dt = time.perf_counter() - t0
    out["timing"] = {
        "repeat": args.repeat,
        "chars": chars,
        "total_s": round(dt, 3),
        "ms_per_sentence": round(dt / (args.repeat * len(texts)) * 1000, 2),
        "ms_per_char": round(dt / chars * 1000, 3),
    }

    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 1 if any("error" in c for c in out["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
