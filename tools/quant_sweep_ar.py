# AR 量化扫描: 对每个 per-group 类型配置 → 转换 GGUF → 跑引擎验收 (logits Δ + 100 token greedy 一致性)
# 用法: python tools/quant_sweep_ar.py [--only name1,name2]
import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
EXE = os.path.join(ROOT, "tests", "test_ar_engine.exe")   # CPU 版 (f32 参考路径)
EXE_VK = os.path.join(ROOT, "tests", "vk", "test_ar_engine_vk.exe")  # Vulkan 版
GOLDEN_VK = os.path.join("..", "golden")
CKPT = os.path.join(ROOT, "models", "s1v3.ckpt")
GOLDEN = os.path.join(ROOT, "tests", "golden")
OUTDIR = os.path.join(ROOT, "models", "sweep")

# (name, spec)  —— 组: attn(qkv_w/out_w) ffn(ffn1/ffn2) predict emb(text/audio)
CONFIGS = [
    ("f32",             "attn=f32,ffn=f32,predict=f32,emb=f32"),
    ("f16",             "attn=f16,ffn=f16,predict=f16,emb=f32"),
    # --- 分组敏感度 (单组 Q8_0) ---
    ("q8_attn",         "attn=q8_0,ffn=f32,predict=f32,emb=f32"),
    ("q8_ffn",          "attn=f32,ffn=q8_0,predict=f32,emb=f32"),
    ("q8_predict",      "attn=f32,ffn=f32,predict=q8_0,emb=f32"),
    ("q8_emb",          "attn=f32,ffn=f32,predict=f32,emb=q8_0"),
    # --- 组合 ---
    ("q8_attn_ffn",     "attn=q8_0,ffn=q8_0,predict=f32,emb=f32"),
    ("q8_attn_ffn_p16", "attn=q8_0,ffn=q8_0,predict=f16,emb=f32"),
    ("q8_attn_ffn_emb16","attn=q8_0,ffn=q8_0,predict=f16,emb=f16"),
    # --- 更小的权重档 (legacy quants) ---
    ("f16_q5_1",        "attn=f16,ffn=q5_1,predict=f16,emb=f32"),
    ("f16_q5_0",        "attn=f16,ffn=q5_0,predict=f16,emb=f32"),
    ("f16_q4_1",        "attn=f16,ffn=q4_1,predict=f16,emb=f32"),
    ("f16_q4_0",        "attn=f16,ffn=q4_0,predict=f16,emb=f32"),
    ("q8_q5_1",         "attn=q8_0,ffn=q5_1,predict=f16,emb=f32"),
    ("q8_q5_0",         "attn=q8_0,ffn=q5_0,predict=f16,emb=f32"),
    ("q8_q4_1",         "attn=q8_0,ffn=q4_1,predict=f16,emb=f32"),
    ("q8_q4_0",         "attn=q8_0,ffn=q4_0,predict=f16,emb=f32"),
    ("f16_q8_0",        "attn=f16,ffn=q8_0,predict=f16,emb=f32"),
    ("q5_1_all",        "attn=q5_1,ffn=q5_1,predict=f16,emb=f32"),
    ("attn_q5_1_ffn_q4_0","attn=q5_1,ffn=q4_0,predict=f16,emb=f32"),
    ("q4_1_all",        "attn=q4_1,ffn=q4_1,predict=f16,emb=f32"),
    ("q4_0_all",        "attn=q4_0,ffn=q4_0,predict=f16,emb=f32"),
]


def run(cmd, env=None, timeout=1800):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cpu", choices=["cpu", "vk"])
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    os.makedirs(OUTDIR, exist_ok=True)

    rows = []
    for name, spec in CONFIGS:
        if only and name not in only:
            continue
        gguf = os.path.join(OUTDIR, f"{name}.gguf")
        t0 = time.time()
        r = run([PY, os.path.join(ROOT, "tools", "convert_ar.py"), "--ckpt", CKPT, "--out", gguf, "--spec", spec])
        if r.returncode != 0:
            print(f"[{name}] convert FAILED: {r.stderr.strip()[-200:]}")
            continue
        size_mb = os.path.getsize(gguf) / 1e6
        if args.device == "vk":
            r = subprocess.run([EXE_VK, os.path.abspath(gguf), os.path.abspath(GOLDEN)],
                               capture_output=True, text=True,
                               env={**os.environ, "GSV_AR_ACC": "1", "GSV_AR_DEVICE": "vulkan",
                                    "GGML_VK_DISABLE_COOPMAT2": "1"}, timeout=1800)
        else:
            r = run([EXE, gguf, GOLDEN], env={"GSV_AR_ACC": "1"})
        out = r.stdout + r.stderr
        m1 = re.search(r"logits max\|Δ\| = ([0-9.eE+-]+)", out)
        m4 = re.search(r"token 差异 (\d+)/(\d+) \(首次分歧 @(-?\d+)\)", out)
        logits_d = float(m1.group(1)) if m1 else float("nan")
        tok_bad = int(m4.group(1)) if m4 else -1
        first_bad = int(m4.group(3)) if m4 else -1
        ok = (tok_bad == 0)
        rows.append((name, size_mb, logits_d, tok_bad, first_bad, ok, time.time() - t0))
        print(f"[{name:18s}] {size_mb:6.1f}MB  logitsΔ={logits_d:<10.3g} tokens bad={tok_bad}/3 @{first_bad:<4d} "
              f"{'PASS' if ok else 'FAIL'}  ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    print()
    print("== 汇总 (按体积升序, 仅列通过) ==")
    for name, size_mb, logits_d, tok_bad, first_bad, ok, _ in sorted(rows, key=lambda x: x[1]):
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:20s} {size_mb:7.1f} MB  logitsΔ={logits_d:<10.3g} tokens={3-tok_bad}/3  {flag}")


if __name__ == "__main__":
    main()
