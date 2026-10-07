# AR 量化验收套件: 对给定 GGUF 跑三项检查
#   ① 首步 logits max|Δ| (vs CPU-f32 golden)   ② 100-token greedy 逐 token 一致
#   ③ 教师强制(TF)探针: 100 步采样分布的 TV 距离 (上下文固定为 f32 参考 token 流)
# 用法: python tools/accept_quant.py models/sweep/q8_attn_ffn.gguf [--device cpu|vk]
# 参考分布缓存: /tmp/tf_ref_<device>.bin (首次自动用 models/gsv-ar-f32.gguf 生成)
import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden")
F32_GGUF = os.path.join(ROOT, "models", "gsv-ar-f32.gguf")


def engine_cmd(device, gguf):
    if device == "vk":
        exe = os.path.join(ROOT, "tests", "vk", "test_ar_engine_vk.exe")
        env = {"GSV_AR_DEVICE": "vulkan", "GGML_VK_DISABLE_COOPMAT2": "1"}
    else:
        exe = os.path.join(ROOT, "tests", "test_ar_engine.exe")
        env = {}
    return exe, {**os.environ, **env}


def run_engine(device, gguf, extra_env):
    exe, env = engine_cmd(device, gguf)
    env.update(extra_env)
    r = subprocess.run([exe, os.path.abspath(gguf), os.path.abspath(GOLDEN)],
                       capture_output=True, text=True, env=env, timeout=1800)
    return r.stdout + r.stderr


def probe_tv(device, gguf, ref_bin):
    exe, env = engine_cmd(device, gguf)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        out_bin = tmp.name
    env.update({"GSV_AR_TF": "1", "GSV_AR_PROB_DUMP": out_bin})
    subprocess.run([exe, os.path.abspath(gguf), os.path.abspath(GOLDEN)],
                   capture_output=True, text=True, env=env, timeout=1800)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "compare_probs.py"), ref_bin, out_bin],
                       capture_output=True, text=True, timeout=600)
    os.unlink(out_bin)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--device", default="cpu", choices=["cpu", "vk"])
    ap.add_argument("--ref", default="")
    args = ap.parse_args()

    ref_bin = args.ref or os.path.join(tempfile.gettempdir(), f"tf_ref_{args.device}.bin")
    if not os.path.exists(ref_bin):
        print(f"[accept] 生成参考分布: {os.path.basename(F32_GGUF)} -> {ref_bin}")
        exe, env = engine_cmd(args.device, F32_GGUF)
        env.update({"GSV_AR_TF": "1", "GSV_AR_PROB_DUMP": ref_bin})
        subprocess.run([exe, os.path.abspath(F32_GGUF), os.path.abspath(GOLDEN)],
                       capture_output=True, text=True, env=env, timeout=1800)
        assert os.path.exists(ref_bin), "参考分布生成失败"

    size_mb = os.path.getsize(args.gguf) / 1e6
    acc = run_engine(args.device, args.gguf, {"GSV_AR_ACC": "1"})
    m1 = re.search(r"logits max\|Δ\| = ([0-9.eE+-]+)", acc)
    m4 = re.search(r"token 差异 (\d+)/(\d+) \(首次分歧 @(-?\d+)\)", acc)
    tv = probe_tv(args.device, args.gguf, ref_bin)

    print(f"=== 验收: {os.path.basename(args.gguf)} ({size_mb:.1f} MB, device={args.device}) ===")
    print(f"  首步 logits Δ : {m1.group(1) if m1 else 'n/a'}")
    print(f"  100-token greedy: 差异 {m4.group(1)}/{m4.group(2)} (首次分歧 @{m4.group(3)})" if m4 else "  100-token: n/a")
    print(f"  分布 TV       : {tv.splitlines()[0] if tv else 'n/a'}")
    print(f"  判定          : {'近无损' if (m4 and m4.group(1) == '0') else '有分歧'} / "
          f"TV{'<0.005 ✓' if tv and float(re.search(r'mean=([0-9.]+)', tv).group(1)) < 0.005 else '≥0.005 ✗'}")


if __name__ == "__main__":
    main()
