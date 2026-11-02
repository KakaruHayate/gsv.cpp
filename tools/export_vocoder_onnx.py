# V5 vocoder (HiFi-GAN Generator, 48kHz) → ONNX 导出 + 对拍 + 基准
# 严格 fp32, 不做任何量化/图优化器改写以外的变换 (ONNX 导出本身不改变数值)
#
# 要点:
#  - 模型类取自 GPT_SoVITS/module/models.py::Generator, 配置与 TTS.py::init_vocoder 的 v4/v5 分支一致
#  - 关键顺序: 先 remove_weight_norm() 再 load_state_dict (官方 checkpoint 是"已物化 weight/bias"的版本)
#  - 输入 mel [B, 100, T] (denorm_spec 之后), 输出 wav [B, 1, T*480]
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("GSV_REPO", os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "repo")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from module.models import Generator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V4_V5_VOCODER = dict(
    initial_channel=100,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 6, 2, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[20, 12, 4, 4, 4],
    gin_channels=0,
    is_bias=True,
)


def build_vocoder(ckpt_path):
    m = Generator(**V4_V5_VOCODER)
    m.remove_weight_norm()          # 必须在 load_state_dict 之前 (checkpoint 为物化版本)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("conv_pre") for k in sd):
        sd = sd["model"]
    info = m.load_state_dict(sd)
    print(f"[export] load_state_dict: {info}")
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "vocoder.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "vocoder_fp32.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--bench-t", type=int, default=1000, help="基准用 mel 帧数 (1000 帧 = 10s @100fps)")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    m = build_vocoder(args.ckpt)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"[export] Generator 参数量 {n_par/1e6:.1f}M (fp32 ≈ {n_par*4/1e6:.0f} MB)")

    # ---- 导出 (fp32, 动态 T) ----
    T_dummy = 100
    dummy = torch.randn(1, 100, T_dummy)
    with torch.no_grad():
        y = m(dummy)
    print(f"[export] dummy: mel {tuple(dummy.shape)} -> wav {tuple(y.shape)} (x{ y.shape[-1]//dummy.shape[-1] } 上采样)")

    torch.onnx.export(
        m, dummy, args.out,
        input_names=["mel"], output_names=["wav"],
        dynamic_axes={"mel": {0: "B", 2: "T"}, "wav": {0: "B", 2: "T_x480"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    import hashlib
    sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    print(f"[export] 写出 {args.out}  {size_mb:.1f} MB  sha256:{sha}...")

    # ---- ONNX Runtime 对拍 (fp32, 多长度) ----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_cpu = ort.InferenceSession(args.out, sess_options=so, providers=["CPUExecutionProvider"])
    print("[verify] providers:", sess_cpu.get_providers())

    worst = 0.0
    for T in (100, 257, 500, 1000):
        x = torch.randn(1, 100, T)
        with torch.no_grad():
            y_t = m(x).numpy()
        y_o = sess_cpu.run(None, {"mel": x.numpy()})[0]
        d = np.abs(y_t - y_o).max()
        corr = np.corrcoef(y_t.ravel()[:200000], y_o.ravel()[:200000])[0, 1]
        worst = max(worst, float(d))
        print(f"  T={T:5d}: shape {y_o.shape}  max|Δ|={d:.3e}  corr={corr:.9f}  |ref|max={np.abs(y_t).max():.4f}")
    print(f"[verify] worst max|Δ| = {worst:.3e}  {'PASS' if worst < 1e-3 else 'FAIL'}")

    if args.skip_bench:
        return

    # ---- 基准 (10s 音频 / 1000 帧) ----
    T = args.bench_t
    x = torch.randn(1, 100, T)
    x_np = x.numpy()
    print(f"\n[bench] mel [1,100,{T}] -> wav [1,1,{T*480}] ({T*480/48000:.1f}s @48kHz)")

    def timeit(fn, n=5, sync=None):
        fn()                     # warmup
        if sync: sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if sync: sync()
        return (time.perf_counter() - t0) / n * 1000

    with torch.no_grad():
        t_cpu = timeit(lambda: m(x))
    print(f"  torch CPU          : {t_cpu:8.1f} ms")
    if torch.cuda.is_available():
        m_cuda = m.cuda()
        xc = x.cuda()
        with torch.no_grad():
            t_gpu = timeit(lambda: m_cuda(xc), sync=torch.cuda.synchronize)
        print(f"  torch CUDA (fp32)  : {t_gpu:8.1f} ms")
        m_cpu = m_cuda.cpu()
    t_ort_cpu = timeit(lambda: sess_cpu.run(None, {"mel": x_np}))
    print(f"  ORT  CPU           : {t_ort_cpu:8.1f} ms")
    for prov in ("DmlExecutionProvider", "CUDAExecutionProvider"):
        if prov in ort.get_available_providers():
            try:
                se = ort.InferenceSession(args.out, sess_options=so, providers=[prov, "CPUExecutionProvider"])
                t_ort_gpu = timeit(lambda: se.run(None, {"mel": x_np}))
                print(f"  ORT  {prov:18s}: {t_ort_gpu:8.1f} ms")
            except Exception as e:
                print(f"  ORT  {prov}: 不可用 ({type(e).__name__})")


if __name__ == "__main__":
    main()
