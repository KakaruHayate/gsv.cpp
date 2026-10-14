# vocoder golden dump: torch Generator 前向 (fp32, CPU) -> mel/wav 对拍基准
#   用法与 tools/export_vocoder_onnx.py 共用同一份 Generator 构造逻辑
#   落盘 float32 raw:
#     voc.{T}.mel.bin   [100, T]  (denorm_spec 之后的量程 U[-12,2], torch 布局 m*T+t)
#     voc.{T}.wav.bin   [T*480]   (torch fp32 输出)
# 注意: 这个脚本只用 torch + onnxruntime, 不依赖 repo 的 G2PW/text 前端, 因此可以在
#   独立的 uv 环境里跑 (不动 diffsinger env)。
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.environ.get("GSV_REPO", os.path.abspath(os.path.join(ROOT, os.pardir, "repo")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from module.models import Generator  # noqa: E402

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


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-18s %s" % (name, tuple(arr.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "vocoder.pth"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "vocoder_fp32.onnx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--T", type=int, nargs="*", default=[1, 100, 200, 257, 300, 500, 1000])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    m = Generator(**V4_V5_VOCODER)
    m.remove_weight_norm()                       # 必须在 load_state_dict 之前
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print("load_state_dict:", m.load_state_dict(sd, strict=False))
    m = m.eval()

    for T in args.T:
        g = torch.Generator().manual_seed(1234 + T)
        # denorm_spec 之后的量程: spec_min=-12, spec_max=2
        mel = torch.empty(1, 100, T).uniform_(-12, 2, generator=g)
        wav = m(mel)                               # [1, 1, T*480]
        dump(args.out, "voc.%d.mel" % T, mel[0])
        dump(args.out, "voc.%d.wav" % T, wav[0, 0])
        print("[T=%d] wav %s range [%.4f, %.4f]" % (T, tuple(wav.shape), wav.min(), wav.max()))
    print("vocoder golden done ->", args.out)


if __name__ == "__main__":
    main()
