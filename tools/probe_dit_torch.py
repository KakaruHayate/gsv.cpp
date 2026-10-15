"""用 torch 的 DiT 复算我们 dump 出来的 DiT 输入, 对比输出 mel 的音高。

用途: 端到端输出"语义对但音高偏低"时, 判断是 DiT 的输入合成有问题, 还是模型本身如此。
输入 (fea_ref / fea_todo / mel2) 由 server 用 GSV_ENGINE_DUMP=<dir> 落盘。
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, "..", "repo"))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
sys.path.insert(0, HERE)
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from tools_dit_common import build_dit            # noqa: E402
from module.models_v5 import synthesize_v5_mel, CFMV5  # noqa: E402


def rb(path, dt):
    return np.fromfile(path, dtype=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "tests", "_dump"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "_dump", "torch_mel.npy"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    fr = rb(os.path.join(args.dump, "fea_ref.f32"), "<f4")
    ft = rb(os.path.join(args.dump, "fea_todo.f32"), "<f4")
    m2 = rb(os.path.join(args.dump, "mel2.f32"), "<f4")
    R, N, Tm = fr.size // 512, ft.size // 512, m2.size // 100
    print("dumped: fea_ref[512,%d] fea_todo[512,%d] mel2[100,%d]" % (R, N, Tm))

    dit = build_dit(os.path.join(ROOT, "models", "s2Gv5turbo.pth"))
    cfm = CFMV5(100, dit).eval()
    # synthesize_v5_mel 只用到 model.cfm 与 model.version
    class Shim:
        pass
    m = Shim(); m.cfm = cfm; m.version = "v5turbo"

    ref_feat = torch.tensor(fr, dtype=torch.float32).reshape(1, 512, R)
    tgt_feat = torch.tensor(ft, dtype=torch.float32).reshape(1, 512, N)
    ref_mel = torch.tensor(m2, dtype=torch.float32).reshape(1, 100, Tm)

    # CFMV5.inference 的初始噪声来自 torch.randn; 我们 C++ 侧用 gsv_rng -> 两边噪声不同,
    # 输出自然不同。为了做**同输入**对比, 这里把 torch 的 randn 换成一个固定的、从文件读入的
    # 噪声 (即我们 C++ 侧用的那份; 由 server 的 GSV_ENGINE_DUMP 一起落盘)。
    noise_path = os.path.join(args.dump, "noise.f32")
    fixed = None
    if os.path.exists(noise_path):
        fixed = torch.tensor(np.fromfile(noise_path, dtype="<f4"))
        print("using fixed noise from", noise_path, tuple(fixed.shape))
    else:
        torch.manual_seed(args.seed)
        print("no noise.f32 in dump -> 用 torch.manual_seed(%d) 的 randn (与 C++ 不同!)" % args.seed)

    orig_randn = torch.randn
    if fixed is not None:
        def rnd_like(size, *a, **k):
            return fixed.reshape(size).clone()
        torch.randn = rnd_like
    try:
        with torch.inference_mode():
            mel = synthesize_v5_mel(m, ref_feat, tgt_feat, ref_mel, args.steps, args.cfg)
    finally:
        torch.randn = orig_randn
    print("torch mel", tuple(mel.shape))
    np.save(args.out, mel[0].float().numpy())
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
