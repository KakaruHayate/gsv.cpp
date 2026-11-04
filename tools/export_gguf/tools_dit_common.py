# 共享: 从 s2Gv5turbo.pth 构造 fp32 F5 DiT (v5: use_step_embedding=False)
# 供 dump_golden_dit.py / dump_golden_cfm.py 使用
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, "..", "repo"))
for p in (REPO, os.path.join(REPO, "GPT_SoVITS")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from GPT_SoVITS.f5_tts.model.backbones.dit import DiT  # noqa: E402


def build_dit(ckpt, verbose=False):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    est = {k[len("cfm.estimator."):]: v for k, v in sd.items() if k.startswith("cfm.estimator.")}
    assert len(est) == 363, len(est)
    dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, mel_dim=100, text_dim=512,
              conv_layers=4, use_step_embedding=False).eval().float()
    miss, unexp = dit.load_state_dict(est, strict=False)
    assert not miss and not unexp, (miss, unexp)
    if verbose:
        n = sum(v.numel() for v in dit.state_dict().values())
        print("[dit] params: %.1fM" % (n / 1e6))
    return dit
