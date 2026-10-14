"""用 repo 的 AR (torch) 跑同样的 (phones, bert, prompt), 看它生成多少语义 token。

目的: 我们 C++ 引擎的 AR 在真实输入上只出 11 个 token (刚好是 EOS 解锁的边界), 需要判定
这是模型行为还是我们的移植问题。这里把**同一份输入**喂给 torch 版 AR, 两边对拍 token 数。
输入 (phones / bert) 直接取自我们 server 的 bert 命令, 保证两侧完全一致。
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("GSV_REPO", os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "repo")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from AR.models.t2s_lightning_module import Text2SemanticLightningModule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "s1v3.ckpt"))
    ap.add_argument("--npz", default=os.path.join(ROOT, "tests", "_ar_probe.npz"),
                    help="frontend 落盘的 phones/bert/prompt (见 --dump)")
    ap.add_argument("--early-stop", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()

    z = np.load(args.npz)
    phones = torch.LongTensor(z["phones"].astype(np.int64)).unsqueeze(0)     # [1, P]
    bert = torch.tensor(z["bert"], dtype=torch.float32).unsqueeze(0)          # [1, 1024, P]
    prompt = torch.LongTensor(z["prompt"].astype(np.int64)).unsqueeze(0)      # [1, T]
    print("inputs: phones=%s bert=%s prompt=%s" % (tuple(phones.shape), tuple(bert.shape), tuple(prompt.shape)))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Text2SemanticLightningModule(ckpt["config"], "****", is_train=False)
    model.load_state_dict(ckpt["weight"])
    model = model.eval().float()
    dec = model.model
    print("AR: vocab=%d EOS=%d" % (dec.vocab_size, dec.EOS))

    x_lens = torch.LongTensor([phones.shape[1]])
    torch.manual_seed(args.seed)
    with torch.no_grad():
        y, idx = dec.infer_panel(
            phones, x_lens, prompt, bert,
            top_k=15, top_p=1.0, temperature=1.0,
            early_stop_num=args.early_stop, repetition_penalty=1.35,
        )
    # y = prompt + 生成 (末位是停止 token)
    gen = y[0, prompt.shape[1]:].tolist()
    print("torch AR: 总长=%d, 生成=%d (idx=%d)" % (y.shape[1], len(gen), idx))
    print("  生成前 40 个:", gen[:40])
    print("  末位 (停止 token):", gen[-1] if gen else None, "(EOS=%d)" % dec.EOS)
    if idx == 0 and y.shape[1] - prompt.shape[1] == 11:
        print("  -> torch 也停在 11: **模型行为**, 我们的引擎移植正确")
    else:
        print("  -> torch 生成了 %d 个; 与我们的 11 不同 -> 需要查引擎" % len(gen))


if __name__ == "__main__":
    main()
