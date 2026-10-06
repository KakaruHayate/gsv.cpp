# GSV V5 golden 导出：AR 段（Text2SemanticDecoder）
# 从 v5 配套的 s1 ckpt 导出 AR 推理的逐段中间量，供 C++ 对拍。
# 注意：GPT-SoVITS V5 使用 s1v3.ckpt（v2 架构 AR，d=512, 16 heads, 24 层? 以 ckpt config 为准）
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.environ.get("GSV_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from AR.models.t2s_lightning_module import Text2SemanticLightningModule  # noqa: E402
from AR.models.utils import logits_to_probs  # noqa: E402


def dump(name, t):
    t = t.detach().cpu().float().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print(f"  dumped {name}: shape={t.shape} dtype=float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "s1v3.ckpt"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--prompt-len", type=int, default=24)
    parser.add_argument("--phones-len", type=int, default=32)
    args = parser.parse_args()
    outdir = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden")
    os.makedirs(outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(ckpt["weight"])
    model = model.eval().float()
    torch.manual_seed(42)  # 注意: 必须在 LightningModule 构造之后(构造会消耗 RNG 流)
    dec = model.model
    print("AR config: hidden_dim=%d head=%d n_layer=%d vocab=%d phones=%d EOS=%d" % (
        dec.model_dim, dec.num_head, dec.num_layers, dec.vocab_size, dec.phoneme_vocab_size, dec.EOS))

    # 合成输入: prompt semantic codes + text phones + bert features
    prompt = torch.randint(0, 1024, (1, args.prompt_len))
    phones = torch.randint(0, dec.phoneme_vocab_size, (1, args.phones_len))
    bert = torch.randn(1, 1024, args.phones_len)

    dump("ar.prompt", prompt)
    dump("ar.phones", phones)
    dump("ar.bert", bert)

    with torch.no_grad():
        # embedding 阶段
        x = dec.ar_text_embedding(phones)
        x = x + dec.bert_proj(bert.transpose(1, 2))
        x = dec.ar_text_position(x)
        dump("ar.x0", x)

        y = prompt
        y_emb = dec.ar_audio_embedding(y)
        y_pos = dec.ar_audio_position(y_emb)
        dump("ar.y_pos", y_pos)
        xy_pos = torch.concat([x, y_pos], dim=1)

        x_len = x.shape[1]
        y_len = y_emb.shape[1]
        x_attn_mask_pad = F.pad(torch.zeros((x_len, x_len), dtype=torch.bool), (0, y_len), value=True)
        y_attn_mask = F.pad(torch.triu(torch.ones((y_len, y_len), dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
        xy_attn_mask = xy_attn_mask.unsqueeze(0).expand(1, -1, -1)  # [1, S, S]
        dump("ar.xy_pos", xy_pos)
        xy_attn_mask_f = torch.zeros_like(xy_attn_mask, dtype=torch.float)
        xy_attn_mask_f.masked_fill_(xy_attn_mask, float("-inf"))
        dump("ar.xy_attn_mask", xy_attn_mask_f)

        # 首步: 完整 transformer（带 mask），导出每层 K/V 供 C++ 校验 KV cache 初值
        # 直接走 t2s_transformer 的 process_prompt（融合 QKV + SDPA + KV cache 语义）
        xy_dec, k_cache, v_cache = dec.t2s_transformer.process_prompt(xy_pos, xy_attn_mask, None)
        dump("ar.step0.dec", xy_dec)
        for li, (k_l, v_l) in enumerate(zip(k_cache, v_cache)):
            dump(f"ar.step0.k_cache_l{li}", k_l)
            dump(f"ar.step0.v_cache_l{li}", v_l)

        logits = dec.ar_predict_layer(xy_dec[:, -1])
        dump("ar.step0.logits", logits)

    print("AR golden done ->", outdir)
