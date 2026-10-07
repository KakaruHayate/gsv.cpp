# BERT golden 导出 (chinese-roberta-wwm-ext-large):
#   - 固定文本 -> token ids (tokenizer 在 Python 侧, 与决策一致)
#   - HF fp32 前向 -> hidden_states[-3] (第 21 层输出) -> 后处理 [1:-1] + 转置
# 供 tests/test_bert_ggml.cpp 对拍
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXTS = [
    "这是一个简单的示例，真没想到这么简单就完成了。",
    "然而，他红了20年以后，他竟退出了大家的视线。",
    "The King and His Stories. Once there was a king.",
    "短。",
]


def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print(f"  dumped {name}: shape={t.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "chinese-roberta-wwm-ext-large"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "golden"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForMaskedLM.from_pretrained(args.model).eval()
    cfg = m.config
    print(f"BERT: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"inter={cfg.intermediate_size} eps={cfg.layer_norm_eps} act={cfg.hidden_act}")

    for i, text in enumerate(TEXTS):
        enc = tok(text, return_tensors="pt")
        ids = enc["input_ids"]                       # [1, T] 含 [CLS]/[SEP], 无 padding
        with torch.no_grad():
            out = m(**enc, output_hidden_states=True)
        hs = out.hidden_states
        h_last3 = hs[-3]                             # 第 21 层输出 [1, T, 1024]
        # 管线后处理: 去掉 [CLS]/[SEP], 转置为 [1024, T-2]
        feat = h_last3[0][1:-1].transpose(0, 1).contiguous()
        dump(args.out, f"bert.{i}.ids", ids[0].float().reshape(1, -1))
        dump(args.out, f"bert.{i}.hidden", h_last3)          # [1, T, 1024] 模型原始输出
        dump(args.out, f"bert.{i}.feat", feat)               # [1024, T-2] 管线用特征
        if i == 0:                                           # 逐层对拍用 (0: embedding+LN, k: 第 k-1 层)
            for k, h in enumerate(hs[:cfg.num_hidden_layers - 1]):
                dump(args.out, f"bert.0.hs{k}", h)
        print(f"  text{i}: tokens={ids.shape[1]} feat={tuple(feat.shape)}  '{text[:18]}...'")

    # 汇总
    dump(args.out, "bert.n_texts", torch.tensor([len(TEXTS)], dtype=torch.float32))
    print("BERT golden done ->", args.out)


if __name__ == "__main__":
    main()
