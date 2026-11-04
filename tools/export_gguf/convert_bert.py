# chinese-roberta-wwm-ext-large (BertForMaskedLM) -> GGUF
# 只导出 encoder 需要的张量 (cls.predictions.* 丢弃): 24 层 post-LN BERT, hidden 1024, 16 头, FFN 4096, eps 1e-12, gelu(erf)
# 布局约定同 convert_ar.py: torch [out,in] row-major 直写 = ggml [ne0=in, ne1=out]
import argparse, os, sys
import torch
import gguf

QT = {
    "f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0, "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K, "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q5_1": gguf.GGMLQuantizationType.Q5_1, "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_1": gguf.GGMLQuantizationType.Q4_1, "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/chinese-roberta-wwm-ext-large")
    ap.add_argument("--out", default="models/gsv-bert-f32.gguf")
    ap.add_argument("--spec", default="", help="attn=q8_0,ffn=q8_0,emb=f32 (默认 f32)")
    args = ap.parse_args()

    SPEC = {}
    for kv in filter(None, args.spec.split(",")):
        k, _, v = kv.partition("=")
        SPEC[k.strip()] = v.strip().lower()

    def ft(group):
        if group in ("norm", "emb", "type"):
            return gguf.GGMLQuantizationType.F32
        return QT.get(SPEC.get(group, "f32"), gguf.GGMLQuantizationType.F32)

    from transformers import AutoModelForMaskedLM
    m = AutoModelForMaskedLM.from_pretrained(args.model).eval()
    sd = m.state_dict()
    cfg = m.config
    H = cfg.hidden_size

    w = gguf.GGUFWriter(args.out, "gsv.bert")
    w.add_architecture()
    w.add_string("general.name", "gsv-bert")
    w.add_uint32("bert.n_layer", cfg.num_hidden_layers)
    w.add_uint32("bert.hidden", H)
    w.add_uint32("bert.head", cfg.num_attention_heads)
    w.add_uint32("bert.inter", cfg.intermediate_size)
    w.add_float32("bert.ln_eps", cfg.layer_norm_eps)
    w.add_uint32("bert.vocab", cfg.vocab_size)
    w.add_uint32("bert.max_pos", cfg.max_position_embeddings)
    # 需要的层数: 取 hidden_states[-3] -> 第 (n_layer-2) 层的输出, 即跑 0..n_layer-3 层
    w.add_uint32("bert.layers_used", cfg.num_hidden_layers - 2)

    def add(name, t, group):
        arr = t.detach().float().numpy()
        qt = ft(group)
        if qt == gguf.GGMLQuantizationType.F16:
            arr = arr.astype("float16")
        elif qt not in (gguf.GGMLQuantizationType.F32,):
            import numpy as np
            arr = gguf.quants.quantize(np.ascontiguousarray(arr), qt)
        w.add_tensor(name, arr, raw_dtype=qt)

    P = "bert."
    add(P + "word_emb", sd["bert.embeddings.word_embeddings.weight"], "emb")
    add(P + "pos_emb", sd["bert.embeddings.position_embeddings.weight"], "emb")
    add(P + "type_emb", sd["bert.embeddings.token_type_embeddings.weight"], "type")
    add(P + "emb_ln_w", sd["bert.embeddings.LayerNorm.weight"], "norm")
    add(P + "emb_ln_b", sd["bert.embeddings.LayerNorm.bias"], "norm")

    for li in range(cfg.num_hidden_layers):
        s = f"bert.encoder.layer.{li}."
        d = f"{P}l{li}."
        # QKV 融合成一次 matmul (q,k,v 行拼接), 行序与 torch (q|k|v) 切片一致
        qkv_w = torch.cat([sd[s + "attention.self.query.weight"], sd[s + "attention.self.key.weight"],
                        sd[s + "attention.self.value.weight"]], dim=0)
        qkv_b = torch.cat([sd[s + "attention.self.query.bias"], sd[s + "attention.self.key.bias"],
                        sd[s + "attention.self.value.bias"]], dim=0)
        add(d + "qkv_w", qkv_w, "attn")
        add(d + "qkv_b", qkv_b, "norm")
        add(d + "attn_out_w", sd[s + "attention.output.dense.weight"], "attn")
        add(d + "attn_out_b", sd[s + "attention.output.dense.bias"], "norm")
        add(d + "attn_ln_w", sd[s + "attention.output.LayerNorm.weight"], "norm")
        add(d + "attn_ln_b", sd[s + "attention.output.LayerNorm.bias"], "norm")
        add(d + "ff1_w", sd[s + "intermediate.dense.weight"], "ffn")
        add(d + "ff1_b", sd[s + "intermediate.dense.bias"], "norm")
        add(d + "ff2_w", sd[s + "output.dense.weight"], "ffn")
        add(d + "ff2_b", sd[s + "output.dense.bias"], "norm")
        add(d + "out_ln_w", sd[s + "output.LayerNorm.weight"], "norm")
        add(d + "out_ln_b", sd[s + "output.LayerNorm.bias"], "norm")

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", args.out)

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
