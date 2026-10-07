# GSV AR (Text2SemanticDecoder) -> GGUF converter
# 权重命名: transformer.block{L}.{qkv,out,norm1,ffn1,ffn2,norm2}, embed.*, predict.*
# 来源: AR/modules/transformer.py TransformerEncoderLayer (post-LN, ReLU FFN)
#       qkv_w = self_attn.in_proj_weight [3D, D] (fused), qkv_b 同
#       out_w = self_attn.out_proj.weight [D, D]
#       norm1/norm2 = LayerNorm [D]
#       ffn1 = linear1 [4D, D] (ReLU), ffn2 = linear2 [D, 4D]
# 注意: ggml mul_mat(W, x) 对应 torch F.linear(x, W^T)，torch 权重存 [out,in]，
#       ggml 存 [in, out]（ne0=in），直接按内存顺序写即可（row-major [out,in] == ggml [ne0=in, ne1=out] 转置视图）。
#       本仓库沿 rvc.cpp/audio.cpp 约定: gguf 张量形状按 torch 原样记录，加载时由 C++ 按需转置视图。
import argparse
import os

import torch
from safetensors.torch import save_file  # noqa: F401  (备选)
import gguf  # pip install gguf (llama.cpp 生态)


def add_tensor(writer, name, tensor, dtype=gguf.GGMLQuantizationType.F32):
    DTYPE = dtype
    t = tensor.detach().float().contiguous()
    # torch [out,in] 直接写: numpy row-major last-dim-fastest == ggml ne0=last dim=in,
    # 恰好是 ggml mul_mat(W,x) 需要的 [ne0=in, ne1=out] 布局，无需转置。
    arr = t.numpy()
    if dtype == gguf.GGMLQuantizationType.F16:
        arr = arr.astype("float16")     # gguf-py 的 raw_dtype 只打标, 需自行转数据
    elif dtype == gguf.GGMLQuantizationType.Q8_0:
        import numpy as _np
        arr = gguf.quants.quantize(_np.ascontiguousarray(arr), gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor(name, arr, raw_dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s1v3.ckpt")
    ap.add_argument("--out", default="models/gsv-ar-f32.gguf")
    ap.add_argument("--f16", action="store_true", help="权重存 F16 (norm/bias/emb 保持 F32)")
    ap.add_argument("--q8", action="store_true", help="权重存 Q8_0 (norm/bias/emb 保持 F32)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    weight = ckpt["weight"]
    m = config["model"]
    D = m["hidden_dim"]
    layers = m["n_layer"]

    if args.q8:
        F16 = gguf.GGMLQuantizationType.Q8_0
    elif args.f16:
        F16 = gguf.GGMLQuantizationType.F16
    else:
        F16 = gguf.GGMLQuantizationType.F32
    def ft(name):
        # 量化敏感项保持 F32: norm/bias/embedding/alpha
        low = name.lower()
        if ("norm" in low) or name.endswith("_b") or ("emb" in low) or ("alpha" in low) or ("bert" in low):
            # bert_proj: 引擎在 host 侧做投影, 必须是 F32 (或改为图内实现)
            return gguf.GGMLQuantizationType.F32
        return F16

    writer = gguf.GGUFWriter(args.out, "gsv.ar")
    writer.add_architecture()
    writer.add_string("general.name", "gsv-ar")
    writer.add_uint32("ar.hidden_dim", D)
    writer.add_uint32("ar.head", m["head"])
    writer.add_uint32("ar.n_layer", layers)
    writer.add_uint32("ar.vocab_size", m["vocab_size"])
    writer.add_uint32("ar.phoneme_vocab_size", m["phoneme_vocab_size"])
    writer.add_uint32("ar.eos", m["EOS"])
    writer.add_float32("ar.bert_dim", 1024)

    def w(key):
        return weight[key]

    # embeddings
    add_tensor(writer, "ar.text_emb", w("model.ar_text_embedding.word_embeddings.weight"), ft("ar.text_emb"))
    add_tensor(writer, "ar.audio_emb", w("model.ar_audio_embedding.word_embeddings.weight"), ft("ar.audio_emb"))
    add_tensor(writer, "ar.bert_proj", w("model.bert_proj.weight"), ft("ar.bert_proj"))
    add_tensor(writer, "ar.bert_proj_b", w("model.bert_proj.bias"))
    # 位置编码: learnable alpha 标量 + 固定 sine PE 表 [T_max, D]（运行时生成或导出）
    add_tensor(writer, "ar.text_pe_alpha", w("model.ar_text_position.alpha"))
    add_tensor(writer, "ar.audio_pe_alpha", w("model.ar_audio_position.alpha"))
    # final norm（norm_first=False 时 TransformerEncoder 可能无最终 norm，跳过）
    # 输出层
    add_tensor(writer, "ar.predict", w("model.ar_predict_layer.weight"), ft("ar.predict"))

    for li in range(layers):
        p = f"transformer.block{li}"
        add_tensor(writer, f"{p}.qkv_w", w(f"model.h.layers.{li}.self_attn.in_proj_weight"), ft("w"))
        add_tensor(writer, f"{p}.qkv_b", w(f"model.h.layers.{li}.self_attn.in_proj_bias"))
        add_tensor(writer, f"{p}.out_w", w(f"model.h.layers.{li}.self_attn.out_proj.weight"), ft("w"))
        add_tensor(writer, f"{p}.out_b", w(f"model.h.layers.{li}.self_attn.out_proj.bias"))
        add_tensor(writer, f"{p}.norm1_w", w(f"model.h.layers.{li}.norm1.weight"))
        add_tensor(writer, f"{p}.norm1_b", w(f"model.h.layers.{li}.norm1.bias"))
        add_tensor(writer, f"{p}.ffn1_w", w(f"model.h.layers.{li}.linear1.weight"), ft("w"))
        add_tensor(writer, f"{p}.ffn1_b", w(f"model.h.layers.{li}.linear1.bias"))
        add_tensor(writer, f"{p}.ffn2_w", w(f"model.h.layers.{li}.linear2.weight"), ft("w"))
        add_tensor(writer, f"{p}.ffn2_b", w(f"model.h.layers.{li}.linear2.bias"))
        add_tensor(writer, f"{p}.norm2_w", w(f"model.h.layers.{li}.norm2.weight"))
        add_tensor(writer, f"{p}.norm2_b", w(f"model.h.layers.{li}.norm2.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
