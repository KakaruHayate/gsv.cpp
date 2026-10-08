# GSV HuBERT (chinese-hubert-base) -> GGUF converter
# Arch: wav2vec2-style 7-layer CNN frontend (feat_extract_norm=group: GroupNorm after layer0 only)
#   + feature_projection (LN(512) -> Linear 512->768)
#   + pos_conv_embed (weight-norm(dim=2) conv1d k=128 groups=16, folded to plain weight at export)
#   + post-norm 12-layer transformer (hidden 768, 12 heads, FFN 3072, GELU)
# Source: models/chinese-hubert-base/pytorch_model.bin (transformers HubertModel)
# Layout follows convert_ar.py: torch [out,in] written as-is (row-major == ggml [ne0=in, ne1=out]).
# conv1d weight torch [OC,IC,K] -> permute to [K,IC,OC] (ggml_conv_1d: ne0=K, ne1=IC, ne2=OC).
import argparse
import os
import numpy as np
import torch
import gguf

CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
N_CONV = 7
N_LAYER = 12
HIDDEN = 768
N_HEAD = 12
POS_CONV_GROUPS = 16
POS_CONV_KERNEL = 128


def parse_spec(spec):
    # "attn=f16,ffn=f16" → dict; 支持 f16/f32/q8_0
    out = {}
    if spec:
        for kv in spec.split(','):
            k, v = kv.split('=')
            out[k.strip()] = v.strip().lower()
    return out

QT = {'f32': gguf.GGMLQuantizationType.F32,
      'f16': gguf.GGMLQuantizationType.F16,
      'q8_0': gguf.GGMLQuantizationType.Q8_0}

SPEC = {}

def pick(name):
    # 分组: conv / attn(q/k/v/out) / ffn / bias|ln 一律 f32
    low = name.lower()
    if low.endswith('_b') or '.norm' in low or '.b_' in low or '_wb' in low: return 'f32'
    if 'pos_conv' in low: return 'f32'   # pos_conv 的 host/GPU 处理路径假设 F32
    if 'feat_conv' in low: return SPEC.get('conv', 'f32')
    if any(t in low for t in ('.q_w', '.k_w', '.v_w', '.out_w')): return SPEC.get('attn', 'f32')
    if 'ffn' in low: return SPEC.get('ffn', 'f32')
    return 'f32'   # 其余 (emb 等) 一律 f32

def add_tensor(writer, name, tensor):
    qt = QT[pick(name)]
    t = tensor.detach().float().contiguous()
    arr = t.numpy()
    if qt == gguf.GGMLQuantizationType.F16:
        arr = arr.astype(np.float16)   # gguf-py 按数组 dtype 落盘; raw_dtype=F16 但数组 F32 会写 4B
    elif qt != gguf.GGMLQuantizationType.F32:
        raise NotImplementedError('use dequantize for ' + str(qt))
    writer.add_tensor(name, arr, raw_dtype=qt)


def add_conv1d(writer, name, weight):
    w = weight.detach().float().permute(2, 1, 0).contiguous()
    arr = w.numpy()
    if QT[pick(name)] == gguf.GGMLQuantizationType.F16:
        arr = arr.astype(np.float16)
    writer.add_tensor(name, arr, raw_dtype=QT[pick(name)])


def main():
    global SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/chinese-hubert-base/pytorch_model.bin")
    ap.add_argument("--out", default="models/gsv-hubert-f32.gguf")
    ap.add_argument("--spec", default="", help="e.g. conv=f16,attn=f16,ffn=f16")
    args = ap.parse_args()
    SPEC = parse_spec(args.spec)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    writer = gguf.GGUFWriter(args.out, "gsv.hubert")
    writer.add_architecture()
    writer.add_string("general.name", "gsv-hubert")
    writer.add_uint32("hubert.hidden_dim", HIDDEN)
    writer.add_uint32("hubert.head", N_HEAD)
    writer.add_uint32("hubert.n_layer", N_LAYER)
    writer.add_uint32("hubert.n_feat_conv", N_CONV)
    writer.add_array("hubert.conv_kernel", CONV_KERNEL)
    writer.add_array("hubert.conv_stride", CONV_STRIDE)
    writer.add_uint32("hubert.pos_conv_groups", POS_CONV_GROUPS)
    writer.add_uint32("hubert.pos_conv_kernel", POS_CONV_KERNEL)
    writer.add_float32("hubert.layer_norm_eps", 1e-5)

    def w(key):
        if key not in sd:
            raise KeyError(key)
        return sd[key]

    for i in range(N_CONV):
        add_conv1d(writer, "hubert.feat_conv.%d.w" % i, w("feature_extractor.conv_layers.%d.conv.weight" % i))
    add_tensor(writer, "hubert.feat_conv.0.norm_w", w("feature_extractor.conv_layers.0.layer_norm.weight"))
    add_tensor(writer, "hubert.feat_conv.0.norm_b", w("feature_extractor.conv_layers.0.layer_norm.bias"))

    add_tensor(writer, "hubert.feat_proj.norm_w", w("feature_projection.layer_norm.weight"))
    add_tensor(writer, "hubert.feat_proj.norm_b", w("feature_projection.layer_norm.bias"))
    add_tensor(writer, "hubert.feat_proj.w", w("feature_projection.projection.weight"))
    add_tensor(writer, "hubert.feat_proj.b", w("feature_projection.projection.bias"))

    g = w("encoder.pos_conv_embed.conv.weight_g").float()
    v = w("encoder.pos_conv_embed.conv.weight_v").float()
    v_norm = v.pow(2).sum(dim=(0, 1), keepdim=True).sqrt()
    pos_w = g * v / v_norm
    add_conv1d(writer, "hubert.pos_conv.w", pos_w)
    add_tensor(writer, "hubert.pos_conv.b", w("encoder.pos_conv_embed.conv.bias"))

    add_tensor(writer, "hubert.enc_norm_w", w("encoder.layer_norm.weight"))
    add_tensor(writer, "hubert.enc_norm_b", w("encoder.layer_norm.bias"))

    for li in range(N_LAYER):
        p = "hubert.layer.%d" % li
        src = "encoder.layers.%d" % li
        add_tensor(writer, p + ".q_w", w(src + ".attention.q_proj.weight"))
        add_tensor(writer, p + ".q_b", w(src + ".attention.q_proj.bias"))
        add_tensor(writer, p + ".k_w", w(src + ".attention.k_proj.weight"))
        add_tensor(writer, p + ".k_b", w(src + ".attention.k_proj.bias"))
        add_tensor(writer, p + ".v_w", w(src + ".attention.v_proj.weight"))
        add_tensor(writer, p + ".v_b", w(src + ".attention.v_proj.bias"))
        add_tensor(writer, p + ".out_w", w(src + ".attention.out_proj.weight"))
        add_tensor(writer, p + ".out_b", w(src + ".attention.out_proj.bias"))
        add_tensor(writer, p + ".ln1_w", w(src + ".layer_norm.weight"))
        add_tensor(writer, p + ".ln1_b", w(src + ".layer_norm.bias"))
        add_tensor(writer, p + ".ffn1_w", w(src + ".feed_forward.intermediate_dense.weight"))
        add_tensor(writer, p + ".ffn1_b", w(src + ".feed_forward.intermediate_dense.bias"))
        add_tensor(writer, p + ".ffn2_w", w(src + ".feed_forward.output_dense.weight"))
        add_tensor(writer, p + ".ffn2_b", w(src + ".feed_forward.output_dense.bias"))
        add_tensor(writer, p + ".ln2_w", w(src + ".final_layer_norm.weight"))
        add_tensor(writer, p + ".ln2_b", w(src + ".final_layer_norm.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
