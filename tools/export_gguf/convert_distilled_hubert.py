# GSV distilled HuBERT (student_v0.pt) -> GGUF converter
# Arch (2.97M params, ~32x smaller than chinese-hubert-base):
#   7-layer conv frontend (320x downsample to 50 Hz), GroupNorm(8) + SiLU after every conv
#     conv kernels/strides: (10,5) (3,2) (3,2) (3,2) (3,2) (2,2) (2,2), c=192
#   + TCN: 6 residual blocks, kernel 3, dilation d (padding=d), GroupNorm(8) + SiLU
#   + 2-layer pre-norm TransformerEncoder (d_model 192, nhead 4, FFN 768, GELU)
#   + 1x1 conv head 192 -> 768 (the teacher's hidden size; outputs 50 Hz like CN-HuBERT)
# Source: GPT_SoVITS/pretrained_models/distilled_hubert/student_v0.pt
#   (weights shipped with the GPT-SoVITS-CPUFast fork; distill report upstream)
#
# Naming contract consumed by src/gsv_dhubert.cpp:
#   front.{i}.c_w / c_b / g_w / g_b          conv1d + GroupNorm
#   tcn.{i}.c1_w / c1_b / n1_w / n1_b / c2_w / c2_b / n2_w / n2_b
#   attn.{i}.qkv_w / qkv_b / o_w / o_b / ff1_w / ff1_b / ff2_w / ff2_b / ln1_w / ln1_b / ln2_w / ln2_b
#   head_w / head_b
# Layout follows convert_hubert.py: linear weights kept as torch [out,in] (== ggml [ne0=in, ne1=out]),
# conv weights permuted from [OC,IC,K] to [K,IC,OC].
import argparse
import json
import os

import numpy as np
import torch
import gguf

CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]
DILATIONS = [1, 2, 4, 8, 16, 32]
HIDDEN = 192
N_HEAD = 4
FFN = 768
N_ATTN = 2
GN_GROUPS = 8
OUT_DIM = 768
GN_EPS = 1e-5


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "student"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and hasattr(next(iter(inner.values())), "numel"):
                return inner
        if obj and hasattr(next(iter(obj.values())), "numel"):
            return obj
    raise SystemExit(f"cannot find a state dict in {path}")


def conv_w(w):
    # torch [OC, IC, K] -> ggml [K, IC, OC]
    return w.detach().float().permute(2, 1, 0).contiguous().numpy()


def lin_w(w):
    return w.detach().float().contiguous().numpy()


def vec(v):
    return v.detach().float().contiguous().numpy()


def main():
    ap = argparse.ArgumentParser(description="Distilled HuBERT (student_v0) -> GGUF")
    ap.add_argument("--ckpt", required=True, help="student_v0.pt")
    ap.add_argument("--out", required=True, help="output .gguf")
    a = ap.parse_args()

    sd = load_state(a.ckpt)
    writer = gguf.GGUFWriter(a.out, "gsv-dhubert")
    writer.add_uint32("dhubert.n_conv", len(CONV_KERNEL))
    writer.add_uint32("dhubert.n_tcn", len(DILATIONS))
    writer.add_uint32("dhubert.n_attn", N_ATTN)
    writer.add_uint32("dhubert.hidden", HIDDEN)
    writer.add_uint32("dhubert.n_head", N_HEAD)
    writer.add_uint32("dhubert.ffn", FFN)
    writer.add_uint32("dhubert.gn_groups", GN_GROUPS)
    writer.add_uint32("dhubert.out_dim", OUT_DIM)
    writer.add_float32("dhubert.gn_eps", GN_EPS)
    writer.add_array("dhubert.conv_kernel", CONV_KERNEL)
    writer.add_array("dhubert.conv_stride", CONV_STRIDE)
    writer.add_array("dhubert.dilations", DILATIONS)

    n = 0
    for i in range(len(CONV_KERNEL)):
        p = f"front.{i}."
        writer.add_tensor(f"front.{i}.c_w", conv_w(sd[p + "conv.weight"]))
        writer.add_tensor(f"front.{i}.c_b", vec(sd[p + "conv.bias"]))
        writer.add_tensor(f"front.{i}.g_w", vec(sd[p + "norm.weight"]))
        writer.add_tensor(f"front.{i}.g_b", vec(sd[p + "norm.bias"]))
        n += 4

    for i in range(len(DILATIONS)):
        p = f"tcn.{i}."
        writer.add_tensor(f"tcn.{i}.c1_w", conv_w(sd[p + "c1.weight"]))
        writer.add_tensor(f"tcn.{i}.c1_b", vec(sd[p + "c1.bias"]))
        writer.add_tensor(f"tcn.{i}.n1_w", vec(sd[p + "n1.weight"]))
        writer.add_tensor(f"tcn.{i}.n1_b", vec(sd[p + "n1.bias"]))
        writer.add_tensor(f"tcn.{i}.c2_w", conv_w(sd[p + "c2.weight"]))
        writer.add_tensor(f"tcn.{i}.c2_b", vec(sd[p + "c2.bias"]))
        writer.add_tensor(f"tcn.{i}.n2_w", vec(sd[p + "n2.weight"]))
        writer.add_tensor(f"tcn.{i}.n2_b", vec(sd[p + "n2.bias"]))
        n += 8

    for i in range(N_ATTN):
        p = f"attn.layers.{i}."
        writer.add_tensor(f"attn.{i}.qkv_w", lin_w(sd[p + "self_attn.in_proj_weight"]))
        writer.add_tensor(f"attn.{i}.qkv_b", vec(sd[p + "self_attn.in_proj_bias"]))
        writer.add_tensor(f"attn.{i}.o_w", lin_w(sd[p + "self_attn.out_proj.weight"]))
        writer.add_tensor(f"attn.{i}.o_b", vec(sd[p + "self_attn.out_proj.bias"]))
        writer.add_tensor(f"attn.{i}.ff1_w", lin_w(sd[p + "linear1.weight"]))
        writer.add_tensor(f"attn.{i}.ff1_b", vec(sd[p + "linear1.bias"]))
        writer.add_tensor(f"attn.{i}.ff2_w", lin_w(sd[p + "linear2.weight"]))
        writer.add_tensor(f"attn.{i}.ff2_b", vec(sd[p + "linear2.bias"]))
        writer.add_tensor(f"attn.{i}.ln1_w", vec(sd[p + "norm1.weight"]))
        writer.add_tensor(f"attn.{i}.ln1_b", vec(sd[p + "norm1.bias"]))
        writer.add_tensor(f"attn.{i}.ln2_w", vec(sd[p + "norm2.weight"]))
        writer.add_tensor(f"attn.{i}.ln2_b", vec(sd[p + "norm2.bias"]))
        n += 12

    writer.add_tensor("head_w", conv_w(sd["head.weight"]))
    writer.add_tensor("head_b", vec(sd["head.bias"]))
    n += 2

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(json.dumps({"out": a.out, "tensors": n, "size_MB": round(os.path.getsize(a.out) / 1e6, 2)}))


if __name__ == "__main__":
    main()
