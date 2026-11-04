# V5 DiT (CFM estimator) 权重 -> GGUF
#   s2Gv5turbo.pth 的 cfm.estimator.* (363 张量, checkpoint 存 fp16) -> models/gsv-dit-f32.gguf
#
# 布局约定:
#   2D Linear (torch [out,in]) 行主直写 = ggml ne{in, out} (mul_mat src0 = [in, out])
#   input_embed.proj.weight [1024,712] 拆两份: dit.input_embed.proj.weight_mel [100,1024]
#                                              dit.input_embed.proj.weight_ctx [612,1024]
#   conv_pos (Conv1d 1024->1024 k=31 groups=16) 权重重排为 [G, K*ICg, OCg]
#       A[g, kw + K*ic, o'] = w[g*OCg + o', ic, kw]   -> ggml ne{K*ICg, OCg, G}
#       (im2col_fast_1d 的行序 = kw + K*ic, 与 wns1 的 [K,IC,OC] reshape 一致)
#   dwconv (Conv1d 512->512 k=7 groups=512, depthwise) 重排为 [K, C] = torch [C,K] 转置
#       -> ggml ne{C, K}; 第 kw 个 tap = view_1d(w, C, kw*C*4)
#   rotary_embed.inv_freq [32] = checkpoint fp16 存储值 (与 fp32 公式差 1.6e-4, 必须用存档值)
#   text_embed.freqs_cis [4096,512] 非持久 buffer -> 转换时按 fp32 公式算 (torch 端初始化的同一数值)
import argparse, os, sys
import numpy as np
import torch


def precompute_freqs_cis(dim, end, theta=10000.0):
    # 与 GPT_SoVITS/f5_tts/model/modules.py 完全同式 (fp32)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)   # [end, dim]


def split_conv_pos(w):
    """torch [OC=1024, ICg=64, K=31], groups=16 -> numpy (G, OCg, ICg*K)
    ggml 视角 ne = {ICg*K, OCg, G}; 第 g 组的 [ICg*K, OCg] 块 = mul_mat src0 (稠密)
    行序 r = ic*K + kw 与 im2col_fast_1d 的 [K,IC,OC] 展平一致"""
    w = np.ascontiguousarray(w.detach().float().numpy())
    OC, ICg, K = w.shape
    G = 16
    OCg = OC // G
    assert OCg * G == OC and ICg == 64 and K == 31
    return w.reshape(G, OCg, ICg * K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="models/gsv-dit-f32.gguf")
    ap.add_argument("--spec", default="", help="e.g. lin=f16 (attn/ffn/adaln 的 2D 权重)")
    args = ap.parse_args()

    import gguf
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    est = {k[len("cfm.estimator."):]: v for k, v in sd.items() if k.startswith("cfm.estimator.")}
    assert len(est) == 363, len(est)

    spec = {}
    for kv in (args.spec or "").split(','):
        if kv:
            k, v = kv.split('=')
            spec[k.strip()] = v.strip().lower()

    def group_of(name):
        low = name.lower()
        if low.endswith(".bias") or "grn." in low or low.endswith(".norm.weight") \
           or "inv_freq" in low or "freqs_cis" in low or "dwconv.weight" in low:
            return "fixed"                    # F32 强制: bias(add)/grn/norm(mul)/host 读取
        if "conv_pos_embed" in low:
            return "conv"
        if low.startswith("transformer_blocks."):
            if ".attn." in low: return "attn"
            if ".ff." in low:   return "ffn"
            if ".attn_norm." in low: return "adaln"
            return "other"
        if low.startswith("norm_out."):  return "adaln"
        if low.startswith("proj_out.") or low.startswith("input_embed.proj."): return "proj"
        if low.startswith("text_embed."): return "text"
        if low.startswith("time_embed."): return "time"
        return "other"

    def pick(name):
        g = group_of(name)
        if g == "fixed":
            return "f32"
        if name.lower().endswith(".weight"):
            return spec.get(g, spec.get("lin", "f32"))   # 组名优先, "lin" 仍是全体 2D 权重的伞形键
        return "f32"

    w = gguf.GGUFWriter(args.out, "gsv.dit")
    w.add_architecture()
    w.add_string("general.name", "gsv-dit")
    w.add_uint32("dit.dim", 1024)
    w.add_uint32("dit.depth", 22)
    w.add_uint32("dit.heads", 16)
    w.add_uint32("dit.dim_head", 64)
    w.add_uint32("dit.ff_mult", 2)
    w.add_uint32("dit.mel_dim", 100)
    w.add_uint32("dit.text_dim", 512)
    w.add_uint32("dit.conv_layers", 4)
    w.add_uint32("dit.conv_pos_groups", 16)
    w.add_uint32("dit.conv_pos_kernel", 31)
    w.add_uint32("dit.rope_dims", 64)          # RoPE 只作用于 q/k 前 64 维 (= head 0)
    w.add_uint32("dit.text_max_pos", 4096)
    w.add_uint32("dit.time_freq_dim", 256)
    w.add_float32("dit.norm_eps", 1e-6)

    n_f16 = 0
    def add(name, t):
        nonlocal n_f16
        if isinstance(t, np.ndarray):
            arr = np.ascontiguousarray(t)
        else:
            arr = np.ascontiguousarray(t.detach().float().numpy())
        dt = pick(name)
        if dt == "f16":
            arr = arr.astype(np.float16)
            n_f16 += 1
        w.add_tensor("dit." + name, arr,
                     raw_dtype=gguf.GGMLQuantizationType.F16 if dt == "f16"
                     else gguf.GGMLQuantizationType.F32)

    # ---- time_embed (SinusPositionEmbedding 256 + MLP 256->1024->1024) ----
    for k, v in est.items():
        if k.startswith("time_embed."):
            add(k, v)

    # ---- text_embed (freqs_cis + 4 x ConvNeXtV2Block) ----
    add("text_embed.freqs_cis", precompute_freqs_cis(512, 4096))
    for k, v in est.items():
        if k.startswith("text_embed."):
            if k.endswith("dwconv.weight"):          # [512,1,7] -> [K,C] 转置
                add(k, v.reshape(512, 7).t())
            else:
                add(k, v)

    # ---- input_embed: proj 拆 mel/ctx 两段 + conv_pos (grouped conv) ----
    proj_w = est["input_embed.proj.weight"].float()       # [1024, 712]
    add("input_embed.proj.weight_mel", proj_w[:, :100].contiguous())
    add("input_embed.proj.weight_ctx", proj_w[:, 100:].contiguous())
    add("input_embed.proj.bias", est["input_embed.proj.bias"])
    for k, v in est.items():
        if k.startswith("input_embed.conv_pos_embed."):
            if k.endswith(".weight"):
                add(k, split_conv_pos(v.float()))
            else:
                add(k, v)

    # ---- rotary inv_freq (fp16 存档值) + 22 x DiTBlock + norm_out/proj_out ----
    add("rotary_embed.inv_freq", est["rotary_embed.inv_freq"])
    n_tb = 0
    for k, v in est.items():
        if k.startswith("transformer_blocks."):
            add(k, v); n_tb += 1
        elif k.startswith("norm_out.") or k.startswith("proj_out."):
            add(k, v)
    assert n_tb == 308, n_tb

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote %s: 363 -> %d tensors (f16=%d)" % (args.out, 363 - 1 + 3, n_f16))   # proj 拆 2 份, freqs_cis 新增


if __name__ == "__main__":
    main()
