# V5 条件段权重 -> GGUF (s2Gv5turbo.pth: quantizer/bridge/ref_enc/wns1/enc_p; cfm=DiT 另做)
# 布局约定同 convert_ar.py / convert_bert.py: torch [out,in] row-major 直写 = ggml [ne0=in, ne1=out]
# 注意: wns1 的 in_layers 是 PyTorch weight-norm (weight_g/weight_v) -> 转换时物化 w = g * v/||v||
import argparse, os, sys, collections
import numpy as np
import torch

def wn_materialize(g, v, eps=1e-12):
    # PyTorch weight_norm(dim=0): w = g * v / ||v||  (||v|| 在 dim 0 之外的维度上算范数)
    norm = v.norm(dim=list(range(1, v.dim())), keepdim=True).clamp_min(eps)
    return g * v / norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="models/gsv-cond-f32.gguf")
    ap.add_argument("--spec", default="", help="e.g. wns1=f16")
    args = ap.parse_args()

    import gguf
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)

    w = gguf.GGUFWriter(args.out, "gsv.cond")
    w.add_architecture()
    w.add_string("general.name", "gsv-cond")
    # 元信息 (从张量形状反推, 已人工核对)
    w.add_uint32("cond.rvq.bins", 1024)
    w.add_uint32("cond.rvq.dim", 768)
    w.add_uint32("cond.rvq.n_q", 1)
    w.add_uint32("cond.hidden", 192)      # enc_p inter_channels
    w.add_uint32("cond.bridge_out", 512)
    w.add_uint32("cond.semantic_frame_rate_hz", 25)

    spec = {}
    for kv in (args.spec or "").split(','):
        if kv:
            k, v = kv.split('=')
            spec[k.strip()] = v.strip().lower()

    def pick(name):
        # wns1: in_layers/res_skip/conv 权重可 f16; bias/rvq/bridge 一律 f32
        low = name.lower()
        if low.endswith('.bias') or low.endswith('_b') or 'codebook' in low:
            return 'f32'
        if 'bridge' in low:
            return spec.get('bridge', 'f32')
        if name.startswith('wns1.') and ('weight_w' in low or '.conv.weight' in low or '.weight' in low):
            return spec.get('wns1', 'f32')
        return 'f32'

    def add(name, t):
        arr = t.detach().float().numpy()
        dt = pick(name)
        if dt == 'f16':
            arr = arr.astype(np.float16)
        w.add_tensor(name, np.ascontiguousarray(arr),
                     raw_dtype=gguf.GGMLQuantizationType.F32 if dt == 'f32'
                     else gguf.GGMLQuantizationType.F16)

    # ---- RVQ (n_q=1: 只有 codebook) ----
    add("rvq.codebook", sd["quantizer.vq.layers.0._codebook.embed"])          # [1024, 768] -> ggml [768, 1024]

    # ---- bridge: Conv1d(192->512, k=1) + LeakyReLU ----
    for k, v in sd.items():
        if k.startswith("bridge."):
            add(k.replace("bridge.0.", "bridge."), v)

    # ---- ref_enc (MelStyleEncoder) ----
    n_ref = 0
    for k, v in sd.items():
        if k.startswith("ref_enc."):
            add(k, v); n_ref += 1

    # ---- wns1 (Encoder: WN 8 层, gin=512), weight-norm 物化 ----
    n_wn = 0
    skip_g = set()
    for k, v in sd.items():
        if not k.startswith("wns1."):
            continue
        if k.endswith(".weight_g"):
            base = k[:-len("_g")]
            add(base + "_w", wn_materialize(v, sd[base + "_v"]))
            n_wn += 1
        elif k.endswith(".weight_v"):
            continue
        else:
            add(k, v)
            n_wn += 1

    # ---- enc_p (TextEncoder: ssl/text 两路 + MRTE) ----
    n_ep = 0
    for k, v in sd.items():
        if k.startswith("enc_p."):
            add(k, v); n_ep += 1

    # ---- 其余小件 (ssl_proj / linear_mel, 推理不一定用但一并带上) ----
    for k, v in sd.items():
        if k.startswith("ssl_proj.") or k.startswith("linear_mel."):
            add(k, v)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.out}: ref_enc={n_ref}, wns1={n_wn}, enc_p={n_ep}")

if __name__ == "__main__":
    main()
