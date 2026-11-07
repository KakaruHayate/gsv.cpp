# GPT-SoVITS v2 / v2Pro / v2ProPlus SoVITS (s2G*.pth) -> GGUF converter
#
# Scope: the S2 part only (the AR/GPT part is s1v3/s1bert25hz and is handled by --kind ar).
# v2-family SoVITS is a VITS decoder, not a cond+DiT pair:
#
#   phones ──enc_p(text_embedding + encoder_text + MRTE(cross-attn) + proj)──► m_p, logs_p
#   ref audio ──cnhubert──► ssl_proj ──quantizer.vq codebook──► ge   (timbre / global embedding)
#   z_p = m_p + noise*exp(logs_p);  z = flow(z_p, reverse=True, g=ge);  y = dec(z, g=ge)
#
# so the GGUF carries five groups instead of v5's cond/dit/refcode split:
#   enc_p.*      text encoder (v2 layout: dual encoder + MRTE)
#   flow.*       ResidualCouplingBlock (10 layers of WaveNet coupling)
#   dec.*        HiFi-GAN generator (conv_pre, 4× upsample, resblocks, conv_post, cond)
#   ref_enc.*    ReferenceEncoder (spectral convs + temporal + slf_attn + fc)
#   quantizer.*  VQ codebook used for the reference's global embedding
#   ssl_proj.*   HuBERT feature -> semantic token projection
#   enc_q.*      posterior encoder: training only, dropped by default (--keep-enc-q)
#
# Tensor naming: the checkpoint keys are already unambiguous and stable across v2, v2Pro and
# v2ProPlus, so they are kept verbatim (dots preserved). Conv1d weights are written as-is
# (torch [OC,IC,K]); the GGUF dims reversal makes ggml see them as [K,IC,OC], which is the
# layout every ggml conv op expects. See convert_cond.py for the same convention.
import argparse
import json
import os

import numpy as np
import torch
import gguf

# arch metadata per family; the checkpoint itself decides the group shapes
KNOWN = {
    "v2": {"sample_rate": 32000, "n_layers_flow": 4, "gin_channels": 512, "hidden": 192},
    "v2pro": {"sample_rate": 32000, "n_layers_flow": 4, "gin_channels": 512, "hidden": 192},
    "v2proplus": {"sample_rate": 44100, "n_layers_flow": 4, "gin_channels": 512, "hidden": 192},
}

CONV_SUFFIXES = ("conv_pre.weight", "conv_post.weight", "cond.weight", "weight")
SKIP_PREFIXES = ("enc_q.",)


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("weight", "state_dict", "model"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and hasattr(next(iter(inner.values())), "numel"):
                return inner
        if obj and hasattr(next(iter(obj.values())), "numel"):
            return obj
    raise SystemExit(f"cannot find a state dict in {path}")


def detect_version(sd, override=None):
    if override:
        return override
    if not any(k.startswith("flow.flows.") for k in sd):
        raise SystemExit("this checkpoint has no flow.flows.*: not a v2-family SoVITS")
    # v2 / v2Pro / v2ProPlus share the same key layout, so the family cannot be inferred from
    # tensor names -- only the sampling rate and a few conditioning widths differ. Default to
    # v2 (32 kHz) and require --version for the others instead of guessing.
    return "v2"


def is_conv1d(name, t):
    if t.dim() != 3:
        return False
    low = name.lower()
    return any(low.endswith(s) for s in CONV_SUFFIXES) or low.endswith(".weight") and (
        "conv" in low or ".ups." in low or ".resblocks." in low or "c_" in low
    )


def main():
    ap = argparse.ArgumentParser(description="GPT-SoVITS v2-family SoVITS -> GGUF")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default=None, choices=sorted(KNOWN))
    ap.add_argument("--f16", action="store_true", help="store weights as f16 (bias/norm stay f32)")
    ap.add_argument("--keep-enc-q", action="store_true", help="keep the posterior encoder (training only)")
    a = ap.parse_args()

    sd = load_state(a.ckpt)
    version = detect_version(sd, a.version)

    kept = {k: v for k, v in sd.items() if a.keep_enc_q or not k.startswith(SKIP_PREFIXES)}
    writer = gguf.GGUFWriter(a.out, "gsv-v2sovits")
    writer.add_string("v2sovits.version", version)
    meta = KNOWN.get(version, KNOWN["v2"])
    for k, v in meta.items():
        writer.add_uint32(f"v2sovits.{k}", v)
    writer.add_uint32("v2sovits.n_tensors", len(kept))
    n_conv = 0
    for name, t in kept.items():
        if not torch.is_tensor(t):
            continue
        if is_conv1d(name, t):
            # 不要 permute。gguf.GGUFWriter 写 dims 时按 shape[::-1] 落盘（见 gguf_writer.py
            # 的 ti.shape[n_dims-1-j]），所以 torch [OC,IC,K] 原样写出后，ggml 侧 ne 就是
            # [K,IC,OC] —— 正是 ggml_conv_1d / ggml_conv_direct_1d 要的核布局。
            # 早期这里多做了一次 permute(2,1,0)，ggml 里就变成 [OC,IC,K]，所有 3D 卷积
            # （enc_p / ref_enc / dec / flow 的 weight_v）全部错位，v2 家族文件因此不可用。
            arr = t.detach().float().contiguous().numpy()
            n_conv += 1
        else:
            arr = t.detach().float().contiguous().numpy()
        low = name.lower()
        keep_f32 = arr.ndim == 1 or "norm" in low or low.endswith(".bias")
        if a.f16 and not keep_f32:
            arr = arr.astype(np.float16)
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    groups = {}
    for name, t in kept.items():
        if torch.is_tensor(t):
            g = name.split(".")[0]
            groups[g] = groups.get(g, 0) + t.numel()
    print(
        json.dumps(
            {
                "out": a.out,
                "version": version,
                "tensors": len(kept),
                "conv1d_permuted": n_conv,
                "params_M": round(sum(v.numel() for v in kept.values() if torch.is_tensor(v)) / 1e6, 2),
                "size_MB": round(os.path.getsize(a.out) / 1e6, 2),
                "groups_M": {k: round(v / 1e6, 2) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
