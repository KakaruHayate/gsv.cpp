#!/usr/bin/env python3
"""Export the GPT-SoVITS V5 HiFi-GAN vocoder (vocoder.pth) to GGUF for gsv.cpp.

The checkpoint is the plain 48 kHz HiFi-GAN generator used by V5 (no NSF source):
  conv_pre (100 -> 512, k=7) -> LeakyReLU
  ups.{0..4}: ConvTranspose1d (k 20/12/4/4/4, stride 10/6/2/2/2)
  each level: 3 x ResBlock1 (k=3, dilations 1/3/5) summed, mean-scaled, LeakyReLU
  conv_post (16 -> 1, k=7) -> tanh

Usage:
  python tools/export_vocoder_gguf.py --ckpt models/vocoder.pth --out models/gsv-vocoder-f16.gguf
  python tools/export_vocoder_gguf.py --ckpt models/vocoder.pth --out models/gsv-vocoder-f32.gguf --f32
"""

import argparse
import os
import sys

import numpy as np
import torch

UPS = [10, 6, 2, 2, 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/vocoder.pth")
    ap.add_argument("--out", default="models/gsv-vocoder-f16.gguf")
    ap.add_argument("--f32", action="store_true", help="store weights as F32 instead of F16")
    args = ap.parse_args()

    try:
        from gguf import GGUFWriter
    except ImportError:
        print("[export] pip install gguf", file=sys.stderr)
        return 2

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not any(k.startswith("conv_pre.") for k in sd):
        sd = {k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith("generator.")}

    fl = np.float32 if args.f32 else np.float16

    def w(name: str) -> np.ndarray:
        return sd[name].detach().float().cpu().numpy().astype(fl)

    def b(name: str) -> np.ndarray:
        # biases stay F32: ggml binary ops need matching dtypes
        return sd[name].detach().float().cpu().numpy().astype(np.float32)

    def conv_w(name: str) -> np.ndarray:
        # ggml ne is reversed w.r.t. the numpy shape, so torch [OC, IC, K] lands as
        # ggml [K, IC, OC] without any transpose.
        return np.ascontiguousarray(w(name))

    def up_w(name: str) -> np.ndarray:
        # torch ConvTranspose1d [IC, OC, K] -> ggml [K, OC, IC] (same reversal).
        # F32: the Vulkan conv_transpose_1d kernel requires an F32 kernel tensor.
        return np.ascontiguousarray(sd[name].detach().float().cpu().numpy().astype(np.float32))

    ups_k = [int(sd[f"ups.{i}.weight"].shape[2]) for i in range(len(UPS))]

    gw = GGUFWriter(args.out, "gsv-vocoder")
    gw.add_uint32("gsv.voc.mel", int(sd["conv_pre.weight"].shape[1]))
    gw.add_uint32("gsv.voc.sr", 48000)
    gw.add_uint32("gsv.voc.hop", int(np.prod(UPS)))
    gw.add_array("gsv.voc.ups", UPS)
    gw.add_array("gsv.voc.ups_k", ups_k)
    gw.add_array("gsv.voc.res_k", [3, 3, 3])
    gw.add_array("gsv.voc.res_dil", [1, 3, 5])
    gw.add_uint32("gsv.voc.res_blocks", 3)
    gw.add_bool("gsv.voc.f32", bool(args.f32))

    gw.add_tensor("voc.conv_pre.weight", conv_w("conv_pre.weight"))
    gw.add_tensor("voc.conv_pre.bias", b("conv_pre.bias"))
    for i in range(len(UPS)):
        gw.add_tensor(f"voc.ups.{i}.weight", up_w(f"ups.{i}.weight"))
        gw.add_tensor(f"voc.ups.{i}.bias", b(f"ups.{i}.bias"))
    for j in range(15):
        for c in (1, 2):
            for k in range(3):
                base = f"resblocks.{j}.convs{c}.{k}"
                gw.add_tensor(f"voc.{base}.weight", conv_w(base + ".weight"))
                gw.add_tensor(f"voc.{base}.bias", b(base + ".bias"))
    gw.add_tensor("voc.conv_post.weight", conv_w("conv_post.weight"))
    gw.add_tensor("voc.conv_post.bias", b("conv_post.bias"))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    print(f"[export] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{'f32' if args.f32 else 'f16'} weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
