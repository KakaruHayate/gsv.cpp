# GGUF exporter for GPT-SoVITS V5

Turns the official V5 checkpoints into the GGUF artifacts consumed by gsv.cpp. Everything the
engine loads is GGUF — the ONNX vocoder path was removed, the vocoder is a ggml HiFi-GAN with
sub-pixel upsampling (`voc.upsub.*`, F32, never quantised).

**You usually do not need this**: prebuilt GGUFs are published at
<https://huggingface.co/KakaruHayate/GSV.CPP>. Run the exporter only when you convert your own
fine-tune or want a different quantisation.

## Components

| kind | input | output | notes |
|---|---|---|---|
| `ar` | `s1v3.ckpt` (Text2Semantic) | `.gguf` | `--f16` / `--q8` / `--spec attn=q8_0,ffn=q8_0` |
| `cond` | `s2Gv5turbo.pth` / `s2Gv5dev.pth` | `.gguf` | enc_p / ref_enc / wns1 / bridge / rvq in one file |
| `dit` | same | `.gguf` | CFM estimator (DiT), `--spec lin=f16` |
| `refcode` | same | `.gguf` | ssl_proj + codebook (reference prompt tokens) |
| `vocoder` | `vocoder.pth` | `.gguf` | 48 kHz HiFi-GAN, **strict F32**, sub-pixel layout |
| `bert` | `chinese-roberta-wwm-ext-large/` | `.gguf` | needs `transformers`; prebuilt file is available |
| `hubert` | `chinese-hubert-base/` | `.gguf` | needs `transformers`; prebuilt file is available |

## Usage

```bash
# from the repository root
python tools/export_gguf/export.py --kind ar --ckpt models/s1v3.ckpt \
    --out build/gsv-ar-f16.gguf --f16 --self-test

python tools/export_gguf/export.py --kind ar --ckpt models/s1v3.ckpt \
    --out build/gsv-ar-q8.gguf --q8 --self-test

python tools/export_gguf/export.py --kind cond --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-cond-f16.gguf --spec "wns1=f16,bridge=f16,enc_p.ffn=f16" --self-test

python tools/export_gguf/export.py --kind dit --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-dit-f16all.gguf --spec "lin=f16" --self-test

python tools/export_gguf/export.py --kind refcode --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-refcode.gguf --self-test

python tools/export_gguf/export.py --kind vocoder --ckpt models/vocoder.pth \
    --out build/gsv-vocoder-f32.gguf --self-test

# optional, prebuilt versions are published on Hugging Face
python tools/export_gguf/export.py --kind bert   --ckpt chinese-roberta-wwm-ext-large \
    --out build/gsv-bert-f16.gguf --self-test
python tools/export_gguf/export.py --kind hubert --ckpt chinese-hubert-base \
    --out build/gsv-hubert-f16.gguf --self-test
```

`--self-test` re-opens the artifact with `gguf.GGUFReader` and checks the tensor count / names
(295 / 298 / 365 / 4 for ar / cond / dit / refcode).

For `ar`, `--f16` expands to `--spec attn=f16,ffn=f16,predict=f16,emb=f16` and `--q8` to
`--spec attn=q8_0,ffn=q8_0`; an explicit `--spec` wins. Norms, biases and `alpha` always stay F32.

## Reference artifacts

Sizes from the official V5 checkpoints (CPU export; same numbers as the Hugging Face repo):

| kind | file | size |
|---|---|---|
| ar (`--f16`) | `gsv-ar-f16.gguf` | 151 MiB |
| ar (`--q8`) | `gsv-ar-q8.gguf` | 83 MiB |
| cond (`wns1=f16,bridge=f16,enc_p.ffn=f16`) | `gsv-cond-f16.gguf` | 91 MiB |
| dit (`lin=f16`) | `gsv-dit-f16all.gguf` | 651 MiB |
| refcode | `gsv-refcode.gguf` | 7.5 MiB |
| vocoder (F32, sub-pixel) | `gsv-vocoder-f32.gguf` | 61 MiB |
| bert | `gsv-bert-f16.gguf` | 662 MiB |
| hubert | `gsv-hubert-f16.gguf` | 198 MiB |

## Requirements

```
torch numpy gguf          # gguf = pip install gguf  (llama.cpp ecosystem)
transformers              # bert / hubert only
```

| kind | source | notes |
|---|---|---|
| `dhubert` | `distilled_hubert/student_v0.pt` | 102 tensors / 2.97M params. 7-layer conv frontend + 6-block dilated TCN + 2-layer pre-norm transformer + 1x1 head to 768. Arch metadata (kernels, strides, dilations, GroupNorm groups, eps) goes into GGUF KV. C++ consumer: `src/gsv_dhubert.{h,cpp}` (probe: `tools/dhubert_probe.cpp`, torch 参考: `tools/dump_ref_dhubert.py`; 对拍用 `tools/dhubert_compare.py`)。 |
| `v2sovits` | `s2G2333k.pth` / `s2Gv2Pro.pth` / `s2Gv2ProPlus.pth` | v2-family SoVITS (VITS decoder: enc_p + MRTE, flow, HiFi-GAN dec, ref_enc, VQ). Keys kept verbatim, `enc_q` dropped (training-only). **C++ consumer (`src/gsv_v2sovits.cpp`) not written yet.** |

## Version matrix

- V5 (`s1v3.ckpt` + `s2Gv5turbo.pth` / `s2Gv5dev.pth`): complete.
- v2 / v3 / v4: **not supported** (different decoder/DiT layouts, and gsv.cpp only implements V5).

## Layout notes

- Tensor names and shapes are what `src/gsv_*.cpp` expect; the converters document each group
  inline (for example `voc.upsub.*` for the phase-major vocoder layout).
- `--spec` group names are listed per component in the converter sources.

## Licence

MIT (same as the parent repository). The GGUF layout is reverse-engineered from the checkpoints;
`--self-test` is the minimum check before feeding an artifact to a runtime.
