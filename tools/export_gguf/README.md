# GGUF / ONNX exporter for GPT-SoVITS V5

One entry point to turn the official V5 checkpoints into the artifacts consumed by
ggml/ONNX runtimes (reference implementation: https://github.com/KakaruHayate/gsv.cpp).

## Components

| kind | input | output | notes |
|---|---|---|---|
| `ar` | `s1v3.ckpt` (Text2Semantic) | `.gguf` | `--f16` / `--q8` / `--spec attn=q8_0,ffn=q8_0` |
| `cond` | `s2Gv5turbo.pth` | `.gguf` | enc_p / ref_enc / wns1 / bridge / rvq in one file |
| `dit` | `s2Gv5turbo.pth` | `.gguf` | CFM estimator (DiT), `--spec lin=f16` |
| `refcode` | `s2Gv5turbo.pth` | `.gguf` | ssl_proj + codebook (reference prompt tokens) |
| `vocoder` | `vocoder.pth` | `.onnx` | **strict fp32, never quantised** |

**Out of scope:** HuBERT and BERT. They are plain HuggingFace checkpoints and are
exported by the runtime project, not by GPT-SoVITS.

## Usage

```bash
# from the repository root
python tools/export_gguf/export.py --kind ar  --ckpt models/s1v3.ckpt \
    --out build/gsv-ar-f16.gguf --f16 --self-test

python tools/export_gguf/export.py --kind cond --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-cond-f16.gguf --spec "wns1=f16,bridge=f16,enc_p.ffn=f16" --self-test

python tools/export_gguf/export.py --kind dit  --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-dit-f16.gguf --spec "lin=f16" --self-test

python tools/export_gguf/export.py --kind refcode --ckpt models/s2Gv5turbo.pth \
    --out build/gsv-refcode.gguf --self-test

python tools/export_gguf/export.py --kind vocoder --ckpt models/vocoder.pth \
    --out build/vocoder_fp32.onnx --self-test
```

`--self-test` re-opens the artifact with `gguf.GGUFReader` and checks the tensor
count / names (295 / 298 / 365 / 4 for ar / cond / dit / refcode).

For `ar`, `--f16` expands to `--spec attn=f16,ffn=f16,predict=f16,emb=f16` and
`--q8` to `--spec attn=q8_0,ffn=q8_0`; an explicit `--spec` wins. Norms, biases and
`alpha` always stay F32.

Measured with the official V5 checkpoints (RTX 2070 host, CPU export):

| kind | size | tensors |
|---|---|---|
| ar (`--f16`) | 149.3 MiB | 295 |
| cond (`--spec wns1=f16,bridge=f16,enc_p.ffn=f16`) | 120.6 MiB | 298 |
| dit (`--spec lin=f16`) | 650.5 MiB | 365 |
| refcode | 7.5 MiB | 4 |
| vocoder | 55.1 MiB (ONNX) | — |

## Requirements

```
torch numpy gguf          # gguf = pip install gguf  (llama.cpp ecosystem)
onnx                      # vocoder export only
```

The converters expect the checkpoints at the paths you pass; `--spec` group names
are documented per component in `docs/quant_tiers.md` of gsv.cpp.

## Version matrix

- V5 (`s1v3.ckpt` + `s2Gv5turbo.pth`): complete.
- v2/v3/v4: **not supported** by this exporter (different decoder/DiT layouts).

## Licence

MIT (same as the parent repository). The GGUF layout documented by the converters
is reverse-engineered from the checkpoints; `--self-test` is the minimum check
before feeding an artifact to a runtime.
