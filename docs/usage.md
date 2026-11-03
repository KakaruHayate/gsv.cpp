# Usage guide

This engine supports **GPT-SoVITS V5 only** — the `v5turbo` (4-step DMD) and `v5dev`
(32-step) models. v2 / v3 / v4 checkpoints are not supported.

## 1. Get a build

Prebuilt binaries are produced by CI for two platforms, both with the **Vulkan** ggml backend
and the ggml HiFi-GAN vocoder:

| platform | artifact | contents |
|---|---|---|
| Windows x64 | `gsv.cpp-windows-x64-vulkan.zip` | `gsv_server.exe` + ggml DLLs |
| Linux x64 | `gsv.cpp-linux-x64-vulkan.tar.gz` | `gsv_server` + ggml shared libraries |

Download them from the **Actions → latest workflow run → Artifacts** of this repository (or
the Releases page once published).

Runtime requirements:

- A Vulkan 1.2+ driver. CPU-only works too: `--device cpu`.
- The Vulkan backend needs a device with >= 32 KB compute shared memory and working f16 matmul
  pipelines. Some newer/experimental drivers (e.g. Moore Threads MUSA on the MTT S70: 16 KB
  shared memory, `createComputePipeline` fails for `matmul_f16_f32_f16acc`) cannot run it; use
  `--device cpu` there.
- x86-64 with AVX2 for the CPU backend (the published Windows/Linux builds are compiled with
  an AVX2 baseline rather than `-march=native`, so they run on any AVX2 CPU).
- Linux binaries are built on Ubuntu 22.04 (glibc 2.35+).
- Python 3.10+ for the text sidecar (`tools/text_sidecar.py`).

## 2. Get models

Converted models are published at **<https://huggingface.co/KakaruHayate/GSV.CPP>**:

| file | role | notes |
|---|---|---|
| `gsv-ar-f16.gguf` | AR (s1v3) | fp16, reference precision |
| `gsv-ar-q8.gguf` | AR (s1v3) | Q8_0 attention/FFN, ~2× smaller |
| `gsv-cond-f16.gguf` | cond (enc_p / ref_enc / wns1 / rvq) | v5turbo |
| `gsv-dit-f16all.gguf` | DiT / CFM | v5turbo |
| `gsv-refcode.gguf` | reference encoder (CN-HuBERT projection) | v5turbo |
| `gsv-v5dev-cond-f16.gguf` | cond | v5dev |
| `gsv-v5dev-dit-f16all.gguf` | DiT / CFM | v5dev |
| `gsv-v5dev-refcode.gguf` | reference encoder | v5dev |
| `gsv-bert-f16.gguf` | BERT (Chinese RoBERTa) | shared |
| `gsv-hubert-f16.gguf` | CN-HuBERT | shared |
| `gsv-vocoder-f32.gguf` | HiFi-GAN vocoder (48 kHz, ggml, sub-pixel) | shared |
| `resampy_kaiser_best.bin` | resampling filter table | shared |

The vocoder is a GGUF file: the ONNX vocoder (`vocoder_fp32.onnx`) is no longer supported.
`gsv-vocoder-f32.gguf` uses the sub-pixel layout (phase-major `voc.upsub.*` tensors); older
exports with `voc.ups.*` tensors cannot be loaded.

AR is published in two precisions; pick one (q8 is the default choice). The v5dev files are
only needed if you want the 32-step model. All files go into the model directory:

```
models/
  gsv-ar-q8.gguf
  gsv-cond-f16.gguf
  gsv-dit-f16all.gguf
  gsv-refcode.gguf
  gsv-bert-f16.gguf
  gsv-hubert-f16.gguf
  gsv-vocoder-f32.gguf
  resampy_kaiser_best.bin
```

Extra voices (e.g. v5dev or a fine-tuned checkpoint) can be described with a manifest:

```json
{
  "name": "v5dev",
  "version": "v5dev",
  "ar": "gsv-ar-q8.gguf",
  "cond": "gsv-v5dev-cond-f16.gguf",
  "dit": "gsv-v5dev-dit-f16all.gguf",
  "refcode": "gsv-v5dev-refcode.gguf",
  "bert": "gsv-bert-f16.gguf",
  "hubert": "gsv-hubert-f16.gguf",
  "vocoder": "gsv-vocoder-f32.gguf"
}
```

Put it in `models/voices/<name>/manifest.json`; relative paths resolve against that
directory. Switch at runtime with `GET /set_gpt_weights?weights_path=<name>`.

Licenses for the model files are those of the upstream GPT-SoVITS project (MIT); see
[licenses/GPT-SoVITS-LICENSE.txt](../licenses/GPT-SoVITS-LICENSE.txt).

## 3. Convert your own trained models

Training/fine-tuning happens in the upstream [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
repository. Point the exporter at the resulting checkpoints:

```bash
# AR (s1v3.ckpt) — two precisions
python tools/export_gguf/export.py --kind ar --ckpt s1v3.ckpt --out models/gsv-ar-f16.gguf
python tools/export_gguf/export.py --kind ar --ckpt s1v3.ckpt --out models/gsv-ar-q8.gguf \
    --spec attn=q8_0,ffn=q8_0

# cond / DiT / refcode from the SoVITS checkpoint (s2Gv5turbo.pth or s2Gv5dev.pth)
python tools/export_gguf/export.py --kind cond    --ckpt s2Gv5turbo.pth --out models/gsv-cond-f16.gguf
python tools/export_gguf/export.py --kind dit     --ckpt s2Gv5turbo.pth --out models/gsv-dit-f16all.gguf --spec lin=f16
python tools/export_gguf/export.py --kind refcode --ckpt s2Gv5turbo.pth --out models/gsv-refcode.gguf

# BERT / HuBERT (or reuse the published files)
python tools/export_gguf/export.py --kind bert   --ckpt chinese-roberta-wwm-ext-large --out models/gsv-bert-f16.gguf
python tools/export_gguf/export.py --kind hubert --ckpt chinese-hubert-base          --out models/gsv-hubert-f16.gguf

# vocoder (vocoder.pth -> GGUF; F32 weights, phase-major sub-pixel upsampling)
python tools/export_vocoder_gguf.py --ckpt models/vocoder.pth --out models/gsv-vocoder-f32.gguf

# resampling filter table
python tools/export_resampy_filter.py --out models/resampy_kaiser_best.bin
```

Notes:

- The exporter requires PyTorch + `gguf`; `--self-test` re-reads the written file and checks
  tensor shapes/counts:
  `python tools/export_gguf/export.py --kind dit --ckpt ... --out ... --self-test`.
- For a fine-tuned v5 model, keep the same `version` in your voice manifest
  (`v5turbo` or `v5dev`) so the default sample steps / CFG match the training recipe.
- Quantization of AR is the only aggressive one; cond/DiT/refcode stay F16/F32. See
  `tools/export_gguf/README.md` for the `--spec` syntax.
- Do not convert v2/v3/v4 checkpoints: the graph layout differs and is not supported.

## 4. Run

```bash
# text frontend
python -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r tools/text_sidecar_requirements.txt      # Windows: .venv-sidecar\Scripts\pip
.venv-sidecar/bin/python tools/text_sidecar.py --port 9882

# engine + HTTP service
./gsv_server --device vulkan --port 9880
# CPU-only: ./gsv_server --device cpu
```

### Choosing a GPU

With more than one Vulkan device (multi-GPU, or an iGPU next to a dGPU), list them first:

```bash
./gsv_server --list-devices
# ggml devices (2):
#   #0  GPU   Vulkan0     NVIDIA GeForce RTX 2070  (7.2/7.9 GiB free)
#   --   CPU   CPU         Intel(R) Xeon(R) CPU @ 1.80GHz
# use --device vulkan:N (or --device N) for GPU #N, --device cpu for CPU
```

Then pass the index: `--device 1` or `--device vulkan:1`. An index that does not exist fails
with `[device] GPU N not found` instead of silently falling back to CPU.

```bash
curl -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" -d @request.json -o out.wav
```

`request.json`:

```json
{
  "text": "你好，欢迎使用语音合成系统。",
  "text_lang": "zh",
  "ref_audio_path": "ref.wav",
  "prompt_text": "参考音频对应的文字。",
  "prompt_lang": "zh",
  "sample_steps": 4,
  "cfg_rate": 0.0,
  "seed": 20261014
}
```

If `sample_steps` / `cfg_rate` are omitted they follow the voice version (v5turbo 4 / 0.0,
v5dev 32 / 1.30). Bind to loopback only unless `--token <secret>` is set.

## 5. License

Engine code is MIT (see [LICENSE](../LICENSE)). Third-party components and the model weights
are covered by their own licenses; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
