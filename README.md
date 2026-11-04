# gsv.cpp

**C++/ggml inference engine for GPT-SoVITS V5** — `v5turbo` (4-step DMD) and `v5dev`
(32-step + cache-dit) — with an api_v2-compatible HTTP service.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KakaruHayate/gsv.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/KakaruHayate/GSV.CPP)

English | [中文](README.zh.md)

The whole V5 pipeline runs in C++ on top of [ggml](https://github.com/ggml-org/llama.cpp):
AR → cond (enc_p / ref_enc / wns1 / rvq) → DiT/CFM → HiFi-GAN vocoder (sub-pixel, ggml).
A small Python sidecar does text front-end work (G2P + BERT ids); it needs no torch, and ONNX
Runtime is not used anywhere.

```
text ──► Python sidecar (G2P + BERT ids) ──┐
                                            ├─► gsv_server (C++/ggml) ──► 48 kHz WAV
GGUF models (AR/cond/DiT/refcode/BERT/HuBERT/vocoder) ─┘
```

## Features

- **ggml backends**: Vulkan (NVIDIA/AMD/Intel) and CPU; no CUDA dependency.
- **Quantised weights**: Q8_0 attention/FFN for AR, F16 elsewhere, F32 sub-pixel vocoder.
- **Batched AR**: several texts/segments share one decode pass.
- **cache-dit** (DBCache) for the 32-step v5dev model: faster with near-lossless audio.
- **Request scheduler**: concurrent `/tts` calls are grouped; DiT and vocoder run as a pipeline.
- **HTTP service**: `/tts`, streaming `/tts`, `/batch` + jobs (SSE), `/models`, `/metrics`,
  `/progress`, token auth, reference-audio allow-list, crash dumps.
- **Vendored text frontend** (`frontend/`): G2P for zh/en/ja/ko/yue, BERT tokenization,
  mixed-language segmentation — no torch/transformers.

V5 only: v2/v3/v4 checkpoints are intentionally not supported.

## Quick start

**Windows (one-click)**: download `windows/GSV.CPP-win64-*.zip` from
<https://huggingface.co/KakaruHayate/GSV.CPP>, extract it anywhere and run `Start GAG.bat`.
It contains the engine, an embedded Python with the sidecar dependencies, the models and the
[GAG](https://github.com/KakaruHayate/GPT-SoVITS-Api-GUI) GUI.

**Linux**: grab the prebuilt `gsv_server` from the
[releases](https://github.com/KakaruHayate/gsv.cpp/releases), plus the GGUF models and the
frontend data from <https://huggingface.co/KakaruHayate/GSV.CPP>, then follow
[docs/build.md](docs/build.md) for the sidecar and run steps.

**From source**: see **[docs/build.md](docs/build.md)** (English) or
**[docs/build.zh.md](docs/build.zh.md)** (中文) — full instructions for Linux and Windows,
including the patched llama.cpp/ggml, the Python sidecar and the model layout.

## Models

Prebuilt GGUFs: <https://huggingface.co/KakaruHayate/GSV.CPP> (AR in fp16 and q8, shared
BERT/HuBERT/vocoder, v5turbo + v5dev sets). Voice manifests go in `models/voices/<name>/`.

To convert your own fine-tune, use `tools/export_gguf` — see
[tools/export_gguf/README.md](tools/export_gguf/README.md) (GGUF only; the ONNX vocoder path is
gone).

## Run

```bash
# text frontend (G2P) — see docs/build.md §6 for the data files
python -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r frontend/requirements-zh.txt
.venv-sidecar/bin/python tools/text_sidecar.py --port 9882

# engine + HTTP service
./build/gsv_server --models models --device vulkan --port 9880
```

```bash
curl -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' -d '{
  "text": "你好，欢迎使用语音合成系统。",
  "text_lang": "all_zh",
  "ref_audio_path": "models/ref.wav",
  "prompt_text": "参考音频对应的文本",
  "prompt_lang": "all_zh",
  "sample_steps": 4
}' -o out.wav
```

Reference audio must be 3–10 s (max 30 s) and inside an allowed directory (see
`--allow-path`). The full parameter list is in [docs/api.md](docs/api.md).

## Documentation

| document | contents |
|---|---|
| [docs/build.md](docs/build.md) / [中文](docs/build.zh.md) | from zero to a running server (Linux/Windows) |
| [docs/usage.md](docs/usage.md) | artifacts, models, conversion, service usage |
| [docs/api.md](docs/api.md) | HTTP API: endpoints, streaming, references, errors |
| [docs/benchmark.md](docs/benchmark.md) | full-pipeline timings (torch vs ggml, turbo/dev) |
| [tools/export_gguf/README.md](tools/export_gguf/README.md) | checkpoint → GGUF exporter |

## Licence

MIT (see [LICENSE](LICENSE)). The vendored frontend keeps its upstream licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
Model weights are conversions of GPT-SoVITS checkpoints (MIT) and stay under that licence.
