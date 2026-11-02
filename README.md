# gsv.cpp

C++/ggml inference engine for **GPT-SoVITS V5** (v5turbo / v5dev), with a small HTTP service
that speaks the same API as the original Python `api_v2.py` (so the GAG GUI can talk to it).

The engine re-implements the full V5 pipeline in C++ on top of [ggml](https://github.com/ggml-org/ggml):

```
text ──► Python text sidecar (G2P + BERT) ─┐
                                           ▼
                          AR (s1v3) ──► cond (enc_p/ref_enc/wns1/rvq)
                                           │
                                           ▼
                          DiT (CFM, v5turbo 4 steps / v5dev 32 steps)
                                           │
                                           ▼
                          vocoder (HiFi-GAN, ggml)
                                           │
                                           ▼
                                         48 kHz WAV
```

## Features

- **ggml backends**: Vulkan (NVIDIA/AMD/Intel) and CPU; no CUDA dependency.
- **Quantized weights**: Q8_0 attention/FFN for AR, F16 for the rest; F16 vocoder.
- **Batched AR**: several texts/segments share one decode pass (fallback to per-sequence).
- **cache-dit** (DBCache) for the 32-step v5dev model: ~20% faster with near-lossless audio.
- **Request scheduler**: concurrent `/tts` calls are grouped; DiT and vocoder run as a pipeline.
- **HTTP service**: `/tts`, `/batch` + jobs (SSE), `/models`, `/set_gpt_weights`, `/metrics`,
  token auth, path allow-list, crash dumps.

V5 only: **v5turbo** and **v5dev** checkpoints are supported; v2/v3/v4 are intentionally
not supported.

## Documentation

- [docs/usage.md](docs/usage.md) — build artifacts, model download/conversion, running the service
- [docs/benchmark.md](docs/benchmark.md) — full-pipeline timings (torch vs ggml, turbo/dev)
- [tools/export_gguf/README.md](tools/export_gguf/README.md) — exporter/quantizer reference

Converted models: <https://huggingface.co/KakaruHayate/GSV.CPP> (all GGUF, AR in fp16 and q8).

## Requirements

- Windows 10/11 x64 or Linux x64, a Vulkan 1.2+ GPU (CPU-only also works).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) source tree with the audio ops patch
  (`patches/0001-ggml-audio-patch-port-on-llamacpp.patch`), built with `GGML_VULKAN=ON`.
- MSVC 2019+ (Windows) or GCC/Clang (Linux), C++17.
- Python 3.10+ for the text sidecar (`tools/text_sidecar.py`, installs with
  `tools/text_sidecar_requirements.txt`).
- No ONNX Runtime needed by default: the HiFi-GAN vocoder runs on ggml too. ONNX Runtime is only
  used when a voice points at `vocoder_fp32.onnx`; `third_party/onnxruntime/` keeps the C headers
  for that fallback.

## Build

The engine builds against [llama.cpp](https://github.com/ggml-org/llama.cpp) (for ggml) with the
audio-ops patch in `patches/`. Clone it next to this repo and apply the patch:

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp apply ../patches/0001-ggml-audio-patch-port-on-llamacpp.patch
```

### CMake (Windows / Linux)

Build ggml first (Vulkan recommended, CPU-only also works):

```bash
cmake -S llama.cpp -B llama.cpp/build-vk-rel -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-vk-rel --config Release -j
```

Then build this project, pointing `GSV_GGML_LIB_DIR` at the ggml build output
(`find_library`-style search, or leave empty to add llama.cpp as a subproject):

```bash
cmake -S . -B build -DGSV_GGML_LIB_DIR=llama.cpp/build-vk-rel
cmake --build build --config Release --target gsv_server
# binary: build/Release/gsv_server.exe (Windows) or build/gsv_server (Linux)
```

On Windows the ggml DLLs found next to the libraries are copied next to the executable.
The ggml vocoder needs no extra runtime; for the optional ONNX path, `onnxruntime.dll` (plus the
WebGPU EP if present) must be on `PATH` or next to the executable.

### MSVC script (Windows alternative)

`scripts\build-server.bat` builds the same target with `cl.exe` and copies the ONNX Runtime
DLLs it finds in `.venv-wg\...` when present (only needed for the optional ONNX vocoder path;
set `GSV_VCRT_DIR` if `onnxruntime.dll` fails to load with error 1114).

## Models

Converted weights: <https://huggingface.co/KakaruHayate/GSV.CPP> (all GGUF, AR in fp16 and
q8; see [docs/usage.md](docs/usage.md) for the file list and manifests). To convert your own
fine-tuned checkpoints:

```bat
python tools\export_gguf\export.py --kind ar     --ckpt s1v3.ckpt  --out models\gsv-ar-q8.gguf --spec attn=q8_0,ffn=q8_0
python tools\export_gguf\export.py --kind cond   --ckpt s2Gv5turbo.pth --out models\gsv-cond-f16.gguf
python tools\export_gguf\export.py --kind dit    --ckpt s2Gv5turbo.pth --out models\gsv-dit-f16all.gguf --spec lin=f16
python tools\export_gguf\export.py --kind refcode --ckpt s2Gv5turbo.pth --out models\gsv-refcode.gguf
python tools\export_gguf\export.py --kind bert   --ckpt ... --out models\gsv-bert-f16.gguf
python tools\export_gguf\export.py --kind hubert --ckpt ... --out models\gsv-hubert-f16.gguf
python tools\export_vocoder_gguf.py --ckpt models\vocoder.pth --out models\gsv-vocoder-f16.gguf
```

The model directory should contain `gsv-ar-*.gguf`, `gsv-cond-*.gguf`, `gsv-dit-*.gguf`,
`gsv-refcode.gguf`, `gsv-bert-*.gguf`, `gsv-hubert-*.gguf`, `gsv-vocoder-f16.gguf` and
`resampy_kaiser_best.bin` (see `tools/export_resampy_filter.py`).
Extra voices can be described with `models/voices/<name>/manifest.json`.

## Run

```bat
REM text frontend (G2P + BERT) on port 9882
python -m venv .venv-sidecar
.venv-sidecar\Scripts\pip install -r tools\text_sidecar_requirements.txt
.venv-sidecar\Scripts\python tools\text_sidecar.py --port 9882

REM engine + HTTP service on port 9880
tests\relcpu\gsv_server.exe --device vulkan --port 9880
```

```bat
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

Useful parameters: `voice`, `text_split_method` (`cut0`/`cut5`), `speed_factor`,
`sample_steps`/`cfg_rate` (defaults follow the voice version: v5turbo 4 / 0.0, v5dev 32 / 1.30),
`dbcache` (`off`/`auto`/`f8`/...), `skip_first_step`.

## API

| Endpoint | Description |
|---|---|
| `POST /tts` | synthesize one text, returns `audio/wav` |
| `POST /batch` | queue several texts, returns a job id |
| `GET /jobs/{id}` | job status (also `GET /jobs/{id}/events` for SSE) |
| `GET /jobs/{id}/result/{i}` | one resulting WAV |
| `GET /models` | list voices |
| `GET /set_gpt_weights?weights_path=NAME` | switch voice |
| `GET /health`, `GET /metrics` | status and counters |

Bind to loopback only unless `--token <secret>` is set; `--allow-path` restricts which
reference audio paths the service may read.

## Tests

```bat
scripts\run-svc-tests.bat        REM unit tests for the service core
powershell -File tests\contract_test.ps1   REM HTTP contract tests (mock engine)
```

## License

MIT — see [LICENSE](LICENSE). Third-party components and the model weights keep their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[licenses/GPT-SoVITS-LICENSE.txt](licenses/GPT-SoVITS-LICENSE.txt).
