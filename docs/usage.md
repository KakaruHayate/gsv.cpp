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


### Reproducibility

The text frontend is deterministic across platforms (identical phones / ert_ids). The AR
sampler is not: the same seed on Windows and Linux yields different token trajectories because
floating-point differences in the logits change the sampled codes, so the two wavs differ in
length and content. Within one platform and build the output is bit-exact (same SHA-256).

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
python tools/export_gguf/export_vocoder_gguf.py --ckpt models/vocoder.pth --out models/gsv-vocoder-f32.gguf

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

The text frontend (G2P) is vendored in `frontend/` (upstream layout, MIT). It needs **no
torch/transformers**; see [frontend/README.md](../frontend/README.md) for the data files
(`G2PWModel`, BERT tokenizer, `nltk_data`) and how to point the sidecar at them.

```bash
# text frontend (zh only; add frontend/requirements-multi.txt for en/ja/ko/yue)
python -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r frontend/requirements-zh.txt   # Windows: .venv-sidecar\Scripts\pip
export GSV_G2PW_MODEL=/path/to/G2PWModel                        # dir with g2pW.onnx
export GSV_BERT_DIR=/path/to/chinese-roberta-wwm-ext-large      # vocab.txt + tokenizer_config.json
export NLTK_DATA=/path/to/.nltk_data                            # required by en (g2p_en)
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

### Windows one-click package

`python scripts/package_windows.py --out dist/GSV.CPP-win64` assembles a portable folder:
GAG sources, `runtime/gsv` (engine + DLLs + VC runtime), an embeddable Python with the GAG and
frontend dependencies, `runtime/frontend` (sidecar + frontend code), `runtime/models` (GGUF,
G2PW, tokenizer, nltk) and a `Start GAG.bat`. It expects a `GAG` checkout next to this repo.
Useful flags: `--reuse-models` (keep an existing `runtime/models`), `--no-models`,
`--skip-pip` (reuse the embedded Python).

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

### DiT algorithm switches (optional, default = production config)

Per-request JSON fields (also accepted as GET query params). The defaults are the shipped
production values; both switches are runtime gates, so changing them does not rebuild the
DiT graph.

| field | default | meaning |
|---|---|---|
| `dit_cfg_off_t` | 0.7 | skip the Uncond branch once `t >= dit_cfg_off_t` (equivalent to cfg_scale = 1.0 for those steps). **The skip ratio is `1 - dit_cfg_off_t`**: 0.7 skips the last 30% of steps. Only affects voices with cfg > 0 (v5dev); for v5turbo (cfg 0) it is an exact no-op. This is a real fidelity/speed trade, not a free win — see the table below. |
| `dit_uncond_reuse` | 0 | 0 = off; 1 = compute the Uncond branch only on even steps and reuse it on odd steps; 2 = linear extrapolation `2*neg[k-1] - neg[k-2]` on odd steps. |

Example (v5dev, skip the last 40% of steps + odd-step Uncond reuse):

```json
{ "text": "...", "voice": "v5dev", "dit_cfg_off_t": 0.6, "dit_uncond_reuse": 1 }
```

Measured on RTX 2070 / Vulkan with the DiT-only fixture (T=1000 frames = 500 reference + 500
generated, 32 steps, cfg 1.30) against the all-off run at the same seed. Deltas are on the DiT
mel output (`relL1` / correlation of `mel_out`) — vocoded waveform correlation is dominated by
phase and is not a usable fidelity proxy:

| setting | skip | mel relL1 | mel corr | DiT time |
|---|---|---|---|---|
| `dit_cfg_off_t=0.95` | 5% | 2.5e-4 | 1.00000 | 1.01x |
| `dit_cfg_off_t=0.90` | 10% | 6.5e-4 | 1.00000 | 1.02x |
| `dit_cfg_off_t=0.85` | 15% | 8.3e-4 | 1.00000 | 1.02x |
| `dit_cfg_off_t=0.80` | 20% | 1.2e-3 | 1.00000 | 1.06x |
| `dit_cfg_off_t=0.7` (default) | 30% | 1.6e-3 | 0.99999 | 1.11x |
| `dit_cfg_off_t=0.60` | 40% | 2.0e-3 | 0.99999 | 1.20x |
| `dit_cfg_off_t=0.50` | 50% | 2.4e-3 | 0.99998 | 1.33x |
| `dit_cfg_off_t=0.7` + `dit_uncond_reuse=1` | 30% | 4.2e-3 | 0.99999 | 1.45x |
| `dit_cfg_off_t=0.7` + `dit_uncond_reuse=2` | 30% | 2.6e-3 | 0.99999 | 1.39x |
| `dit_cfg_off_t=0.6` + `dit_uncond_reuse=1` | 40% | 4.1e-3 | 0.99998 | 1.54x |

The mel delta grows roughly linearly with the skip ratio, so pick the point you can afford:
0.90 (10%) is nearly free, 0.70 with `dit_uncond_reuse=1` is the speed sweet spot. Both stack
with cache-dit, and both are exact no-ops on v5turbo (cfg = 0, verified bit-identical).

#### Reference

The trick is to stop paying for the Uncond branch once the trajectory is close to clean data:
`x_t` is nearly denoised there, so the cond/uncond velocity fields have converged and dropping
the CFG amplification costs little. `dit_uncond_reuse` goes one step further and reuses (or
linearly extrapolates) the last computed Uncond velocity instead of evaluating the network.

### Timeouts

The engine lock / scheduler / AR-batch waits used to be hardcoded at 300 s. They are now
configurable:

- global default: `--lock-timeout-ms N` (default 300000)
- per request: `"timeout_ms": N` in the JSON body (or GET query param); `<= 0` uses the global
  default. Applies to the engine lock, the request scheduler and the AR batch wait.

The HTTP read timeout for the text sidecar is separate (180 s, `svc_core.h`).

## 5. License

Engine code is MIT (see [LICENSE](../LICENSE)). Third-party components and the model weights
are covered by their own licenses; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
