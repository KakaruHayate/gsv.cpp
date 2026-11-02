# Benchmark

Full pipeline (frontend → AR → cond → DiT → vocoder), same text / reference audio / seed.
Times in milliseconds, median of three warm runs (rows 2–6). Rows 2–6 were re-measured on
2026-11-03 with the v0.1.1 Windows build (ggml-only F32 HiFi-GAN vocoder, RoPE/adaLN/mish
fusion included); rows 1 and 4 are the unchanged torch reference.

| # | configuration | total | G2P/frontend | cond | AR | DiT | vocoder | cache |
|---|---|---|---|---|---|---|---|---|
| 1 | v5turbo torch | **3142** | 162 | 251 | ~2103* | 561 | 65 | — |
| 2 | v5turbo ggml (AR fp16) | **1239** | 102 | 98 | 578 | 287 | 140 | — |
| 3 | v5turbo ggml (AR q8) | **1196** | 101 | 101 | 510 | 290 | 141 | — |
| 4 | v5dev torch | **11716** | 163 | 247 | ~2326* | 8924 | 56 | — |
| 5 | v5dev ggml | **5686** | 97 | 123 | 546 | 4710 | 139 | — |
| 6 | v5dev ggml + cache-dit | **4619** | 102 | 124 | 375 | 3853 | 140 | **24/64 = 37.5%** |

The vocoder is now the ggml F32 HiFi-GAN and its cost scales with output length: ~140 ms for the
3.4 s clip in this table (the old ONNX vocoder was ~114 ms here), ~35 ms for a 0.7 s clip. The
earlier "ggml is 3x faster" reading compared different clip lengths; for equal-length audio the
F32 ggml vocoder is currently slightly slower than ONNX, which is the remaining optimization
target (subpixel/direct-conv upsampling). Note that AR varies by roughly +/-20% between runs on this
machine (CPU sidecar + GPU sharing), which dominates the spread of the totals.

\* torch's AR loop is captured as a single CUDA graph and cannot be timed separately; this
column is `total − (frontend + cond + DiT + vocoder)` (AR + scheduling overhead).

## Environment

- CPU: Intel Xeon E5-2675 v3 @ 1.80 GHz (16 threads)
- GPU: NVIDIA GeForce RTX 2070 8 GB, driver supporting Vulkan 1.2
- OS: Windows 10 x64
- ggml: Vulkan backend (`--device vulkan`), weights as described per row
  (AR fp16 = `gsv-ar-f16.gguf`, AR q8 = `gsv-ar-q8.gguf`; cond/DiT F16; refcode F16)
- vocoder: ggml HiFi-GAN, F32 weights (`gsv-vocoder-f32.gguf`), Vulkan backend
- torch: 2.12.1 + CUDA 13.0, fp16, `use_cuda_graph=True`, batch size 1, text split `cut0`
- sample steps: v5turbo 4 / cfg 0.0; v5dev 32 / cfg 1.30
- cache-dit: DBCache `Fn=8`, threshold 0.02, warmup 8, downsample 4

Notes:

- Text: `你好，欢迎使用语音合成系统。`
- Reference: 16 kHz Chinese prompt, ~3.4 s output audio in every row.
- Timings vary by a few percent between runs; the torch numbers include CUDA graph replay
  (two warm-up runs with the same text are done before the measured run).

## DiT fusion (RoPE + adaLN)

`gsv_dit.cpp` builds the DiT step graph with a fused RoPE (`ggml_rope_ext`, head 0 only, with a
`pos` tensor and freq factors that reproduce the checkpoint `inv_freq`) and a fused
`LN(x)·(1+scale)+shift` (`ggml_layernorm_affine` with `wb = [1+scale, shift]`), replacing the
per-block cos/sin multiply chain and the `norm → add → mul → add` sequence.

Same machine, turbo-q8, same text / seed, warm:

| DiT | before | after | delta |
|---|---|---|---|
| v5turbo DiT | 333 ms | 291 ms | −12.6% |
| v5turbo total | 1130 ms | 1033 ms | −8.6% |

Audio correlation against the pre-fusion build is 0.9955; two runs with the same seed are
bit-identical (corr 1.000000), so determinism is preserved. The `mish` epilogue fusion (A4 in the
DiT report) is also in: `ggml_add_act` gained an `act=3` (mish) branch on CPU and in the Vulkan
shader, and `conv_pos` folds bias + mish into one node. Its effect is below the 0.2% noise floor,
so the numbers above do not change. `tests/test_add_act.cpp` checks all four activations against
a host reference on CPU and on every GPU backend; CI runs it under Mesa lavapipe, which exercises
the Vulkan shader branch (max|diff| 2.4e-7).
