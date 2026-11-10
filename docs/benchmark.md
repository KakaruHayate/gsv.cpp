# Benchmark

Full pipeline (frontend → AR → cond → DiT → vocoder), same text / reference audio / seed.
Times in milliseconds, median of three warm runs (rows 2–6). Rows 2–6 were re-measured on
2026-11-03 with the current development build (ggml-only F32 HiFi-GAN vocoder with sub-pixel upsampling,
RoPE/adaLN/mish fusion included); rows 1 and 4 are the unchanged torch reference.

| # | configuration | total | G2P/frontend | cond | AR | DiT | vocoder | cache |
|---|---|---|---|---|---|---|---|---|
| 1 | v5turbo torch | **3142** | 162 | 251 | ~2103* | 561 | 65 | — |
| 2 | v5turbo ggml (AR fp16) | **945** | 102 | 93 | 363 | 275 | 68 | — |
| 3 | v5turbo ggml (AR q8) | **1028** | 101 | 95 | 425 | 279 | 67 | — |
| 4 | v5dev torch | **11716** | 163 | 247 | ~2326* | 8924 | 56 | — |
| 5 | v5dev ggml | **5415** | 102 | 126 | 456 | 4617 | 68 | — |
| 6 | v5dev ggml + cache-dit | **4740** | 101 | 120 | 422 | 4005 | 69 | **17/64 = 26.6%** |

The vocoder is the ggml F32 HiFi-GAN with **sub-pixel upsampling**: the five transposed
convolutions are replaced by one 3-tap conv1d on the input grid plus a phase interleave
(`src/gsv_vocoder_ggml.cpp::upsample_subpixel`), and every convolution runs through
`ggml_conv_direct_1d_fused` (no im2col buffer; bias/LeakyReLU, the resblock input LeakyReLU and
the residual add all fold into the same kernel). For the 3.4 s clip in this table the vocoder
takes **~68 ms**, down from ~140 ms with `ggml_conv_transpose_1d`, ~115 ms with sub-pixel + im2col
and ~75 ms with per-conv direct conv (-51% overall). The old ONNX vocoder was ~114 ms. AR varies
by roughly ±20% between runs on this machine (CPU sidecar + GPU sharing), which dominates the
spread of the totals.

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

## DiT algorithm switches (2026-11-06, same machine/text/seed)

Optional per-request switch. Quality is measured on the DiT mel output (`mel_out`) against the
all-off run — vocoded waveform correlation is dominated by phase and is not a usable fidelity
proxy. The DiT column is the stage time from the server log (`dit`).

| setting | mel relL1 | mel corr | DiT time | verdict |
|---|---|---|---|---|
| all off (upstream behaviour) | 0 | 1.0 | — | default |
| `dit_uncond_reuse=1` | 4.2e-3 | 0.99999 | ~-30% | only for voices with cfg > 0 (v5dev) |
| `dit_uncond_reuse=2` | 2.6e-3 | 0.99999 | ~-28% | safer variant of the above |

Notes: the switch does nothing when cfg = 0 (v5turbo / the `default` voice), because there is no
Uncond branch to reuse then. It stacks with cache-dit.

## CPU backend threading

ggml's barrier spins while waiting, so on SMT CPUs the default hardware_concurrency() (logical
threads) oversubscribes and slows requests down. gsv_server now defaults to physical cores when
--threads is not given (hardware_concurrency()/2 when > 8).

Windows, Xeon E5-2675 v3 (16 cores / 32 threads, 1.8 GHz), turbo-q8, 4 steps, 3.04 s audio:

| threads | total | DiT | cond | AR | vocoder |
|---|---|---|---|---|---|
| 32 (old default) | 39.9 s | 28.8 s | 1.80 s | 2.26 s | 3.07 s |
| 16 (new default) | 35.4 s | 29.0 s | 1.20 s | 1.52 s | 1.62 s |
| 8 | 35.9 s | 28.8 s | 1.15 s | 1.54 s | 1.93 s |

DiT is memory-bound in this configuration and does not improve past 8 threads; the win is in the
AR/cond/vocoder stages. --threads N overrides the default.

## AR decode: KV cache concat removal (2026-11-08)

The decode graph used to copy the whole KV history twice per layer per token
(`ggml_concat(cache[0..L), new_column)`, K and V) only to make the "write → read" dependency
explicit; the copy is mathematically redundant. The new column is now written straight into the
resident cache column `len` (a `cpy`/`set_rows` side-effect root expanded **before** the logits
root, so node order still guarantees write-then-read) and flash attention reads the cache view
directly. Both decode paths are covered: `run_decode()` (CPU) and `run_decode_cached()` (the
fixed-shape graph reused on the GPU).

AR-only measurement, teacher forcing with a fixed context so every build sees identical inputs,
`gsv-ar-q8.gguf`, 179 steps, interleaved A/B (ABBA) with a fresh process per run:

| path | before | after | delta |
|---|---|---|---|
| CPU, 4 threads, prompt 200 | 24.84 ms/token | **12.55 ms/token** | −49% |
| CPU, 8 threads, prompt 400 | 23.9 ms/token | **12.9 ms/token** | −45% |
| Vulkan (RTX 2070), prompt 200 | 2.88 / 2.91 / 3.06 ms/token | **2.52 / 2.54 / 2.73** | −12% |
| Vulkan, prompt 400 | 3.33 / 3.42 / 3.54 ms/token | **3.13 / 3.20 / 3.29** | −7.5% |
| Vulkan, batch 2, prompt 200 | 787.6 / 683.2 ms | **629.8 / 623.7 ms** | −10…−20% |

Correctness: on CPU the per-token probability dump is **bit-identical** before/after
(md5 unchanged over 61 steps × 1025 vocab), and the generated token sequence matches on both
backends (`be29fd0385f0553f` for the 179-step fixture, `7b7e35445ea1c39d` for batch 2). On Vulkan
the graph-cached path now feeds flash attention a **non-contiguous** cache view, so it takes a
different kernel path: the probabilities differ from the concat build by `mean|d| 9.6e-7 /
max|d| 2.2e-2`, i.e. **smaller** than the normal CPU-vs-Vulkan backend spread on the same fixture
(`mean 4.7e-5 / max 5.9e-2`), with identical argmax on all steps. The largest deviation sits on a
`top_k=15` pivot tie (rank 15 vs 16), not on a mask error.

Note for reviewers: on the GPU path the `set_rows` → `flash` dependency is not a graph edge, it
relies on node order being execution order (CPU is sequential; Vulkan was deterministic across
repeated runs). `GSV_AR_NO_GRAPHCACHE=1` still forces the CPU-style path if this ever needs to be
bisected.

## AR decode: where the time goes, and why n-gram speculation does not pay off (2026-11-10)

### What the AR decode already does

`src/gsv_ar.cpp`, in the order the per-token cost is saved:

| optimisation | effect (measured) |
|---|---|
| KV cache resident on the backend, new column written in place (`set_rows`/`cpy` side effect, no `concat`) | 0.1.3: −7…−12% on Vulkan, −49% on CPU (section above) |
| fixed-shape decode graph reused across steps (bucket = `L + 25%`), rebuilt only when `L` outgrows the bucket | build 0.01 ms/step (vs ~0.8 ms rebuilding) |
| constant mask + tail slots masked by index, so the mask buffer is not resized per step | mask rebuild 0.000 ms/step |
| flash attention (`ggml_flash_attn_ext`) with an F16 mask | part of `compute` |
| fused LayerNorm+affine (`ln_affine`), fused bias+act (`add_act`) | `GSV_NO_FUSE=1` reverts |
| BERT projection in-graph via the augmented `[W|b]` weight (no 30 M-MAC host loop) | prefill only |
| positional table precomputed once (`pe_tab`), host-side embedding add | emb+pe 0.002 ms/step |
| batched AR: several sequences share one decode pass, per-sequence seeds, KV released after | B=2: −10…−20% |
| vectorised sampler (`argmax(probs/q)`, bit-identical to torch) | sampling 0.066 ms/step |
| F16 weights everywhere, Q8_0 for the AR option | 83 MB vs 296 MB read per token |

Per-step breakdown, `gsv-ar-q8.gguf`, RTX 2070 / Vulkan, 156 steps, warm (`GSV_AR_PROFILE=1`,
`tests/probe_ar_profile.py`):

| stage | per step | share |
|---|---|---|
| graph compute | **3.53 ms** | 79% |
| host→device uploads (x / mask / idx, 3 `tensor_set`) | 0.62 ms | 14% |
| logits read-back + loop | 0.22 ms | 5% |
| sampling | 0.07 ms | 2% |
| graph (re)build, mask rebuild, emb+PE | 0.01 / 0.000 / 0.002 ms | ~0% |

So the remaining cost is the graph itself, not framework overhead: the prefill pass costs
**0.3 ms/position** for 159 positions against **3.53 ms/position** one-at-a-time, i.e. the same
weights read once per pass instead of once per token. Anything that puts several tokens into one
pass (speculation, larger batches) is therefore the only big lever left — though that 12x is an
upper bound, not what small batches deliver (B=2 is only ~1.5x, see below).

### Prompt-lookup (n-gram) speculation: measured, and rejected

The idea is that speech tokens repeat locally (held vowels, silence, repeated prosody), so the
last 2–3 tokens can be matched against the prompt/generated history and the following tokens used
as a draft. We tested it on real streams before writing any decoder code
(`tests/probe_lookup_accept.py`, `tests/probe_token_stream.py`, engine dump `sem.i32` +
`prompt_codes.i32` via `GSV_ENGINE_DUMP`):

| text | AR tokens | distinct 2-grams / total | longest run of equal tokens |
|---|---|---|---|
| 50 chars | 73 | 265 / 276 | 2 |
| 15 chars (all 「啊」) | 27 | 219 / 230 | 2 |
| 23 chars (「哈哈…慢慢慢…」) | 11 | 203 / 214 | 2 |

Every repeated n-gram in that table comes from the **reference audio** part of the history
(`[486,486,…]` ×12, the silence/long-vowel run in `prompt_codes`), never from the generated
stream. Simulated acceptance over all four texts: **0 drafts proposed** for n = 2 and 3, K = 3/5/8.
The counts are what you would expect from near-i.i.d. draws over the 1024-entry codebook
(276 bigram positions predict ~37 chance collisions; we see 11, all from the prompt).

The premise holds for *acoustic* codec tokens (EnCodec/SoundStream-style RVQ, or mel frames),
where a held note is literally `[a,a,a,a,…]`. In GPT-SoVITS V5 the AR emits **semantic** tokens
(one-stage VQ over HuBERT features, 1024 bins at ~25 Hz) that are information-dense and
uncorrelated: the model reaches EOS in 11–156 tokens for these texts, and consecutive codes carry
no local redundancy. Silence in the *output* is represented by the DiT/CFM stage, not by repeated
semantic tokens.

Consequence: implementing lookup decoding would add a draft-search and a multi-token verify path
for a 0% hit rate, i.e. it can only slow the decoder down. Not implemented.

### What would actually use the headroom

The 12x prefill-vs-decode gap needs a draft source that is *correlated with the AR's own output*,
which n-grams are not. The next section measures how much a wider pass is actually worth, and why
the draft-based families (lookahead, self-speculative) still fail for a second, independent
reason.

## AR decode: width is cheap, the distribution is flat (lookahead / self-speculative) (2026-11-10)

Follow-up to the section above, after Lookahead Decoding and self-speculative (layer-skipping)
decoding were proposed. Both rest on the same assumption: putting W tokens into one pass must cost
less than W separate passes. That part is true, and it is measured below. What kills them here is
the second half of the trade — the acceptance rate.

### 1. Width scaling (`tests/bench_ar_width.cpp`)

Teacher-forced decode (`oracle_tokens`, so every run does exactly 160 steps and no sequence exits
early), `gsv-ar-q8.gguf`, RTX 2070 / Vulkan, second call (KV allocated, graph and pipelines warm):

| B | total | ms/step | ms/token | vs B=1 | tok/s |
|---|---|---|---|---|---|
| 1 | 388.3 ms | 2.43 | 2.43 | 1.00x | 412 |
| 2 | 407.1 ms | 2.55 | 1.27 | 1.91x | 786 |
| 4 | 539.4 ms | 3.37 | 0.84 | 2.88x | 1187 |
| 8 | 815.9 ms | 5.10 | 0.64 | 3.81x | 1569 |

So the compute really is amortised: eight sequences cost 2.1x the per-step time of one. The gain
is sub-linear (not 8x) because the decode matvecs are bandwidth-bound, but 3.8x on throughput is
still the largest lever in the decoder.

### 2. Acceptance ceiling (`tests/probe_ar_peakiness.py`)

`GSV_AR_PROB_DUMP` gives the post-sampler distribution per step (74 steps, same text/seed):

| sampler | mean p(top1) | min p(top1) | top-5 mass |
|---|---|---|---|
| top_k=15, temp=1.0 (default) | **0.211** | 0.088 | 0.591 |
| top_k=15, temp=0.7 | 0.269 | 0.099 | 0.680 |
| top_k=5, temp=1.0 | 0.339 | 0.224 | 1.000 |

Every hard-draft scheme (n-gram, lookahead) verifies by *matching a token*, so its acceptance is
exactly `p(draft token)` — at best `p(top1)` = 0.21 at the default sampler. (This is the caveat
the Lookahead write-ups already flag: it is designed for greedy search.)

### 3. Return on one verify pass

One pass of width W returns `1 + a + a² + a³` tokens (1 sampled base token + 3 drafts, each
verified at rate `a`), and costs `C(W)` from the table above:

| acceptance a | tokens/pass (W=4) | ms/token | vs 2.43 baseline |
|---|---|---|---|
| 0.21 (measured) | 1.26 | 2.68 | **0.91x — slower** |
| 0.35 | 1.53 | 2.20 | 1.10x |
| 0.50 | 1.88 | 1.80 | 1.35x |
| 0.70 | 2.53 | 1.33 | 1.83x |
| 1.00 (greedy) | 4.00 | 0.84 | 2.88x |

### 4. Per-method verdict

- **Prompt lookup / n-gram** — measured 0 drafts (previous section). Dead.
- **Lookahead Decoding** (Fu et al., NeurIPS 2023) — the Jacobi lookahead branch would produce
  drafts in a wide pass, so `C(4)` applies, but acceptance is the same hard-token match, i.e.
  `a ≈ 0.21`: row 1 of the table, a *slowdown*. It also needs the n-gram pool (0 hits here), a
  W-wide fixed-shape graph and a 2D window mask, i.e. the most work of the three for a negative
  return.
- **Self-speculative (layer skipping)** — worse, because drafting W−1 tokens is *sequential*:
  `(W−1) × (k/24) × 2.43 ms` on top of the verify pass. For W=4 and half the layers that is
  +3.65 ms, so even at `a = 0.7` (generous for a 12-layer draft) the result is
  7.0 ms / 2.53 tokens = 2.8 ms/token, i.e. **slower than plain decode**. It only breaks even at
  `k ≈ 6` layers *and* high agreement, and a quarter-depth model will not agree with a flat
  distribution like this one.

Root cause for all three: at top_k=15/temp=1.0 the semantic-token distribution is spread over the
candidates (top-5 = 0.59), so the model itself is only 21% confident. Speculation converts
confidence into speed; there is no confidence to convert. Greedy decoding (`top_k=1`) would make
the whole family viable at ~2.9x — but it also removes the prosody variation users select with the
sampler, so it is a product decision, not an optimisation.

### 5. Consequence

Width is exploitable across *requests*, not within one: `kMaxBatch = 8` with a 20 ms grace window
already fills it under concurrent load, giving the 3.8x AR throughput on the table (the honest
real-world number is lower than the bench, ~1.5x at B=2 in the earlier section, because sequences
finish at different steps and the batch thins out). For single-request latency the only remaining
lever is a cheaper AR (distillation or lower precision), not speculation.

## What is not parallelised

- **Pipeline overlap** (AR of segment i+1 while DiT/vocoder runs segment i) is not implemented:
  ggml exposes a single backend queue, so both stages would serialize on the device anyway, and
  on CPU they compete for the same cores. Batches are formed *across* requests instead
  (/batch, AR batcher, request scheduler).
- **Larger conv tiles** in conv_direct_1d were not beneficial: the kernel already uses 6-wide
  time blocks x 16-wide output blocks, and DiT (the dominant CPU stage) is memory-bound, so
  thread count beyond 8 did not move it.
