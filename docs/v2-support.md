# v2 / v2Pro / v2ProPlus 支持（VITS 解码链路）

`--version v2` 让 gsv.cpp 跑 GPT-SoVITS **v2 家族**的 S2 解码器。它不是 v5 的配置开关：
v2 家族用的是 **VITS**（enc_p → flow → dec 一步出波形），而 v5 是 cond(enc_p+bridge+wns1)
→ DiT/CFM → mel → vocoder。两者共用的只有 HuBERT / BERT / AR / ref_enc。

```
sem codes (25 Hz, 1024 码本)
  └─ quantizer.vq.decode + nearest ×2      → [768, 2Tc]       gsv_v2sovits::decode_codes
  └─ enc_p (ssl_proj, encoder_ssl×3, text_embedding,          gsv_encp（与 v5 逐层同构，直接复用）
           encoder_text×6, MRTE, encoder2×3, proj) → m_p, logs_p [192, T]
  z_p = m_p + ε·exp(logs_p)·noise_scale     （ε~N(0,1), noise_scale 默认 0.5）
  z   = flow(z_p, g=ge, reverse=True)       → [192, T]        gsv_v2sovits::flow_reverse
  wav = dec(z, g=ge)                        → [T·640] @32 kHz gsv_v2sovits::decode_wav
```

参考音频侧（与 v5 相同，`prep_reference`）：

```
16k + 0.3s 零尾 → HuBERT(教师 or 蒸馏) → ssl_proj(k=2,s=2) → VQ → prompt_codes[25 Hz]
32k + maxx    → spectrogram(2048/640) → [:, :704] → ref_enc → ge[512]
（v2 不需要 DiT，所以不再算 mel2）
```

## 文件与职责

| 文件 | 内容 |
| --- | --- |
| `src/gsv_v2sovits.{h,cpp}` | 码本 + ×2、flow（4×ResidualCouplingLayer mean_only）、dec（HiFi-GAN 5 段上采样）；enc_p 复用 `gsv_encp`，ref_enc 复用 `gsv_refenc` |
| `src/gsv_engine.cpp` | `cfg.version` / `cfg.s2_file` / `GSV_VERSION`；`prep_reference` 与 `synthesize` 的 v2 分支；HuBERT 前端选择 |
| `src/gsv_engine.h` | `version()` / `sample_rate()` / `hubert_frontend()` |
| `tests/probe_v2sovits.cpp` | 分段对拍 + 端到端 benchmark |
| `tests/v2_dump_reference.py` | 从 torch（GPT-SoVITS-CPUFast）导出对拍 fixture |

## 权重与 GGUF 约定

`tools/export_gguf/export.py --kind v2sovits --ckpt s2G2333k.pth --out gsv-v2sovits.gguf --version v2`
（`--version` 取 `v2|v2pro|v2proplus`）。张量名就是 checkpoint 的键名：

```
enc_p.*            TextEncoder（与 v5 同构，前缀一样）
ref_enc.*          MelStyleEncoder(704 → 512)
ssl_proj.*         Conv1d(768, 768, k=2, stride=2)   ← v2 的 25 Hz 下采样
quantizer.vq.layers.0._codebook.embed   [1024, 768]
flow.flows.{0,2,4,6}.*                 ResidualCouplingLayer（奇数是 Flip，无参数）
dec.{conv_pre,ups.0..4,resblocks.0..14,conv_post,cond}.*
```

加载时做两件**必须**的变换（都在 `gsv_v2sovits.cpp`，形状驱动、失败即报错）：

1. **weight_norm 折叠**。GGUF 存的是 `weight_g` / `weight_v`（torch 布局，未重排）。
   `W = v · g/‖v‖₂`，范数沿 dim0 之外的轴：
   - `Conv1d` v:[OC,IC,K]、g:[OC] → 折叠成 ggml `[K,IC,OC]`
   - `ConvTranspose1d` v:[IC,OC,K]、g:[IC]（**dim0 = in**）→ 同上
   结果统一存 F32（`ggml_conv_direct_1d` 的 CPU 实现要求 F32 核）。约 26.5 M 参数 ≈ 106 MB。
2. **ConvTranspose1d → 相位主序子像素卷积**。推导见 `gsv_v2sovits.cpp` 文件头：
   每个相位 `r` 是输入的一个普通相关，`k = d·s + (r + pad)`，`d` 可能取 −1（用下一帧），
   所以按权重形状自动求 `[d_lo, d_hi]`、`M = d_hi−d_lo+1`、`P = d_hi`，
   折叠成 `[M, IC, OC·s]` 的核，一次 `ggml_conv_direct_1d_fused` + 相位交织得到上采样结果。
   对 v2（s=10,8,2,2,2 / K=16,16,8,2,2）得到 `M = 3,3,5,1,1`、`P = 1,1,2,0,0`，
   输出行数正好等于输入帧数（`OL = T + d_lo + d_hi = T`）。

v2Pro / v2ProPlus 与 v2 的差别只有 `gin_channels`（1024）、`upsample_initial_channel`（v2ProPlus
是 768）、dec 上采样核（20/16/8/2/2）与采样率。除了**上采样步长**（K=16 既可能 s=10 也可能 s=8，
无法从形状唯一反推）按 `v2sovits.version` 查 `configs/s2*.json` 的表，其余全部由权重形状校验。
v2Pro/v2ProPlus 的 `sv_emb`（20480 维声纹条件）需要一个外部声纹模型，本实现不接；
文件里没有 `sv_emb` 时按 repo 的 `version != v2pro` 分支走（ge 只来自 ref_enc）。

## 怎么跑

```bash
# 1) 导出（一次性）
python tools/export_gguf/export.py --kind v2sovits --ckpt s2G2333k.pth \
       --out models/gsv-v2sovits.gguf --version v2 --f16
python tools/export_gguf/export.py --kind ar --ckpt s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt \
       --out models/gsv-ar-v2-q8.gguf --q8          # 或 --f16 出 F16 档

# 2) 模型目录里需要：
#    gsv-v2sovits.gguf  gsv-ar-v2-q8.gguf  gsv-hubert-f16.gguf（或 gsv-dhubert.gguf）
#    gsv-bert-f16.gguf  resampy_kaiser_best.bin（tools/export_resampy_filter.py）

# 3) 跑
GSV_VERSION=v2 gsv_server --models models --device cpu
```

代码里等价于：

```cpp
gsv_engine_cfg c;
c.version   = "v2";                      // 或 "v2pro" / "v2proplus"；空 = 读 GSV_VERSION，再空 = v5
c.s2_file   = "models/gsv-v2sovits.gguf";
c.hubert_frontend = "";                  // "" = 按 GGUF 的 general.architecture 自动识别
                                         //      "distilled" = 蒸馏 HuBERT, "hubert" = 教师
c.v2_noise_scale = 0.5f;
auto * eng = gsv_engine::load(c);
eng->prep_reference(ref_wav, prompt_codes, ge, mel2, mel2_T);   // mel2 留空
eng->ar_generate(phones, bert, prompt_phones, prompt_bert, prompt_codes, 2000, 4, seed, sem);
eng->synthesize(sem, phones, prompt_codes, prompt_phones, ge, mel2, mel2_T,
                0, 0.0f, seed, wav, /*speed=*/1.0);
```

`steps` / `cfg`（DiT 的采样步数与 CFG）在 v2 路径被忽略 —— v2 没有扩散过程，
唯一的随机性是 `noise_scale`。

### 模型需求

| 组件 | v5 | v2 |
| --- | --- | --- |
| AR | `gsv-ar-q8.gguf` | `gsv-ar-v2-q8.gguf`（同一套 `--kind ar`） |
| S2 | cond + DiT + vocoder 三个文件 | `gsv-v2sovits.gguf` 一个文件 |
| HuBERT | 教师 | 教师**或**蒸馏（`GSV_HUBERT_FRONTEND=distilled`） |
| BERT | 必需 | 必需（enc_p 的 text_embedding 输入） |
| ref_enc | 在 cond 文件里 | 在 v2sovits 文件里 |

## 对拍

`tests/probe_v2sovits.cpp` 逐段比较 torch 与本实现的中间张量（`tests/v2_dump_reference.py`
产出 fixture，同一份 ssl/spec704 两侧共用，因此比较与重采样实现无关）。

见下表（4 vCPU EPYC 7K62，单线程/4 线程 CPU，参考音频 4.2 s）。

fixture：`ref.wav` → 4.12 s @ 32 kHz，103 codes → 206 帧（`/tmp/v2fix`，由 `tests/v2_dump_reference.py` 生成）。
命令：`probe_v2sovits gsv-v2sovits.gguf /tmp/v2fix --threads 4`。

| 阶段 | 张量 | 元素数 | max_abs | rms_diff | 参考 rms | cos | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ssl_proj + VQ | codes[103] | 103 | — | — | — | match 103/103 (100.00%) | ✅ |
| ref_enc | ge[512] | 512 | 2.168e-04 | 6.759e-05 | 9.162e-01 | 1.000000 | ✅ |
| enc_p | m_p[192,206] | 39552 | 5.949e-04 | 6.817e-05 | 4.558e-01 | 1.000000 | ✅ |
| enc_p | logs_p[192,206] | 39552 | 1.292e-04 | 3.152e-05 | 4.668e-01 | 1.000000 | ✅ |
| flow | z[192,206] | 39552 | 1.402e-04 | 1.141e-05 | 1.239e+00 | 1.000000 | ✅ |
| dec | wav[131840] | 131840 | 1.293e-02 | 1.925e-03 | 8.480e-02 | 0.999917 | ✅ |

波形峰值 0.5471（参考 0.5395），131840 采样 = 103 codes × 2 × 640，长度完全一致。
残差全部来自 GGUF 里的 F16 权重（`--f16` 导出）与 F32 累加的差异；用 `--f32` 导出可进一步收敛。

同一设备上的分段耗时（4 线程，4.12 s 音频）：ssl_proj+VQ 6.0 ms、ref_enc 4.6 ms、
enc_p 96.3 ms、flow 45.6 ms、dec 862.1 ms。

## Benchmark

<!-- BENCH_TABLE -->

## 已知差异

- **重采样**：torch 侧（GPT-SoVITS-CPUFast）用 PyAV/FFmpeg 重采样，本工程的 `prep_reference`
  用 resampy kaiser_best 的复刻（与上游 torch 一致）。只有参考音频的重采样这一步受影响，
  对拍时两侧共用同一份 ssl/spec704，所以不影响数值结论。
- **noise_scale 的随机数**：torch 用全局 RNG，本实现用 `gsv_rng(seed)`，逐位不可比；
  对拍时用 fixture 里的 `zp_ref.f32` 固定噪声，端到端听感用 seed 控制。
- **v2Pro 声纹条件**（`sv_emb`）未接，见上。
- **GPU**：dec/flow 用 `ggml_conv_direct_1d_fused`（F32 核）+ 相位交织，CPU 走定制的
  direct 卷积；Vulkan 上依赖 `patches/` 里的 `conv_direct_1d.comp`。
