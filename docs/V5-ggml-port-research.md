# GPT-SoVITS V5（cuda_graph_accel_v5）ggml 推理移植调研报告

> 调研范围：仅 V5 推理链路（`v5dev` / `v5turbo`）。
> 代码基线：`RVC-Boss/GPT-SoVITS@cuda_graph_accel_v5`（a303508；本地 clone 到工作区外的 `./repo`，路径可用环境变量 `GSV_REPO` 指定）。
> **状态更新（2026-10-06 第二轮）**：§6 选型问题已确认（见 §6 决策记录）；G2PW 版本核实与 v5turbo 机制核实结论已并入正文；v3/v4 condition 缓存 / DiffSinger 对照 / cache-dit 调研见 §3.5。
>
> **状态更新（2026-10-09）**：DiT/CFM 段已实现并对拍通过（[dit_ggml.md](dit_ggml.md)）；cache-dit 已按 §3.5.3 落地为可选特性——**32 步档有效（CPU −30~35%、Vulkan e2e −3~7%），4 步 turbo 无效**（v5turbo 确认为 **DMD（Distribution Matching Distillation）** 训练的 shortcut 模型，开发者证实；命中即劣化，与 §3.5.4 判断一致）。量化最小档 = 全 F16 651 MiB（Q8/Q6 FAIL）。

---

## 0. 一句话结论

V5 = **AR Transformer（类 llama）+ F5 式 DiT（CFM 流匹配）+ SoVITS 残余编码器 + HuBERT/BERT 前端 + HiFi-GAN vocoder** 的五段式链路。
DiT 部分与 F5-TTS v1 Base 完全同构（dim=1024/depth=22/heads=16/ff_mult=2/text_dim=512/conv_layers=4/100 mel），audio.cpp 与 CrispASR 各有一份已验证的 ggml 实现，是主要蓝本；AR 部分照 llama.cpp 抽象即可；HuBERT/BERT 有现成参考但各有必须补的缺口；你自己的 ggml-audio-patch 中 Supertonic 五件套恰好覆盖 ConvNeXt 块、pc-nsf-hifigan 系列覆盖 vocoder 卷积。
**重要简化**：V5（v3v4set，`use_vocoder=True`）**不支持流式**（TTS.py 直接 raise），不需要 SOLA/overlap 逻辑；v2Pro 的 SV（ERes2NetV2）模型 V5 不用；BigVGAN/AP_BWE 超分只有 v3 用。

---

## 1. V5 推理链路全景

### 1.1 数据流（非流式，单句）

```
文本
 ├─ 文本前端（纯 CPU 规则 + 中文 G2PW ONNX 模型）
 │   LangSegmenter → zh/en/ja 归一化 → clean_text → phones[] + word2ph[]
 │   （zh 额外走 G2PW 多音字模型 = 一个 ONNX BERT）
 │
 ├─ BERT 特征：chinese-roberta-wwm-ext-large
 │   token ids → 24 层 BERT-large → hidden_states[-3]（1024 维）
 │   → 按 word2ph 逐字重复到音素级 → bert_features [1024, n_phones]
 │
 └─ 参考音频（每个 ref 一次，缓存）
     resample→16k → CNHuBERT(chinese-hubert-base) → last_hidden_state [768,T]
     → extract_latent: ssl_proj(Conv1d 768→768, stride 2) → RVQ decode(1024 bins 单层)
     → prompt_semantic codes [1,1,T/2]            （AR 的 prompt）
     refer_spec = mel(ref@32k, n_fft 1280, hop 320, 100 mel)（SoVITS 侧条件）
     ref_audio@32k → mel_fn_v4 → norm_spec → reference mel [100,≤500]（DiT 的 ref mel）

推理：
 1) AR（Text2SemanticDecoder）
    输入: phones embedding + bert_proj(bert_features) + sine-PE(α)
          + prompt semantic codes embedding
    自回归生成 semantic tokens（vocab 1025, EOS=1024, ≤1500 步，最少 10 token）
    每步: fused-QKV 12~24 层 post-LN transformer → Linear(512→1025)
          → repetition_penalty→top_k→top_p→temperature→multinomial
 2) decode_encp（SoVITS 条件特征）
    codes → RVQ decode → ×2 nearest 插值 → enc_p(TextEncoder: 6 层
    rel-pos-attn Encoder + MRTE cross-attn(text, ge) + encoder2)
    → bridge(Conv1d 192→512 + LeakyReLU) → ×2 nearest 插值 → wns1(WN×8, k=5, gin=512)
    → fea [512, T_feat]（DiT 的 condition）
    其中 ge = MelStyleEncoder(refer mel[:, :704]) 全局风格向量 [512,1]
 3) CFM/DiT（synthesize_v5_mel + CFMV5.inference）
    分块: ref mel 500 帧 + 目标 chunk ≤640 帧（总 ≤1000）；后续块用
    ref 前 468 帧 + 上一块生成结果尾部 32 帧做 rolling prompt
    x = randn×0.875（temperature），ref 段置 0
    32 个 Euler 步（v5turbo: 4 步）：v = DiT(x, prompt_x, t)；
    cfg=1.30（v5turbo: 0.0）时 v += cfg·(v − v_null)，v_null 为 drop_audio_cond 分支
    x += step·v；prompt 段反复置 0
    static cache：text_embed（4×ConvNeXtV2 后）与 proj(cat(cond,text)) 一次性预计算，
    每步只更新 x 和 t
 4) vocoder
    denorm_spec → HiFi-GAN Generator（conv_pre → 5×ConvTranspose1d
    [10,6,2,2,2]=480 → MRF 3×ResBlock1(k=3/7/11, d=1/3/5) → conv_post → tanh）
    → 48 kHz 波形 → 峰值归一 → int16
```

### 1.2 各段张量/频率关键数

| 项 | 值 | 出处 |
|---|---|---|
| AR vocab | 1025（1024 语义 + EOS），phones 词表 522（v2 symbols） | `configs/s1longer-v2.yaml`, `symbols2.py` |
| AR 结构 | 由 ckpt config 驱动；v2 默认 d=512, 16 heads, FFN 2048, 24 层；v1 big d=1024；`default_config` 12 层 8 头 | `AR/models/t2s_model.py` |
| mel（DiT/vocoder 域） | 32 kHz, n_fft 1280, hop 320, 100 mel → 100 fps；`norm_spec`: (x−(−12))/14×2−1 | `TTS.py:159` `mel_fn_v4`, `norm_spec` |
| DiT | dim 1024, depth 22, heads 16 (hd 64), ff_mult 2, text_dim 512, conv_layers 4, mel 100；**v5 无 step embedding（d_embed=None），无 long_skip** | `models.py:1304` |
| CFM | temperature 0.875, steps 32/4, cfg 1.30/0.0, REF 500 帧, CHUNK ≤640, TOTAL 1000, rolling tail 32 帧 | `models_v5.py` |
| vocoder | upsample [10,6,2,2,2] = 480（48 kHz 输出），512 ch，ResBlock1×3，bias=True | `TTS.py:735-759` |
| HuBERT | 16 kHz，Wav2Vec2FeatureExtractor 默认归一化（do_normalize=True 零均值单位方差），输出 768 维 | `feature_extractor/cnhubert.py` |
| BERT | 取 `hidden_states[-3:-2]`（24 层中的第 22 层输出），1024 维 | `TextPreprocessor.get_bert_feature` |

---

## 2. 分模块移植要点（结构与 ggml 落点）

### 2.1 DiT（CFM estimator）— 蓝本最充分

结构逐算子核对（`f5_tts/model/backbones/dit.py` + `f5_tts/model/modules.py`）：

| torch 组件 | 结构 | ggml 方案 |
|---|---|---|
| InputEmbedding.proj | Linear(712→1024)，cat(x, cond, text_embed) | `ggml_mul_mat`；static cache 把 cond·W_tail 项预计算成常量 |
| ConvPositionEmbedding | grouped Conv1d k=31, g=16 + Mish×2 + 残差（带 mask） | audio.cpp 手写 lower（slice→im2col→per-group mul_mat→concat）可直接抄；或用 `ggml_conv_1d_dw`/自研 direct conv |
| TextEmbedding ConvNeXtV2×4 | dwconv k=7 → LN(1e-6) → pw1 → GELU(erf) → **GRN** → pw2 + 残差 | **audio.cpp 已逐算子实现（cos 1.0）**；你的 patch 二 `SUPERTONIC_DEPTHWISE_1D`/`LAYER_NORM_CHANNEL`/`BIAS_GELU` 恰好对应 dwconv/LN/GELU 融合；GRN 用 ReduceSum(时间)→sqrt→ReduceMean(通道)→div 组合 |
| AdaLayerNormZero | LN(无仿射)·(1+scale)+shift，6 路调制切分 | CrispASR `core/adaln.h` modulate6/gated_residual 直接复用；ones 行广播加 |
| Attention | Q/K/V 三个独立 Linear（非 fused）→ reshape 16×64 → RoPE(θ=10000) → **非因果全序列 SDPA**（mask 为 padding mask，batch=1 时全 1 可置 null） | `ggml_flash_attn_ext`；**必须先 `ggml_cont` 物化**（audio.cpp 两次血泪回归）；无 KV cache，每步全序列重算 |
| TimestepEmbedding | host 算 256 维 sinusoid → Linear(256→1024)→SiLU→Linear | sinusoid 留 host（每步一个 [1,256] 输入叶），MLP 进图 |
| norm_out/proj_out | AdaLN-Zero-Final + Linear(1024→100) | 同 AdaLN 方案 |
| long_skip | **V5 未启用**（构造不传） | 无需实现 |
| step embedding (d_embed) | **V5 关闭**（use_step_embedding=False） | 无需实现；比 F5 v1 还少一条分支 |

CFM 采样循环全部留 host（audio.cpp `synthesize_chunk` 同款）：timestep 表、Euler 积分、cfg 组合、`x[..., :prompt_len]=0` 置零。`prepare_static_cache` 语义与 audio.cpp 的"单次 build 图 + 每步只换输入叶"完全对齐——**V5 的 static cache 就是为此设计的**，ggml 版每步只需更新 `x` 与 `t` 两个输入。CFG torch 实现是两次 forward（null 分支 drop_audio_cond）；audio.cpp 用 B=2 批图，CrispASR 实测 batch-CFG 无收益，建议跟随 torch 两次 forward（v5turbo cfg=0 直接跳过 null 分支，正好 4 步纯推）。

### 2.2 AR（Text2SemanticDecoder）— llama.cpp 模板

- 12~24 层 **post-LN** transformer（`norm_first=False` 推理路径），ReLU FFN，无 dropout；T2SBlock 已把 QKV 融合成单 `qkv_w`（in_proj_weight），**融合 QKV 是现成的**。
- 位置编码：**learnable sine PE + 可学习标量 α**（`pe[:, pos]·α + x`），不是 RoPE——KV 内 K 不旋转，比 llama.cpp 还简单；进图前 `ggml_get_rows` 查 [T_max,512] 表加 α 即可。
- 首步 `process_prompt`：x 段全可见 + y 段 causal + padding mask；之后每步单 token 无 mask。
- 采样链：repetition_penalty（对已有 token 的 logit 做 `x<0 ? x·r : x/r`，语义与 llama.cpp `init_penalties` 一致）→ top_k → top_p → temperature → multinomial；业务规则（最少 10 token、EOS=1024、early_stop=hz×max_sec、1500 上限、流式 mute-token 切分）留 host。
- KV cache：单路 MHA、单序列，llama.cpp `llama_kv_cache` 大幅简化即可；V5 的 v2Pro 线多 batch 并行场景 ggml 首版可先做 bs=1。
- Accel/CUDA-Graph 那套（`GPT_SoVITS/Accel`）是 PyTorch 侧加速，ggml 版天然取代，不需要移植。

### 2.3 BERT（chinese-roberta-wwm-ext-large）

- 就是 BERT-large（24 层，hidden 1024, 16 heads, FFN 4096, eps 1e-12），post-LN。
- bert.cpp encoder 图可近乎直接用，改动三处：① 输出改为第 21 层（index，`hidden_states[-3]`）的逐 token hidden states（去掉 mean-pool 尾巴）；② LayerNorm eps 参数化（bert.cpp 旧 ggml 硬编码 1e-5）；③ **tokenizer 弃用**（bert.cpp 正则会丢 CJK）——分词与 word2ph 对齐留在 Python/前端层，C++ 只吃 token ids 输出 [N,1024]。
- 若要 GPU/GGUF：llama.cpp PR #5423 合并版或 iamlemec fork；或把 encoder 图并入自家工程（推荐，和其他模块统一）。

### 2.4 HuBERT（chinese-hubert-base）

- wav2vec2-base 架构：7 层 CNN（k10/s5 + 6×k3/s2，**GroupNorm（feat_extract_norm="group"）**+ GELU）→ feature proj 768 → **pos_conv（groups=16, k=128, same-pad, weight_norm）** → 12 层 **post-norm**（do_stable_layer_norm=False）transformer → `encoder.layer_norm` → 取 last_hidden_state。
- 输入还要 HF 风格 z-score 归一化（do_normalize=True）。
- 参考适配度：**py-ai-dev/wav2vec2.cpp 契合度最高**（GroupNorm 正确、post-norm 分支、weight_norm 处理、z-score、GGUF、83 个测试），缺口三处都是小改：lm_head 改可选、转换脚本适配 HuBERT 键名（无 `wav2vec2.` 前缀）、暴露 encode() 输出。engineerchuan 版价值在全 ggml 图 + GPU 通路，但需补 GroupNorm/post-norm/z-score/pos_conv 权重重排（其手动合并 weight_norm 的轴有 bug，应直接取参数化后的 `conv.weight`）。rvc.cpp 的 contentvec 实现证明"host 手写 CNN 前端 + ggml 12 层 encoder"的混合路线也成立且快（同为 wav2vec2 CNN，且 rvc.cpp 与 GPT-SoVITS 同生态）。
- 下游 `extract_latent`：ssl_proj（Conv1d 768→768 stride 2）+ 单层 RVQ decode（`ggml_get_rows` 查 1024×768 码本 + `F.interpolate ×2` nearest）—— trivial。

### 2.5 SoVITS 残余（SynthesizerTrnV3 非 DiT 部分）

全部是中小算子，无外部依赖：

- **enc_p/TextEncoder**：ssl_proj(1×1 conv) → 6 层 Encoder（`attentions.MultiHeadAttention` **window_size=4 的相对位置注意力**（emb_rel_k/v 偏置表 + pad/cut 位移），注意这不是标准 MHA）→ MRTE（cross-attn(ssl_enc, text_enc) + ssl_enc + ge，4 头）→ encoder2（3 层）→ proj。encoder 内 FFN 是 **k=5 的 Conv1d FFN**（非逐位置 MLP，带 same padding）。
- **ref_enc/MelStyleEncoder**：Linear+Mish×2 → 2 层 Conv1dGLU（门控卷积）→ 2 头 self-attn → fc → masked temporal mean pool → ge [512,1]。
- **wns1/Encoder(WN×8)**：WaveNet 残差块，k=5 dilation=1，`fused_add_tanh_sigmoid_multiply`（gin=512 条件注入），res-skip 结构。**难度再评估（2026-10-06）**：其实这是链路里被高估的"难点"——它的全部构件（k5 膨胀卷积、1×1 投影、tanh/sigmoid 门控乘加、残差累加、LeakyReLU 条件）都是 ggml 现成算子或 audio-patch 现成算子（`CONV_DIRECT_1D(_fused)` 恰好是 k≥3 直接卷积 + 融合，WN 每层正是"conv5 → 门控切分 → res_skip"结构）；权重无 weight_norm 转置陷阱（转换时物化折叠）；计算量 8 层 × 512 通道 × T=1000 ≈ **24.9 GMAC，全链路只跑一次**（对比 DiT 单步 184 GMAC × 4~32 步）。rvc.cpp 的 WN 实现确实效率差（已停止维护、host 手写），但它的问题是把 conv 做成逐 tap 散射——照 audio-patch 的 im2col/mul_mat 或 direct-conv 路线就是常规工作，**属于"工作量中等"而非"生态盲区"**。若最终决定走 ONNX（与 G2PW/vocoder 同栈），`export_torch_script_v3v4.py` 的 `models_onnx.py::SynthesizerTrnV3.forward` 已把 bridge→wns1→fea 路径做成可导出的整体（v3/v4 时期已 trace 过），"张量倒手"（ONNX 输入输出与 ggml 内存互拷）可通过 ONNX Runtime 的 IOBinding/内存共享把 ge/fea 的 device 指针直接喂给 session 来规避，但要接受跨 runtime 的 dtype/布局对齐成本。
- **bridge**：1×1 conv 192→512 + LeakyReLU；两次 `F.interpolate` nearest ×2。
- 移植要点：相对位置注意力的 `_get_relative_embeddings`（pad + slice + matmul-with-relative + shift）需要图内组合（pad→im2col 式点积→rel_shift 风格 view，CrispASR fastconformer.h 的 rel_shift 无拷贝 view 可参考）；Conv1dGLU/门控乘加都是基础算子；权重全部 weight_norm，**转换时物化折叠**（rvc.cpp `materialize_weight_norm` / CrispASR hifigan 转换同款做法）。

### 2.6 vocoder（gsv-v5-pretrained/vocoder.pth，HiFi-GAN）

- 标准 HiFi-GAN Generator：conv_pre → 5×ConvTranspose1d（[10,6,2,2,2]，k=[20,12,4,4,4]）→ 每级 3×ResBlock1（k=3/7/11，d=1/3/5，两轮 conv+LeakyReLU 0.1 残差）→ 求和平均 → LeakyReLU → conv_post → tanh；**is_bias=True**。
- 你在 pc-nsf-hifigan.cpp + ggml-audio-patch 里已经走通整条：`ggml_conv_transpose_1d_ext`（全参数含 groups/output_padding/padding）、`CONV_DIRECT_1D(_fused)`、`ADD_LEAKY_RELU`。**唯一注意**：这里 K=2 的 ups（k=[20,12,4,4,4] 中有两个 k=4，无 k=2；直接卷积门控 `K≥3` 不影响 convT 路径），以及 convT 权重是 PyTorch `[IC,OC,K]` → ggml `[K,OC/g,IC/g]` 布局转换；V5 的 vocoder 无 NSF 激励源（纯 HiFi-GAN，无 sine），比 pc-nsf-hifigan 还少一段。

### 2.7 mel / resample / 其他非模型组件

- **mel_spectrogram_torch**：reflect-pad → STFT（Hann，非中心）→ 幅度谱 → **librosa mel filterbank（默认 Slaney 尺度+Slaney norm，不是 HTK！）** → `ln(clamp(x, min=1e-5))`（`spectral_normalize_torch`）。**这是移植最容易踩的坑**：F5-TTS 的 mel 是 torchaudio HTK，GPT-SoVITS 全家（v3 24k/v4v5 32k）都是 librosa Slaney。audio.cpp/game.cpp 的 mel 都是各自模型口径，V5 要新写一个 Slaney 100-band@32k 前端（CrispASR `core/mel.h` 支持 HTK/Slaney 两种 filterbank，可参考）。
- **resample**：TTS.py 有两条路径——torchaudio Resample（kaiser 窗 sinc 插值）与 resampy kaiser_best（match_librosa=True，ref 音频走这条）。C++ 需复现其中之一；kaiser_best 的滤波器表可预生成。
- RVQ decode、norm/denorm、interpolate、峰值归一：基础算子。
- 文本前端（LangSegmenter、zh_normalization、pypinyin、G2PW、symbols 映射、word2ph）：见 §4 选型问题 5。

---

## 3. 参考仓库调研结论映射

| 仓库 | 对 V5 的价值 | 关键文件 |
|---|---|---|
| **0xShug0/audio.cpp** | **DiT 主蓝本**。F5-TTS v1 ggml 实现已验证（22 块/dim1024/16 头/conv_layers 4，与 V5 同构；ConvNeXt+GRN、grouped-conv CPE、AdaLN、RoPE 全有，cos 1.0 对拍）。V5 比它少 step-embedding 分支 | `src/community_models/f5_tts/dit_modules.cpp`、`runtime.cpp`、`synthesize.cpp`、`tools/community_models/convert_f5_tts.py` |
| **CrispASR** | 第二 F5 参考（`src/f5_tts.cpp`，单次 build + ODE 只换输入）；`core/adaln.h`（AdaLN 六路调制 helper）、`core/hifigan.h`（HiFi-GAN 全套）、`core/mel.h`（HTK/Slaney 双 filterbank）、`gguf_loader`（mmap/分层卸载）；LEARNINGS.md 是精度/后端坑字典（F5 条件通路必须 F32、flash-attn F16 累加 NaN → 手动 SDPA 兜底、常量被 gallocr alias、重复调用测试） | `src/f5_tts.{h,cpp}`、`src/core/{adaln.h,hifigan.h,mel.h,gguf_loader.h}`、`docs/ggml-optimisation-playbook.md` |
| **game.cpp** | 工程骨架参考（GAME 模型非 DiT，但其 EBF/JEBF backbone 实践通用）：PersistentStage 按序列长缓存图、跨图 NONE 型张量传递、QKV 融合进 GGUF、dwconv direct/legacy 双路径 + supports_op 探测、GGUF 量化规则（dwconv→F16/norm/bias→F32）、注入 RNG 逐 bit 对拍；**cache-dit（DBCache）C++ 静态图实现**（三段式子图 + device 侧 L1 门控 + tail delta 残差复用，见 §3.5.3） | `src/{model.cpp,ops_attn.cpp,ops_basic.cpp}`、`src/model_impl.h`（SegmenterCacheState）、`docs/benchmark-7channel.md` |
| **bert.cpp (skeskinen)** | BERT 起子实现：post-LN encoder 图 + 转换脚本；改动点=输出第 21 层 + eps 1e-12 + 弃用其 CJK 不兼容 tokenizer | `bert.cpp`、`models/convert-to-ggml.py` |
| **py-ai-dev/wav2vec2.cpp** | **HuBERT 首选**：GroupNorm + post-norm + z-score + GGUF + weight_norm 正确处理，缺口小（lm_head 可选/键名/输出暴露） | `src/wav2vec2.cpp`、`scripts/convert_to_gguf.py` |
| **engineerchuan/wav2vec2.cpp** | HuBERT 的 GPU/全图参考：conv/encoder/ctc 三图拆分、pos_conv 分组卷积 16 路 view+concat 写法、新 ggml backend 调度 | `src/wav2vec2.cpp` |
| **rvc.cpp** | 同生态验证：contentvec（=wav2vec2 CNN+encoder）host CNN + ggml encoder 混合路线、weight_norm 物化、tap-major conv 权重重排、golden 对拍流水线、分段 CLI 落盘 .f32 调试 | `docs/PROGRESS.md`、`src/contentvec.cpp`、`tools/convert_rvc.py` |
| **llama.cpp** | AR 蓝本：`llm_graph_context`（build_norm/build_ffn/build_attn）、`llama_kv_cache`（可大幅简化）、sampler 链（top_k/top_p/penalties/dist），BERT 合并版（PR #5423） | `src/models/llama.cpp`、`src/llama-graph.*`、`src/llama-kv-cache.*`、`src/llama-sampler.cpp` |
| **DiffSinger (openvpi)** | V3/V4 condition 缓存的思想源头（编译期图手术 `graph_extract_conditioner_projections` 把循环不变量提出 If/Loop；fs2+diffusion 双子图合并单文件）；对 V5 的启示见 §3.5.2——缓存清单判据（"输入只来自 cond 的算子可外提"）、t 标量进图、host 循环即等价物 | `deployment/exporters/acoustic_exporter.py`、`utils/onnx_helper.py`、`deployment/modules/rectified_flow.py` |
| **ggml-audio-patch（你的）** | 算子直接命中：`IM2COL_FAST_1D`（HuBERT CNN/DiT conv）、`conv_transpose_1d_ext`（vocoder 上采样）、`ADD_LEAKY_RELU`+`CONV_DIRECT_1D(_fused)`（vocoder/resblock）、`SUPERTONIC_DEPTHWISE_1D/LAYER_NORM_CHANNEL/BIAS_GELU/EDGE_PAD_1D`（ConvNeXtV2 文本块 + CPE）、`GRU`（本链路暂无 RNN，留作备用）；另附移植套路文档（新算子固定触点清单、Vulkan 十余处清单、Windows 宏坑） | `patches/*.patch`、`docs/operators_zh.md`、`docs/porting_notes_zh.md` |

**三方 DiT 精度结论交叉验证**：audio.cpp 全 F32 + 权重可选 F16（3090 实测 F5 尺寸下 F16 反而慢）；CrispASR F16 权重 + 条件通路 F32（"Errors compound through 32 ODE steps × 22 layers"）、flash-attn 在 sm_60 级 GPU 会 NaN 需手动 SDPA 兜底；game.cpp 全 F32 默认 + Q8 规则表。→ V5 建议首版全 F32 打 parity，再按层选量化。

#### 3.5.4 v5turbo 的 4 步机制核实（2026-10-06 补充，模型包实证）

对 `gsv-v5-pretrained/s2Gv5turbo.pth`（766,406,144 B，HF `gsv-v5-pretrained`）做了头部分析（zip/pickle 结构 + 2 MB 头部键名枚举）：

- **结构上与 v5dev 完全同构**：完整 22 块 `transformer_blocks`、`time_embed` 有而 `d_embed` **不存在**、无 `long_skip`、enc_p 三段 encoder 层数（ssl 3 + text 6 + encoder2 3）与 MRTE/quantizer/ssl_proj 全部在位——**不是更小的 student 网络，也没有 shortcut 分支**。
- ckpt 内嵌 config 与 v4 口径一致（sampling_rate 32000、semantic_frame_rate 25hz 等），`info` 显示由 `final_G_20018400_60_G_207000_40.pth` fp16 化并对齐 v4 keys 而来；训练代码（s2_train*.py）中无任何 turbo/distill 专用路径。
- 结合仓库 `head2version`（b"08"→v5turbo）与 commit（ac846f5/3b95794 均只加推理端支持）判断：**v5turbo = 同一架构的独立训练 checkpoint，靠 shortcut 自蒸馏类训练目标把 32 步 Euler 的教师行为压进少步推理（4 步、cfg=0）**；2026-10-09 经开发者确认具体为 **DMD（Distribution Matching Distillation）**。训练细节未随分支发布（开源训练脚本无对应实现），但推理端没有隐藏分支——ggml 侧 v5turbo 与 v5dev 共用同一张图，只是 `steps=4, cfg=0` 的调度参数不同，且 cfg=0 可整体跳过 null 分支。**结论：它不是"单纯降低推理质量"的后处理，但也与 cache-dit 无关**——cache 是否要做的判据是"32 步档是否作为长尾质量档保留"：若 ggml 版把 v5dev 32 步作为质量档、v5turbo 4 步作为速度档，两者已构成质量-速度阶梯，cache-dit 优先级维持"首版不做、32 步档 parity 后作可选实验"（§3.5.3）。

### 3.5 Condition 缓存 / shortcut 时间步 / cache-dit（第二轮补充调研）

#### 3.5.1 V3/V4 → V5 的缓存演进（本地代码证据链）

**V3/V4 的 CFM（`module/models.py` CFM 类）有两套机制**：

1. **shortcut 风格 step embedding（训练期自蒸馏）**：训练 forward（`CFM.forward`）以 30% 概率采 `d = 1/2^base`（base∈[2,8)），先算 `v_pred_1 = est(xt, t, d)`，推一步得 `x_mid`，再算 `v_pred_2 = est(x_mid, t+d, d)`，目标速度取两者均值，同时把 `dt = 2d` 作为第三个分支的条件——即 shortcut models 的"一次预测多步速度"自蒸馏。推理时 `d = 1/n_timesteps` 作为 `dt_base_bootstrap` 进图（`d_embed` 与 `time_embed` 相加：`t = time_embed(t) + d_embed(dt)`），t 从 0 起步。
2. **conditioner cache（推理期运行时缓存）**：`use_conditioner_cache=True` 时，首步算出的 `text_embed`（ConvNeXt 文本块输出）与 `dt`（step embedding 输出）被缓存（`text_cache`/`dt_cache`/`text_cfg_cache`），后续步直接复用——因为 text_embed 不依赖 t，dt 是常数。cfg 负分支（`cfg_drop_text=True` 时 text 全零）用独立的 `text_cfg_cache`。这套缓存于 2025-07（b921165）进入主干。

**V5 的变化**：`use_step_embedding=False`（`d_embed=None`，shortcut 分支整体移除，只保留时间步 Euler），`prepare_static_cache` 把缓存粒度从"text_embed + dt"扩大为四项：`condition`（input_embed.proj 剔除 mel 段权重的静态投影，即 `proj.weight[:, mel_dim:]` 部分的输出）、`negative_condition`（cfg 负分支同款，cond 置零版）、`mask`（sequence_mask）、`rope`（预计算频率表）。每步动态部分只剩：`F.linear(x, proj.weight[:, :100])` + 加缓存 + 22 块主干 + AdaLN（t 只进 AdaLN 与 time_embed）。`export_torch_script_v3v4.py` 的 `ExportDitEmbed/ExportDiT/ExportCFM` 拆分就是这一思想的 TorchScript 版。

#### 3.5.2 DiffSinger 的对应做法（参照 openvpi/DiffSinger）

DiffSinger 把声学模型拆成 `fs2.onnx`（FastSpeech2 条件编码器，只跑一次）+ `diffusion.onnx`（去噪循环被 TorchScript 导成 ONNX If/Loop 子图，一次 `session.run` 完成全部 N 步），再 `onnx.compose.merge_models` 合并回单文件。**condition 缓存在它那边是编译期图手术**：`utils/onnx_helper.py::graph_extract_conditioner_projections()` 按权重名正则找出循环体内每层 backbone 的 `conditioner_projection`（1×1 Conv，把条件投影到层输入，**不依赖时间步**），搬到顶层 If 之前，循环体内下游改接别名 `/diffusion/backbone/cache.*`，然后 onnxsim 清理。时间步处理：t 是 `(1,)` float32 标量输入，采样器侧乘 `time_scale_factor=1000`（等价 GPT-SoVITS 在 `SinusPositionEmbedding(x, scale=1000)` 里做），**dt 不进图**；推理无 shortcut（最接近的是训练期 `use_dual_timestep` 双时间条件，与 GPT-SoVITS 的 shortcut 自蒸馏是两条不同路线）。

**与 GPT-SoVITS 的对应关系**：GPT-SoVITS 继承的是 DiffSinger 的"条件通路静态化/循环不变量外提 + reflow/Euler 少步采样 + spec 归一化"思想，但实现路线不同——DiffSinger 编译期自动提取（ONNX 图手术），GPT-SoVITS 手工拆类 + 运行时缓存（TorchScript trace 无法自动提取）。`d_embed`/dt 进图、text/dt 运行时缓存、500 帧滚动 chunk 都是 GPT-SoVITS 自己的设计。

**对 V5 ggml 的结论**：V5 的 `prepare_static_cache` 已经达到 DiffSinger 的缓存粒度（还多了 rope/mask）。ggml 建图时按同一判据划静态/动态——"输入只来自 cond/text 的算子"全部提出循环外，做成图内常量（audio.cpp 的 ConstStage 模式，注意放 arena 外私有 buffer）；每步图输入只剩 `x` 和标量 `t`（host 传 float，正弦表+MLP 每步重算成本可忽略）。ggml 无循环原语，**host 端逐步循环 + 静态图复用**就是 DiffSinger If/Loop 的等价物，不必也没有必要做图内循环。

#### 3.5.3 game.cpp 的 cache-dit（DBCache）与 vipshop/cache-dit 对照

game.cpp 实现了 vipshop/cache-dit 的 DBCache（Dual Block Cache）C++/静态图子集：把 block stack 切成 front（Fn 块，每步必算，用于算门控度量）/ middle（命中时整段跳过）/ back（Bn 块，可选，实测劣化默认关）三段，缓存的是 **tail delta 残差**（`x_mid = x_front(当前步) + tail_delta(上次全算时的增量)`），命中判据是 front 输出的归一化 L1 相对差 `fd < threshold`（默认 0.25），外加 warmup、复用窗口（首末步强制全算）、连续命中上限、UCache 式累计误差门（game.cpp 自加）四个闸门。工程上最有价值的是**静态图三段式持久子图（PersistentStage）+ NONE 型 leaf 张量跨图引用（`ggml_cpy` 写入）+ 全 device 侧 fd 判定（每步只回读 1 个 float）**——旧版 host 侧判定在 Vulkan+Q8 上曾回退 +20%。实测：CPU F32 segmenter 时间 −48%，CUDA −19%，质量帧级指标中性。

**搬到 V5 F5 DiT 的适用性判断**：

- **可行但有前提**：cache-dit 对 AdaLN 模型（Flux/Wan）本就默认支持，"块内 t 调制"不是硬障碍（上游对 AdaLN 用残差缓存 + 收紧阈值）。但 V5 与 GAME 有个关键结构差异：GAME 的 t 只注入在 block stack 输入端（tail 各块无 t 依赖），F5 的每块 AdaLN 调制都随 t 变——跳过块时调制参数过期，误差比纯激活漂移更严重。
- **步数经济学是最大风险**：GAME nsteps=8 中砍一半；V5 v5dev 32 步可期（cache-dit 上游经验全来自 20–50 步模型），**v5turbo 4 步档命中率大概率得不偿失**（步间 t 跳变 0.25，fd 容易超阈值，还白付 front 段+判定开销）。
- 若做，建议配置：Fn=8/22 起步（上游 F8B0 惯例）、warmup≈8（32 步档）、阈值 0.08–0.12（比 GAME 的 0.25 严）、累计误差门默认开、门控把 AdaLN 调制参数差并进去（或命中步仍重算廉价的 AdaLN 线性，只缓存 attention+FFN 主体）；v5turbo 不启用。
- **优先级建议**：这是三档优化里投入产出比最低的一档——static cache（已在 V5 设计里）> 4 步蒸馏（v5turbo 模型本身）> cache-dit 步间缓存。ggml 首版不做，32 步档打完 parity 后作为可选实验特性，且评估必须用帧级指标（mel L1、说话人相似度）而非听感。

**三档机制总览**（V5 CFM 循环能省什么）：

| 机制 | 缓存/跳过内容 | V5 状态 | ggml 落点 |
|---|---|---|---|
| 静态条件缓存（V5 已有 ≈ DiffSinger 图手术） | text_embed 投影、cfg 负分支、rope、mask——循环不变量 | V5 `prepare_static_cache`，每步动态只剩 x 投影 + 主干 + AdaLN | 图内常量（ConstStage，arena 外），每步图输入仅 x/t |
| 步间缓存（cache-dit DBCache） | 相邻步相似时跳过 middle 块 | torch 版未实现（无此代码路径）；上游对 AdaLN 模型支持 | 可选实验：三段式子图 + device 侧 fd 门控；仅 32 步档 |
| shortcut 少步（v5turbo） | 模型本身蒸馏到 4 步 | V5 已移除 shortcut 分支，v5turbo 直接 4 步纯推 | 无需实现 d_embed/dt；比 V3/V4 导出还少一条分支 |

---

## 4. 链路上完整模型/组件清单（供你判断选型）

**必选（V5 推理硬依赖）**

| # | 模型/组件 | 角色 | 规模 | 参考实现 |
|---|---|---|---|---|
| 1 | G2PW 多音字模型 | 中文 g2p（ONNX BERT-base 级 + 词典） | ~110M（仅中文需要） | **已定案：保持 ONNX，不移植 ggml**（见 §6-Q5）。版本核实见下注 |
| 2 | chinese-roberta-wwm-ext-large | 文本 BERT 特征（−3 层） | 326M | bert.cpp / llama.cpp#5423 |
| 3 | chinese-hubert-base | ref 音频语义特征 + prompt codes | 95M | py-ai-dev/wav2vec2.cpp（+rvc.cpp 路线） |
| 4 | Text2SemanticDecoder (AR) | 语义 token 自回归 | 13~60M（ckpt 决定） | llama.cpp |
| 5 | SynthesizerTrnV3 条件侧（enc_p/MRTE/wns1/bridge/ref_enc/RVQ） | 文本+风格→DiT condition | ~40M | 自研（无现成；算子简单） |
| 6 | DiT (F5 式 CFM) | mel 生成 | ~260M | audio.cpp / CrispASR |
| 7 | HiFi-GAN vocoder 48k | mel→波形 | ~55M | pc-nsf-hifigan.cpp + 你的 patch / CrispASR hifigan.h |

**明确不需要（V5 排除）**

| 组件 | 排除原因 |
|---|---|
| BigVGAN v2 24k | 仅 v3 模型使用 |
| AP_BWE 48k 超分 | 仅 v3 音频后处理；v5 直出 48k |
| ERes2NetV2 SV | 仅 v2Pro/ProPlus；V5 风格向量走 MelStyleEncoder |
| whisper encoder（feature_extractor/whisper_enc.py） | 备用 content extractor，默认链路不加载 |
| SOLA/流式拼接、decode_streaming | **V5 不支持流式**（use_vocoder 分支直接 raise） |
| UVR5 / ASR / 训练侧模型 | 非推理链路 |

> **G2PW 版本核实注**（2026-10-06 实测）：GPT-SoVITS 的 `text/g2pw/onnx_api.py`（`model_version="1.1"`）从 ModelScope `kamiorinn/g2pw` 拉 **`G2PWModel_1.1.zip`**（PaddleSpeech 官方镜像 `paddlespeech.cdn.bcebos.com/.../g2p/new/G2PWModel_1.1.zip`，md5 `f8b605...`，635 MB 的 `g2pW.onnx`）；mozillazg/pypinyin-g2pW README 用的是 GitYCC/g2pW 发布的 **`G2PWModel-v2-onnx.zip`**（esun-ai storage，635 MB 的 `g2pw.onnx`）。两个 zip 实测比对：**图结构完全一致**（ir 6 / opset 12 / 1205 节点 / 318 个 initializer，输入输出签名相同），initializer 逐个 md5 比对仅 `bert.embeddings.word_embeddings.weight` 一个张量数值不同，且 max|Δ|≈1.9e-6（float32 词嵌入的舍入噪声级，非重新训练）。zip 内 `version` 文件分别为 `v2.0`（PaddleSpeech 1.1 包）与 `v3.0`（esun-ai v2-onnx 包，命名口径不同）。**结论：GPT-SoVITS 用的 G2PWModel_1.1 与 pypinyin-g2pW 的 v2-onnx 是同一版 v2 权重的两次打包，二者可互换使用；G2PW 权重只有 ONNX 格式发布**（g2pW 仓库另发过 torch 版 `G2PWModel-v2.zip`，但 pypinyin-g2pW/GPT-SoVITS/PaddleSpeech 三家的推理代码全部只走 ONNX 路径）。

**非神经网络但需复现**

mel 谱（Slaney 100@32k + 24k@v3 口径）、resampy kaiser_best / torchaudio 重采样、G2PW 之外的文本规范化与 phone 映射（symbols2 表）、word2ph 对齐、AR 采样规则、CFM 分块/rolling prompt 调度。

---

## 5. 高风险坑清单（移植前先立 parity 基线）

1. **mel filterbank 是 Slaney 不是 HTK**；`norm_spec` 线性映射 (−12→+2) 前有 `ln(clamp(x,1e-5))`。
2. AR 是 **post-LN + ReLU + 融合 QKV + α-sine PE**，别套 RoPE/pre-LN 模板；BERT 是 post-LN eps 1e-12；HuBERT 是 **GroupNorm + post-norm**；三者三种 norm 排布，建议每段独立 golden 对拍（rvc.cpp/CrispASR 的 per-stage cosine harness 模式）。
3. TextEncoder 的注意力带 **window_size=4 相对位置偏置**、FFN 是 **k=5 卷积**——不是普通 transformer，照 `attentions.py` 逐步实现。
4. DiT：RoPE/flash-attn 输入必须 `ggml_cont`；图内常量（PE 表/ones 行/positions）必须放 arena 外私有 buffer；叶节点必须 `ggml_set_input/output`（audio.cpp 三条血泪注释）；gallocr 每 compute 可能 alias 输入槽（CrispASR：pos_in 每步必须重传）。
5. 全链 weight_norm（wns1/MelStyleEncoder/vocoder/pos_conv）在**转换时物化折叠**。
6. ggml 卷积权重布局 `[K, IC, OC]` 与 PyTorch 相反；convT 是 `[K, OC/g, IC/g]`。
7. flash-attn 的 F16 累加在部分后端不可控，DiT 需保留手动 SDPA 路径。
8. BERT 只吃 token id（CJK 分词留前端）；HuBERT 输入要 z-score。

---

## 6. 决策记录（2026-10-06）

| # | 议题 | 决策 | 落地含义 |
|---|---|---|---|
| 1 | ggml 基线 | **llama.cpp 的 ggml 为基线**（AR 是主旋律），之后把 ggml-audio-patch 的算子向其移植 | bert.cpp 已合并进 llama.cpp，BERT encoder 图现成；audio-patch 目前基于 v0.19.0，需按 llama.cpp 的枚举/分发结构重做适配（patch 套路文档 §porting-notes 适用） |
| 2 | 权重格式 | **统一 GGUF**；量化规则参考 game.cpp，**"最小验证版 + 全量版"双档**发布 | 最小验证版：全 F32/F16 无量化（parity 锚点）；全量版：按 game.cpp pattern 规则（dwconv→F16、norm/bias→F32、matmul 类 Q8_0 起步） |
| 3 | 精度 | 锚点对齐 torch 端（is_half → fp16 路线）；**vocoder 必须保持原版精度、禁止量化**；**vocoder 推荐走 ONNX Runtime**（导出参考 DiffSinger：backbone trace + 整体导出）；其余模块后期做 game.cpp 式逐层量化（不敏感层降精度） | vocoder 是波形末端，Q8 的量化误差直接听得出（CrispASR/ggml-audio-patch 双方实测口径一致）；ONNX vocoder 与 ggml 主体之间注意"张量倒手"（mel 张量 → ORT session → waveform），用 IOBinding/共享内存规避拷贝 |
| 4 | CFG 实现 | 跟随 torch 原版：两次 forward（cond + drop_audio_cond null 分支），不做 B=2 合图 | v5turbo cfg=0 时跳过 null 分支即纯 4 步；v5dev 32 步 × 2 forward |
| 5 | 文本前端 | **只有 ONNX 模型的部分继续 ONNX**（G2PW 已核实只有 ONNX，§4 注）；有其他格式/无模型的组件走正常移植 | 分词/pypinyin/规范化的归属随 Q5 判断：G2PW 留 ONNX 后，BERT encoder 建议 ggml（llama.cpp 内置），引擎 API 收 phones + bert token ids |
| 6 | HuBERT | **参考 rvc.cpp 的路线**（host 手写 CNN 前端 + ggml encoder），但**不参考其实现质量**（rvc.cpp 已停止维护、conv 逐 tap 散射效率差） | CNN 前端可用 audio-patch 的 `IM2COL_FAST_1D`/direct conv 重写，encoder 部分套 llama.cpp 图写法；py-ai-dev 版仍可作 GroupNorm/post-norm 语义对照 |
| 7 | 后端优先级 | **产品优先级 Metal→CUDA→Vulkan→CPU；自研实现优先级 Vulkan→CPU→CUDA**；Metal 写实现不测（无苹果设备），按 audio-patch 早期 commit 的测试任务单模式，交有条件用户测试并以 PR 回流 | Vulkan 是第一个自研 GPU 后端：pipeline cache（patch 五）+ conv direct（patch 四）直接复用；CUDA 用 llama.cpp 现成路径起步 |
| 8 | AR batch | **做 batch 能力**，并评估 llama.cpp 其他推理加速（continuous batching / ubatch 等） | llama.cpp 的 batch 维度在 ubatch 层，V5 的多句并行场景（webui batch=20）可借；KV cache 按序列分槽 |
| 9 | 工程形态 | 后期做**轻巧但全功能的推理端**（单库引擎 + C API 形态），首版仍按 rvc.cpp 式分段对拍开发 | 分段 CLI 只作为开发脚手架，交付物是引擎库；API 形状收 phones + bert ids（若前端留 Python） |
| 10 | cache-dit | v5turbo 的 4 步是**独立训练的 shortcut 蒸馏模型**（§3.5.4 核实），非质量阉割；v5dev 32 步作为质量档保留，**cache-dit 首版不做**，作为 32 步档的可选后续实验 | 若做，按 §3.5.3 配置：Fn≈8/22、阈值 0.08–0.12、warmup≈8、累计误差门开、帧级指标消融 |

### 6.1 决策后的遗留判断点（不阻塞开工，随开发确认）

1. **wns1/WN 走 ggml 还是 ONNX**：难度再评估后（§2.5）倾向 ggml 原生（算子全现成、只跑一次 24.9 GMAC、无跨 runtime 倒手）；若工程上想省事与 vocoder 同走 ONNX 也可，代价是 ge/fea 两次跨 runtime 传张量。建议先 ggml，卡住再降级 ONNX。
2. **BERT encoder 栈**：建议 ggml（llama.cpp 内置图现成），G2PW 独占 ONNX Runtime；若想简化依赖可让两者都走 ORT，代价是引入 ORT 运行时依赖到引擎主路径。
3. **"张量倒手"边界的最小化**：若 vocoder/（可能的）wns1 走 ONNX，把 ONNX 边界尽量外推到"整段 mel → 整段波形"一级（DiffSinger 式整体导出），不要在帧级循环内跨 runtime。

---

## 7. 建议的实施顺序（按 §6 决策推导）

1. **骨架**：内嵌 llama.cpp ggml，移植 audio-patch 算子（按 porting-notes 套路，Vulkan 先行）；GGUF 转换脚本 + "最小验证版（全 F32）"权重包。
2. **第一段 AR**：`Text2SemanticDecoder` → llama.cpp 图（post-LN/ReLU/QKV 融合/α-sine PE/KV cache/采样链），对拍 ckpt logits golden；batch 通路留接口。
3. **第二段条件编码**：HuBERT（host CNN + ggml encoder，audio-patch conv 核重写 rvc.cpp 路线）→ RVQ decode → enc_p/MRTE/ref_enc/bridge/wns1（全部原生 ggml）→ fea。
4. **第三段 DiT**：audio.cpp dit_modules 移植 + V5 static cache（ConstStage 常量）+ host 端 CFM 循环；先 v5turbo（4 步 cfg=0）再 v5dev（32 步 cfg=1.30 双 forward）。
5. **vocoder ONNX**：HiFi-GAN 按 DiffSinger 方式整体导出（fp32 无量化），IOBinding 对接；mel 前端（Slaney 100@32k）+ resampy kaiser_best 复现。
6. **全量量化版**：game.cpp pattern 规则逐层量化（vocoder 除外），帧级指标消融（mel L1/说话人相似度/WER）。
7. **Metal**：实现 + 写测试任务单，交社区用户按 audio-patch 早期模式验证并 PR 回流。
8. **后续可选**：cache-dit（仅 32 步档）、AR continuous batching、G2PW→ggml BERT 复刻（若要去 ORT 依赖）。
