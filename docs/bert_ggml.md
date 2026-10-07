# BERT 前端（chinese-roberta-wwm-ext-large）ggml 实现与精度/量化结论

日期：2026-10-07 · 机器：Xeon E5-2675 v3（16C/32T）· RTX 2070 · Vulkan（`GGML_VK_DISABLE_COOPMAT2=1`）· torch 2.12.1+cu130

## 0. 结论（先给答案）

| 项 | 结果 |
|---|---|
| 模块 | `src/gsv_bert.{h,cpp}`：22 层 post-LN encoder（**只跑 0..21 层**，等价 `hidden_states[-3]`），F32 计算 + `flash_attn_ext` |
| 精度（CPU，vs HF fp32） | 4 文本最终特征 max\|Δ\| = **1.0e-5**，cos = 1.00000000；逐层 23 个隐藏状态全部对齐（max 9.5e-6） |
| 精度（Vulkan，vs HF fp32） | max\|Δ\| = 1.2e-2 / 均值 4.0e-4（cos 0.9999999）——见 §4 的容限论证，**在产品指标上安全** |
| 速度（T=25） | ggml CPU **108~116 ms（16 线程，已含融合算子）** / ggml Vulkan-f16 **8.5 ms** / Vulkan-f32 10.6 ms；torch CPU-fp32 169 ms / torch GPU-fp32 49 ms / torch GPU-fp16 50 ms（**比 torch 最快路径快 ~5.9×**） |
| 最小近无损精度 | **F16：662 MB**（1.9×↓）——CPU 均值 2.7e-4 / Vulkan 2.0e-3 |
| ✗ 拒绝 | **Q8_0 及以下**（均值 ≥1.2e-2，max 达 0.17~1.0）：BERT 对隐藏状态级误差远比 AR 敏感（§5） |
| Vocoder 无关 | BERT 只影响 semantic token，与声码器无耦合；不影响"声码器严格不量化"的决策 |

体积/精度总表（对 HF fp32 golden，4 文本取最差）：

| 变体 | 体积 | CPU max / mean | Vulkan max / mean | 判定 |
|---|---|---|---|---|
| f32（参考） | 1238 MB | 1.0e-5 / 5.0e-7 | 1.2e-2 / 4.0e-4 | 基准 |
| **f16（推荐）** | **662 MB** | 4.1e-3 / 2.7e-4 | 4.1e-2 / 2.0e-3 | ✅ 两边都在 token 稳定区内 |
| q8_0 attn+ffn（emb f16） | 392 MB | 1.7e-1 / 1.2e-2 | 1.0e-1 / 7.1e-3 | ✗ 已越过分歧阈值 |
| q6_k attn+ffn（emb f16） | 280 MB | 4.3e-1 / 3.1e-2 | 2.2e-1 / 2.0e-2 | ✗ |
| q5_k attn+ffn（emb f16） | 242 MB | 1.0 / 4.5e-2 | 5.8e-1 / 3.7e-2 | ✗ |

## 1. 模型与切层依据

- 模型：`models/chinese-roberta-wwm-ext-large`（HF `BertForMaskedLM`，24 层 post-LN，d=1024，16 头，FFN 4096，eps=1e-12，gelu(erf)，vocab 21128，max_pos 512）。
- 管线只要 `res["hidden_states"][-3]`（`TTS_infer_pack/TextPreprocessor.py::get_bert_feature`）：`hidden_states` 共 25 项（0=embedding 输出，k=第 k−1 层输出），故 `[-3]` = **第 21 层输出**，20/22/23 层的结果根本用不到 → **只加载/只计算 0..21 层**（`bert.layers_used = n_layer-2 = 22`，省 2/24 算力与权重）。
- post-processing 在 C++ 侧一并提供：`encode()` 给原始 `[hidden, T]`（= `hidden_states[-3]`），`encode_feat()` 给管线特征 `[hidden, T-2]`（去 `[CLS]/[SEP]` + 转置）。
- **分词仍留在 Python/ONNX 侧**（决策 #5：文本前端保持 ONNX）：C++ 只吃 token ids。`word2ph` 的按字重复展开属文本前端，不在本模块。

## 2. 实现要点（`src/gsv_bert.cpp`）

图（单次全序列，无 KV cache，T ≤ 512）：

```
x = word_emb(ids) + pos_emb(pos 0..T-1) + type_emb(0)
x = LN(x, emb_ln)                                   # eps=1e-12
for li in 0..21:
    qkv = W_qkv x + b_qkv                           # 一次 matmul: [3D,T]，q/k/v 行拼接（转换时融合）
    qh,kh,vh = view(qkv, HD,T,NH,1, nb1=12D, nb2=4*HD)   # 直接构造 flash 布局视图, 不 permute/不 cont
    o = flash_attn_ext(qh, kh, vh, mask, 1/sqrt(HD), 0, 0)      # mask = 全 0 F16 [T,T,1,1]
    o = reshape_2d(reshape_4d(o, D,T,1,1), D, T)                # (HD,NH) 折叠 = torch 的 head 拼接顺序
    x = LN(x + (W_o o + b_o), attn_ln)              # post-LN ①
    x = LN(x + (W_2 gelu_erf(W_1 x + b_1) + b_2), out_ln)       # post-LN ②
```

踩过的坑（都已修，写在这里省下一次调试）：

1. **无 mask 分支**：Vulkan 的 flash attention 内核按"有 mask"编译（`mask != nullptr` 参与 pipeline 选择），故统一传**全 0 的 F16 mask**（等价无 mask，CPU/Vulkan 都走已验证路径）。注意 mask 只在 `n_layers>0` 参与图，`n_layers=0` 时若仍 `tensor_set` 会触发 `GGML_ASSERT(buf != NULL)`。
2. **`get_rows` 类型**：F16 词表查表返回 F16，需 `ggml_cast` 到 F32 再相加（否则 F32 计算链被截断为 F16，或在 `ggml_add` 上类型不匹配）。
3. **逐层导出不能靠 `ggml_set_output`**：图根以外的中间张量即使打了 OUTPUT 标记，`ggml_backend_tensor_get` 仍可能读到被复用/覆盖的 arena 数据（实测只有最后一个节点正确）。逐层调试改成**按层数重建图**（`run(..., n_layers=k)`，语义与 HF `hidden_states[k]` 严格一致），见 `encode_layers()`。
4. `INTER` 是 Windows 头文件里的宏，别用作成员名（改用 `FFD`）。
5. 批处理脚本必须**纯 ASCII**：UTF-8 中文注释的 `.bat`（LF 行尾 + 936 代码页）会被 cmd 解析错位。

## 3. 验收（`tests/test_bert_ggml.cpp` + `tools/dump_golden_bert.py`）

```bash
python tools/dump_golden_bert.py        # 生成 golden
GSV_BERT_THREADS=8 tests/test_bert_ggml.exe                              # CPU（Debug 构建）
GSV_BERT_DEVICE=vulkan GGML_VK_DISABLE_COOPMAT2=1 tests/rel/test_bert_rel.exe --bench
GSV_BERT_LAYERS=1 ...                                                    # 逐层对拍
```

golden（HF fp32，4 个文本：中文 25/24 token、英文 16、短句 4）：

| 文件 | 形状 | 用途 |
|---|---|---|
| `bert.{i}.ids.bin` | `[1,T]` | token ids（分词在 Python 侧） |
| `bert.{i}.hidden.bin` | `[1,T,1024]` | `hidden_states[-3]` |
| `bert.{i}.feat.bin` | `[1024,T-2]` | 管线特征（去 CLS/SEP + 转置） |
| `bert.0.hs0..22.bin` | `[1,T,1024]` | 逐层（0=embedding+LN，k=第 k−1 层输出） |

CPU 结果（复现见上）：4 文本 `max|Δ| = 3.8e-6 ~ 1.05e-5`，cos = 1.0；逐层 k=0..22 全部 `cos=1.0`、`max|Δ| ≤ 9.5e-6`（k=0 为 9.5e-7）→ 结构、gelu(erf)、post-LN 顺序、head 拼接顺序、CLS/SEP 切法全部正确。

## 4. Vulkan 为什么是 1e-2 级，以及它是否安全

- k=0（embedding+LN，无 matmul/attention）Vulkan 也是 9.5e-7 → 误差来自层内。关闭 coopmat2 已不够：**ggml-vulkan 的 flash attention 内核只支持 F16 的 K/V**（`pipeline_flash_attn_f32_f16`；`GGML_VK_DISABLE_COOPMAT` 打开后误差不变，排除 coopmat2 之外的 matmul 贡献）→ K/V 被降到 f16，这是误差主源。
- **容限标定（新方法，`test_ar_engine` 的 `GSV_AR_BERT_NOISE`）**：BERT 特征均值 `|feat| = 0.80`。给 AR 引擎的 bert 输入注入确定性的高斯噪声 σ，看 100-token greedy 是否还与 f32 参考一致：

  | σ（特征绝对噪声） | 100-token greedy | 说明 |
  |---|---|---|
  | 0 / 1e-4 | 0/3 差异 | ✅ |
  | **3.7e-4（= Vulkan-f32 后端自身误差）** | **0/3 差异** | ✅ |
  | 1e-2（= 1.25% 相对） | 0/3 差异 | ✅ 仍是 token 稳定区 |
  | 5e-2（= 6% 相对） | **1/3 差异（@35 步）** | ✗ 开始分歧 |

  即：**token 稳定的硬阈值在 σ≈1e-2 与 5e-2 之间**，而 Vulkan 后端的误差（4.0e-4）比 1e-2 还小 25 倍，比分歧点小 125 倍。
- 结论：**Vulkan 后端可以用于 BERT**（误差相当于给 AR 喂了 0.05% 相对扰动，token 流不变）；若追求"分布级不可分"，用 f32 权重 + CPU，或改为手写 F32 attention（不采用 flash，代价是 T=512 时注意力显存/耗时上升）。

## 5. 基准（T=25，单次前向，含输入上传/输出回读）

| 实现 | T=25 | T=64 | T=128 | T=256 | T=512 |
|---|---|---|---|---|---|
| torch CPU fp32（24 层） | 169 | 303 | 513 | 953 | 1244 |
| torch GPU fp32（24 层） | 49 | 48 | 56 | 54 | 61 |
| torch GPU fp16（24 层） | 50 | 53 | 48 | 53 | 47 |
| ggml CPU f32（22 层, 8 线程） | 149 | 357 | 671 | 1350 | 2899 |
| ggml CPU f16（22 层, 16 线程） | 110 | 237 | 455 | 881 | 1788 |
| **ggml Vulkan f32** | **10.6** | 7.9 | 14.2 | 23.1 | 50.7 |
| **ggml Vulkan f16** | **8.5** | 7.5 | 11.9 | 21.3 | 43.4 |

要点：

- **torch GPU 的 BERT 是 launch-bound**：T 从 4 到 512 恒定 ~50 ms（HF eager 前向 ~600 个小 kernel）；ggml Vulkan f16 8.5 ms → **约 5.9× 于 torch 最快路径**。
- **本次提速（QKV 融合 + q/k/v 视图直送 flash）**：每层 25 → 18 dispatch。Vulkan f16 10.1 → 8.5 ms（-16%）、
  Vulkan f32 13.1 → 10.6 ms（-19%）、CPU f32 167 → 149 ms（-11%）；CPU 侧主要瓶颈是算力（82 GFLOPS @8 线程），
  故算子数减少的收益小，换 16 线程 + f16 权重才有 1.5×（149 → 110 ms）。
- CPU 上 ggml-f32 与 torch-fp32 持平（167 vs 169 ms）→ ggml 的 CPU 路径没有额外开销，但也说明 BERT 短序列在 CPU 上就是 ~0.17 s 量级（一次性成本，占整句 TTS 比重很小：AR 一字约 20~30 token × ~5 ms/step ≫ BERT）。
- **F16 权重在 CPU 上不提速**（166 ms，F16→F32 转换等价开销），只省内存；在 Vulkan 上提速 ~25%（10.1 vs 13.1 ms）。量化权重（q6_k）反而**更慢**（15.0 ms，去量化开销）→ 小 batch 场景下"量化换速度"不成立，量化只为省内存。
- torch 侧数据由 `tools/bench_bert.py` 生成（含 `output_hidden_states=True` 的全部 25 层，与官方 `get_bert_feature` 用法一致）。

## 5.1 BERT 还有多少提速空间

**时间都花在哪**：按 T 扫描拆固定/变动成本（Vulkan f16）：T=25 → 8.5 ms，T=512 → 43.4 ms，
即每多一个 token 只多 ~0.072 ms，反推 **T=25 时约 79% 是与序列长度无关的固定成本**
（~400 次 dispatch + 每次调用的建图/gallocr），只有 ~21% 是真算力（25 token 时 13.9 GFLOP，
8.5 ms ↔ 1.6 TFLOPS 有效，仅为 RTX 2070 fp32 峰值的 21%）。所以**减少 dispatch 数就是它的提速路线**。

**能做的（工程层）—— 已做完**：把 `norm+mul+add` 合成 `ggml_layernorm_affine`（3 处/层：embedding LN +
两个 post-LN），把 `add+gelu_erf` 合成 `ggml_add_act`（1 处/层），每层 18 → 14 dispatch（算子实现见
[ar_latency.md §2.5](ar_latency.md)，CPU + Vulkan 双实现，默认开启）。实测：

| 配置 | 未融合 | 融合 | 变化 |
|---|---|---|---|
| CPU f16（16 线程，T=25） | 110.0 / 111.2 ms | **108.2** / 123.7 ms | 噪声内（本机漂移 ±15%） |
| Vulkan f16（T=25） | 9.46 / 10.41 ms | 8.87 / **7.90** ms | 方向为正，噪声内 |
| （第一版融合的配对测量）CPU f32/f16（16 线程） | 127.5 / 131.7 ms | **116.2 / 115.7 ms** | **-9% / -12%** |

Vulkan 上没收益的原因：BERT 的 matmul 形状（[1024,1024]×[1024,25]）走的是带 bias 融合的多头 matmul
路径，我新增的 elementwise 融合在这条路径上省下的 dispatch 占比很小；CPU 上没有这种后端融合，
所以收益实打实。**顺带记录一条推导结论**：BERT 的 LN 仿射**不能**折叠进下游 matmul——post-LN 结构里
LN 的输出同时喂给残差流和下一个 matmul，折叠会改变残差（AR 同理）。

**不建议做的（模型层）**：

1. **剪层/换更小的预训练模型**：本实现已经只用 22/24 层（`hidden_states[-3]` 之后的两层本来就白跑）。
   再往下剪会改变特征分布，而 BERT 特征直接进 AR 的 `bert_proj`——§4 的噪声探针表明
   σ≈1e-2 就开始影响 token，因此"换模型/蒸馏"都必须重训并重新标定验收，不是工程优化。
2. **再量化**：实测 Q8_0 及以下越界（§0 表），F16 已是下限。
3. **蒸馏/替换为小模型**：需要训练数据与 GPU 时间，收益上限只有 ~10 ms/句
   （BERT 在整句 TTS 中占比 <5%，AR 是 0.5~1 s、DiT 是 100~200 ms 量级），
   风险（特征分布漂移 → 音质/韵律变化）远大于收益。

**结论**：BERT 段在 GPU 上（8.5 ms）已比 torch 最快路径快 5.9×，剩余工程空间约 -15~20%（需要
Vulkan 融合算子，可与 AR 的同类改造一起做）；模型层面不做改动。CPU 侧若必须用 CPU，
用 16 线程 + f16 权重（110 ms/句）即可，它同样是整句里的小头。

## 6. 量化：为什么 BERT 比 AR 敏感（与 `docs/quant_ar.md` 的对照）

AR 的验收是**概率空间**（TV / token 一致性，predict 层把 512 维隐状态投影到 1025 词表，误差被"再投影 + argmax"吸收），BERT 这里是**隐藏状态空间**（22 层 post-LN 残差流的逐元素误差，直接进入 AR 的 `bert_proj`）。加上 BERT-Large 的离群激活（个别维度量级远高于均值），CPU 端 Q8_0 还会**再量化激活**（per-32 块）→ Q8 在 CPU 上比 Vulkan 更差（1.2e-2 vs 7.1e-3 均值），与 AR 当时的结论方向一致但幅度大得多。

因此：

- **最小近无损 = F16（662 MB）**：误差（均值 2.7e-4 CPU / 2.0e-3 Vulkan）≈ f32-权重-Vulkan 后端自身误差量级，token 稳定区内有 5~30 倍余量。
- Q8_0 / Q6_K / Q5_K 全部落在 token 分歧阈值附近或以上 → 不采用。
- 若下一步要做"分层混合精度"（例如只把最后几层保 F16、前面量化），BERT 的收益上限只有 ~4 MB/层，优先级低；AR 侧的混合精度收益更大。

## 7. 与链路的衔接

BERT 段已就绪的能力：`gsv_bert::load(gguf, {device, n_threads})` → `encode_feat(ids, T, feat)` → `feat [1024, T-2]` 行主序，正是 `gsv_ar_request::bert` 需要的布局（`bert_proj` 按 `k*T+t` 读）。剩余接线：

1. 文本前端（ONNX G2PW + 分词）产出 token ids（Python/ONNX 侧）；
2. `gsv_bert` 产出特征 → `gsv_ar` 生成 semantic tokens；
3. HuBERT/RVQ→enc_p/MRTE/ref_enc/bridge/wns1 与 DiT/vocoder 各自接入。

对 AR 侧的建议：**BERT 走 Vulkan 时，AR 的 `bert` 输入误差与 f32 参考的差异已在 token 稳定区内**，无需为 BERT 单独降级后端。
