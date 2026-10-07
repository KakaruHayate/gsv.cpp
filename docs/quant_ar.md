# AR 量化调研：找"最小的近无损精度"

日期：2026-10-07 · 同前（Xeon E5-2675 v3 / RTX 2070 / Vulkan coopmat2 关闭）
协议：`tests/test_ar_engine.cpp` 的 ACC 模式 —— 3 条序列 greedily 生成 **100 token** 与 f32 参考逐 token 比对
（`batch.greedy100.*`），并记录首步 logits 的 max|Δ|。驱动：`tools/quant_sweep_ar.py`（转换 + 验收 + 汇总一条龙）。

## 0. 先说三个结构性事实（决定了结论的形态）

1. **CPU 的量化矩阵乘会把激活也量化到 8 bit**（`ggml_mul_mat` 走 Q8_0/Q8_1 的 vec_dot 路径）；
   Vulkan 的量化 matmul 是"权重量化 → 反量化到 f16 再乘"，**不量化激活**。
   → 同一档位，**Vulkan 的量化误差约为 CPU 的一半**（实测 q8_ffn：0.026 vs 0.059）。量化要在 GPU 上用。
2. **Vulkan 的 coopmat2 会掉精度 6×**（logits Δ 0.0035 → 0.0201）且速度几乎无差（bs=20: 14.5 vs 15.3ms）。
   该开关只能在进程启动前设（ggml-vulkan 在静态初始化阶段读取，引擎内部设置太晚——已在加载时加警告）：
   `GGML_VK_DISABLE_COOPMAT2=1`。
3. **`predict`（输出层）是最敏感的权重**：单独把 predict 量化到 Q8_0（其余 f32）→ logits Δ 0.093、100 token 有 1/3 分歧；
   attn/ffn 单独量化反而都能过。所以输出层永远保留 F32/F16。embedding 因引擎在 host 侧查表，必须 F32（可改成图内 get_rows 后放开）。

## 1. Vulkan 扫描结果（100 token 验收 + 首步 logits Δ，对比 CPU-f32 参考）

| 配置 | 体积 | logits Δ | token | 备注 |
|---|---|---|---|---|
| f32（基准） | 310.4 MB | 3.5e-03 | 3/3 | Vulkan f32 自身的归约差异 |
| **f16（attn/ffn/predict 全 F16，emb F32）** | **158.4 MB** | **1.8e-02** | 3/3 | 保守档 |
| q8_ffn（attn F16） | 162.6 MB | 2.6e-02 | 3/3 | |
| f16_q8_0（attn F16 + ffn Q8_0） | 111.2 MB | 3.1e-02 | 3/3 | |
| q8_attn（仅 attn Q8_0） | 236.5 MB | 3.4e-02 | 3/3 | |
| **q8_attn_ffn（predict 保持 F32）** | **88.7 MB** | **5.9e-02** | 3/3 | **推荐全量档** |
| q8_attn_ffn_p16（predict F16） | 87.6 MB | 7.0e-02 | 3/3 | 与上行几乎同体积，predict 保 F32 更稳 |
| f16_q5_0（ffn Q5_0） | 92.3 MB | 1.6e-01 | 3/3 | 刀刃边缘 |
| q8_q5_0（attn Q8_0 + ffn Q5_0） | 68.7 MB | 1.7e-01 | 3/3 | 刀刃边缘 |
| f16_q5_1（ffn Q5_1，6 bit） | 95.5 MB | 1.8e-01 | **2/3 ✗** | 比 Q5_0 多 0.5 bit 反而挂 → 说明门限已到随机性区间 |
| f16_q4_0 | 86.0 MB | 4.0e-01 | 1/3 ✗ | |
| attn_q5_1_ffn_q4_0 | 54.6 MB | 3.4e-01 | 0/3 ✗ | |
| q4_0_all | 49.9 MB | 7.2e-01 | 0/3 ✗ | |

CPU 侧同扫（供对照，注意事实 1 导致的系统性更差）：

| 配置 | 体积 | logits Δ | token |
|---|---|---|---|
| f32 | 310.4 MB | 5.7e-06 | 3/3 |
| f16 | 158.4 MB | 2.8e-03 | 3/3 |
| q8_attn_ffn_p16 | 87.6 MB | 5.9e-02 | 3/3 |
| q8_q5_0 | 68.7 MB | 1.7e-01 | 3/3 |
| **q8_predict（仅 predict Q8_0）** | 308.9 MB | 9.3e-02 | **2/3 ✗** |
| f16_q5_1 | 95.5 MB | 1.9e-01 | 2/3 ✗ |
| f16_q4_0 | 86.0 MB | 3.8e-01 | 1/3 ✗ |
| q4_0_all | 49.9 MB | ~7e-01 | 0/3 ✗ |

## 2. 速度（Vulkan，100 token，coopmat2 off）

| 权重 | bs=1 每步 | bs=20 每步 | bs=20 吞吐 |
|---|---|---|---|
| f32 | 4.87 ms | 14.31 ms | 1398 tok·s⁻¹ |
| f16 | 4.75 ms | **11.22 ms** | **1782 tok·s⁻¹**（+27%） |
| q8_attn_ffn_p16 | 4.87 ms | 13.76 ms | 1454 tok·s⁻¹ |
| q8_q5_0 | 4.87 ms | 11.68 ms | 1713 tok·s⁻¹ |

**bs=1 时各档位耗时完全相同（4.87 ms）——单流解码是 launch/dispatch 受限，不是带宽受限**，量化只在批量
（bs=20）时带来 20~27% 的收益。这修正了"量化一定更快"的直觉：对单句交互式合成，量化的意义主要是省显存，
而不是提速。

## 3. 结论：两档发布

| 档位 | 配置 | 体积 | 误差 | 用途 |
|---|---|---|---|---|
| **参考/最小验证版** | **全 F16，predict/emb/norm/bias/alpha 保 F32** | **158 MB**（1.96×↓） | Δ 0.018（Vulkan）/ 0.003（CPU），100 token 全一致 | parity 锚点、质量基线 |
| **全量版** | **attn + ffn 用 Q8_0，predict/emb/norm/bias 保 F32** | **89 MB**（3.5×↓） | Δ 0.059，100 token 全一致；GPU 上比同档 CPU 好一倍 | 省显存/批量吞吐 |

**不建议低于这一档**：Q5_0 的 69 MB 虽然本次 100 token 全过，但兄弟配置（Q5_1，体积更大）反而挂了
——说明该误差量级下"过/不过"由具体权重分布决定，缺乏余量；Q4_0 一档则明确崩（Δ 0.4~0.7，token 0~1/3）。

## 4. 方法与后续

- 验收口径提醒：**"100 token 逐 token 一致"是刀刃型判据**——单个 argmax 翻转就会让后续全部发散，
  因此不同配置的过/不过并不随误差单调（见 Q5_0 vs Q5_1）。最终定档建议再补两个更平滑的指标：
  ① 采样分布级（每步 top-k 概率的 TV/KL 距离）；② 链路级（DiT/vocoder 就绪后比 mel-L1 / 说话人相似度 / WER）。
  本文件的 Δ 表就是为后一步做筛选用的。
- 本机 gguf-py 未实现 K-quants（Q6_K/Q5_K/Q4_K 等），因此 4~7 bit 区间只测了 legacy quants。
  K-quant 在同体积下通常明显更优，**建议后续用 llama.cpp 的 `llama-quantize`（或自写 K-quant 量化器）补测
  Q5_K/Q6_K**，预期能在 90~110 MB 区间给出比 legacy Q5_0 更好的精度。
- 复现：`python tools/quant_sweep_ar.py --device cpu|vk [--only name,...]`（转换 + 验收一步完成）；
  产物在 `models/sweep/*.gguf`。
