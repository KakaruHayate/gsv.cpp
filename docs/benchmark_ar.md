# AR 段基准（gsv.cpp vs Python/torch）

日期：2026-10-07 · 机器：Xeon E5-2675 v3（16C/32T，AVX2）· GPU：RTX 2070 · torch 2.12.1+cu130
工作负载：3 条序列（32/26/20 音素 + 24 prompt token，bert 特征随机固定），greedy 生成 100 token
（`tests/bench_ar.cpp` / `tools/bench_ar.py`，同一份 golden 输入，两边线程数一致 = 16）

## 1. 解码吞吐（每 token）

| 实现 | bs=1 | bs=3 | bs=8 |
|---|---|---|---|
| **gsv.cpp（ggml CPU, F32 权重）** | **18.1 ms / 55.4 tok·s⁻¹** | **24.1 ms / 124.3 tok·s⁻¹** | **37.7 ms / 212.4 tok·s⁻¹** |
| gsv.cpp（F16 权重） | 14.4 ms / 69.7 | 20.0 ms / 150.0 | 34.8 ms / 230.2 |
| gsv.cpp（Q8_0 权重） | 9.7 ms / 102.8 | 15.9 ms / 188.4 | 29.7 ms / 269.6 |
| torch CPU f32（16 线程） | 37.8 ms / 26.4 | 41.9 ms / 71.6 | 56.1 ms / 142.6 |
| torch CUDA f32 | 40.6 ms / 24.7 | 42.5 ms / 70.5 | 41.3 ms / 193.9 |
| torch CUDA f16 | 40.7 ms / 24.6 | 44.8 ms / 67.0 | 42.1 ms / 190.1 |

**F32 下相对最快 torch 的加速**：bs=1 **2.1×**、bs=3 **1.7×**、bs=8 1.1×（对 CUDA）/ 1.5×（对 CPU）。

首步（prefill，56 token 前缀）：gsv.cpp 72–95 ms（bs=1）vs torch CPU 63 ms / torch CUDA 34 ms —— 我们的
prefill 偏慢（ggml 的 F32 `mul_mat` 在 M≈56 的形状上效率一般），但每句只跑一次。

## 2. 权重精度（对 torch F32 参考）

| 权重 | 首步 logits max\|Δ\| | greedy token 一致性 |
|---|---|---|
| F32 | 5.7e-06 | 3/3 一致 |
| F16（norm/bias/emb/bert_proj/alpha 保 F32） | 2.8e-03 | 3/3 一致（本测试） |
| Q8_0（同上保留项） | 1.6e-01 | 2/3 分歧 ✗ |

结论：**AR 的量化上限是 F16**（+25% 速度，token 序列未变）；Q8_0 不可接受——与调研报告中 CrispASR 的
结论一致（"F5 条件通路必须 F32，Q8 会劣化"）。Q8_0 的误差主要来自权重 8-bit 量化本身（ggml 量化核
用 F32 累加，F16 累加误差反而不是主因：F16 权重时 ggml 的 `vec_dot` 走 F16 累加路径，误差 2.8e-3）。

## 3. 性能剖析与已做优化

`GSV_AR_PROFILE=1` 给出的每步分解（bs=1，F32，16 线程）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| 建图 + gallocr | 2.7 ms | 1.9 ms |
| **KV cache 回传** | **17.5 ms**（bs=8 时 97 ms） | **≈0 ms** |
| 计算（graph_compute） | 17.2 ms | 16.4 ms |
| 每步合计 | 32.5 ms | 18.1 ms |

优化手段：**KV cache 常驻后端内存**（`ggml_backend_alloc_ctx_tensors` 单独 buffer），首步用图内
`ggml_cpy` 写入，解码步只写新列、flash 直接读视图——彻底消除每步 O(L) 的全量回传与 host 端
strided gather。（其间踩到一个坑：`cpy` 写 cache 与 flash 读 cache 视图在同图内**没有依赖边**，
执行顺序不确定，表现为"bs=3 时 15 步就 EOS"；修复是让 flash 读 `concat(旧列视图, cpy输出)`，
把写→读依赖显式化。）

剩余成本主要是计算本身，且是内存带宽受限：F32 权重每 token 要读 ~300 MB（24 层 × 3.1 M 参数 × 4B）
→ 16.4 ms ≈ 18 GB/s 有效带宽。进一步提速的路径：F16 权重（已测，+25%）、BLAS（`GGML_BLAS=ON`
接 MKL/OpenBLAS）、或 GPU 后端（按计划 Vulkan 先行）。

## 4. 与 torch 对比的注意事项

- torch 的每步耗时在 CPU 与 CUDA 上几乎相同（≈40 ms），说明它被**每步 Python 调度 / kernel 发射**
  主导，而不是算力；因此上表不是"算力差距"，而是"当前路径的实现效率差距"。
- 仓库里的生产加速路径是 `GPT_SoVITS/Accel`（CUDA Graph + FlashAttention，PyTorch 侧），
  我们尚未与之对比——那需要 ggml 先有可用的 GPU 后端（Vulkan/CUDA）。**这是 AR 段性能的下一步**。
- 本表两边都用 greedy（我们的采样链 `top_k=1 + rep=1.0` 与 torch argmax 逐 token 一致，见
  `tests/test_ar_batch.cpp` 的 [4]），所以生成步数完全一致，比较是公平的。

## 5. 复现

```bash
# C++（Release ggml + native）
cmake -S llama.cpp -B llama.cpp/build-rel -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build llama.cpp/build-rel --target ggml --config Release -j
# 编译 bench（见 scripts/build-tests.bat 的 cl 命令，链接 build-rel 的 Release 库）
tests/bench_ar.exe models/gsv-ar-f32.gguf tests/golden 100 16
GSV_AR_PROFILE=1 tests/bench_ar.exe models/gsv-ar-f32.gguf tests/golden 100 16   # 分阶段

# torch
python tools/bench_ar.py --n-gen 100 --threads 16

# 权重格式
python tools/convert_ar.py --out models/gsv-ar-f16.gguf --f16
python tools/convert_ar.py --out models/gsv-ar-q8.gguf  --q8
```
