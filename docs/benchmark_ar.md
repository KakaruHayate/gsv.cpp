# AR 段基准：gsv.cpp（CPU / Vulkan）vs Python 实现

日期：2026-10-07 · 机器：Xeon E5-2675 v3（16C/32T，AVX2）· GPU：RTX 2070（sm75，8GB）· torch 2.12.1+cu130
工作负载：3 条序列（32/26/20 音素 + 24 prompt token）greedy 生成 100 token（bs>3 时循环复用这 3 条）。
所有实现使用同一份 golden 输入（`tests/golden/batch.*`），生成步数一致（greedy），可直接比较。

## 0. Python 侧"最快的 s1 路径"是哪条

仓库有四种 AR (s1) 推理路径，实测/判定（`tools/acceleration.py::resolve_acceleration`）：

| 路径 | 条件 | 本机可用性 |
|---|---|---|
| `flash_attn_varlen_cuda_graph` | 需要 flash-attn + **算力 ≥ sm80** | ✗（RTX 2070 = sm75） |
| **`torch_static_cuda_graph`** | 仅需 `torch.cuda.CUDAGraph/graph/Stream` + SDPA | ✅ **本机最快** |
| torch eager（fp16/fp32，`infer_panel_*`） | 始终可用 | ✅（慢 2.5×） |
| MLX | 仅 Apple | ✗ |

结论：**这条机器上最快的 Python 路径是 `torch_static_cuda_graph`（CUDA Graph + SDPA，fp16）**，
通过 `tools/acceleration.py::create_acceleration` → `GPT_SoVITS.Accel.adapter.AccelInference.infer_batch` 驱动
（`tools/bench_ar_accel.py` 就是封装这个调用；首次调用含图捕获约 2\sim6 s）。

## 1. 主结果：解码每 token 耗时 / 吞吐

> **2026-10-07 更新（AR 优化三轮之后，同场配对测量）**：下表是**当前** ggml 实现与 torch 最快路径
> 在**同一时段**重测的结果（旧的优化前数字与优化过程见 [docs/ar_latency.md](ar_latency.md)）。
> 工作负载不变：3 条序列（32/26/20 音素 + 24 prompt token）greedy 生成 100 token。

| | bs=1 | bs=3 | bs=8 | bs=20 |
|---|---|---|---|---|
| **ggml + Vulkan f32**（当前）每 token | **2.61 ms** | 3.94 | 7.39 | 14.80 |
| ggml + Vulkan f16（158MB） | 2.38 | — | — | — |
| ggml + Vulkan q8_attn_ffn（85MB，近无损） | **2.27 ms** | — | — | — |
| torch CUDA Graph fp16（最快 Python 路径） | 14.82 ms | 17.70 | 25.02 | 40.89 |
| **倍率（每 token）** | **5.7× / 6.2× / 6.5×**（f32/f16/q8） | 4.5× | 3.4× | 2.8× |

| | bs=1 | bs=3 | bs=8 | bs=20 |
|---|---|---|---|---|
| ggml Vulkan f32 总吞吐 | **383 tok/s** | 761 | 1082 | **1351 tok/s** |
| torch CUDA Graph fp16 总吞吐 | 64.5 tok/s | 154 | 268 | 376 |
| **倍率（吞吐）** | **5.9×** | 4.9× | 4.0× | **3.6×** |

首步（prefill，56 token 前缀；torch 侧含 CUDA Graph 捕获）：

| | bs=1 | bs=3 | bs=8 | bs=20 |
|---|---|---|---|---|
| ggml Vulkan f32 | **16.8 ms** | 12.5 | 18.4 | 36.5 |
| torch CUDA Graph fp16 | 83.4 ms | 198.4 | 510.2 | 1277.9 |
| **倍率** | **5.0×** | 15.9× | 27.7× | 35× |

端到端整句（TTFT + 100 token）：

| | ggml Vulkan | torch CUDA Graph | 倍率 |
|---|---|---|---|
| bs=1（f32） | 16.8 + 261 = **278 ms** | 83.4 + 1482 = 1565 ms | **5.6×** |
| bs=1（q8_attn_ffn 近无损档） | 16.3 + 227 = **243 ms** | 1565 ms | **6.4×** |
| bs=20 | 36.5 + 1480 = 1517 ms | 1277.9 + 4089 = 5367 ms | 3.5× |

**精度对照（同一 workload）**：ggml f32 与 golden 逐 token 完全一致；torch CUDA Graph 走 fp16，
与 golden 对比 bs=1 差 1/1、bs=3/8/20 差 3/3 条序列（`--accel` 输出自带该检查）→ 我们是
"更快 + 精确 f32"，torch 最快路径是"更慢 + fp16 漂移"。

（旧的优化前数字供参考：ggml Vulkan 5.00/6.46/8.83/15.27 ms·token⁻¹，torch 14.47/16.32/23.24/37.50，
首步 43.7/73.6/180.7/439.6 vs 77.0/206.5/503.8/1221.8 —— 当时倍率 2.5~3.2×。）

## 2. 数值正确性

| 实现 | 首步 logits max\|Δ\|（vs torch f32 golden） | greedy token 序列 |
|---|---|---|
| gsv.cpp CPU f32 | 5.7e-06 | 3/3 与参考一致 |
| gsv.cpp Vulkan f32（禁用 coopmat2，默认） | 3.5e-03 | 3/3 一致 |
| gsv.cpp Vulkan f32（coopmat2 开启） | **2.0e-02** | 3/3 一致（本 workload） |
| gsv.cpp F16 权重（CPU） | 2.8e-03 | 3/3 一致 |
| gsv.cpp Q8_0 权重（CPU） | 1.6e-01 | 2/3 分歧 ✗ |
| torch CUDA Graph fp16 | —（参考即 f32） | 0/3 一致（fp16 数值漂移，属预期） |

**Vulkan 的 `coopmat2` 会显著掉精度**（0.0035 → 0.0201，6×）且性能几乎无差（bs=20: 15.27 vs 14.51 ms）→
引擎默认禁用（`gsv_ar_cfg::disable_coopmat2 = true`，加载时设置 `GGML_VK_DISABLE_COOPMAT2=1`）。

## 3. 性能剖析与已做优化（CPU 侧）

`GSV_AR_PROFILE=1` 的每步分解（bs=1，f32，16 线程）：

| 阶段 | 优化前 | 优化后 |
|---|---|---|
| 建图 + gallocr | 2.7 ms | 1.9 ms |
| **KV cache 回传** | **17.5 ms**（bs=8 时 97 ms） | **≈0** |
| 计算（graph_compute） | 17.2 ms | 16.4 ms |
| 合计 | 32.5 ms | 18.1 ms |

手段：**KV cache 常驻后端内存**（单独 buffer），首步图内 `ggml_cpy` 写入、解码步只写新列、flash 直接读
cache 视图。踩坑：`cpy` 写 cache 与 flash 读 cache 视图在同图内**无依赖边**，顺序不定（表现为"bs=3 时 15 步
就 EOS"）；修复是让 flash 读 `concat(旧列视图, cpy输出)` 显式化写→读依赖。

CPU 剩余成本是内存带宽（f32 权重每 token 读 ~300 MB → 16.4 ms ≈ 18 GB/s）。这也解释了 Vulkan/Gpu 的
优势：同样的 300 MB 走显存带宽（~450 GB/s）。

## 4. Vulkan 路径的工程要点（本次新增）

- 引擎按 `gsv_ar_cfg::device` 选择设备（`""`=CPU，`vulkan`/`gpu`=第一个 GPU 设备）；
- 权重加载统一为 **`no_alloc` 读 meta → `ggml_backend_alloc_ctx_tensors` 在后端分配 → 逐张量从文件
  `ggml_backend_tensor_set` 上传**（CPU/GPU 同一路径）；
- 前端（bert_proj / embedding / alpha）改为**加载时拷到 host 副本**——GPU 后端下 `tensor->data` 不可直读
  （踩坑：Q8_0 权重时 host 直读量化字节 → logits 全 NaN；现已强制这些权重为 F32 并在加载时校验）；
- 只用单后端（Vulkan）跑图，未用 `ggml_backend_sched`——AR 用到的算子（mul_mat / flash_attn_ext / norm /
  relu / cast / cpy / concat / cont）Vulkan 后端全部原生支持，实测无需回退。

## 5. 复现

```bash
# CPU (Release + native)
cmake -S llama.cpp -B llama.cpp/build-rel -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build llama.cpp/build-rel --target ggml --config Release -j 16
# Vulkan (Release + native)
cmake -S llama.cpp -B llama.cpp/build-vk-rel -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -G "Visual Studio 16 2019" -A x64
cmake --build llama.cpp/build-vk-rel --target ggml --config Release -j 16

# 对拍 / 基准
GSV_AR_DEVICE=vulkan tests/vk/test_ar_engine_vk.exe models/gsv-ar-f32.gguf tests/golden      # Vulkan 正确性
GSV_AR_DEVICE=vulkan tests/vk/bench_ar_vk.exe    models/gsv-ar-f32.gguf tests/golden 100 8   # Vulkan 基准
GSV_AR_PROFILE=1 tests/bench_ar.exe              models/gsv-ar-f32.gguf tests/golden 100 16  # CPU + 分阶段
python tools/bench_ar_accel.py --n-gen 100                  # Python 最快路径
python tools/bench_ar.py --n-gen 100 --threads 16           # torch eager
```

## 6. 结论与后续

1. **AR 段在 GPU 上已经明显超过 Python 最快路径**（2.5\~3.2×），且是精确 f32（Python 的加速路径是 fp16，
   token 会漂移）。这条链路（ggml + Vulkan + 常驻 KV cache）可以直接作为后续 Vulkan 工作的模板。
2. prefill（首步）在同量级但不占优（Vulkan 43.7 vs CUDA Graph 77.0 ms — 已领先 1.8×），
   若后续句子变长可以再优化（分块 prefill / 更优的 attention 配置）。
3. 剩余可选项：f16 权重（CPU 实测 +25%；Vulkan 上待测）、更深度的量化（AR 上限为 F16）。
4. 该基准未包含采样链开销（greedy 时 top_k=1 走一遍 chain，开销 <0.1ms/步）与 Python 侧前端
   （phones/bert 预处理），因为两者对所有实现是同一量级。
