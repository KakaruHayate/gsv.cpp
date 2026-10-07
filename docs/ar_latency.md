# AR 单流（bs=1）延迟优化

日期：2026-10-07 · 机器：Xeon E5-2675 v3（16C/32T）· RTX 2070 · Vulkan（`GGML_VK_DISABLE_COOPMAT2=1`）· Release 构建
工作负载同 [docs/benchmark_ar.md](benchmark_ar.md)：3 条序列（32/26/20 音素 + 24 prompt token）greedy 生成 100 token。

## 0. 结论（先给答案）

| 配置 | 每 token 延迟 | 吞吐 | 相对优化前 |
|---|---|---|---|
| **Vulkan f32（bs=1）** | **5.08 → 3.89 ms** | 197 → 257 tok/s | **-23%** |
| Vulkan f32（bs=3 / bs=8） | 6.84 → 5.49 / 9.07 → 8.10 ms | 546 / 987 tok/s | -20% / -11% |
| Vulkan **+ q8_attn_ffn 权重**（bs=1） | **3.54 ms** | **283 tok/s** | -30%（含精度档） |
| CPU f32（bs=1，8 线程） | 19.64 → 17.6 ms | 57 tok/s | 建图/算子优化对 CPU 影响很小（CPU 受权重带宽限制） |
| **CPU + q8_attn_ffn 权重**（bs=1） | **9.1 ms** | **110 tok/s** | **-54%**（近无损档，见 §3） |
| CPU f16（bs=1） | 13.4 ms | 75 tok/s | -32% |
| CUDA（bs=1） | 见 §5 | | |

一句话：**GPU 单流延迟的瓶颈是"每步固定开销"（dispatch + 建图），不是一个 token 的算力**；
本次砍掉 144 次拷贝 dispatch、48 次冗余图遍历、去掉每步一次 cast，Vulkan 单流 -23%；
CPU 路径的杠杆是**权重精度**（f32→Q8_0 近无损档直接 -54%），不是算子数。

## 1. 每步开销剖析（优化前，Vulkan bs=1）

`GSV_AR_PROFILE=1`：

| 阶段 | 每步 | 占比 |
|---|---|---|
| 建图 + gallocr | 1.42 ms | 24% |
| 输入/输出拷贝 | ~0.00 ms | 0% |
| `graph_compute`（含 GPU 执行） | 4.61 ms | 76% |

节点计数（每层）：mul_mat qkv 1 + bias 1 + **cont q/k/v 3** + **permute+cont 3** + cpy 2 + concat 2 +
flash 1 + out 1 + out_b 1 + residual 1 + norm 1 + mul 1 + add 1 + ffn 3 + relu 1 + ffn 1 + add 1 +
residual 1 + norm 1 + mul 1 + add 1 = **28 dispatch/层 → 674/步**（+ ~170 个 view 节点）。

**"启动受限"的判定**：bs=3 的每步只比 bs=1 多 1.46 ms（≈0.73 ms/序列）→ bs=1 的 ~86% 是
与 batch 无关的固定开销 = 逐节点 dispatch（674 × ~6 µs）+ 建图。

## 2. 本次已做（无新算子，数值完全等价）

1. **解码步 q/k/v 用 strided 视图直送 flash**（省掉每层 6 次 `ggml_cont` = 144 dispatch/步）。
   S=1 时 `permute(reshape(q, HD,NH,1,B), 0,2,1,3)` 是恒等变换：qkv 的内存序本来就是 (hd, nh)，
   直接构造 `view_4d(HD,1,NH,B, nb=S维任意, nb2=4*HD, nb3=4*3D)` 即 flash 需要的布局。
   K/V 同样以视图交给 KV cache 的 `ggml_cpy`（cpy 按索引拷贝，src/dst 步长可以不同）。
   CPU 与 Vulkan 都按传入步长寻址（CPU 只要求内维连续 `nb0 == type_size`；Vulkan 的 FA 通过
   push constant 接收 q/k/v 三个步长），实测**逐位一致**。
2. **mask 直接以 F16 上传**（原来是 F32 上传 + 图内 `ggml_cast`），省 1 dispatch/步。
3. **删掉 48 次冗余 `ggml_build_forward_expand`**：KV 写入的 `cpy` 已经通过
   `concat(旧列视图, cpy输出)` 连到 flash，是 logits 的祖先；再单独建根会让
   `ggml_build_forward_expand` 重新遍历整条链 48 次（建图 1.42 → 0.99 ms）。
4. 建图上下文 16384 → 2048 张量（无性能差，减小内存抖动）。
5. 引擎 `device` 选择支持 `"cuda"`（原来只认 `"vulkan"/"gpu"`）。

> 关键坑（已写在 README):`cpy` 写 cache 与 flash 读 cache 视图之间**必须有数据依赖边**，
> 否则 Vulkan 不插 barrier、顺序不定（表现为随机早停）。本次没有动这个结构，只是把
> "写 cache 的数据"从 `cont` 结果换成了视图。

## 3. 权重精度档对单流延迟（本次新增的实测）

| 模型 | 体积 | CPU bs=1 | Vulkan bs=1 | 100-token 门（CPU / Vulkan） |
|---|---|---|---|---|
| f32 | 296 MB | 17.6 ms | 3.89 ms | — / —（参考） |
| f16 | 151 MB | 13.4 ms | 3.95 ms | PASS / PASS |
| **q8_attn_ffn（近无损档）** | **85 MB** | **9.1 ms** | **3.54 ms** | **PASS / PASS** |
| k_q6k | 67 MB | 9.0 ms | — | PASS（`docs/quant_ar.md`） |

- CPU 是**权重带宽受限**：f32 每 token 读 ~300 MB，Q8_0 只读 ~90 MB → 延迟几乎按体积线性下降；
  8 线程与 16 线程无差别（17.6 vs 17.9 ms）进一步印证是带宽而非算力瓶颈。
- Vulkan 不受益于量化（每步固定开销不变，只是权重读得少一点）：3.89 → 3.54 ms（-9%）。
- 结论：**CPU 路径默认用 q8_attn_ffn（85 MB，近无损，已过 100-token 门）**；GPU 路径用 f32/f16 即可。

## 4. 剩余空间（已量化，未实施）

| 方向 | 预计收益 | 代价/风险 |
|---|---|---|
| **融合算子**：`norm + mul + add` → 1 个 `layernorm_affine`（每层 2 处，省 4/层）；`add + relu` 复用 patch 已有的 `ADD_LEAKY_RELU`（CPU-only，需补 Vulkan shader，省 1/层） | 每层 22 → 17 dispatch，bs=1 ≈ **3.3 ms（-17%）** | 需在 ggml 加算子 + Vulkan shader + 重放 audio-patch（约 250 行，工作路径与 patch 里 116 个算子一致） |
| 图复用（把 KV 列偏移从图里拿掉，建图 0.99 ms → 0） | ≈ **2.9 ms（-25%）** | 需要 mutate `tensor->data`（非 ggml 公开 API）或给 `ggml_cpy` 加"可变视图"语义；属于 hack，需谨慎 |
| prefill（TTFT）：56 token 前缀 45.5 ms | 与 decode 同源（674 dispatch + chunk 化 attention） | 分块 prefill / 更大 tile；当前已领先 torch CUDA Graph 1.7× |
| FA 的 K/V 常驻 F16（Vulkan 的 FA 内核本来就只吃 F16 K/V） | 省一半 KV 流量，Vulkan 上数值等价 | 会改变 CPU 端数值（f16 化 K/V），需重跑验收 |

## 5. CUDA 后端（产品优先级 CUDA > Vulkan）

用 `llama.cpp/build-cuda126`（CUDA 12.6 + sm75）构建同一套 ggml，引擎无需改代码即可选到 CUDA
设备（`GSV_AR_DEVICE=cuda`，本次新增别名）。数值/延迟见下表（构建完成后填入）：

| 后端 | bs=1 | bs=3 | bs=8 |
|---|---|---|---|
| Vulkan f32 | 3.89 ms | 5.49 | 8.10 |
| CUDA f32 | 见下方表格更新 | | |

## 6. 复现

```bash
# 构建 (Release + Vulkan)
scripts\build-ar-vk.bat
# 单流延迟 + 分阶段剖析
GGML_VK_DISABLE_COOPMAT2=1 GSV_AR_DEVICE=vulkan GSV_AR_PROFILE=1 tests/rel/bench_ar_rel.exe models/gsv-ar-f32.gguf tests/golden 100 8
# CPU + 近无损档
GSV_AR_THREADS=8 tests/rel/bench_ar_rel.exe models/sweep/q8_attn_ffn.gguf tests/golden 100 8
# 量化门（100 token 一致）
GSV_AR_ACC=1 tests/rel/test_ar_engine_rel.exe models/sweep/q8_attn_ffn.gguf tests/golden
```
