# AR 单流（bs=1）延迟优化

日期：2026-10-07 · 机器：Xeon E5-2675 v3（16C/32T）· RTX 2070 · Vulkan（`GGML_VK_DISABLE_COOPMAT2=1`）· Release 构建
工作负载同 [docs/benchmark_ar.md](benchmark_ar.md)：3 条序列（32/26/20 音素 + 24 prompt token）greedy 生成 100 token。

## 0. 结论（先给答案）

| 配置 | 每 token 延迟 | 吞吐 | 相对优化前 |
|---|---|---|---|
| **Vulkan f32（bs=1）** | **5.08 → 3.89（阶段 1）→ 3.29（融合算子）→ 2.29（图缓存）→ 2.40~2.45 ms（LN 吸收 bias + prefill 视图）** | 197 → **408~417 tok/s** | **-52~53%** |
| Vulkan f32（bs=3 / bs=8） | 6.84 → **4.11~4.14 ms** / 9.07 → **7.00~7.05 ms** | 726 / 1143 tok/s | **-40% / -23%** |
| Vulkan **+ q8_attn_ffn 权重**（bs=1） | **3.54 ms**（图缓存尚未叠加该档） | **283 tok/s** | -30%（含精度档） |
| **TTFT（首步/prefill，56 token 前缀）** | **Vulkan 42 → 15.5 ms（-63%）** / CPU 153 → 96 ms | | host 侧 bert_proj 挪进图（见 §4.2） |
| CPU f32（bs=1，8 线程） | 同期配对：20.97 → **19.29 ms** | | 融合算子 -8%（跨时段绝对值 17.3~21.0 ms 波动） |
| **CPU + q8_attn_ffn 权重**（bs=1） | **9.1 ms** | **110 tok/s** | **-54%**（近无损档，见 §3） |
| CPU f16（bs=1） | 13.4 ms | 75 tok/s | -32% |

一句话：**GPU 单流延迟的瓶颈是"每步固定开销"（dispatch + 建图），不是一个 token 的算力**。
三轮优化分别打掉：144 次拷贝 dispatch + 48 次冗余图遍历 + 每步一次 cast（阶段 1，-23%）；
每层 5 次 elementwise → 2 个新算子（阶段 2，-19%，见 §2.5）；每步 0.78 ms 的建图
（阶段 3 图缓存，-32%，见 §2.6）。合计 **5.08 → 2.29 ms（-55%）**。
CPU 路径的杠杆不同，是**权重精度**（f32→Q8_0 近无损档直接 -54%），不是算子数；
图缓存在 CPU 上反而更慢（带宽），只在 GPU 启用。

> **测量噪声警告**：这台机器上 Vulkan 的 bs=1/bs=8 数字在不同时段会漂 ±10~20%（GPU 时钟/host 负载）。
> 只有**同一时段、相邻运行的配对比较**才有意义；跨时段/跨文档的绝对值请当作 ±20% 看待
> （§5 记录了一次 "同一配置 3.46 → 4.78 ms" 的漂移）。

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
5. 引擎 `device` 选择支持 `"cuda"` 别名（原来只认 `"vulkan"/"gpu"`；CUDA 后端本次未验证，原因见 §5）。

> 关键坑（已写在 README):`cpy` 写 cache 与 flash 读 cache 视图之间**必须有数据依赖边**，
> 否则 Vulkan 不插 barrier、顺序不定（表现为随机早停）。本次没有动这个结构，只是把
> "写 cache 的数据"从 `cont` 结果换成了视图。

## 2.5 阶段 2：两个融合算子（ggml 新增，CPU + Vulkan）

每层原本有 5 次"小算子"：`norm + mul(n1w) + add(n1b)` ×2 处、`add(f1b) + relu`。它们的张量都很小
（[512, 1, B] 或 [2048, 1, B]），但**每次都是独立 dispatch**，而且 `mul`/`add` 带 `[D]` 向量是
**广播型 elementwise**——Vulkan 上走的是通用 fastdiv 路径（每元素一次索引计算），比普通 elementwise 更慢。

新增两个算子（`patches/` 里的 audio-patch 现在含 118 个算子，CPU + Vulkan 双实现）：

| 算子 | 语义 | 替换 | 省 |
|---|---|---|---|
| `ggml_layernorm_affine(ctx,x,r,bias,wb,eps)` | `(x+r+bias-mean)/sqrt(var+eps)*w+b`，一次遍历（`r`/`bias` 可空）；`wb` 是 `[w;b]` 打包张量 | `add(r) + add(bias) + norm + mul + add` | 4 dispatch ×2 处/层 |
| `ggml_add_act(ctx,a,b,act)` | `act(a+b)`，b 按 ne0 广播（`act`: 0=none,1=relu,2=gelu_erf） | `add + relu/gelu` | 1 dispatch/层 |

**关于 `wb=[w;b]` 打包**：`ggml_vk_op_f32` 只有 4 个 src 槽，而"残差 + bias + 权重 + 偏置"要 5 个输入；
解法是在**引擎加载时**把每处 LN 的 `w|b` 读回 host 再拼成一个 `[2*ne0]` 张量（放进一个专用小 buffer，
AR ~200KB / BERT ~530KB），**不动 GGUF 格式也不动转换器**。这样 `out_b`、`f2b`、`attn_out_b`、
`ff2_b` 这些 per-ne0 的 bias 加法也被 LN 吃掉（每层再省 2 次 dispatch）。
副作用：Vulkan 首步 logits Δ 从 3.46e-3 降到 **2.39e-3**（少一次广播型 elementwise 往返）。

每层 dispatch 22 → **15**（AR）/ 18 → **12**（BERT）；Vulkan 端 2 个 shader
（`layernorm_affine.comp` 每 workgroup 一行、`add_act.comp` 每线程一元素；erf 用上游
`geglu_erf.comp` 同款 A&S 近似，因为 glslc 没有 erf 内建）。

**配对实测（同时段相邻运行，n_gen=100）**：

| 场景 | 未融合 | 融合 | 变化 |
|---|---|---|---|
| Vulkan bs=1 | 4.23 / 4.54 ms | **3.29 / 3.33 ms** | **-24%** |
| Vulkan bs=3 | 5.65 / 5.85 ms | **4.84 / 4.88 ms** | **-16%** |
| Vulkan bs=8 | 8.04 / 8.23 ms | **7.30 / 7.35 ms** | **-10%** |
| CPU bs=1（8 线程，只含第一版融合） | 20.97 ms | **19.29 ms** | -8% |
| BERT Vulkan f16（T=25） | 9.46 / 10.41 ms | 8.87 / **7.90** ms | 方向为正，噪声内 |
| BERT CPU（16 线程） | 110.0 / 111.2 ms | **108.2** / 123.7 ms | 噪声内（本机 BERT 效应小于漂移） |

- 收益集中在**小 batch（单流）**与 CPU：bs 越小，"固定开销"占比越高，省 dispatch 越值钱。
- BERT 在 Vulkan 上收益落在噪声里（它的 matmul 形状是 [1024,1024]×[1024,25]，TP=25 行，
  后端把 bias 融进 matmul 的路径不同），但 CPU 上有实打实的 -9~12%。
- 验收：`tests/test_fused_ops.cpp` 用 5 种形状（D=56/64/512/1024/4096）× 2 种激活，与"未融合链"
  逐元素对拍（CPU + Vulkan 全 PASS）；AR/BERT 的 golden 对拍数值不变（5.25e-6 / 1e-5，token 全一致）。
- 开关：默认开启（加载时用 `ggml_backend_supports_op` 探测，不支持则自动回退）；
  `GSV_NO_FUSE=1` / `GSV_NO_FUSE_LN=1` / `GSV_NO_FUSE_ACT=1` 可分别关闭（A/B 用）。

## 2.6 阶段 3：解码图缓存（固定 key 数 + `set_rows` 追加 KV）

阶段 2 之后每步 3.3 ms 里仍有 **0.78 ms 是"重建解码图"**（24%）——因为 KV 追加的
`cpy` 目标是"第 len 列"，len 每步变 → 图每步都要重建。做法是把这一点从图结构里拿掉：

1. **key 数固定**：flash 读 `concat(cache[0..Lmax), 新列)` → 形状恒为 `(HD, Lmax+1, NH, B)`，
   `Lmax` 是桶大小（`L + 25%` 余量，上限为已分配的 cache 长度）；
2. **KV 追加用 `ggml_set_rows`**（dst=cache, src=新列视图, idx=输入张量）：列号从"图的形状"
   变成"输入数据"，每步只上传一个 i32（`len`）。set_rows 的约束正好满足：
   只要求 `nb0 == type_size`（行内连续），我们的 strided 视图满足；
3. **mask 屏蔽无效槽**：`k < pad_len[b]`（左 padding）与 `k ∈ [cache.len, Lmax)`（未写入的槽，
   跨调用/跨桶后是旧数据），只放行历史 `[0, cache.len)` 与图内新列（索引 `Lmax`）；
4. 于是 `(Lmax, B)` 不变时**图与显存计划只建一次**，每步只做 `tensor_set(x/mask/idx)` + `compute`。

**收益（配对，n_gen=100）**：

| 场景 | 逐步建图 | 图缓存 | 变化 |
|---|---|---|---|
| Vulkan bs=1 | 3.40 / 3.39 ms | **2.29 / 2.44 ms** | **-32%** |
| Vulkan bs=3 | 4.40 / 4.42 ms | **4.09 / 4.29 ms** | -7~9% |
| Vulkan bs=8 | 6.74 / 6.93 ms | **6.56 / 6.56 ms** | -3~5% |
| CPU bs=1 / bs=3 / bs=8 | 16.0 / 23.6 / 41.8 ms | 16.5 / 26.9 / 58.0 ms | **+3% / +14% / +39%（更慢）** |

- 桶策略很重要：先用了"2 的幂桶"（64/128/256），bs=1 只到 -24% 且 CPU 慢到 +56%；
  换成"紧桶 `L + max(8, L/4)`"后每步多算的 key 几乎为零、重建也被摊掉，GPU 上变成 -32%。
- **只在 GPU 启用**：固定 key 数的代价是每步多拷/多算 `(Lmax-L)` 列的 K/V——GPU 带宽富裕、
  几乎免费，而它省下的建图占 24%；CPU 是带宽受限，实测净亏（bs=8 时 +39%），所以
  `use_graph_cache` 按设备类型判定（`GSV_AR_NO_GRAPHCACHE=1` 可强制关闭做 A/B）。
- **测试抓到的一个真 bug**：尾部屏蔽最初写成从 `L = cache.len+1` 开始，把"未写入的第
  `cache.len` 列"也放行了——GPU 上因为该列是 0 值侥幸通过 token 门，CPU 上 early-stop
  用例直接分歧（2/3）。改成从 `cache.len` 屏蔽后 CPU/Vulkan 全绿。这也说明
  `tests/test_ar_engine` 的 [2]/[3] 两条生成路径 + 100-token 门是必要的。

## 3. 权重精度档对单流延迟（本次新增的实测）

| 模型 | 体积 | CPU bs=1 | Vulkan bs=1 | 100-token 门（CPU / Vulkan） |
|---|---|---|---|---|
| f32 | 296 MB | 17.6 ms | 3.89 ms | — / —（参考） |
| f16 | 151 MB | 13.4 ms | 3.95 ms | PASS / PASS |
| **q8_attn_ffn（近无损档）** | **85 MB** | **9.1 ms** | **3.54 ms** | **PASS / PASS** |
| k_q6k | 67 MB | 9.0 ms | — | PASS（`docs/quant_ar.md`） |

（本表为阶段 1 之后的测量；融合算子会把 bs=1 的 Vulkan 数字再降 ~15~19%，见 §2.5。）

- CPU 是**权重带宽受限**：f32 每 token 读 ~300 MB，Q8_0 只读 ~90 MB → 延迟几乎按体积线性下降；
  8 线程与 16 线程无差别（17.6 vs 17.9 ms）进一步印证是带宽而非算力瓶颈。
- Vulkan 不受益于量化（每步固定开销不变，只是权重读得少一点）：3.89 → 3.54 ms（-9%）。
- 结论：**CPU 路径默认用 q8_attn_ffn（85 MB，近无损，已过 100-token 门）**；GPU 路径用 f32/f16 即可。

## 4. 剩余空间（已量化，未实施）

| 方向 | 预计收益 | 代价/风险 |
|---|---|---|
| ~~融合算子 `layernorm_affine` / `add_act`~~ | ~~bs=1 -17%~~ | **已在 §2.5 完成**：bs=1 -19%、bs=3 -14%、CPU -8% |
| ~~图复用~~ | ~~-25%~~ | **已在 §2.6 完成**（用 `set_rows` 把 KV 列索引变成输入，**没有**动 `tensor->data`）：Vulkan bs=1 -32% |
| ~~prefill（TTFT）~~ | **已完成 -63%**（见 §4.2）：host 侧 bert_proj 挪进图 | 剩余 15.5 ms ≈ 之前的「同形状热态」成本，再压空间小 |
| ~~FA 的 K/V 常驻 F16~~ | **不做**（结构上不成立） | 图缓存需要 `concat(缓存, 新列)`，而新列来自 F32 投影输出 → 缓存必须同为 F32；要 f16 缓存就得加一次 cast/层（+24 dispatch），省下的流量在 Vulkan 上本就几乎为零（FA 内核内部已转 f16），CPU 上也被 cast 抵消 |

### 4.1 prefill（TTFT）：剖析与优化（-63%）

`GSV_AR_PROFILE=1` 现在同时输出 prefill 的分段（建图 / 上传 / 图执行 / 回读）。最初的构成：

| 部分 | 耗时 | 说明 |
|---|---|---|
| `run_first` 图执行 | ~26 ms | 与**权重精度无关**（f32/f16/q8 端到端都是 41~42 ms）→ 既不是权重带宽也不是 matmul 吞吐；把每层 6 次 `cont` 换成视图直送（-144 dispatch）也**没有可测变化** → 每节点提交开销（~45 µs/节点 vs decode 6 µs） |
| **host 侧输入构建** | ~14~26 ms | `build_inputs` 里的 `bert_proj`：`[D,1024]×[1024,T]` ≈ 30M MAC，单线程三重循环，且 bert 特征是 **k 主序跨步读**（`r.bert[k*T+t]`，内层 stride = T）→ 缓存极不友好，实测代价远高于算力估计 |
| 建图 + 上传 + 回读 | ~2 ms | — |

**做的优化**：把 bert 投影挪进图。用与融合 LN 同一套「加载时打包」机制造一个**增广权重** `[W | b]`
（`[BERT_DIM+1, D]`），图上以 `feat [BERT_DIM+1, S, B]` 为输入（文本位置 = bert 特征 + 末行 1；
prompt/padding 位置 = 全 0，这样 bias 只在文本位置生效），于是 `x = xy_host + mul_mat([W|b], feat)`，
host 侧那条 30M MAC 循环整个消失。

| | 优化前 | 优化后 | 变化 |
|---|---|---|---|
| Vulkan 首步（56 token 前缀） | 41.7~45.8 ms | **15.5~16.6 ms** | **-63%** |
| CPU 首步（8 线程） | 153.4 ms | **96.3 ms** | -37% |

- 优化后的 15.5 ms 已经等于之前测到的「**同形状第二次 prefill**」成本（同一进程内实测 49 ms → 15 ms），
  说明剩下的基本是每节点提交/分配开销，继续压的空间小；
- 记录一个已量化但未做的可选项：把 prefill 图按 S 分桶（形状稳定 → 描述符/显存计划可复用），
  预计 15.5 → ~8 ms（约整句 2.5%），收益低于复杂度，暂不做。

### 4.2 测量噪声记录（重要）

同一份二进制、同一配置重复跑，Vulkan bs=1 出现过 **3.46 / 3.91 / 4.06 / 4.39 / 4.78 ms** 的漂移
（跨时段；同一时段内重复一般 ±3%）。原因是这台机器上 bs=1 的每步只有 ~4 ms，GPU 时钟与 host 负载
都会直接落在数字上。所以本文件的所有结论都基于**同时段配对测量**；跨时段比较请忽略 ±20% 以内的差异。
判定方法：先跑 A、再跑 B、再跑 A（ABA），若 A 的两次差异小于 B 的偏移才认为 B 有效。

## 5. CUDA 后端（本次未做，原因记录）

**不做**：llama.cpp 的 CUDA flash attention 内核要求 `head_dim ∈ {40,64,72,80,96,112,128,256,...}`，
而 AR 的 head_dim = 32（D=512, NH=16）→ `ggml_cuda_flash_attn_ext_supported()` 返回 false，
引擎直接调 `ggml_backend_graph_compute`（无 scheduler 回退）时会在 `fattn.cu` 里 `GGML_ABORT`。

- 想走 CUDA 只有两条路：把注意力换成等价的 `mul_mat + soft_max + mul_mat`（实测过：CPU 上
  逐 token 一致，Vulkan 上 logits Δ 只从 3.46e-3 降到 2.3e-3 —— 说明 Vulkan 的误差主要来自
  matmul 而非 FA 的 f16 K/V，且要付 +2 dispatch/层的代价），或者改 head 划分。**两者收益都不成立，
  按项目所有者决定放弃 CUDA 线**（Metal/Vulkan/CPU 已覆盖目标平台）。
- 构建踩坑留档：本机只注册了 CUDA 13.0 的 MSBuild 集成（CUDA 13 不支持 VS2019），用 VS 生成器时
  MSBuild 会用 nvcc 13.0 编译 `.cu`；且 `fattn-*/mmq-*` 模板实例文件在 -j16 下 30 分钟编不完。
  若将来重试：用 Ninja 生成器让 CMake 直接驱动 nvcc 12.6。

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
