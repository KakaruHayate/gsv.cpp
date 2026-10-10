# DiT / CFM（V5 生成器）ggml 实现

日期：2026-10-09 · 机器同前（Xeon E5-2675 v3 + RTX 2070 + Vulkan）

## 0. 结构与语义侦察（先钉死参考实现）

V5 的 mel 生成 = `synthesize_v5_mel` → `CFMV5.inference`（`repo/GPT_SoVITS/module/models.py:1118` +
`models_v5.py`）。estimator 是 **F5 风格的 DiT**（22 层、dim 1024、16 头 × head_dim 64、FFN 2048），
v5 变体 `use_step_embedding=False`、`noise_temperature=0.875`。

用 torch 探针逐条钉死的语义（全部写进 `tools/dump_golden_dit.py` 头部注释，对拍时不可改动）：

1. **RoPE 只作用于 q/k 的前 64 维**（= attention 的 0 号头）：x_transformers 的
   `apply_rotary_pos_emb` 用 `rot_dim = freqs.shape[-1] = dim_head = 64` 截断，其余 15 个头不旋转。
   角度表 = `n * inv_freq[i]`（同对相邻两维共享），`rotate_half` 是交错（GPT-NeoX）配对。
2. **`inv_freq` 用 checkpoint 里的 fp16 存储值**：与 fp32 公式差 ~1.6e-4，用公式重算会破坏大 n 的相位。
3. 时间嵌入正弦 = `cat(sin, cos)(1000 * t * exp(-i*log(10000)/127))`，i<128；MLP 256→1024→1024。
4. **FFN 的 GELU 是 tanh 近似**（`approximate="tanh"`）；ConvNeXtV2 的 GELU 是精确 erf 版。
5. **static cache**（`prepare_static_cache`）：`condition / negative_condition [1024,T]` 一次算好；
   每步只算 `F.linear(x, proj[:, :100]) + static + conv_pos_embed + 22 块`。卷积位置编码里
   `masked_fill` 在 conv **前、后各一次**，且第二次前面还有一次 **Mish**（`Conv1d→Mish→Conv1d→Mish`）。
6. **mask 语义**：SDPA bool mask（padding key 全屏蔽）；attention `to_out` 之后还有一次
   `x.masked_fill(~mask, 0)`（pad 行清零）；CFM 循环里 `x[..., :prompt_len] = 0` **每步都做**。
7. CFM 循环：`x = randn * 0.875`；`x += (1/steps) * v`；cfg 分支 `v = v + (v - neg) * cfg_rate`；
   `sample()` 返回 `result[..., P:]`（v5_inference 的裁剪）。

## 1. 实现

- `tools/convert_dit.py` → `models/gsv-dit-f32.gguf`（1.35 GB）。要点：
  - `input_embed.proj.weight [1024,712]` 拆成 `weight_mel [100,1024]` + `weight_ctx [612,1024]`；
  - conv_pos（`Conv1d 1024→1024 k=31 groups=16`）重排为 `[G, K*ICg, OCg]`（行序 `r = ic*K + kw`，
    与 `im2col_fast_1d` 的输出行序一致）；dwconv 转置为 `[K, C]`；
  - `text_embed.freqs_cis` 是非持久 buffer，按 torch 同式在转换时重算（fp32）。
- `src/gsv_dit.{h,cpp}`：三张图 + host 侧 CFM 循环。
  - **cache 图**：`prompt_x + mu → text_embed(4×ConvNeXtV2) + concat + proj → condition / negative_condition`，
    `prepare()` 时算一次并拷入持久张量（同 T 重复 prepare 只重算这一步）；
  - **step 图 ×2（pos/neg）**：`linear(proj[:100]) + static → conv_pos(两层分组 conv + Mish) → 22×DiTBlock`
    （AdaLN 调制 + 16 头注意力 + head0 RoPE + FFN）`→ norm_out → proj_out`；
  - **16 头注意力融合（2026-10-10）**：QKV 三次整块 matmul（原为每头 3 个小 matmul ×16）→ head0 前
    64 行切片 `cont` 后 rope64、再与其余 960 行 `concat`（rope64 内部 reshape 要求连续，直接切
    strided view 会断言）→ q/k/v 以 `(hd, T, nh)` 步长视图直送 `ggml_flash_attn_ext`
    （F16 mask `[T,T,1,1]`；FA 输出元素序 = `hd + 64·nh + 1024·t`，即 [DIM, T] 行主序，直接 reshape）
    → to_out + pad 行掩码。**单步图节点 7783 → 1975（−75%）**；`GSV_DIT_ATTN_LEGACY=1` 回退旧逐头路径。
    精度：CPU 与旧路径接近逐位一致（FA F32）；Vulkan 数值不变（FA 的 F16 K/V 误差被既有 F16 权重噪声
    掩盖）——两后端 110 项对拍全 PASS。
  - **三张图各自独立 gallocr**（踩坑 #1，见下），构建时规划一次，运行期不再重规划；
  - 对外 API 一律 torch 布局：`x/prompt/mu/vel` 都是 `[1,C,T]` 字节序；`sample()` = CFM.inference 等价，
    `synthesize()` = 分块 + rolling prompt（`V5_REFERENCE_FRAMES=500`、`TAIL=32`、块 640）等价。
- `tests/test_dit.cpp`：单步三用例（t96 / pad `x_lens<T` 掩码 / t0）+ CFM 4 步 cfg0 + 8 步 cfg1.30
  （pos/neg 逐次对拍）+ 分块（block=5 强制 3 块）+ `GSV_DIT_BENCH` 生产尺寸基准；
  `GSV_DIT_DEBUG=1` 下约 25 个中间张量可用 `debug_read()` 读回逐层对拍
  （`dbg_te_pos/te_cn0..2`、`dbg_x_lin`、`dbg_convpos`、`dbg_b0_*`、`dbg_q0_raw/roped` 等）。

## 2. 对拍结果

golden（`tools/dump_golden_dit.py` + `dump_golden_cfm.py`，torch 真实实现逐步拦截）：

| 用例 | CPU (f32) | Vulkan |
|---|---|---|
| 单步 vel（t96/pad/t0） | 1.7e-4 / 2.8e-4 / 1.3e-4 | 1.2e-2 ~ 2.3e-2 |
| convpos | ≤ 1.0e-4（refmax 132） | 3.9e-1 |
| b21_out（22 层累计） | 7.3e-1（refmax 3556） | 2.9e+1 |
| q0_raw/roped（head0） | 3.3e-6 | 1.6e-2 |
| AdaLN chunk（gate/shift/scale） | ≤ 3.6e-7 | ≤ 3.0e-7 |
| `te_pos`（mu+pos emb） | **0（逐位一致）** | 0 |
| CFM s4c0 端到端 | 1.2e-4 | 6.6e-3 |
| CFM s8c13（cfg，16 次正/负前向） | 5.3e-4 | 1.5e-2 |
| 分块 synthesize（3 块 rolling） | 9.2e-5 | 4.9e-3 |

判定阈值：`d < 2e-2 × max|ref|`（尺度自适应），CPU 与 Vulkan 均 **ALL PASS**（110 项检查）。
（2026-10-10 注意力融合后重跑两后端仍全 PASS；各探针数值与上表同量级，CPU vel 1.61e-4。）

## 3. 生产尺寸基准（T=1000，prompt 500，单步，v5turbo 4 步形态）

| 后端 | 旧（逐头注意力） | 融合注意力（现默认） | vs torch |
|---|---|---|---|
| torch CPU f32（16 线程） | 2108.7 | — | 参照 |
| **ggml CPU f32（16 线程）** | 2992.8 | **1968.3** | **−34%，反超 torch CPU（1.07×）** |
| torch CUDA f16（RTX 2070） | 113.3 | — | 参照 |
| **ggml Vulkan（RTX 2070）** | 107.0 | **62.9** | **−41%，快 torch CUDA 1.8×** |

（同段配对 ABA，a-b-a 漂移 <2%。此前"Vulkan ≈ torch CUDA、CPU 1.44× 落后"的结论已随注意力融合翻转。）

> CPU 注意 ggml 默认 4 线程：不设 `GSV_DIT_THREADS` 时 ~6356 ms，16 线程 3039 ms，32 逻辑核反而回退（3564 ms）。
> torch 侧脚本 `tools/bench_dit.py --device cpu --threads 16`（同 workload）。

> **coopmat2**：本段从首测起就是默认口径（未设 `GGML_VK_DISABLE_COOPMAT2`）。受控 A/B（T=1000 单步）：
> coopmat2 关（回退 KHR_coopmat）177.8 ms、连 KHR_coopmat 也关 346.8 ms、开 **107.7 ms** ——
> 张量核已吃满 3.2×，且 110 项对拍在 coopmat2 开的条件下全 PASS。其他组件的 coopmat2 复评
> （BERT/HuBERT/条件段，2026-10-10）以本段为先例，见各 doc 复评节；AR 是唯一必须关闭的（logits Δ 6×）。

## 4. 踩坑记录（本次定位的 4 个根因，值 O(1) 级别，全部由逐层探针定位）

1. **gallocr 共享**：cache 图与 step 图、pos/neg step 图之间共享一个 `ggml_gallocr` 时，后一张图的
   auto-reserve 会 free/重建整块 vbuffer 并复位 tallocr，先建图张量的 `data` 指针全部悬空
   （表现为 `ggml_gallocr_alloc_graph` 内部崩溃 / tensor_set 写已释放内存）。改为三张图各自独立
   gallocr、构建期各规划一次。
2. **分组 conv 的组内通道偏移少乘 T**：`view_2d(x, T, ICG, nb1=T*4, off=g*ICG*4)`——Domain A `[T,C]`
   下通道步长是 `T*4` 字节，正确偏移 = `g*ICG*nb1`。错值让 15/16 个组都读到错位的通道窗口
   （convpos 误差直接 O(refmax)，且误差集中在组边界通道）。
3. **GRN 的通道均值**：torch `Gx.mean(dim=-1)` 是全通道标量均值；`ggml_mean` 只归约 ne0，而
   `gx` 的 ne0=1（`[1,C]`）→ 退化成恒等，Nx≈1。改用 `ggml_sum(gx) * (1/DIM)`（1/1024 为 2 的幂，
   与 torch 除法逐位一致）。
4. **attention to_out 后的 masked_fill**：torch AttnProcessor 末尾对 pad 行清零；漏掉后 pad 行
   的注意力输出会一路累积（b0_attn ~4、22 层后 O(1e3)）。

另外对拍脚本侧两个约定修正：golden 的 `noise` 是**乘 0.875 之前**的原始 randn（`CFMV5.noise_temperature`）；
`CFM.inference` 返回的是**完整 x**（含 prompt 列），`sample()` 输出是 `result[..., P:]`。

## 5. cache-dit（DBCache step 间缓存，可选特性）

按 vipshop/cache-dit 上游语义实现（本地对照 `cache-dit-main` 的 `CachedBlocks_Pattern_Base.forward`；
GAME/game.cpp 的移植是它在离散扩散上的特例，本项目直接对上游）：

- **三段**：front（前 Fn 块，每步必算，同时产出判据量）/ middle（其余块，命中时跳过）/ back（Bn，未启用）。
- **判据**：Fn 残差 `R = x_F - trunk` 的相对 L1 diff `fd = Σ|R - R_prev| / (Σ|R_prev| + ε)`
  （device 侧归约，每步只回读 1 个 float；R 缓冲按上游 `downsample_factor` 在时间轴下采样 ds=4）。
- **复用**：Mn 残差 `x_M - x_F`（上次全算时存档）；命中时 `x_mid = x_F(当前步) + Mn残差(上次全算)`。
- **闸门**：warmup（默认 8 步强制全算）、max_cached_steps / max_continuous_cached_steps /
  max_accumulated_residual_diff_threshold（默认全关，与上游一致）。
- **cfg 分支独立记账**（enable_separate_cfg + cfg_diff_compute_separate）：pos/neg 两套缓冲与计数器。
- 工程：front/add/mid/head 每张段图独立 gallocr；全部缓存张量 device 常驻
  （NONE 张量 + `ggml_cpy` 跨图写，段图之间零 host 搬运 —— 同 game.cpp 的 PersistentStage 方案）。

配置：`gsv_dit_cfg.dbcache_*`（测试串 `GSV_DIT_DBCACHE="fn=8,thr=0.12,warmup=8,ds=4"`；
`GSV_DIT_DBCACHE_TRACE=1` 逐步打印 HIT/MISS/fd 与分段耗时）。

### 消融（T=96, s32c13 = 32 步 cfg=1.30, sample() 端到端, 与无缓存输出对比；数含注意力融合）

| 档 (Fn/thr) | 命中 (pos/neg) | vs 无缓存 max/mean\|d\| | CPU 时间 | Vulkan 时间 |
|---|---|---|---|---|
| 无缓存 | — | — | 28178 ms | 1096 ms |
| F8/.08 | 16/16 | 0.041 / 0.0062 | 19522 ms（**−31%**） | 1093 ms（±0） |
| F8/.12 | 18/18 | 0.064 / 0.0097 | 18456 ms（**−35%**） | 1024 ms（−7%） |
| F12/.12 | 20/21 | 0.122 / 0.0137 | 20359 ms（−28%） | 1137 ms |
| F16/.20 | 22/22 | 0.201 / 0.0295 | 23089 ms（−18%） | 1307 ms |

（同配置的**逐头旧路径**下 s32c13 参考 CPU 为 33355 ms —— 即注意力融合本身使 e2e 参考 −16%；
T=1000 单步的融合增益见 §3。）

生产尺寸（T=1000，Vulkan，受控三档对照）：单体 **62.1 ms/步**；全 miss 分段 64.3 ms/步
（**分段开销只剩 +3.5%**，融合前为 +14~24% —— 单步节点数 7783→1975 后 submit/fd 的固定开销大减）；
高命中场景（thr .12，90% 命中）**27.6 ms/步（−56%）**。真实扩散分布 `s32c13_big` 32 步 e2e：
参考 4166 ms → F8/.08 3107 ms（−25%）/ F8/.12 2961 ms（**−29%**，命中 56%），质量 mean|d| 与
融合前一致（0.0063 / 0.0100）。**结论：注意力融合后 cache-dit 的相对收益从"−7%"升到
"−25~29%"，32 步档建议开（F8/thr 0.12）；4 步 turbo 依旧不适用。**

**4 步 turbo（v5turbo）：不要用 cache**。v5turbo 是 **DMD（Distribution Matching Distillation）
训练的 shortcut 模型**（开发者确认；§见调研文档），步间 Δt=0.25，fd 远超任何合理阈值：
F8/.10 全程 0 命中（输出与无缓存逐位一致）；激进档（F12/.15、F8/.25）勉强命中 1-2 次，
mean|d| 立刻跳到 0.021~0.026（电机可闻级别劣化）。DMD 蒸馏出的轨迹每步都是"特制"的，
用邻近步残差近似天然不适用——与调研文档 §3.5.4 的判断（"v5turbo 4 步与 cache-dit 无关"）一致。

## 6. 量化扫描（凹档）

`tools/convert_dit.py --spec "attn=f16,ffn=f16,..."` 已支持按组分档
（组：attn / ffn / adaln / proj / text / time / conv；bias、grn、ConvNeXt 的 LN、dwconv、freqs_cis
恒 F32）。`quantize_gguf` 增加了 `dit.*` 分组。验收 = test_dit 全套对拍（尺度自适应 2e-2 阈值）。

| 档位 | 体积 | CPU t96 vel max\|d\| | CPU 32 步逐步 max\|d\| | 判定 |
|---|---|---|---|---|
| F32（基线） | 1289 MiB | 1.7e-4 | 1.2e-3 | — |
| attn F16 | 1113 MiB | 2.3e-4 | 1.2e-3 | PASS |
| ffn F16 | 1113 MiB | 7.4e-4 | 2.3e-3 | PASS |
| attn+ffn F16 | 937 MiB | 3.8e-4 | 2.2e-3 | PASS |
| +adaln F16 | 669 MiB | 5.3e-4 | 4.2e-3 | PASS |
| **全 F16（lin=f16）** | **651 MiB（1.98×↓）** | **9.4e-4** | **6.4e-3** | **PASS（最小档）** |
| attn+ffn+adaln Q8_0 | 358 MiB | 2.5e-2 | 1.0e-1（32 步内 6 项失败） | **FAIL**（临界，同 HuBERT 结论） |
| 同 Q6_K | 283 MiB | 7.7e-2 | 大幅失败 | FAIL |

- **最小档 = 全 F16（651 MiB）**。Vulkan 上 F16 与 F32 **逐位一致**（Vulkan 内部本就 F16 计算，
  同 cond 段结论），速度不变（127.8 vs 129.8 ms，噪声级）；CPU 全 F16 略慢 ~7%
  （3559 vs 3333 ms，F16 权重的 vec_dot 转换开销）。Q8_0 的误差与 HuBERT 同因
  （ggml 量化 mul_mat 连激活一起量化），attn 尤为敏感（q0 通道 1.5e-1）。
- 扫描暴露并修复的引擎问题：attention 的 per-head 权重切片与 conv_pos 组偏移原先按
  "元素数×type_size" 算字节偏移（F32 下碰巧正确、F16 下正确但**量化块类型越界崩溃**），
  改为按 `nb[1]`/`nb[2]` 行/组步长计算（F32/F16/量化统一）。`quantize_gguf` 的 F16 源量化路径
  原先把 F16 数据按 `float*` 读（2 倍越界读 → 段错误），已补 F16→F32 反量化。

## 7. 后续

- vocoder 保持 ONNX fp32 不量化（既定约束）；端到端引擎接线（A~E 项）见调研文档 §7。
- cache-dit 保留为 32 步档可选特性（默认关）；若要进一步压 head 开销，方向是
  fd 判定与 add 融合进 head 图（少一次 submit）。
- ~~16 头注意力融合~~ **已完成**（2026-10-10，−34% CPU / −41% Vulkan，见 §1/§3）；之后同类余量：
  q/k/v 改 `ggml_cpy` 原地写 head0（省两次 `cont`+`concat`）、prompt 列的 q/out 投影按需裁剪（理论 ~20%）。
