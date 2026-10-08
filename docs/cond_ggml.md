# 条件段（V5 `decode_encp` 链）ggml 实现

日期：2026-10-07 · 机器同前（Xeon E5-2675 v3 + RTX 2070 + Vulkan）

## 0. V5 条件段的真实调用链（先侦察清楚）

V5 推理入口 `TTS_infer_pack/TTS.py::using_vocoder_synthesis`（`version ∈ {v5dev, v5turbo}`）：

```python
fea_ref, ge = vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_spec)        # 参考侧
fea_todo, _ = vits_model.decode_encp(semantic_tokens,        phones,        refer_spec, ge)    # 目标侧（复用 ge）
mel = synthesize_v5_mel(vits_model, fea_ref, fea_todo, mel2, sample_steps, cfg_rate)           # DiT (CFM)
wav = vocoder(denorm_spec(mel))                                                                # HiFi-GAN
```

`SynthesizerTrnV3.decode_encp`（**v5 用这个类**，`TTS.py:634`）的模块顺序与形状
（权重全部来自 `gsv-v5-pretrained/s2Gv5turbo.pth`，该文件 **同时含 DiT 与条件段**：`cfm` 363 / `enc_p` 235 /
`wns1` 55 / `ref_enc` 18 / `quantizer` 4 / `bridge` 2 / `ssl_proj` 2 / `linear_mel` 2 张量）：

| 步骤 | 模块 | 权重形状（关键） | 算子 | 每句调用 |
|---|---|---|---|---|
| 0 | `ref_enc` = `MelStyleEncoder(704→512)` | `spectral.0.fc [128,704]`（对**频带**做 Linear）, `temporal.{0,1}.conv1 [256,128,5]`, `slf_attn.{w_qs,w_ks,w_vs,fc} [128,128]`（**单头**自注意力）, `fc.fc [512,128]` | mul_mat + conv5 + attn + pool | 1（可得 `ge` 后缓存） |
| 1 | `quantizer.decode(codes)` = **RVQ** | `vq.layers.0._codebook.embed [1024,768]`，**n_q=1** | **纯 `get_rows`** | 1（目标侧） |
| 2 | `F.interpolate(×2, nearest)`（25hz→50hz） | — | `ggml_interpolate(NEAREST)` | — |
| 3 | `enc_p`（`TextEncoder`：`ssl_proj [192,768,1]` + `encoder_ssl`/`encoder_text` + **MRTE** 交叉注意力） | `attn_layers.*.emb_rel_k/emb_rel_v [1,9,96]`（**相对位置**注意）+ `conv_q/k/v [192,192,1]` | mul_mat + **rel-pos attn** + concat | 1 |
| 4 | `bridge` = `Conv1d(192→512,k=1) + LeakyReLU` | `bridge.0.weight [512,192,1]` | mul_mat + leaky_relu | 1 |
| 5 | `F.interpolate(×2, nearest)` | — | idem | — |
| 6 | `wns1` = `Encoder(512,512,512,k=5,dil=1,n=8, gin=512)` | `enc.in_layers.{i}.weight_g/_v [1024,512,5]`（**weight-norm，需物化**）, `pre/res_skip` 1×1 conv | conv5-direct + 1×1 conv + tanh/sigmoid 门控 | 1 |

**精度事实**：`TTS.py:673` `if is_half and device != "cpu": vits_model.half()` → 官方管线里
**整个条件段跑 fp16**（而且 `s2Gv5turbo.pth` 里这些权重本身就是 fp16）→ 对这些模块 **fp16 就是"参考精度"**，
不需要像 vocoder 那样死守 fp32。

## 1. 选型：先做 RVQ（延续性最高）

| 模块 | 与已完成工作的延续性 | 需要的新东西 |
|---|---|---|
| **RVQ** ✅ 已实现 | **最高**：`get_rows`（＝AR/BERT 的 embedding 查表）+ `ggml_interpolate(NEAREST)`（原生算子，Vulkan 有 `upscale_nearest` 管线）；**布局无关**；而且是**整条链的入口**，输入正是我们 AR 产出的语义 token | 无 |
| bridge | 高：1×1 conv = `mul_mat` + bias + leaky_relu（3 dispatch，零新算子） | 无（但要等 enc_p 产 `x`） |
| ref_enc | 中高：mul_mat + **conv5**（patch 的 `CONV_DIRECT_1D`，从未用过）+ 单头 attn（flash ✓） | 首次使用 conv 算子 + **channels-last `[T,C]` 约定** |
| wns1 | 中：conv5-direct + 1×1 conv + 门控 | 同上 + weight-norm 物化（转换器已做） |
| enc_p | 中：相对位置注意力（patch 有 `REL_POS_BIAS`，从未用过）+ MRTE 交叉注意力 + 两个编码器 | 同上 + 相对位置语义核对（工作量最大） |

结论：**RVQ 是延续性最高且最"入口"的一块**（零新算子、布局无关、输入与 AR 直接对接），
所以先做它；bridge 随后（2 行）；conv 系（ref_enc/wns1/enc_p）要先定 channels-last 约定，放后面。

## 2. RVQ 实现与验收

`src/gsv_cond.{h,cpp}`（这个模块将成为条件段的容器，后续 bridge/ref_enc/wns1/enc_p 都加进来）：

```cpp
gsv_cond * m = gsv_cond::load("models/gsv-cond-f32.gguf", {.device = "vulkan"});
m->rvq_decode(codes, T, /*upsample_x2=*/true, out);   // out: [768, 2T], idx = d + t*768
```

- 转换器 `tools/convert_cond.py`：把 `s2Gv5turbo.pth` 里**整个条件段**导出为 `models/gsv-cond-f32.gguf`
  （177 MB；`cfm`=DiT 留给 DiT 的转换器）——wns1 的 `weight_g/weight_v` 在转换时物化为 `weight_w`（55→38 张量）。
- golden `tools/dump_golden_rvq.py`：4 个用例（zeros / ramp / rand / **AR golden 的真实 40 个 token**），
  dump `codes [T]` / `quant [T,768]` / `up [2T,768]`（已转成 ggml 展平顺序，便于逐位比较）。
- 对拍 `tests/test_cond_rvq.cpp`：**CPU 与 Vulkan 都 bit-exact（0 个不等元素）** —— gather 与 nearest
  都是纯搬运，所以这里用"逐位一致"而不是容差。

## 3. 顺带修的两件事

1. **`models/s2Gv5turbo.pth` 本地副本是坏的**（zip 里 `data.pkl` CRC 错，`torch.load` 报
   `unpickling stack underflow`）→ 已从 hf-mirror 重新下载并校验（766,406,144 B，`zipfile.testzip()` 通过，
   681 张量可正常加载）。**这是条件段全部工作的前置**。
2. `MRTE` 在 v5 里**不是独立模块**，它就在 `enc_p` 内部（`enc_p.mrte.*`），所以"实现 MRTE"=实现 `enc_p` 的一部分。

## 4. 后续顺序（按延续性）

1. ~~RVQ + ×2 nearest~~ ✅
2. `bridge`（1×1 conv + LeakyReLU，零新算子；等 `enc_p` 的 `x` 到位后可立刻接上）
3. `ref_enc`（Linear + conv5 + 单头自注意力 + pooling；产出 `ge`，参考侧一次、可缓存）
4. `wns1`（conv5-direct ×8 + 门控 + gin 条件注入；链条最后一段，输出直接喂 DiT）
5. `enc_p`（相对位置注意力 + MRTE + 双编码器；工作量最大，放最后）

其中 3/4/5 需要先定一个**布局约定**：patch 的 conv 算子（`CONV_DIRECT_1D` / `IM2COL_FAST_1D`）按
`x [T, C]`（时间在最内层）设计，而我们 AR/BERT/RVQ 一路用的是 `[C, T]`（通道在最内层，适配 `mul_mat`）。
两条路可选：(a) 在模块边界做一次 transpose（每模块 1~2 次，代价小、语义清楚），
(b) 权重转置存储 + 全程 `[T,C]`（省 transpose 但所有张量语义反转）。建议 (a)，在写 `ref_enc` 时定稿并记录。

## 5. HuBERT（音频特征前端，gsv.cpp 路线）✅

> 背景另一条线（host C++ 版 test_hubert.cpp，ORCATERM）已 parity PASS 但 401ms；本节记录 ggml 图实现。

结构（chinese-hubert-base, wav2vec2 风格）：
- 7 层 conv 前端 k=[10,3,3,3,3,2,2] s=[5,2,2,2,2,2,2]，conv0 后 GroupNorm(**512**) + gelu，其余层 gelu
  - **注意**：transformers 的 `HubertGroupNormConvLayer` 用的是 `GroupNorm(num_groups=512, num_channels=512)`
    —— 即**逐通道**（over T）归一，不是 GN(32)！GGUF 里名字叫 `feat_conv.0.norm_w/b`。
- `feature_projection`：LN(512) + Linear(512→768)
- `pos_conv`：grouped k=128 g=16 weight-norm（导出时已折叠）；**groups=16 的 conv 不适配 ggml_conv_1d，
  保留 host 实现**（49 帧 ~230M MAC，OpenMP ~15ms，未来可用 batched mul_mat 重写）
- 12 层 post-LN transformer：d=768, heads=12, FFN 3072, gelu_erf —— 与 BERT 同款（fused QKV 可后做）

### 实现要点（踩坑记录）

1. **conv 权重布局**：ggml im2col 要求 a 按 `ne=[K, IC, OC]` 列主 = **k 最内**。
   而 host conv 语义（ORCATERM converter 的 `permute(2,1,0)` + gguf-py 反转）字节是 k 最外。
   两者相反！→ load() 时对 7 个 conv 权重做一次 host 转置回写（k↔oc 互换），图内 reshape 为 [K,IC,OC]。
2. **字节序恒等**：`[T,D] 行主`（torch/torch golden）与 `[D,T] 列主`（ggml）**字节序相同**，
   图输入输出直接整块 memcpy，不要写转置循环（写了必错）。
3. **FA mask 是 input 张量**：galloc 不清零 → 必须显式写 0，否则 softmax 读到垃圾出 NaN。
4. **每图独立 gallocr**：共用 gallocr 时第二个图 alloc 触发扩容会 free 旧 buffer，
   第一个图所有节点指针悬空（新缓冲复用同一段虚拟内存），计算互相涂写。
5. **host 不得直接解引用 GPU 权重**（`t->data`）：Vulkan 下会段错误；一律 `ggml_backend_tensor_get`。
   pos_conv/enc_norm 的 4 个权重每次 encode 拷回 host（~3.4MB，可忽略）。
6. **C++ 实参求值顺序**：`f(need(nm), need((snprintf(nm,...), nm)))` —— MSVC 从右往左求值，
   右半先覆盖共享的 `nm`，左半拿到错权重。名字格式化+查表必须封装成单次调用（见 `W()` lambda）。

### 对拍与性能

精度（vs torch golden）：

| 后端 | max\|d\| |
|------|-----------|
| host C++（ORCATERM） | 8.6e-6 |
| **ggml CPU（全 F32，含 F32 im2col）** | **6.68e-6** |
| ggml Vulkan（FA 走 F16 K/V） | 1.49e-2 |

耗时（同一会话、同一输入、warmup 后 30 次迭代取平均；RTX 2070 + i9 32 逻辑核）：

| 实现 | 后端 | avg | min | 相对 |
|------|------|-----|-----|------|
| **ggml（本实现）** | **Vulkan** | **7.1 ms** | **6.9** | **4.0× 快于 torch CUDA** |
| torch（transformers，F32） | CUDA | 28.4 ms | 25.6 | |
| torch（transformers，F16） | CUDA | 27.8 ms | 26.5 | F16 无收益 |
| **ggml（本实现）** | **CPU 24T** | **106 ms** | **101** | — |
| *torch 参考（同机）* | *CPU* | *87 ms* | *76* | *torch 快 ~1.2×* |
| host C++（ORCATERM，OpenMP 全核） | CPU | 289 ms | 273 | ggml 快 2.7× |

> 数值取自机器空载时段；测量期间机器常有外部负载（avg 可高出 30-50%），故以 min 为准。

**CPU 对 CPU 的分段对比**（这是唯一还有差距的地方）：

| 段 | ggml CPU | torch CPU | 结论 |
|----|----------|-----------|------|
| CNN 前端 + feat_proj | ~40 ms | 19.7 ms | **慢 2×** ← 唯一落后项 |
| pos_conv + enc LN | 2.9 ms | 6.2 ms | 快 2× |
| transformer 12 层 | ~49 ms | 52.8 ms | 略快 |
| 合计 | ~106 ms | 87 ms | 差 1.2× 全在 CNN |

CNN 落后的原因：oneDNN 用的是直卷积 + AVX-512 级阻塞；我们走
`im2col(fast 1D) → F32 GEMM`，GEMM 吞吐 ~123 GFLOPS vs oneDNN ~250 GFLOPS。
ggml CPU 后端的 GEMM 效率不是本仓库能改的，故此项暂到此为止。

**CPU 对 CPU**：ggml 图 139 ms vs host C++ 289 ms（2.1×）。host 版瓶颈在 cnn 137 / tr 141 ms，
ggml 版把 transformer 压到 ~51 ms 的同时 CNN 也从 137 → ~78 ms。

**CUDA 对 Vulkan**：torch CUDA 25.7 ms vs ggml Vulkan 12.0 ms —— **ggml Vulkan 快 2.1×**。

拆解 Vulkan 的 12 ms（`GSV_HUBERT_TIMING=1`）：

| 段 | 耗时 | 说明 |
|----|------|------|
| g1 CNN 前端（GPU） | ~2 ms | 7 层 conv 走 im2col(F16)+mul_mat |
| pos_conv + enc LN（**host**） | ~3.5 ms | 见下方 GEMM 结构说明 |
| g2 12 层 transformer（GPU） | ~5.7 ms | 每层 ~0.48 ms，现在最大的单项 |

### 优化历程（36.2 ms → 7.8 ms，4.6×）

前两轮（非算法性瓶颈，-25 ms）：

1. **每帧从 GPU 回传 18.9 MB pos_conv 权重**：为 Vulkan 正确性引入的 `tensor_get` 被放在
   `encode()` 里（每次调用执行）。改为 load() 时拷回一份 host 常驻副本。Vulkan −20 ms。
2. **pos_conv 权重访问跨 3 KB 步长**（旧布局 `[k][ic][g*48+oc]`，每次只用 32 B/64 B line，
   预取失效）：load() 时重排为 `[g][ocg][ic][k]`（k 最内），内核按
   「任务 = (组, 8 个输出通道)，o 外层、`(ic,k)` 内层，累加器 = 全部 49 帧（7 个 YMM）」实现
   —— 每个 w 元素只读一遍、纯顺序流。OMP=24：10.5 → 3.1 ms。

第三轮（-4 ms）：

3. **pos_conv 工作区常驻**：`xt`(540 KB) 与输出缓冲原本每次 encode 分配+清零；
   改为成员缓冲、padding 区只建一次、输出零清零（内核完整覆写）。
   transpose 段 0.91 → 0.02 ms。
4. **CPU 路径的 host 段仍在串行等待**（GPU 全闲）：把 pos_conv 整体搬进 g1 图（仅 GPU 后端，
   `GSV_HUBERT_POS_GPU=1`，默认按设备自动）：
   - `layernorm_affine` 直接输出 post-LN 的 encoder 输入，host 段归零
   - 16 个分组 `conv_1d`（`pos_w_perm` 的 `[oc][ic][k]` 布局 == conv_1d 需要的 `ne=(K,IC,OC)`
     列主，且每组的权重/输入切片都是连续 slab）、对称 padding 64 + 裁掉末帧、`cont+concat`
     链组装出 `[768, 49]`、bias+gelu 在图内
   - 省掉 GGUF 里 pos_conv.w 的 18.9 MB 上传（直接从文件读字节到 host 构建重排表）
   - 精度不变：CPU 图内路径 1.785e-3，Vulkan 1.487e-2（仍由 FA 的 F16 K/V 主导）
5. **g2 图瘦身**（`GSV_HUBERT_DIET`，默认开）：FA 的三个 q/k/v `cont` 去掉（视图直送，FA 两端
   都按步长寻址）；两处 post-LN 的 `norm+mul+add`（3 op）换成 `layernorm_affine`（1 op，bias
   吸收进残差）；ffn1 的 bias+gelu 用 `add_act` 合一。每层 24 → 12 次 dispatch。
   实测收益 ~0.2 ms（g2 已不是 dispatch 受限，而是 GEMM 吞吐限制：4.2 GMAC / 4.3 ms ≈ 1.0 TFLOP/s，
   对 2070（无 coopmat、F32 权重）已接近其 Vulkan 后端上限）。

CPU 侧现况：g1 ≈ 40 ms、g2 ≈ 49 ms、host ≈ 3 ms（合计 ~106 ms，空载）——差距全在 CNN（见上表）。
Vulkan 侧：g1（含图内 pos_conv）≈ 3.2 ms、g2 ≈ 3.9 ms、host ≈ 0。

### 复用 ggml-audio patch 自带的 kernel

`patches/0001-ggml-audio-patch-port-on-llamacpp.patch` 已经带了 0xShug0/audio.cpp 移植过来的
conv 系列 kernel，本轮直接复用（不新增 op）：

| kernel | 用法 | 结果 |
|--------|------|------|
| `ggml_conv_1d_fast_1d_im2col` | CNN 7 层 conv（1D 专用 im2col：逐行 memcpy + 零填充，取代通用 im2col 循环） | **Vulkan 9.1 → 7.1 ms**；CPU g1 43 → 40 ms |
| `ggml_im2col_fast_1d` (dst_type=F32) | 同上，但用 F32 im2col 缓冲（CPU 默认，`GSV_HUBERT_IM2COL_F32=0` 可关） | 省掉 CPU 侧 F16→F32 的 wdata 转换；**CPU 精度 1.77e-3 → 6.68e-6**，速度持平 |
| `ggml_conv_direct_1d` | 试用于图内 pos_conv 的 16 个分组 conv（stride-1、免 im2col、bias 融合） | **反效果**（Vulkan +11 ms）：该 kernel 的 chunked pipeline 有可观的每次 dispatch 固定开销，16 个小分组摊薄不了；已回退。它适合"单次大 conv"，不适合 16 个 [49×48] 的小分组 |

（`conv_direct_1d` 的 Vulkan shader 注释里写明是按 RTX 2070 级硬件调优的，选型启发式按 OC 挑 tile
变体；我们的 OC=48 属于最小档，固定开销占主导。）

测试：`tests/test_hubert_ggml.cpp`（`GSV_HUBERT_DEVICE=vulkan` 切后端，`GSV_HUBERT_NTHREADS=N` 调线程、
`GSV_HUBERT_BENCH=N` 基准，`GSV_HUBERT_ENCIN=<file>` 可用 golden enc_in 单测 g2）；
torch 侧基准：`tools/bench_hubert_torch.py`（`python tools/bench_hubert_torch.py`，diffsinger env）。
