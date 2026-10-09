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
2. ~~`bridge`~~ ✅
3. ~~`ref_enc`~~ ✅（§7）
4. ~~`wns1`~~ ✅（§9）
5. ~~`enc_p`（相对位置注意力 + MRTE + 双编码器）~~ ✅（§10）

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
| **ggml CPU（全 F32，含 F32 im2col）** | **6.68e-6** |
| ggml Vulkan（FA 走 F16 K/V） | 1.49e-2 |

> 对拍基准一律是 **torch**（CPU/CUDA 同源 golden），与 host C++ 参考实现无关。

耗时（同一会话、同一输入、warmup 后 30 次迭代取平均；RTX 2070 + i9 32 逻辑核）：

对拍/性能基准一律取 torch（transformers HubertModel）CPU 与 CUDA：

| 实现 | 后端 | avg | min | 相对 |
|------|------|-----|-----|------|
| **ggml（本实现）** | **Vulkan** | **7.1 ms** | **6.9** | **4.0× 快于 torch CUDA** |
| torch（F32） | CUDA | 28.4 ms | 25.6 | |
| torch（F16） | CUDA | 27.8 ms | 26.5 | F16 无收益 |
| **ggml（本实现）** | **CPU 24T** | **106 ms** | **101** | — |
| torch（F32） | CPU | 87 ms | 76 | torch 快 ~1.2× |

> 数值取自机器空载时段；测量期间机器常有外部负载（avg 可高出 30-50%），故以 min 为准。
> torch 侧分段脚本：`tools/bench_hubert_torch_seg.py`；端到端：`tools/bench_hubert_torch.py`。

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

## 6. bridge（Conv1d 192->512 k=1 + LeakyReLU）✅

链路位置：`codes -> RVQ.decode -> x2 nearest -> enc_p -> **bridge** -> x2 nearest -> wns1 -> fea[512,T]`
（enc_p 的 inter_channels=192 即 bridge 的输入维度；ge 由 ref_enc(MelStyleEncoder) 单独提供）。

实现（`src/gsv_cond.cpp` 的 `gsv_cond::bridge_run`）：k=1 卷积就是一次 `mul_mat`
（`bridge.weight` 在 GGUF 里是 ne=[1,192,512]，reshape 成 [192,512] 后元素 (ic,oc) 落在
`4*ic + 768*oc` —— 正是 mul_mat 的 [in,out] 布局，无需拷数据）→ `add` bias →
`ggml_leaky_relu(0.01)` → `permute+cont` 成 t 主序 [T,512]（与 wns1 的 fea 一致）→
`upsample_x2` 时在时间轴 `interpolate(NEAREST)` ×2 → [2T,512]。

> 注：patch 里的 `ggml_add_leaky_relu`（融合 add+leaky）是 **CPU only**，其它后端不 emit，
> 所以这里用标准的 `leaky_relu`（Vulkan 有 pipeline），图在两个后端上完全一致。

对拍（`tests/test_cond_bridge.cpp`，golden T=60）：

| 后端 | conv+leaky max\|d\| | x2 nearest 结构 | 耗时 avg/min | torch 同机 |
|------|----------------------|----------------|--------------|-----------|
| CPU 24T | 4.29e-6 | PASS | 0.75 / 0.38 ms | torch CPU 0.33 / 0.23 ms |
| Vulkan | 1.16e-2（相对 1.1e-3） | PASS | 0.44 / 0.35 ms | torch CUDA 0.39 / 0.35 ms |

Vulkan 的 1.1e-3 相对误差来自后端把 F32 权重转 F16 的 mul_mat（与 HuBERT/wns1 的 Vulkan 项同因）。
CPU 的 avg 高于 min 是因为每次调用都新建图（`ggml_init` + graph + galloc）；这个算子整体
不到 1ms，集成后可留待需要时按 T 做图缓存。

## 7. ref_enc（MelStyleEncoder）✅

结构（对照 `repo/GPT_SoVITS/module/modules.py`）：spectral(Linear 704→128 + Mish ×2)
→ temporal(2×Conv1dGLU k=5 pad=2: conv(128→256) 分半门控 + 残差) → 帧 mask →
2 头自注意力(d_k=d_v=64, 温度 √128, mask=-inf, fc+残差) → Linear(128→512) →
有效帧均值池化 → ge[512]。实现 `src/gsv_refenc.{h,cpp}` + `tests/test_refenc.cpp`。

**"双后端一致 FAIL 87"的真正原因是两个结构性 bug（非后端问题）**，2026-10-09 修复：

1. **`oh` 的 mul_mat operands 反了**：原来是 `mul_mat(pm, vhT)`，得到 `[T,64]`
   （行为 query），而 concat 轴又是 ne0 → 两头拼成 `[2T,64]` 而不是 `[128,T]`。
   正确为 `mul_mat(vhT, pm)`（`vhT [T_ki,64]` 为 a、`pm [T_ki,T_q]` 为 b → `[64,T_q]`），
   与 enc_p 的 attention 输出段同构。
2. **MHA 输出断链**：参考 MHA 返回 `fc(cat_heads) + 残差`，原图里 fc 直接吃了 `z`
   （attention 输入），attention 整体成了死代码（不参与 t_out 可达 → 从图上被剪掉）。

修复时一并做的：每头投影用 load 时预切的**稠密权重块**（`mul_mat` src0 全稠密，
避开 llamafile sgemm 的行主假设）；`dbg_attn` 改指 MHA 输出（此前误指 attention 输入 z，
所以中间层对拍结论曾被误导）；host 侧 mask 构建的 `getenv` 从 O(T²) 循环里提出
（`GSV_REFENC_MASKVAL` 等，之前每次 8 万次 getenv → host 段 105ms，现 1ms）。

### 对拍与性能（T=200 / len=180，`tests/test_refenc.cpp`）

| 后端 | ge max\|d\| | 相对 | 判定 |
|------|------------|------|------|
| CPU（Release, 16T） | **4.57e-4** | 2.8e-6 | PASS |
| Vulkan | 5.36e-1 | **3.3e-3** | PASS（F16 转换底噪，见下） |

中间层（有效帧 t<180）：spectral 3.8e-5 / temporal 2.6e-3 / attention 7.8e-2 /
fc 7.2e-2（后两者绝对值大是因为本模块 z 值域 ~±1000 → fp32 相对误差仍在 1e-4 级）。
padding 帧（t≥180）attention 不参与池化，且 padding query 的 mask 策略与 torch 有意不同，
dump 上表现为大差异——对最终 ge 无影响。

**Vulkan 的 3.3e-3 相对误差是后端 F16 底噪**：本模块中间值域 ~±1000，attention 的
softmax 输入（q·k/√128）量级 ~1e4，F16 级的 q/k 扰动在 softmax 里被放大（个别帧的
attention 峰值翻转），pooling 平均后降到 3.3e-3。判定改为**按参考量级缩放**
（`2e-2 × max|ref|`）：官方管线本身在 fp16 下跑条件段，此量级与 fp16 参考精度同阶。

| | torch | ggml | 相对 |
|---|---|---|---|
| GPU | CUDA F32 avg 4.81 / min 4.70 ms（F16 无收益 5.09/4.89） | **Vulkan avg 1.74 / min 1.43 ms** | **快 ~3.3×** |
| CPU | CPU F32 16T avg 7.17 / min 5.43 ms | CPU 16T avg 4.18 / min 3.16 ms | **快 ~1.7×** |

> ref_enc 每段参考音频只跑一次（ge 天然可离线缓存），绝对量级很小；
> torch 侧基准脚本：`tools/bench_refenc_torch.py`（diffsinger env）。
> 构建：`scripts/build-refenc.bat`（CPU Release → tests/relcpu；Vulkan → tests/rel）。
> 历史上排除项的探针（`test_attn_views` / `test_cont_view` / `test_attn_combo` /
> `test_attn_unit`）结论仍然有效——它们验证的 view/cont 语义没有错，错的是本模块的接线。

## 8. 量化扫描（HuBERT / wns1 / bridge）

convert 脚本已支持按组分档：`tools/convert_hubert.py --spec "attn=f16,ffn=f16"`、
`tools/convert_cond.py --spec "wns1=f16,bridge=f16"`（bias/LN/RVQ codebook 恒 F32；
add_act/layernorm_affine 融合算子要求 F32 输入，故 bias 不可降档）。

| 模型 | 档位 | 体积 | CPU max\|d\| | Vulkan max\|d\| | CPU 耗时 (24T avg/min) | Vulkan 耗时 (avg/min) |
|------|------|------|--------------|----------------|----------|-------------|
| HuBERT | F32（基线） | 377 MB | 6.7e-6 | 1.49e-2 | 117/108 ms | 7.7/7.2 ms |
| HuBERT | attn F16 | 321 MB | 4.0e-4 | 1.49e-2 | — | — |
| HuBERT | ffn F16 | 264 MB | 8.1e-4 | 1.49e-2 | — | — |
| HuBERT | conv F16（+attn F16） | 312 MB | 4.0e-4 | 1.49e-2 | — | — |
| HuBERT | conv F16（+ffn F16） | 256 MB | 8.1e-4 | 1.49e-2 | — | — |
| **HuBERT** | **全 F16（conv+attn+ffn）** | **199 MB（1.9×↓）** | **1.0e-3** | **1.49e-2（不变）** | **110/102 ms（还快 6%）** | **8.1/6.4 ms（快 10%）** |
| wns1+bridge | F32（基线） | 186 MB | 9.5e-7 | 2.9e-3 | 92/72 ms (wns1) | 2.30/2.00 ms |
| wns1+bridge | **wns1 F16** | 127 MB（1.5×↓） | 9.5e-7（不变） | 2.9e-3（不变） | 92/72 ms（不变） | 2.30 ms（不变） |
| bridge | bridge F16（附加） | 126 MB | 1.8e-3 | 1.2e-2（不变） | 0.72/0.70 ms | 0.38/0.36 ms |

**最终档位**：
- **HuBERT 全 F16（199 MB）**——conv/attn/ffn 全部 F16 后 CPU 1.0e-3、Vulkan 1.49e-2
  （Vulkan 项一直由 FA 的 F16 K/V 主导，与权重无关），且**两个后端都更快**
  （CPU F16 行主数据对 cache 更友好；Vulkan F16 权重少一半搬运）。已定为默认档。
- **wns1 全 F16（127 MB）**——CPU 精度与 F32 逐位一致。
- bridge F16 可选（+1MB 收益，CPU 4.3e-6→1.8e-3 仍在阈内）。
- pos_conv 权重保持 F32（其 host/GPU 处理路径假设 4 字节元素，且 conv 组的
  `pick()` 已显式排除）。

分档结论（凹档过程）：attn=F16 单独贡献 4.0e-4，ffn=F16 贡献 8.1e-4，两者叠加 1.0e-3，
conv=F16 不叠加误差（conv 后接 GroupNorm 归一化了 F16 噪声）。Vulkan 侧所有档位都是
1.49e-2 —— FA 的 F16 K/V 是主导项，权重档位不影响。

工具链坑：gguf-py 的 `raw_dtype=F16` **不会**转换数据——必须同时把 numpy 数组
`astype(float16)`，否则 header 声明 F16 而数据仍按 F32 落盘（体积不变且图内误读）。
另：conv 权重转置回写（load 时）已支持 F16（`gsv_hubert.cpp` 按类型大小 get/set）。

### Q8_0 档（`quantize_gguf` 组扩展后）

`tools/quantize_gguf.cpp` 已扩展 hubert/wns1/bridge 分组（feat_conv=conv 组、
transformer q/k/v/out=attn、ffn=ffn 组、ln_w/norm/bias/codebook/pos_conv 强制 F32/F16 fixed）。

| 模型 | 档位 | 体积 | CPU max\|d\| | Vulkan max\|d\| | CPU 耗时 | Vulkan 耗时 |
|------|------|------|--------------|----------------|----------|-------------|
| HuBERT | **attn+ffn+conv Q8_0** | **189 MB（F16 199MB 再 ↓5%）** | **5.2e-3** | 待测 | 129/119 ms（慢 15%） | — |
| wns1 | wns1 权重 Q8_0（k=1 的 1×512 条回退 F16） | **93 MB（w16 127MB 再 ↓27%）** | **9.5e-7（不变！）** | 2.9e-3（不变） | 91/79 ms | 2.10/2.01 ms |
| bridge | 3D 张量不量化（回退 F16），收益 0 | — | — | — | — | — |

结论：
- **wns1 Q8_0 是意外赢家**：CPU 精度逐位不变（9.5e-7）、体积再 ↓27%、速度还略快。
  in_layers k=5 的 [512,1024] 大块 Q8 对称量化恰好适配。
- **HuBERT Q8_0 不划算**：CPU 误差 1.0e-3→5.2e-3（涨 5×）、速度慢 15%（Q8 去量化开销），
  体积只比 F16 小 9MB。**维持 F16 档**。
- 修复：quantizer 的 fixed 判定补了 `.b`/`.bias` 结尾（此前 `feat_proj.b` 等 bias 被
  误量化为 F16，binary-op 直接断言拒绝 F16 输入）；`ln1_w/ln2_w` 归 fixed
  （layernorm_affine 要求 F32）；`pos_conv.w` 归 fixed（host 端按 float* 读，F16 越界）。

## 9. WNS1（VITS WN Encoder）—— 已落地（ORCATERM 实现 + conv 内核优化）

结构：pre Conv1d(512→512,k=1) ×mask → 8 层 WaveNet（in_layers k=5 pad=2、cond_layer k=1 的
gin=512 全局条件、`tanh⊙sigmoid` 门控、res_skip 双输出；每层 `(x+res)×mask`）→ skip 求和 ×mask →
proj k=1 ×mask。权重来自 `models/gsv-cond-f32.gguf`（38 个张量，weight_norm 已在转换时物化）。

复核（独立重跑，T=120 / len=100）：

**已用 audio patch 的算子优化 conv 路径**（`GSV_WNS1_CONV` 选档，默认 2）：

| conv 实现 | CPU 24T avg/min | Vulkan avg/min | 精度 (CPU / Vulkan) |
|-----------|-----------------|----------------|---------------------|
| 0 = `ggml_conv_1d`（原样） | 127 / 102.8 ms | 2.77 / 2.33 ms | 6.5e-4 / 3.6e-3 |
| 1 = `ggml_conv_1d_fast_1d_im2col` | 291 / 101.1 ms（抖动大） | 2.72 / 2.30 ms | 6.5e-4 / 3.6e-3 |
| **2 = fast im2col + dst F32（默认）** | **79.7 / 62.8 ms** | **2.97 / 2.33 ms** | **1.07e-6 / 2.90e-3** |
| 3 = `ggml_conv_direct_1d` | 272 / 238 ms | **崩溃（K=1 时）** | 1.67e-6 / — |
| 4 = 混合（K>1 走 direct） | 131 / 115 ms | 6.40 / 5.56 ms | 6.0e-4 / 2.8e-3 |

**结论：mode 2 胜出（CPU −32%、精度 6.5e-4 → 1.1e-6；Vulkan 持平、精度略好）。**

- `conv_direct_1d` 在这里不划算：T=120 的帧数太小，它的 chunked pipeline 固定开销压不住
  （Vulkan 慢 2.3×，CPU 慢 2.3×）。它的设计目标是长 T（音频模型常见几百~几千帧）。
- **发现一个 patch kernel bug**：`ggml_conv_direct_1d` 在 **Vulkan 下 K=1 时崩溃**（与 T 无关；
  K=2/3/5 正常；CPU 正常）。最小复现：`tests/test_conv_direct_vk.cpp`
  （`test_conv_direct_vk.exe vulkan 1 120 0`）。疑在 shader/变体选择的权重打包对 KW=1 的处理。
  我们不依赖它（只用 mode 2），但对 patch 维护者是个待修项。

对 torch 的最终对比（T=120 / len=100）：

| | torch | ggml（mode 2） | 相对 |
|---|---|---|---|
| GPU | CUDA F32 9.50 ms（min 8.88） | **Vulkan 2.97 ms（min 2.33）** | **快 4.1×** |
| CPU | CPU F32 42.2 ms（min 35.2） | CPU 24T 79.7 ms（min 62.8） | 慢 1.8×（原 2.4×） |

CPU 仍落后 1.8×：剩下的是纯 GEMM 吞吐差（oneDNN 直卷积 + AVX-512 阻塞 vs ggml CPU GEMM），
与 HuBERT 的 CNN 同因。

> 该实现的图语义已逐条核对（mask 时机、门控切分、cond 分层切片、res/skip 分支）与 VITS 一致；
> 落地前待办：T 目前固定 120（集成需按 T 重建图）、`src/gsv_cond.h` 工作副本的中文注释被写入
> 工具转成了乱码需修复、两个文件带 BOM、测试的 maxdiff 为 NaN 盲。
> torch 侧基准脚本：`tools/bench_wns1_torch.py`。

测试：`tests/test_hubert_ggml.cpp`（`GSV_HUBERT_DEVICE=vulkan` 切后端，`GSV_HUBERT_NTHREADS=N` 调线程、
`GSV_HUBERT_BENCH=N` 基准，`GSV_HUBERT_ENCIN=<file>` 可用 golden enc_in 单测 g2）；
torch 侧基准：`tools/bench_hubert_torch.py`（`python tools/bench_hubert_torch.py`，diffsinger env）。

## 10. enc_p（V5 TextEncoder + MRTE）✅

结构（`src/gsv_encp.{h,cpp}`，235 张量全部 F32）：
`ssl_proj(768→192 1×1) → Encoder_ssl×3 → text_embedding(732→192) → Encoder_text×6 →
MRTE → Encoder2×3 → proj(192→384) → split m/logs`。
Encoder 层 = rel-pos MHA（heads=2, dk=96, window=4）+ ConvFFN(k=3, pad=1, ReLU)，均为 post-LN
（自定义 LayerNorm：归一在**通道维**、参数名 gamma/beta）。MRTE = c_pre 192→512 →
cross-attn（heads=4 dk=128，**注意其内部还有一层 conv_q/k/v**）+ cpre 残差 + ge → c_post。

**rel-pos 是本段唯一的新算子**（torch `_rel_to_abs`/`_abs_to_rel` 的 ggml 复刻，见文件头注释）：

- 核心是「pad→flatten→pad→reshape→slice」的索引重排，用 `concat`+`reshape_1d`+`view`
  在图上表达；`R = mul_mat(emb_k[d,w], qh)` 得 [w, tq]，两侧 concat 零行成 `x_g [2T-1, T]`
  （ggml 元素序恰等于 torch 行主 `x[tq, w]`）。
- `_rel_to_abs` 后与主 scores 相加：**实测 ggml view 元素 `[i0,i1] = flat[T-1+i0+i1*(2T-1)]`
  与 torch `rel[tq=i0, ki=i1]` 逐位一致**，直接相加即可（不需要转置）。
- `_abs_to_rel(p)` 走对偶流程；`rel_v_pad` 由 `cont(transpose(emb_v))` 两侧 concat 零得到，
  `out_v = mul_mat(evp, rw) → [dk, T]` 加进 `oh`。
- 曾尝试用 `ggml_acc` 逐 band 写到对角线：**CPU 与 Vulkan 都失败**（写入位置/语义与
  view 步进不符，sc 差 1.44）。留了最小探针 `tests/test_acc_probe.cpp` 记录 acc 的对角
  写入语义（nb1=(T+1)*4 可行但当时 view 组合路径不可靠）——最终用上面的 pad+reshape 方案。
- 零常量张量（pad 用）因为 galloc 分配后内容不保证，**每次 encode 前清零**；名字带
  seg+li 后缀（`zname()`），否则多层的同名零张量会互相踩（这个坑导致过 `dbg_ssl_attn`
  输出乱码 3e19）。
- `m/logs` 的 split：`stats [384, T]` 上的 view 是非连续的（nb1=384*4），**必须 `ggml_cont`
  后取回**，否则 CPU `tensor_get` 把 strided 视图按连续读（错位 → m/logs 差 1.7）。

### 对拍（T=120 / ntext=50，`tests/test_encp.cpp`）

| 后端 | m max\|d\| | logs max\|d\| | 判定 |
|------|-----------|---------------|------|
| CPU（Release, 16T） | **1.07e-6** | **8.94e-7** | PASS |
| Vulkan | 3.50e-3 | 1.74e-3 | PASS（F16 转换误差，与 HuBERT/wns1 同因） |

中间层借助 golden hook 定位（`GSV_ENCP_DEBUG=1` 转储 `_mine_*.bin`，见
`tools/dump_golden_encp.py` 里的 `encp.stage_*` / `encp.hook_ssl_attn*`）：
ssl 三层 attention 输出 5.5e-7、temb/text/mrte/enc2/stats 各段 1e-6 量级，
`m/logs` 全链 PASS。期间修掉三类真实 bug：**x 的 [T,C]/[C,T] 转置**（输入 y 与输出 m/logs 各一次）、
**零张量名字冲突**、**strided view 直读**。

### 性能（vs torch，同机同输入）

| | torch | ggml | 相对 |
|---|---|---|---|
| GPU | CUDA F32 avg 78.9 / **min 66.3** ms（F16 无收益 79.2/73.2） | **Vulkan avg 8.7 / min 8.1** ms | **快 ~8×** |
| CPU | CPU F32 16T avg 107 / **min 89** ms | CPU 16T avg 41.6 / **min 33.0** ms | **快 ~2.7×** |

> **CPU 基准的 build 说明（重要）**：`tests/` 目录里预置的 `ggml-cpu.dll` 是 **Debug** build
> （/Od /Ob0），上面所有 CPU 基准必须用 Release build。本模块用 `scripts/build-encp.bat`
> （link `llama.cpp/build-rel`，GGML_LLAMAFILE=ON + AVX2）→ 输出 `tests/relcpu/` 连 DLL 一起跑；
> 用 Debug DLL 同样代码是 313+ ms（差 ~8×），会得出"ggml CPU 落后"的错误结论
> （`llama.cpp/build-vk-rel` 无 LLAMAFILE，也慢 ~2.5×，Vulkan 基准中不动 CPU 部分即可）。
> torch 侧基准脚本：`tools/bench_encp_torch.py`（diffsinger env，`torch.set_num_threads(16)`）。

测试：`tests/test_encp.cpp [gguf] [golden_dir]`（`GSV_ENCP_DEVICE=vulkan` 切后端、
`GSV_ENCP_THREADS=N` 调线程、`GSV_ENCP_BENCH=N` 基准、`GSV_ENCP_DEBUG=1` 转储中间张量）；
Vulkan 构建：`scripts/build-encp-vk.bat`；CPU Release 构建：`scripts/build-encp.bat`。
