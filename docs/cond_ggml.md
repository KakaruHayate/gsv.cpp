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
