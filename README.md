# gsv.cpp

GPT-SoVITS V5 推理的 ggml/C++ 实现（开发中）。

## 状态

| 阶段 | 状态 |
|---|---|
| ggml 基线（llama.cpp）+ audio-patch 算子移植 | ✅ **118 ops**（含本次新增的 transformer 融合算子 `layernorm_affine`/`add_act`：CPU + Vulkan），CPU/Vulkan 编译通过，learned-ops 测试 ALL PASSED |
| AR (Text2SemanticDecoder) step0 全前向 | ✅ 对拍 max\|Δ\| = 2.4e-6 |
| AR 增量 decode（KV cache） | ✅ 3 步对拍 3~6e-6 |
| AR 采样链（rep-penalty/top-p/temperature/top-k/softmax + exp-trick 采样） | ✅ probs 对拍 7e-9；注入 q 的采样索引与 torch 一致 |
| AR batch 能力（多序列 KV cache + padding/causal mask） | ✅ batched 首步 6.2e-6；解码步 4~5e-6；greedy 生成逐 token 一致（含 early-stop 路径） |
| AR 引擎（`src/gsv_ar`：GGUF 加载 + 前端 + 生成循环 + 常驻 KV cache） | ✅ 引擎端到端对拍 5.7e-6 / token 一致；**性能 2.1×（bs=1）/ 1.7×（bs=3）于 torch eager**，见 [docs/benchmark_ar.md](docs/benchmark_ar.md) |
| AR 单流延迟优化（bs=1） | ✅ Vulkan 5.08 → **2.40~2.45 ms/step（408~417 tok/s，-52%）**；bs=3 6.84 → 4.11~4.14（-40%）、bs=8 9.07 → 7.00（-23%）。TTFT（56 token 前缀）42 → **15.5 ms（-63%，bert 投影挪进图）**；三轮: strided 视图/去冗余建图 → 融合算子（LN 连残差/bias 一起吃，`wb=[w;b]` 加载时打包）→ 解码图缓存（`set_rows` 追加 KV，仅 GPU）；CPU 换 q8_attn_ffn 近无损档 19.6 → **9.1 ms（-54%）**；见 [docs/ar_latency.md](docs/ar_latency.md) |
| AR 量化（最小近无损档） | ✅ **F16 158MB（TV 0.0002）/ attn+ffn Q8_0 89MB（TV 0.0035）**；低于此档 TV 翻倍，见 [docs/quant_ar.md](docs/quant_ar.md) |
| BERT 前端（chinese-roberta-wwm-ext-large，22 层 encoder） | ✅ CPU 对拍 max\|Δ\| = 1.0e-5（逐层 23 个隐藏状态全部 cos=1.0）；Vulkan f16 **8.5 ms** / f32 10.6 ms（≈ torch 最快路径 5.9×），CPU **108~116 ms**（16 线程，已含融合算子）；**量化下限 F16 662MB**（Q8 及以下越 token 稳定阈值）；见 [docs/bert_ggml.md](docs/bert_ggml.md) |
| HuBERT 音频前端（wav2vec2 CNN + 12 层 transformer） | ✅ ggml 图实现；对拍 CPU 6.7e-6 / Vulkan 1.5e-2；**Vulkan 7~9 ms vs torch CUDA 26 ms（约 3~4×）**，CPU ~105 ms vs torch CPU 84 ms；复用 audio-patch 的 fast-1D im2col（含 F32 im2col 档），见 [docs/cond_ggml.md](docs/cond_ggml.md) 第 5 节 |
| 条件段（decode_encp 链: RVQ → ×2 → enc_p → bridge → ×2 → wns1） | 🚧 **RVQ ✅**（bit-exact）、**bridge ✅**（CPU 4.3e-6 / Vulkan 1.2e-2，≈ torch 速度，见 docs/cond_ggml.md 第 6 节）、**wns1 ✅**（CPU 6.5e-4 / Vulkan 3.6e-3；Vulkan 2.7 ms vs torch CUDA 9.5 ms = 3.6×，per-head conv 已换 audio-patch F32 im2col，CPU −32%）、**ref_enc 🚧 WIP**（spectral/temporal 已对拍 PASS；attention 段卡点已定位，见 docs/cond_ggml.md 第 7 节）、**enc_p ⬜**（权重已在 GGUF，235 张量） |
| DiT（CFM + static cache，v5turbo 4 步） | ⬜ |
| vocoder（ONNX，fp32 严格不量化） | ✅ 导出 57.8MB / sha256 `13f95a88…`；对拍 max\|Δ\| ≤1.1e-4、corr 1.0；ORT-DML ≈ torch CUDA，ORT-CPU 快 torch 1.85×，见 [docs/vocoder_onnx.md](docs/vocoder_onnx.md) |

## 目录

- `patches/0001-ggml-audio-patch-port-on-llamacpp.patch` — **ggml 基线补丁**（116 个算子 + Vulkan shader + pipeline cache），
  对 `llama.cpp@f0c41e0` 应用；`llama.cpp/` 目录本身不入库（见下方"获取 ggml 基线"）
- `docs/V5-ggml-port-research.md` — 选型与移植调研报告（链路清单、参考仓库映射、已确认决策）
- `docs/benchmark_ar.md` — AR 段基准（vs torch，含精度-速度权衡与剖析）
- `docs/ar_latency.md` — AR 单流（bs=1）延迟优化专项（剖析、已做项、精度档对延迟、剩余空间）
- `docs/quant_ar.md` — AR 量化（验收协议：logits/greedy/TF 分布 TV；最小近无损档）
- `docs/vocoder_onnx.md` — vocoder ONNX 导出（fp32 严格不量化、DML 动态形状陷阱）
- `docs/bert_ggml.md` — BERT 前端（切层依据、逐层对拍、Vulkan 精度容限探针、量化扫描）
- `docs/cond_ggml.md` — 条件段（V5 decode_encp 链）侦察 + RVQ 实现（链条入口，bit-exact）
- `models/` — 权重（不入库）：s1v3.ckpt(AR) / s2Gv5turbo.pth / vocoder.pth / chinese-hubert-base / chinese-roberta-wwm-ext-large
- `src/` — 引擎代码
  - `gsv_sampler.{h,cpp}`：AR 采样链（严格复刻 `AR/models/utils.py::logits_to_probs` 的顺序与语义）+ exp-trick 采样
  - `gsv_ar.{h,cpp}`：AR 引擎（GGUF 加载、phones/bert/prompt 前端、batched 首步/增量解码、常驻 KV cache、生成循环）。`GSV_AR_PROFILE=1` 输出分阶段耗时
  - `gsv_bert.{h,cpp}`：BERT 前端（22 层 post-LN encoder + flash attention；`encode()`/`encode_feat()`/`encode_layers()`）
  - `gsv_cond.{h,cpp}`：条件段容器（RVQ decode + ×2 nearest、bridge Conv1d+LeakyReLU + ×2 nearest）
  - `gsv_wns1.{h,cpp}`：VITS WN Encoder（8 层 WaveNet + gin 条件；conv 内核可选 audio-patch 档，`GSV_WNS1_CONV`）
  - `gsv_refenc.{h,cpp}`：MelStyleEncoder（**WIP**：spectral/temporal PASS，attention 卡点见 docs/cond_ggml.md 第 7 节）
  - `gsv_hubert.{h,cpp}`：HuBERT 音频前端（CNN 前端 + 12 层 post-LN transformer；`GSV_HUBERT_DEVICE`/`BENCH`/`TIMING` 等）
- `scripts/build-tests.bat` — 一键构建 ggml + 全部对拍可执行文件（VS2019 BuildTools，含 `/utf-8`；脚本须保持纯 ASCII）
- `scripts/build-bert-vk.bat` — BERT Release+Vulkan 构建（用 `llama.cpp/build-vk-rel`，产物 `tests/rel/`）
- `scripts/build-ar-vk.bat` — AR 基准/对拍 Release+Vulkan 构建（`tests/rel/bench_ar_rel.exe`、`test_ar_engine_rel.exe`）
- `tools/` — 权重转换与 golden 导出（Python，diffsinger env）
  - `convert_ar.py`：s1v3.ckpt → `models/gsv-ar-f32.gguf`
  - `dump_golden_ar.py`：torch 侧 golden（step0 各段 + K/V cache + decode 步）
  - `dump_golden_ar_sampling.py`：采样链 golden（probs/q/idx）+ batch golden（batched 首步/解码步 + greedy 参考 + phones/bert 输入）
  - `bench_ar.py`：torch 侧 AR 基准（与 bench_ar.cpp 同 workload）
  - `convert_bert.py`：chinese-roberta-wwm-ext-large → `models/gsv-bert-*.gguf`（`--spec attn=...,ffn=...,emb=...`）
  - `dump_golden_bert.py`：BERT golden（ids / hidden[-3] / 管线特征 / 逐层 hs0..22）
  - `bench_bert.py`：torch 侧 BERT 基准（CPU fp32 / GPU fp32 / GPU fp16）
  - `quantize_gguf.cpp`：GGUF 量化器（含 K-quants；`--spec` 分组同转换脚本）
- `tests/` — C++ 对拍（MSVC 链接 `llama.cpp/build-cpu` 的 ggml）
  - `test_ar_step0.cpp`：全前向对拍（支持 GSV_AR_DEBUG_LAYERS / TOKEN / RECALC）
  - `test_ar_decode.cpp`：增量 decode 对拍（KV cache，单序列）
  - `test_ar_batch.cpp`：batch 图 + greedy 生成循环 + 采样链接入（含 mask 解析构造校验）
  - `test_ar_sampler.cpp`：采样链 probs 对拍 + 注入 q 的 argmax 规则
  - `test_ar_engine.cpp`：引擎端到端（前端 + 首步 logits + greedy/early-stop 生成；`GSV_AR_BERT_NOISE=<σ>` 注入前端噪声探针）
  - `test_bert_ggml.cpp`：BERT 对拍/基准（`GSV_BERT_DEVICE`/`MODEL`/`THREADS`/`LAYERS`，`--bench`）
  - `bench_ar.cpp`：AR 基准（bs=1/3/8，见 docs/benchmark_ar.md）
  - `test_min_ffn.cpp`：最小 LN+FFN 对拍（排查用）
  - `golden/`（不入库）由上述 dump 脚本生成

## 获取 ggml 基线

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
cd llama.cpp && git fetch --depth 2 origin f0c41e0 && git checkout f0c41e0
git apply ../patches/0001-ggml-audio-patch-port-on-llamacpp.patch
```

> 注：llama.cpp 的 `ggml/` 目录是从 `ggml-org/ggml` 同步而来（`scripts/sync-ggml.last`），
> 后续若改为直接依赖独立 ggml 仓库，本补丁需按 `ggml/src` 布局重放一遍（补丁本身已按此布局书写）。

## 复现

```bash
# 1) 模型下载（hf-mirror）
#    s1v3.ckpt, gsv-v5-pretrained/{s2Gv5turbo.pth,vocoder.pth}, TencentGameMate/chinese-hubert-base, hfl/chinese-roberta-wwm-ext-large

# 2) GGUF 转换 + golden 导出（conda diffsinger；golden/ 不入库，必须由脚本生成）
python tools/convert_ar.py
python tools/dump_golden_ar.py
python tools/dump_golden_ar_sampling.py            # batch/采样链 golden（test_ar_batch/engine/sampler 需要）
python tools/convert_bert.py                       # f32 1.24GB
python tools/convert_bert.py --spec "attn=f16,ffn=f16,emb=f16,type=f16" --out models/gsv-bert-f16.gguf
python tools/dump_golden_bert.py

# 3) 构建 ggml（CPU）+ 对拍程序
scripts\build-tests.bat
scripts\build-bert-vk.bat          # BERT 的 Release+Vulkan 版（bench 用）

# 4) 对拍
tests\test_ar_step0.exe   models\gsv-ar-f32.gguf tests\golden
tests\test_ar_decode.exe  models\gsv-ar-f32.gguf tests\golden
tests\test_ar_batch.exe   models\gsv-ar-f32.gguf tests\golden
tests\test_ar_sampler.exe tests\golden
tests\test_ar_engine.exe  models\gsv-ar-f32.gguf tests\golden   # GSV_AR_ACC=1 跑 100-token 量化门
tests\test_bert_ggml.exe  --bench                          # GSV_BERT_DEVICE=vulkan 走 GPU
```

## 关键移植结论（踩坑记录）

- **GGUF tensor 布局**：torch `[out,in]` row-major 直接写入 = ggml `ne0=in, ne1=out`，恰好是 `mul_mat(W,x)` 需要的布局（无需转置）。但 embedding 表查表要按 row-major `mem[tok*D+d]` 读。
- **flash_attn_ext 布局**：q/k/v 需 `(hd, S, nh)`（batch 时 `(hd, S, nh, B)`），mask `ne0=n_kv`（batch 时 `[n_kv, n_q, 1, B]`，跨 batch 用 ne3 广播）；输出 `(hd, nh, S[, B])` 直接按 `(hd*nh)` 折叠成 D，**不要再 permute**。
- **`ggml_backend_tensor_get` 是裸 memcpy**：读 **strided view**（如 `view_3d` 取最后一列）会读到错误内存、静默产出错误结果（本次表现为 "只有 batch 0 正确"）。读任何非连续视图前先 `ggml_cont`。
- **KV cache**：布局 `(hd, S, nh, B)`，新列用 `ggml_concat(dim=1)` 拼接（其余维必须相同）。
- **MSVC 必须加 `/utf-8`**：源码含中文注释且为 LF 时，默认 936 代码页会把行末中文字节与换行配对、吞掉换行，导致下一行被并入注释（表现为莫名其妙的 "不是成员" 报错）。
- **torch 对拍脚本**：`torch.manual_seed` 必须在 `Text2SemanticLightningModule` 构造**之后**（构造消耗 RNG 流）。
- **采样链顺序不可调换**：rep-penalty → top-p → temperature → top-k（`<` 比较，等值保留）→ softmax；`idx<11` 时排除 EOS 等价于 `logits[:, :-1]`。
- **图中间张量读不回来**：`ggml_set_output` 只影响 gallocr 的原地复用/回收判定，图根以外的中间张量在 `backend_graph_compute` 之后读到的可能是被复用/覆盖的数据（实测只有最后一个节点正确、其余全是垃圾）。要逐层导出就**按层数重建图**（`encode_layers()`），不要靠 OUTPUT 标记。
- **CUDA 线未做**：llama.cpp 的 CUDA FA 要求 `head_dim >= 40`，AR 的 head_dim = 32 → `ggml_cuda_flash_attn_ext` 直接 abort；引擎直连 `graph_compute`（无 scheduler 回退），故 CUDA 后端不可用（等价的 mul_mat+soft_max 方案实测收益不成立，见 docs/ar_latency.md §5）。
- **Vulkan flash attention 只有 F16 K/V**（`pipeline_flash_attn_f32_f16`）：F32 K/V 会被降精度 → BERT 隐藏状态 1e-2 级偏差、AR logits Δ 0.0035。关 `GGML_VK_DISABLE_COOPMAT2` 不够，关 `GGML_VK_DISABLE_COOPMAT` 也几乎不改善（实测误差不变）。是否可接受用**前端噪声容限探针**判定（`GSV_AR_BERT_NOISE`：σ≤1e-2 时 100-token 逐 token 仍一致，σ=5e-2 开始分歧）。
- **decode 步（S=1）的 permute 是恒等变换**：qkv 的内存序本来就是 (hd, nh)，直接 `view_4d(HD,1,NH,B, nb1=任意, nb2=4*HD, nb3=共轭)` 就是 flash 需要的布局 —— 省掉每层 6 次 `ggml_cont`（144 dispatch/步，Vulkan bs=1 -23%）。CPU 只要求内维连续（`nb0 == type_size`），Vulkan 的 FA 通过 push constant 接收三个步长，两边都按视图步长寻址（实测逐位一致）。同理 BERT 的 `(HD,T,NH,1)` 视图。
- **图根不要重复 expand**：KV 写入的 `cpy` 已通过 `concat` 连到 flash（是 logits 的祖先），再对每个 cpy 单独 `ggml_build_forward_expand` 会让它重新遍历整条链 48 次（建图 1.42 → 0.99 ms/步）。
- **mask 直接上传 F16**：flash 要求 mask 为 F16，图内 `ggml_cast` 是白扔一次 dispatch。
- **Windows 上 `INTER` 是宏**（`windef.h`），不能当成员名。
- **`.bat` 必须纯 ASCII + 无括号歧义**：UTF-8 中文注释的批处理（LF 行尾、936 代码页）会被 cmd 解析错位；`echo (...)` 里的括号会截断 `if (...)` 块。
- audio-patch 补丁路径需从独立 ggml 布局 `src/`→`ggml/src/`、`include/`→`ggml/include/`；vulkan 的 `vk_device_struct` 已拆到 `ggml-vulkan-types.h`；pipeline cache 的静态成员需 `vk_device_struct::` 限定调用。
