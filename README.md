# gsv.cpp

GPT-SoVITS V5 推理的 ggml/C++ 实现（开发中）。

## 状态

| 阶段 | 状态 |
|---|---|
| ggml 基线（llama.cpp）+ audio-patch 算子移植 | ✅ 116 ops，CPU/Vulkan 编译通过，learned-ops 测试 ALL PASSED |
| AR (Text2SemanticDecoder) step0 全前向 | ✅ 对拍 max\|Δ\| = 2.4e-6 |
| AR 增量 decode（KV cache） | ✅ 3 步对拍 3~6e-6 |
| AR 采样链（rep-penalty/top-p/temperature/top-k/softmax + exp-trick 采样） | ✅ probs 对拍 7e-9；注入 q 的采样索引与 torch 一致 |
| AR batch 能力（多序列 KV cache + padding/causal mask） | ✅ batched 首步 6.2e-6；解码步 4~5e-6；greedy 生成逐 token 一致（含 early-stop 路径） |
| AR 引擎（`src/gsv_ar`：GGUF 加载 + 前端 + 生成循环 + 常驻 KV cache） | ✅ 引擎端到端对拍 5.7e-6 / token 一致；**性能 2.1×（bs=1）/ 1.7×（bs=3）于 torch eager**，见 [docs/benchmark_ar.md](docs/benchmark_ar.md) |
| 条件编码段（HuBERT/RVQ/enc_p/MRTE/ref_enc/bridge/wns1） | ⬜ |
| DiT（CFM + static cache，v5turbo 4 步） | ⬜ |
| vocoder（ONNX，DiffSinger 式导出） | ⬜ |

## 目录

- `patches/0001-ggml-audio-patch-port-on-llamacpp.patch` — **ggml 基线补丁**（116 个算子 + Vulkan shader + pipeline cache），
  对 `llama.cpp@f0c41e0` 应用；`llama.cpp/` 目录本身不入库（见下方"获取 ggml 基线"）
- `docs/V5-ggml-port-research.md` — 选型与移植调研报告（链路清单、参考仓库映射、已确认决策）
- `docs/benchmark_ar.md` — AR 段基准（vs torch，含精度-速度权衡与剖析）
- `models/` — 权重（不入库）：s1v3.ckpt(AR) / s2Gv5turbo.pth / vocoder.pth / chinese-hubert-base / chinese-roberta-wwm-ext-large
- `src/` — 引擎代码
  - `gsv_sampler.{h,cpp}`：AR 采样链（严格复刻 `AR/models/utils.py::logits_to_probs` 的顺序与语义）+ exp-trick 采样
  - `gsv_ar.{h,cpp}`：AR 引擎（GGUF 加载、phones/bert/prompt 前端、batched 首步/增量解码、常驻 KV cache、生成循环）。`GSV_AR_PROFILE=1` 输出分阶段耗时
- `scripts/build-tests.bat` — 一键构建 ggml + 全部对拍可执行文件（VS2019 BuildTools，含 `/utf-8`）
- `tools/` — 权重转换与 golden 导出（Python，diffsinger env）
  - `convert_ar.py`：s1v3.ckpt → `models/gsv-ar-f32.gguf`
  - `dump_golden_ar.py`：torch 侧 golden（step0 各段 + K/V cache + decode 步）
  - `dump_golden_ar_sampling.py`：采样链 golden（probs/q/idx）+ batch golden（batched 首步/解码步 + greedy 参考 + phones/bert 输入）
  - `bench_ar.py`：torch 侧 AR 基准（与 bench_ar.cpp 同 workload）
- `tests/` — C++ 对拍（MSVC 链接 `llama.cpp/build-cpu` 的 ggml）
  - `test_ar_step0.cpp`：全前向对拍（支持 GSV_AR_DEBUG_LAYERS / TOKEN / RECALC）
  - `test_ar_decode.cpp`：增量 decode 对拍（KV cache，单序列）
  - `test_ar_batch.cpp`：batch 图 + greedy 生成循环 + 采样链接入（含 mask 解析构造校验）
  - `test_ar_sampler.cpp`：采样链 probs 对拍 + 注入 q 的 argmax 规则
  - `test_ar_engine.cpp`：引擎端到端（前端 + 首步 logits + greedy/early-stop 生成）
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

# 2) GGUF 转换 + golden 导出（conda diffsinger）
python tools/convert_ar.py
python tools/dump_golden_ar.py

# 3) 构建 ggml（CPU）+ 对拍程序
scripts\build-tests.bat

# 4) 对拍
tests\test_ar_step0.exe   models\gsv-ar-f32.gguf tests\golden
tests\test_ar_decode.exe  models\gsv-ar-f32.gguf tests\golden
tests\test_ar_batch.exe   models\gsv-ar-f32.gguf tests\golden
tests\test_ar_sampler.exe tests\golden
```

## 关键移植结论（踩坑记录）

- **GGUF tensor 布局**：torch `[out,in]` row-major 直接写入 = ggml `ne0=in, ne1=out`，恰好是 `mul_mat(W,x)` 需要的布局（无需转置）。但 embedding 表查表要按 row-major `mem[tok*D+d]` 读。
- **flash_attn_ext 布局**：q/k/v 需 `(hd, S, nh)`（batch 时 `(hd, S, nh, B)`），mask `ne0=n_kv`（batch 时 `[n_kv, n_q, 1, B]`，跨 batch 用 ne3 广播）；输出 `(hd, nh, S[, B])` 直接按 `(hd*nh)` 折叠成 D，**不要再 permute**。
- **`ggml_backend_tensor_get` 是裸 memcpy**：读 **strided view**（如 `view_3d` 取最后一列）会读到错误内存、静默产出错误结果（本次表现为 "只有 batch 0 正确"）。读任何非连续视图前先 `ggml_cont`。
- **KV cache**：布局 `(hd, S, nh, B)`，新列用 `ggml_concat(dim=1)` 拼接（其余维必须相同）。
- **MSVC 必须加 `/utf-8`**：源码含中文注释且为 LF 时，默认 936 代码页会把行末中文字节与换行配对、吞掉换行，导致下一行被并入注释（表现为莫名其妙的 "不是成员" 报错）。
- **torch 对拍脚本**：`torch.manual_seed` 必须在 `Text2SemanticLightningModule` 构造**之后**（构造消耗 RNG 流）。
- **采样链顺序不可调换**：rep-penalty → top-p → temperature → top-k（`<` 比较，等值保留）→ softmax；`idx<11` 时排除 EOS 等价于 `logits[:, :-1]`。
- audio-patch 补丁路径需从独立 ggml 布局 `src/`→`ggml/src/`、`include/`→`ggml/include/`；vulkan 的 `vk_device_struct` 已拆到 `ggml-vulkan-types.h`；pipeline cache 的静态成员需 `vk_device_struct::` 限定调用。
