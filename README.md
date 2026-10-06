# gsv.cpp

GPT-SoVITS V5 推理的 ggml/C++ 实现（开发中）。

## 状态

| 阶段 | 状态 |
|---|---|
| ggml 基线（llama.cpp）+ audio-patch 算子移植 | ✅ 116 ops，CPU/Vulkan 编译通过，learned-ops 测试 ALL PASSED |
| AR (Text2SemanticDecoder) step0 全前向 | ✅ 对拍 max\|Δ\| = 2.4e-6 |
| AR 增量 decode（KV cache） | ✅ 3 步对拍 3~6e-6 |
| AR 采样链（top-k/top-p/rep-penalty/EOS） | ⬜ 待做 |
| 条件编码段（HuBERT/RVQ/enc_p/MRTE/ref_enc/bridge/wns1） | ⬜ |
| DiT（CFM + static cache，v5turbo 4 步） | ⬜ |
| vocoder（ONNX，DiffSinger 式导出） | ⬜ |

## 目录

- `llama.cpp/` — ggml 基线（内嵌，含 ggml-audio-patch 全部算子移植；见其 git log）
- `models/` — 权重（不入库）：s1v3.ckpt(AR) / s2Gv5turbo.pth / vocoder.pth / chinese-hubert-base / chinese-roberta-wwm-ext-large
- `tools/` — 权重转换与 golden 导出（Python，diffsinger env）
  - `convert_ar.py`：s1v3.ckpt → `models/gsv-ar-f32.gguf`
  - `dump_golden_ar.py`：torch 侧 golden（step0 各段 + K/V cache + decode 步）
- `tests/` — C++ 对拍（MSVC 链接 `llama.cpp/build-cpu` 的 ggml）
  - `test_ar_step0.cpp`：全前向对拍（支持 GSV_AR_DEBUG_LAYERS / TOKEN / RECALC）
  - `test_ar_decode.cpp`：增量 decode 对拍（KV cache）
  - `test_min_ffn.cpp`：最小 LN+FFN 对拍（排查用）
  - `golden/`（不入库）由 `dump_golden_ar.py` 生成

## 复现

```bash
# 1) 模型下载（hf-mirror）
#    s1v3.ckpt, gsv-v5-pretrained/{s2Gv5turbo.pth,vocoder.pth}, TencentGameMate/chinese-hubert-base, hfl/chinese-roberta-wwm-ext-large

# 2) GGUF 转换 + golden 导出（conda diffsinger）
python tools/convert_ar.py
python tools/dump_golden_ar.py

# 3) 构建 ggml（CPU）
cd llama.cpp && cmake -B build-cpu -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF && cmake --build build-cpu --target ggml -j

# 4) 对拍
./tests/test_ar_step0.exe  models/gsv-ar-f32.gguf tests/golden
./tests/test_ar_decode.exe models/gsv-ar-f32.gguf tests/golden
```

## 关键移植结论（踩坑记录）

- **GGUF tensor 布局**：torch `[out,in]` row-major 直接写入 = ggml `ne0=in, ne1=out`，恰好是 `mul_mat(W,x)` 需要的布局（无需转置）。但 embedding 表查表要按 row-major `mem[tok*D+d]` 读。
- **flash_attn_ext 输入约定**：q/k/v 需 `(hd, S, nh)`（`reshape_3d(hd,nh,S)` + `permute(0,2,1,3)` + `cont`），mask `ne0=n_kv`（行=query）；输出 `(hd, S, nh)` 直接 `reshape_2d(D,S)` 折叠，**不要再 permute**。
- **KV cache**：布局 `(hd, S, nh)`，新列用 `ggml_concat(dim=1)` 拼接（其余维相同）。
- **torch 对拍脚本**：`torch.manual_seed` 必须在 `Text2SemanticLightningModule` 构造**之后**（构造消耗 RNG 流）。
- audio-patch 补丁路径需从独立 ggml 布局 `src/`→`ggml/src/`、`include/`→`ggml/include/`；vulkan 的 `vk_device_struct` 已拆到 `ggml-vulkan-types.h`；pipeline cache 的静态成员需 `vk_device_struct::` 限定调用。
