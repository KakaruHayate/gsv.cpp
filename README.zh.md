# gsv.cpp

**GPT-SoVITS V5 的 C++/ggml 推理引擎** —— 支持 `v5turbo`（4 步 DMD）与 `v5dev`（32 步 + cache-dit），
自带一个与 `api_v2.py` 兼容的 HTTP 服务。

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KakaruHayate/gsv.cpp)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/KakaruHayate/GSV.CPP)

[English](README.md) | 中文

整条 V5 流水线都在 C++ 里跑（基于 [ggml](https://github.com/ggml-org/llama.cpp)）：
AR → cond（enc_p / ref_enc / wns1 / rvq）→ DiT/CFM → HiFi-GAN 声码器（sub-pixel）。
文本前端（G2P + BERT ids）由一个小的 Python 侧车完成，**不需要 torch**，也完全不使用 ONNX Runtime。

```
文本 ──► Python 侧车（G2P + BERT ids）──┐
                                        ├─► gsv_server（C++/ggml）──► 48 kHz WAV
GGUF 模型（AR/cond/DiT/refcode/BERT/HuBERT/vocoder）─┘
```

## 特性

- **ggml 后端**：Vulkan（NVIDIA/AMD/Intel）与 CPU，不依赖 CUDA。
- **量化权重**：AR 的 attention/FFN 用 Q8_0，其余 F16，sub-pixel 声码器 F32。
- **AR 批处理**：多段文本共享一次解码。
- **cache-dit**（DBCache）：v5dev（32 步）加速，音质基本无损。
- **请求调度器**：并发的 `/tts` 会合并调度，DiT 与声码器流水线执行。
- **HTTP 服务**：`/tts`、流式 `/tts`、`/batch` + jobs（SSE）、`/models`、`/metrics`、
  `/progress`、token 鉴权、参考音频白名单、崩溃转储。
- **内置文本前端**（`frontend/`）：中/英/日/韩/粤 G2P、BERT 分词、混合语种切分，无需 torch/transformers。

只支持 V5：v2/v3/v4 检查点不支持。

## 快速开始

**Windows（一键包）**：从 <https://huggingface.co/KakaruHayate/GSV.CPP> 下载
`windows/GSV.CPP-win64-*.zip`，解压到任意目录，双击 `Start GAG.bat`。包内含引擎、带侧车依赖的
嵌入式 Python、模型以及 [GAG](https://github.com/KakaruHayate/GPT-SoVITS-Api-GUI) 界面。

**Linux**：从 [releases](https://github.com/KakaruHayate/gsv.cpp/releases) 取预编译 `gsv_server`，
再从 <https://huggingface.co/KakaruHayate/GSV.CPP> 取 GGUF 模型与前端数据，然后按
[docs/build.zh.md](docs/build.zh.md) 配置侧车并运行。

**从源码构建**：见 **[docs/build.zh.md](docs/build.zh.md)**（中文）或
**[docs/build.md](docs/build.md)**（English）—— Linux 与 Windows 的完整步骤，包括打过补丁的
llama.cpp/ggml、Python 侧车和模型目录布局。

## 模型

预编译 GGUF：<https://huggingface.co/KakaruHayate/GSV.CPP>（AR 有 fp16 与 q8，共享
BERT/HuBERT/声码器，v5turbo 与 v5dev 两套）。音色清单放在 `models/voices/<name>/`。

转换自己的微调模型用 `tools/export_gguf`，见
[tools/export_gguf/README.md](tools/export_gguf/README.md)（只输出 GGUF，ONNX 声码器已移除）。

## 运行

```bash
# 文本前端（G2P）——数据文件见 docs/build.zh.md 第 6 节
python -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r frontend/requirements-zh.txt
.venv-sidecar/bin/python tools/text_sidecar.py --port 9882

# 引擎 + HTTP 服务
./build/gsv_server --models models --device vulkan --port 9880
```

```bash
curl -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' -d '{
  "text": "你好，欢迎使用语音合成系统。",
  "text_lang": "all_zh",
  "ref_audio_path": "models/ref.wav",
  "prompt_text": "参考音频对应的文本",
  "prompt_lang": "all_zh",
  "sample_steps": 4
}' -o out.wav
```

参考音频必须 3–10 秒（最长 30 秒），且位于允许目录内（见 `--allow-path`）。完整参数见
[docs/api.md](docs/api.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/build.zh.md](docs/build.zh.md) / [English](docs/build.md) | 从零构建到可用服务（Linux/Windows） |
| [docs/usage.md](docs/usage.md) | 构建产物、模型、转换、服务用法 |
| [docs/api.md](docs/api.md) | HTTP API：端点、流式、参考音频、错误码 |
| [docs/benchmark.md](docs/benchmark.md) | 全链路耗时（torch vs ggml，turbo/dev） |
| [tools/export_gguf/README.md](tools/export_gguf/README.md) | 检查点 → GGUF 导出器 |

## 许可证

MIT（见 [LICENSE](LICENSE)）。vendored 前端保留其上游许可证；见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `licenses/`。
模型权重由 GPT-SoVITS 检查点转换而来（MIT），沿用上游许可证。
