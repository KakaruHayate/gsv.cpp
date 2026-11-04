# 从零构建 gsv.cpp

本文把一台干净机器带到可用的 `/tts` 服务：引擎、GGUF 模型、Python 文本前端。

支持两条路径：

- **Linux**（x86_64，glibc 2.35+）：参考平台，CI 产出的就是它。
- **Windows**（x86_64，Visual Studio 2022）：同一套代码，MSVC 编译；Hugging Face 上另有预编译
  一键包（`windows/GSV.CPP-win64-*.zip`）。

两个平台都支持纯 CPU 构建（跳过 Vulkan 部分）。

```
文本 ──► Python 侧车（G2P + BERT ids）──┐
                                        ├─► gsv_server（C++/ggml）──► 48 kHz WAV
GGUF 模型（AR/cond/DiT/refcode/BERT/HuBERT/vocoder）─┘
```

## 1. 依赖

### Linux（Ubuntu 22.04+）

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv \
                        libvulkan-dev vulkan-tools glslang-tools mesa-vulkan-drivers
```

- CMake **3.20+**、C++17 编译器、Python **3.10+**；
- `glslang-tools` 提供 `glslc`，ggml-vulkan 编译时需要；
- 用 GPU 需要可用的 Vulkan 驱动（`vulkaninfo --summary`）。

### Windows

- **Visual Studio 2022**，勾选"使用 C++ 的桌面开发"（MSVC + CMake）；
- **Vulkan SDK** 1.3+（<https://vulkan.lunarg.com/>），自带 `glslc.exe`；
- **Git**、**Python 3.10+**；
- 在 *x64 Native Tools Command Prompt for VS 2022* 里构建（或先运行 `vcvars64.bat`）。

## 2. 获取源码

```bash
git clone https://github.com/KakaruHayate/gsv.cpp
cd gsv.cpp
```

引擎需要打过补丁的 **llama.cpp/ggml**（上游没有 conv1d 和音频算子）。提交号固定在
`.github/workflows/ci.yml` 的 `LLAMA_CPP_COMMIT`：

```bash
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout f0c41e0168dfd4b5ef72b21d1a311b24cc7a894a
git -C llama.cpp apply ../patches/0001-ggml-audio-patch-port-on-llamacpp.patch
```

> 这个补丁提供 `ggml_conv_1d`、`ggml_conv_transpose_1d`、`ggml_pad_reflect_1d` 以及 sub-pixel
> 相关算子。不打补丁无法编译。

## 3. 编译 ggml

### Vulkan（GPU）

```bash
cmake -S llama.cpp -B llama.cpp/build-vk-rel \
  -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build-vk-rel --config Release \
  --target ggml ggml-base ggml-cpu ggml-vulkan -j 2
```

Windows 在 configure 阶段加 `-G "Visual Studio 17 2022" -A x64`，库会生成在
`llama.cpp/build-vk-rel/Release/`。

### 纯 CPU

```bash
cmake -S llama.cpp -B llama.cpp/build-cpu-rel \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build-cpu-rel --config Release --target ggml ggml-base ggml-cpu -j
```

## 4. 编译服务端

```bash
cmake -S . -B build -DGSV_GGML_LIB_DIR=llama.cpp/build-vk-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target gsv_server -j 4
```

产物：

- Linux：`build/gsv_server`
- Windows：`build/Release/gsv_server.exe`

测试（可选但推荐）：默认 `-DGSV_BUILD_TESTS=ON` 会生成 `test_add_act`、`test_conv_direct_act`、
`test_svc`，用 `ctest --test-dir build` 运行。

Windows 需要把 `ggml*.dll` 拷到 exe 旁边（在 `llama.cpp/build-vk-rel/Release/`），或把该目录加入
`PATH`；同时需要 VC++ 运行库（`vcruntime140*.dll`、`msvcp140.dll`），预编译包里已经带上。

## 5. 准备模型

引擎只读 GGUF。两种方式：

**a) 下载预编译模型**（推荐）：<https://huggingface.co/KakaruHayate/GSV.CPP>

```
models/
  gsv-ar-q8.gguf              # AR，Q8（默认推荐）
  gsv-cond-f16.gguf           # v5turbo cond
  gsv-dit-f16all.gguf         # v5turbo DiT
  gsv-refcode.gguf            # v5turbo 参考编码器
  gsv-bert-f16.gguf           # 共享 BERT
  gsv-hubert-f16.gguf         # 共享 CN-HuBERT
  gsv-vocoder-f32.gguf        # 共享 HiFi-GAN（sub-pixel，F32）
  resampy_kaiser_best.bin     # 重采样滤波表
```

v5dev 还需要 `gsv-v5dev-cond-f16.gguf`、`gsv-v5dev-dit-f16all.gguf`、
`gsv-v5dev-refcode.gguf`，并在 `models/voices/v5dev/manifest.json` 里描述。

**b) 转换自己的微调模型**：用 `tools/export_gguf`，见
[tools/export_gguf/README.md](../tools/export_gguf/README.md)。BERT/HuBERT 仍需预编译文件或
`--kind bert/hubert`。

## 6. 文本前端（G2P 侧车）

前端已经 vendored 在 `frontend/`，不需要 torch/transformers。

```bash
python3 -m venv .venv-sidecar
.venv-sidecar/bin/pip install -r frontend/requirements-zh.txt      # Windows: .venv-sidecar\Scripts\pip
.venv-sidecar/bin/pip install -r frontend/requirements-multi.txt   # 英/日/韩/粤
```

数据文件（不在 git 里）：

| 内容 | 放到 | 环境变量覆盖 |
|---|---|---|
| G2PW 模型（`g2pW.onnx` + 字典，约 608 MB） | `models/frontend/g2pw/` | `GSV_G2PW_MODEL` |
| BERT tokenizer（`vocab.txt`、`config.json`、`tokenizer_config.json`） | `models/frontend/bert_tokenizer/` | `GSV_BERT_DIR` |
| `nltk_data`（cmudict + `averaged_perceptron_tagger_eng`） | `models/frontend/nltk_data/` | `NLTK_DATA` |
| fast-langdetect 模型（`lid.176.bin`，约 125 MB） | `models/frontend/fast_langdetect/` | `GSV_FAST_LANGDETECT` |

四者都在 Hugging Face 的 `frontend/` 目录下。按上表放置侧车会自动找到，只有自定义路径才需要环境变量。

启动：

```bash
.venv-sidecar/bin/python tools/text_sidecar.py --host 127.0.0.1 --port 9882
```

中文前端预热完成后 `GET /health` 返回 `{"ok": true, "ready": true}`（几秒；其它语言首次使用时预热）。

## 7. 运行

```bash
# GPU
./build/gsv_server --models models --device vulkan --sidecar 127.0.0.1:9882 --port 9880
# CPU
./build/gsv_server --models models --device cpu --sidecar 127.0.0.1:9882 --port 9880
```

`--allow-path` 限制参考音频可读目录（默认当前目录 + 上传临时目录；`--allow-path '*'` 表示不限制）。

检查：

```bash
curl -s http://127.0.0.1:9880/health
curl -s http://127.0.0.1:9880/models
```

合成（完整参数见 [api.md](api.md)）：

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

参考音频必须 3–10 秒（最长 30 秒），且进程有权限读取。

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| CMake 报 `llama.cpp not found` | 把它克隆到仓库旁边，或传 `-DGSV_LLAMA_DIR=/path/to/llama.cpp` |
| ggml 编译报 `glslc not found` | 安装 `glslang-tools`（Linux）/ Vulkan SDK（Windows） |
| `[device] no Vulkan device` | 驱动缺失；可先 `--device cpu`，用 `vulkaninfo` 检查 |
| `text sidecar 不可达` | 侧车没启动 / 端口不对 / 它崩了（看它的 stderr） |
| 启动时 G2PW 想下载模型 | `GSV_G2PW_MODEL` 指错了目录（必须包含 `g2pW.onnx`） |
| `reference audio too long` | 把参考音频裁到 3–10 秒 |
| Windows 缺 `ggml-*.dll` | 拷到 `gsv_server.exe` 旁边，或加入 `PATH` |
| Linux 二进制报 `GLIBC_2.38 not found` | 在你要支持的最老发行版上构建（CI 用 Ubuntu 22.04） |
