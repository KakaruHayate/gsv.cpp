# vocoder + G2PW：ORT ONNX 集成（DML / WebGPU / CPU EP）

日期：2026-10-14 · 机器同前（RTX 2070 / ORT 1.27.0 WebGPU + 1.24.4 DML）

## 0. 结论

| 项 | 结果 |
|---|---|
| `src/gsv_vocoder.{h,cpp}` | ORT C API（**运行时 `LoadLibrary`**，无需 import lib），vocoder ONNX 推理 |
| `src/gsv_g2pw.{h,cpp}` | 同上 + **自带 BERT WordPiece tokenizer**（复刻 repo `wordize_and_map`+HF 切分） |
| EP 支持 | **WebGPU > DML > CPU**（自动挑选；`GSV_*_DEVICE=cpu/dml/webgpu` 强制） |
| vocoder 对拍 | max\|Δ\| ≤ **2.6e-6**（T=1/100/200/257/300/500/1000 全过） |
| g2pw 对拍 | pred **逐条一致**；probs max\|Δ\| = **6.0e-8**（7 个 case，含数字/标点/变长 batch） |
| 端到端 | `tests/demo_e2e.cpp` 跑通 参考音频+phones → wav；`tools/frontend/tts.py` 文本端到端（长句 7.16 s @48k，见 [e2e_server.md](e2e_server.md)） |

**EP 基准**（`test_vocoder.cpp GSV_VOC_BENCH=5` / `test_g2pw.cpp GSV_G2PW_BENCH=5`）：

| 负载 | CPU | DML | **WebGPU** |
|---|---|---|---|
| vocoder T=200（2 s 音频） | 1564 ms | 172 ms | **71 ms** |
| vocoder T=500 | 1564 ms | 224 ms | **174 ms** |
| vocoder T=1000（10 s） | 3039 ms | 308 ms | 333 ms |
| g2pw N=16 batch | 75 ms | 110 ms | **36 ms** |

WebGPU 在短序列上优势最大（vocoder 2.4× / g2pw 3.0× 于 DML）；T=1000 时 DML 反超
（WebGPU 的 dispatch 开销随图增大）。两者都远快于 CPU（vocoder ~10×）。
精度上 **DML 与 WebGPU 相对 CPU 都是 max|Δ| ≈ 8e-7、corr = 1.0**（fp32 无量化）。

## 1. 为什么运行时加载（而不是链接 import lib）

`onnxruntime.dll` **只导出一个符号** `OrtGetApiBase`，而：

- **pip wheel 不带 `.lib`**（nuget 才有）；
- **nuget 只有 DML 包**，WebGPU EP 只发 pip wheel（`onnxruntime-webgpu`）；
- 我们想在同一可执行文件里**两者都支持**并按 DLL 实际能力挑 EP。

故用 `LoadLibraryA` + `GetProcAddress("OrtGetApiBase")` → `base->GetApi(ORT_API_VERSION)`
拿函数表。附带好处：DLL 路径可配置（`GSV_VOC_ORT` / `GSV_G2PW_ORT`），默认按
`.venv-wg`（WebGPU）→ `.venv-ort`（DML）→ PATH 的顺序找。

> **WebGPU 的附加依赖**：`onnxruntime.dll` 同目录必须有 `dxcompiler.dll` + `dxil.dll`
> （pip wheel 自带）。加载时会从 DLL 自身目录显式预加载这两个，否则 WebGPU 初始化
> 静默失败并回落到 CPU。见 `gsv_vocoder.cpp` 的加载段。

## 2. EP 挑选：盲试 + 顺序优先

```cpp
for (const char * n : { "WebGpuExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider" })
    if (SessionOptionsAppendExecutionProvider(s.so, n, nullptr, nullptr, 0) == nullptr)  // 成功
        ...   // 第一个 append 成功的即生效 EP (ORT 按 append 顺序分配)
```

该 EP 未编入当前 DLL 时 `AppendExecutionProvider` 返回错误 status —— 这就是探测手段
（`get_available_providers()` 在 C API 里没有直接等价物，且 1.22+ 改成了
`OrtGetEpDevices` 枚举）。`GSV_*_DEVICE=dml` 之类只保留单一 EP + CPU 兜底。

## 3. vocoder 集成要点

- **输入必须是 `denorm_spec()` 之后的 mel**（值域约 [-12, 2]），不是 [-1,1] 的归一化 mel；
  输出 `[B,1,T*480]`（480 = `prod(upsample_rates)` = 10·6·2·2·2），末端 `tanh` → 严格 |wav| ≤ 1。
- **输出是 48 kHz**：mel 是 32 kHz/hop 320 = 100 fps，×480 = 48000 Hz。repo 同
  (`TTS.py`: `vocoder_configs["sr"]=48000`, `upsample_rate=480`)。**写 wav 头若按 32k 标注，
  播放端会慢放 1.5 倍**（听感 = 男声慢放）—— 见 docs/e2e_server.md §7.9。
- **单动态 session**：DML/WebGPU 换形状会重新 plan（doc §3 的结论仍然成立），按形状建
  session 的创建成本收不回来。
- **严格 fp32 不量化**：导出图内 0 个量化算子；不要对它跑 ORT 量化工具。
- 布局：mel 用 `run()` 时传 torch 布局 `[100,T]`（idx = m·T+t）；DiT/ggml 侧产出是
  `[T,100]`，用 `run_ggml_mel()`（内部转置）。

## 4. G2PW 集成要点

图结构 = **12 层 BERT**（bert-base-chinese，hidden 768、12 头）+ 条件描述符头：

```
seq        = bert(input_ids)                                  [B, L, 768]
h          = seq[position_ids]                                [N, 768]  取多音字所在 token
pred_pos   = argmax(pos_classifier(h))                        [N]       11 类词性
affect     = descriptor_bias[0] + char_descriptor[char_ids]
             + second_order_descriptor[char_ids*11 + pred_pos] [N, 1305]
w          = phoneme_mask * sigmoid(affect)                   [N, 1305]
probs      = clip(softmax(logits)·w / Σ(softmax·w), 1-eps, eps)   eps = 1e-6
pred       = argmax(probs) -> labels[pred]                    (注音符号 ㄅㄆㄇ…)
```

**自带 tokenizer**（不依赖 HF）：repo `text/g2pw/utils.py::wordize_and_map` 的规则是
"连续 ASCII 字母数字成一个 word，其余**逐 code point**一个 word"，每 word 单独过
WordPiece。`gsv_g2pw.cpp` 的 `tokenize_and_map()` 复刻了它（含 `##` 续接、整词
`[UNK]` 覆盖、`.lower()`），对拍 7 个 case 全部逐 token 一致（含 `我吃了3个苹果会不会`
这种数字混排、全角逗号、变长 batch 右 pad）。

**口径**（`dataset.py::prepare_onnx_input`）：

- text **先 `.lower()`**；`[CLS]` + tokens + `[SEP]`；token_type 全 0；attention_mask 全 1；
- batch 内变长**右侧 pad 0**；`phoneme_mask` 是 `[N,1305]` 的 0/1（**不 pad**）；
- `position_id = text2token[query_id] + 1`（`[CLS]` 占位置 0）；
- `max_len = 512`，超长需调用方按 repo 的 `_truncate` 裁到多音字附近；
- **`char_ids`/`position_ids` 是扁平的 `[N]`**，其余是 `[N,L]`；`phoneme_mask` 是 **FLOAT**
  而其它四个是 INT64（onnx 图输入类型，见 §5 踩坑）。

**只做到 argmax 为止**：注音→拼音/音素的映射、文本规整（数字/英文/日文）、
pypinyin 的 `simple_seg` 兜底都留给调用方（repo 的 `text/` 那一层）。

## 5. 踩坑记录

1. **`phoneme_mask` 是 FLOAT 不是 INT64**。六个输入里五个 int64，唯独它是 float32；
   统一按 INT64 建张量会报 `not enough space: expected 10440, got 5220`（字节数差 2×）。
2. **`A->GetAllocatorWithDefaultOptions(nullptr)` 会段错误**。该函数要写出一个指针，
   传 null 即写 null → 崩溃。本模块不需要 allocator，直接删掉这行。
3. **golden `.bin` 是 raw 字节**：int64 张量按 **8 字节/元素**读，不能按 float32（4 字节）读，
   否则元素数翻倍、`N` 全错（表现为"尺寸不自洽"）。
4. **`std::vector<float>(vector<int>)` 不是合法转换**（MSVC C2440）——phoneme_mask 的
   int→float 要显式循环。
5. **C++ 源码含中文字面量时 `printf("...\n")` 里的 `\n` 会被 bash heredoc 吃掉**：用
   Edit 工具改，或 `chr(92)`； heredoc 里的 `\n` 会变成真换行，把字符串字面量截断
   （表现为莫名其妙的 C2001 常量未终止）。
6. **`ggml.dll` 陈旧会导致 0xC0000139（ENTRYPOINT_NOT_FOUND）**：`tests/relcpu/` 里的
   DLL 是旧构建拷进去的，e2e 链接 `build-vk-rel`（唯一含 audio-patch 融合算子的构建）
   后必须重新 `copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\relcpu\`。

## 6. 端到端（`tests/demo_e2e.cpp`）

```
A) 参考音频 -> 三份产出
     16k + 0.3s 零尾 -> HuBERT -> ssl_proj -> RVQ     -> prompt codes [372]
     32k + maxx 归一化 -> spectrogram_torch -> [:,704] -> ref_enc -> ge[512]
     32k -> mel_fn_v4 -> norm_spec                    -> mel2 [100,1430]
B) phones+bert(golden) -> AR -> 语义 tokens
C) decode_encp x2 (参考侧 / 目标侧) -> fea_ref[512,1488] / fea_todo[512,44]
D) DiT synthesize_v5_mel (4 步, cfg=0) -> mel_out [100,44]
E) denorm_spec -> vocoder (WebGPU) -> wav 21120 samples (0.66 s @32k 标注; 实际 48 kHz -> 0.44 s)
```

时序（CPU ggml + WebGPU ORT）：A 11.3 s / B 4.3 s / C 3.3 s / D 20.5 s / E 2.7 s ≈ 36 s。
DiT 的 4 步是 CPU ggml；切 `GSV_E2E_DEVICE=vulkan` 会快一个量级。

**输出是真实语音**（该 demo 早于 48 kHz 修复）：21114/21120 非零、频谱 70% 能量落在
100–4000 Hz 语音带、ZCR 0.0245（浊音）、谱峰 133/247/252 Hz（基频+谐波）。

> AR 在 11 个 token 处正常停止（golden phones 只有 32 个，语义密度匹配）；
> 想听更长的句子需要真正的文本前端接线（G2PW + tokenizer + BERT 前端），
> 那一步会把 phones 从 golden 换成真实输入。

## 7. 复现

```bash
# ORT 环境 (独立 uv env, 不动 diffsinger):
uv venv --python 3.12 .venv-wg
uv pip install --python .venv-wg/Scripts/python.exe onnxruntime onnxruntime-webgpu
uv pip install --python .venv-wg/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-wg/Scripts/python.exe onnx transformers pypinyin requests opencc tqdm x_transformers

# DML 环境 (可选, 用于 EP 对比):
uv venv --python 3.12 .venv-ort
uv pip install --python .venv-ort/Scripts/python.exe onnxruntime-directml

# golden
python tools/dump_golden_vocoder.py      # 或 .venv-wg/Scripts/python.exe
python tools/dump_golden_g2pw.py

# 模型 (vocoder ONNX 已导出; G2PW 需从 modelscope 拉一次, 见 tools/dump_golden_g2pw.py 注释)
# 构建 + 对拍
scripts\build-vocoder.bat && tests\relcpu\test_vocoder.exe
scripts\build-g2pw.bat    && tests\relcpu\test_g2pw.exe

# 端到端 (先 copy ggml dll!)
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\relcpu\
scripts\build-e2e.bat && tests\relcpu\demo_e2e.exe --ref tests/audio/ref_zh_32k.wav --out tests/e2e_out.wav
```
