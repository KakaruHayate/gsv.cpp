# 前后端分离：`gsv_ggml_cli` server + Python 前端

日期：2026-10-14 · 机器同前（RTX 2070 / ggml CPU+Vulkan / ORT WebGPU）

## 0. 结论

把 C++ 侧从"一堆 `test_*.cpp` 入口"收敛成**一个常驻 server**，Python 前端通过 IPC 驱动它。
模型只加载一次；前端可以在 Python 里自由迭代文本处理；换 EP / 调 DiT 参数只动 server 侧。

| 组件 | 文件 | 说明 |
|---|---|---|
| 引擎门面 | `src/gsv_engine.{h,cpp}` | 复用全部 `gsv_*.cpp`，只做编排与布局转换 |
| server | `tools/gsv_ggml_cli.cpp` | 常驻，一次性加载 7 个模型 |
| 传输层 | `src/ipc/*` | 两平面 framing（vendored from openutau-vst-bridge，MPL-2.0） |
| Python 前端 | `tools/frontend/{gsv_client,frontend,tts}.py` | 协议客户端 + 文本前端 + 驱动 |

**端到端**（`python -m tools.frontend.tts --text ...`）：

```
参考音频(4.38s) -> prepRef    prompt codes=124 / ge[512] / mel2[100,438]   2.0 s
文本 "大家好，这是一段用来测试的语音合成。"
                -> bert        [1024, 34]                                    0.3 s
                -> ar          sem=19                                       1.0 s
                -> synth       条件段 + DiT(4步) + vocoder -> wav 1.14 s     20.5 s
```

长句（111 phones）→ 语义 token 112 个 → **6.72 s 语音**（95% 能量在 100–4000 Hz 语音带）。
短句有时停在 11 个 token（0.66 s）——见 §6，**torch 在同一输入上也停 11**，是模型行为。

## 1. 为什么这么切

repo 的 `TTS.py` 本身就把推理分成三段：`set_ref_audio`（参考音频，可缓存）→ `get_phones_and_bert`
（文本前端）→ T2S + CFM + vocoder。我们的切法照抄这个边界：

| 阶段 | 谁做 | 为什么 |
|---|---|---|
| 文本规整 / G2PW 多音字 / 音素化 | **Python**（套 repo 的 `text/`） | 与 ckpt 一起标定的口径，重写必然引入偏差；且文本处理迭代快 |
| BERT 分词 | **Python**（HF BertTokenizer） | 只分词，不加载权重 |
| BERT 前向 → AR → 条件段 → DiT → vocoder | **C++ server** | 我们已对拍通过的 ggml/ORT 实现 |

> BERT 前向放 C++ 侧（`gsv_bert::encode_feat` 就是 `hidden_states[-3]` 且已去 CLS/SEP），
> server 按前端传来的 `word2ph` 做 repo `get_bert_feature` 的逐字重复。前端**不依赖 torch 权重**。
> **非 zh 语言 repo 直接给全零**（`TextPreprocessor.get_bert_inf`），所以 en/ja/ko 连 BERT
> 都不跑 —— 少一次前向。

## 2. 传输：两平面 framing

设计照搬 `KakaruHayate/openutau-vst-bridge` 的 `PROTOCOL.md`（同作者、MPL-2.0，直接 vendored）：

```
连接: TCP 127.0.0.1:<动态端口>。server 在 %TEMP%/gsv-ggml/servers/<name>.json 写 discovery;
      Python 扫该目录, 并**探活**(connect 一次)以剔除崩溃 server 留下的死文件。

控制面 (LF 结尾 UTF-8 行):
  request:<uuid>:<kind> <json>\n       客户端 -> server
  response:<uuid> <json>\n             server -> 客户端   {success, data, error, tensors}
  notification:<kind> <json>\n         server -> 客户端   (连接建立时发 ready)
  close\n                              干净关闭

数据面 (长度前缀二进制帧):
  tensor <name> <xxh64十进制> <len>\n<len bytes raw>
```

**为什么大块走数据面**：wav / mel2 / bert 都是 float32 大块（10 s 音频 1.9 MB）。塞进 JSON 要
base64（+33%）且整体拼内存 —— bridge 的 §15 明确否掉了这条路。

**响应里带 `tensors: [...]` 字段**列出后续帧名：否则客户端分不清"这个响应没有数据帧"和
"数据帧还没到"，只能靠超时猜。

**frame 头是三段式**（name/hash/len）而 bridge 是两段（hash/len）：我们的响应里有多个具名张量。

## 3. 命令集

| kind | 入参 (JSON) | 数据面 (客户端→server) | 回传数据面 |
|---|---|---|---|
| `ping` | – | – | – |
| `prepRef` | `{wavPath}` | – | `codes`(i32) `ge`(f32×512) `mel2`(f32) + `melT` |
| `bert` | `{idsLen, word2phLen}` | `ids`(i32) `word2ph`(i32) | `feat`(f32×1024·P) + `phones` |
| `ar` | `{maxSteps, earlyStop, seed}` | `phones`(i32) `prompt`(i32) [`bert`(f32)] | `sem`(i32) |
| `synth` | `{melT, steps, cfg, seed}` | `sem` `phones` `prompt`(i32) `ge` `mel2`(f32) | `wav`(f32) + `sampleRate` |

## 4. 复用 bridge 的实现（MPL-2.0）

| 文件 | 作用 |
|---|---|
| `stream.h` | `ByteStream` 抽象 —— 测试可喂脚本化字节流 |
| `socket.{h,cpp}` | loopback socket → `ByteStream`（`ws2_32` / BSD） |
| `frame.{h,cpp}` | 两平面 line/frame 解析与构造（改了 frame 前缀与三段式） |
| `reader.{h,cpp}` | `FrameReader`：行 + 精确长度负载**共用一个缓冲**（数据帧部分预读不丢） |
| `hash.{h,cpp}` | XXH64（十进制字符串上线，避开 JSON 2^53 陷阱） |
| `discovery.{h,cpp}` | discovery 文件 + 端口探活 + 剪枝 |
| `session.{h,cpp}` | 请求路由 / 响应封装 / 数据帧收发（重写消息集） |

`reader.h` 的"共用一个缓冲"与"长度上限防对端耗尽内存"是这类 IPC 最容易出 bug 的地方，直接拿来。

**framing 冒烟测试**（`tests/ipc_probe.cpp`，`scripts/build-ipc-probe.bat`）专门覆盖两个经典坑：
数据帧跨两次读（按 1/3/7 字节喂）、以及头已预读但负载未到。

## 5. Python 前端（多语言）

`tools/frontend/frontend.py` 只做**编排**，G2P 全部转调 repo 的 `text/`：

- **zh**：`text.chinese2.g2p`（G2PW 多音字消歧）→ phones/word2ph；
- **en**：`text.english`；ja / ko / yue 同构（需要各自的依赖，未装的会打印提示并跳过）；
- `bert_ids`：`AutoTokenizer` 对 `norm_text` 编码（**含 CLS/SEP**，前向在 C++ 侧）。

两个必须知道的**路径/环境约束**：

1. `text/chinese2.py` 在 **import 时** 就用相对 cwd 的 `GPT_SoVITS/text/G2PWModel` 构造
   `G2PWPinyin`。所以前端必须 `chdir` 到 **repo 根**并把该模型放到位 ——
   `scripts/link-g2pw-model.bat` 用 junction 指向本仓库的 `models/G2PWModel`
   （该路径被 repo 自己的 `.gitignore` 忽略）。
2. 英文 G2P 需要 NLTK 数据；nltk 3.10 要 `averaged_perceptron_tagger_eng`（磁盘上没有、网又下不动），
   故 `scripts/fetch-nltk-data.bat` 从已有的旧名目录复制一份到仓库本地 `.nltk_data`
   （已 gitignore），或把 nltk 钉到 3.8.1。

## 6. 与 torch 的对照（AR 只出 11 个 token 的判案）

短文本经常只生成 11 个语义 token（0.66 s）。11 恰好是 EOS 解锁的边界（AR 前 11 步屏蔽 EOS），
所以看起来像"引擎卡住了"。用**同一份输入**喂 torch 版 AR（`tools/probe_ar_torch.py`）：

```
inputs: phones=(1,34) bert=(1,1024,34) prompt=(1,164)
torch AR: 总长=175, 生成=11 (idx=11)
ours  sem=11
```

**结论：torch 也停在 11 —— 是模型行为，不是移植问题。** 两侧 token **长度一致**；具体取值不同，
因为我们的采样用 xorshift RNG 而 torch 用 `torch.multinomial`（同一分布的不同抽样）。

输入侧已逐项对拍：`bert` 与 HF `hidden_states[-3]` 经 word2ph 重复后
**max|d| = 4.2e-3 / refmax 26（相对 1.6e-4）、corr 0.99999995**（F16 档底噪）；
`prompt codes` 与 torch 100% 一致（refcode 链）；`phones` 直接来自 repo 前端。

> 参考音频必须 **3~10 s**（repo `_set_prompt_semantic` 会 `OSError` 拒掉其余长度）。
> `tests/audio/ref_zh_short_16k.wav`（4.38 s）是合适的默认素材。

## 7. 踩坑记录

1. **`setvbuf(stdout, nullptr, _IOLBF, 0)` 让 MSVC CRT 崩掉**（0xC0000409 fast-fail）。
   零长缓冲 + 行缓冲的组合在 CRT 里会炸；用 `_IONBF`（仓库里其它 test main 都是这么写的）。
   这个坑很隐蔽：崩在 `main` 的第一行，而日志因缓冲根本没出来，看起来像"进程没启动"。
2. **子进程 stdout 用 `PIPE` 而不读 → 死锁**。server 日志（尤其 `--verbose` + ORT 警告）填满
   64 KB 管道缓冲后，server 永久阻塞在写日志上，客户端表现为**某个请求超时**。
   改成重定向到日志文件即可（`GSV_SERVER_LOG` 可指定路径）。
3. **discovery 文件会留死亡条目**：server 崩了文件还在，客户端连上去被拒。
   客户端必须**探活**（connect 一次）再返回端口，并顺手删掉死文件。
4. **峰值归一化要先去掉直流**。vocoder 输出可能带极小 DC 偏置，直接按峰值归一会把偏置当峰值，
   把整段压成静音（实测踩到）。`tts.py::normalize` 先减均值再归一。
5. **`gsv_dit::synthesize` 必须显式传 `mel_ref_T`**（见 docs/dit_ggml.md）：repo 先
   `min(mel2.T, fea_ref.T)` 再取末 500 帧，早期实现误用 R 当列数 → 越界 → NaN。
6. **中文注释写在 `.bat` 里会让 cmd 解析错位**（代码页 + 行尾），脚本保持纯 ASCII。
7. **heredoc 里的 `\n` 会变真换行**，写进 C 字符串字面量就断行（C2001）。用 Edit 工具改。

## 8. 复现

```bash
# 1) ORT + 前端依赖 (独立 uv env, 不动 diffsinger)
uv venv --python 3.12 .venv-wg
uv pip install --python .venv-wg/Scripts/python.exe onnxruntime onnxruntime-webgpu onnx
uv pip install --python .venv-wg/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-wg/Scripts/python.exe transformers pypinyin requests opencc tqdm \
    x_transformers cn2an jieba_fast wordsegment g2p_en nltk==3.8.1 matplotlib pytorch_lightning xxhash

# 2) 一次性设置
scripts\link-g2pw-model.bat        # G2PW 模型 -> repo 期望路径 (junction)
scripts\fetch-nltk-data.bat        # 英文 G2P 的 NLTK 数据 -> 本地 .nltk_data

# 3) 构建 server
scripts\build-cli.bat              # -> tests/relcpu/gsv_ggml_cli.exe
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\relcpu\

# 4) 跑
python tools\frontend\tts.py --ref tests/audio/ref_zh_short_16k.wav --outdir tests/e2e_out \
    --text "你好，欢迎使用语音合成系统。"
# 也可先测协议: python tools\frontend\smoke_client.py
# server 单独起: tests\relcpu\gsv_ggml_cli.exe --device vulkan --verbose
```

## 9. 未做 / 已知限制

- **ja / ko / yue 只接了路由**：缺 `pyopenjtalk` / `jamo` / `ToJyutping` 依赖，装了可用；
- **auto 模式的中英混合**：repo 会逐段拼接 BERT 特征（zh 段真、其余全零），本版退化为全零
  并在 stderr 提示；音素侧不受影响；
- **AR 采样 RNG 与 torch 不同**：分布一致但抽样不同（要逐 token 一致需把 torch 的
  `multinomial` 状态搬过来，属另一件事）；
- **输出质量还可调**：短句有时停在 EOS 边界（torch 亦然），可试不同 seed / 参考音频；
- **单连接串行**：server 每连接一线程，引擎共享（模块按 T 重建图、无跨请求可变状态），
  但同一时刻多请求会串行争用 GPU。
