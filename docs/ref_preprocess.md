# 参考音频预处理：wav / 重采样 / mel_fn_v4（+ 测试素材）

日期：2026-10-10 · 机器同前（RTX 2070 / Vulkan + CPU）

## 0. 结论

- `src/gsv_mel.{h,cpp}`：**wav 读取**（RIFF PCM16/24/32、IEEE float32，多声道取均值）、
  **带限 sinc 重采样**（torchaudio.functional.resample 等价）、**mel_fn_v4**（repo 逐式对齐）、
  `norm_spec/denorm_spec`。第三方依赖只有 vendored 的 `third_party/pocketfft_hdronly.h`（MPL-2.0）。
- 对拍（`tools/dump_golden_ref.py` 生成 golden，`tests/test_mel.cpp`，`scripts/build-mel.bat`）：

| 项 | zh（32k 源 14.3s） | en（16k 源 5.55s） |
|---|---|---|
| 重采样 → 32k / 16k | **0 / 2.4e-7** | **1.8e-7 / 0**（max\|d\|） |
| mel（log 域，vs torch） | 4.3e-4 | 1.3e-3 |
| mel_norm | 6.2e-5 | 1.9e-4 |
| 帧数 T | 1430（一致） | 555（一致） |

重采样基本逐位（残差来自 torch 的 float32 累加）；mel 差 1e-3 量级是 FFT 舍入经 log 放大的结果，
相对 refmax≈11.5 只有 1e-4 —— 下游（ref_enc/DiT）的容差在 1e-2 量级，余量充足。

## 1. 测试素材：参考音频从哪来

机器上现成的音频都是**歌声**（hanser 数据集、银河录 主音）——不适用于参考音频；HF 网络在本机
不可达（`datasets-server` 连接失败）。改用 **Windows 本地 TTS 合成 speech**（零依赖、可复现）：

```powershell
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$f32 = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(32000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SelectVoice('Microsoft Huihui Desktop')          # zh-CN; en 用 'Microsoft Zira Desktop'
$s.SetOutputToWaveFile('tests\audio\ref_zh_32k.wav', $f32)
$s.Speak('大家好，这是一段用于语音合成链路测试的参考音频。……')
```

产物：`tests/audio/ref_zh_32k.wav`（Huihui，32k，14.3s）、`tests/audio/ref_en_16k.wav`
（Zira，16k，5.55s）——两种源采样率也顺带覆盖了上/下重采样两个方向。

## 2. 与 repo 的逐式对齐（`mel_processing.py::mel_spectrogram_torch` + TTS.py 的 `mel_fn_v4`）

参数：`n_fft=win=1280, hop=320, n_mels=100, sr=32000, fmin=0, fmax=None(→sr/2), center=False`。

关键点（易错处）：

1. **无条件 reflect pad**：repo 在调用 `torch.stft(center=False)` **之前**手工 pad
   `int((n_fft-hop)/2) = 480` **两侧对称**（不是 game.cpp 那种 480/481 的非对称写法）。
   于是 `T = 1 + (n + 2·480 − 1280)/320 = n/320`。
2. hann 窗为**周期窗**（`torch.hann_window` 默认 periodic）。
3. 幅度 `|X| = sqrt(re² + im² + 1e-8)`（float32），mel 用 **librosa Slaney** 三角滤波器组
   （htk=False、norm='slaney'、fmax=sr/2），最后 `log(clamp(x, min=1e-5))`。
4. `norm_spec = (x + 12)/14·2 − 1`（spec_min=-12, spec_max=2，`TTS.py:133`）。

## 3. 重采样：torchaudio 的等价公式（用冲激响应反推核对）

`TTS.py` 用 `torchaudio.transforms.Resample(sr0, sr1)`（= `functional.resample`，
sinc_interp_hann / lowpass_filter_width=6 / rolloff=0.99）。其等价直式：

```
y[o] = Σ_m x[m] · 2·fc·sinc(2·fc·(τ_o − m)) · hann((τ_o − m)/W)
    τ_o = o·sr_in/sr_out（输出时刻，输入样本单位）
    fc  = 0.5·min(1, sr_out/sr_in)·rolloff（每输入样本周期）
    W   = 6/(2·fc)（hann 半宽）；核偶对称 → 相位符号无关
```

核对（单冲激，`torchaudio` 实测 vs 上式）：16k→32k 中心 **0.99**、半样本邻 **0.6259**；
32k→16k 中心 **0.495**、2 样本外 **4.67e-3** —— 逐位吻合。C++ 端实现后与 golden 的最大差
**2.4e-7**（残差 = double 累加 vs torch float32）。

> 踩坑：第一版在"约减率网格"（orig/nw）上建核，等价于把核频率乘了 nw 倍——上采样相位全错
> （16k→32k 的 max|d| 达 0.66）。直式的正确形式是**每输入样本周期上按 fc 采样**。

## 4. 复现

```bash
# golden（diffsinger env；torchaudio 2.11 走 torchcodec, 本环境缺 DLL → 脚本用 soundfile/stdlib wave 读 wav）
python tools/dump_golden_ref.py
# 构建 + 对拍（纯 host 侧, CPU Release）
scripts\build-mel.bat && tests\relcpu\test_mel.exe
```

## 5. 参考音频 → prompt semantic tokens（v5 `_set_prompt_semantic` 链的后半段）

`src/gsv_refcode.{h,cpp}` + `tools/convert_refcode.py`（→ `models/gsv-refcode.gguf`：ssl_proj 拆成
w0/w1 两个 [IC,OC] + codebook）+ `tools/export_resampy_filter.py`（→ resampy kaiser_best 表）：

```
16k wav(含 9600 零尾) → z-score(HF Wav2Vec2FeatureExtractor: 零均值单位方差, eps 1e-7)
  → HuBERT(last_hidden_state^T = ssl [768,T], 50Hz)
  → ssl_proj = Conv1d(768,768,k=2,stride=2)  [w0·x[2i] + w1·x[2i+1] + b]      (25Hz)
  → RVQ 编码 argmin(x² − 2x·Eᵀ + E²)  (E = quantizer.vq.layers.0._codebook.embed)
  → codes [T/2] (int32) = AR 的 prompt semantic tokens
```

对拍（`tests/test_refcode.cpp`，golden 由 `tools/dump_golden_refcode.py` 生成；三段素材 zhs/en/zh32）：

| 项 | zhs（16k 源 4.4s） | en（16k 源 5.6s） | zh32（32k 源 14.3s） |
|---|---|---|---|
| match_librosa 重采样 | 0（16k 源直通） | 0 | **1.8e-7** |
| ssl_proj 输出（vs torch） | **3.5e-5** | **2.8e-5** | **3.5e-5** |
| codes（由 golden ssl 输入） | **124/124** | **153/153** | **372/372** |

- **重采样**：复刻 repo 的 `resample(match_librosa=True)`（resampy kaiser_best 表 + 双边窗口 +
  表内线性插值 + 索引 clamp）—— 与 torch 逐位一致（残差 ≤1.8e-7 = double 累加 vs torch f32）。
- **ssl_proj + RVQ**：由 golden 的 ssl 出发，codes 与 torch **逐帧完全一致**。踩坑记录：
  3D 权重 `ne{K,IC,OC}` 的 ic 维被 k 打断 → 切片 view 无法直接喂 mul_mat，转换器改成拆两份；
  `ggml [bins,T2]` 的平铺是 `b + bins*i`（不是行主 `b*T2+i`）；argmin 初值取 `D[bins*i]`。
- **全链（wav→重采样→z-score→我们的 HuBERT→ssl_proj→argmin）**：codes 与 torch 的匹配率
  **83% / 89% / 80%** —— 差异全部来自我们 HuBERT 本身的输出偏差（下表），非 refcode 段。

### 5.1 "HuBERT 变长偏差" 已定位：测试输入口径不一致（非引擎问题）

初测时全链 codes 匹配率只有 80~89%、ssl 差 0.09~6 —— 追查结论：**不是 HuBERT 的实现问题**，
而是**测试与 golden 的输入口径不一致**：`dump_golden_refcode.py` 按 repo 生产路径
（`cnnhuhbert_model.model(wav16k)`）喂**原始 16k 波形**，而测试当时先做了 z-score
（Wav2Vec2FeatureExtractor 的口径）。统一口径后：

| 项 | zhs (T=248) | en (T=307) | zh32 (T=745, 14.3s) |
|---|---|---|---|
| 我们的 HuBERT ssl vs torch | **2.3e-5** | **1.8e-5** | **4.8e-4** |
| 全链 codes vs torch | **124/124 (100%)** | **153/153 (100%)** | **372/372 (100%)** |

分段定位（`tests/probe_hubert_stages.cpp` + `tools/dump_golden_hubert_stages.py`，torch 侧逐段 dump
feature_projection / pos_conv / LN / 逐层）：g1out(feature_projection) 7.2e-6、全链 ssl 7.7e-6、
用 torch 的 LN 输出隔离 g2 为 0 —— 即 CNN 前端 / pos_conv / LN / 12 层 transformer 全部对齐。
**结论：HuBERT 变长（按 T 重建图 + pos_conv 内核 56 帧分块）精度合格**，T=49 的合成输入 golden
仍 6.0e-6；本节的 0.09~6 是口径问题。

> 口径说明（管线要注意）：repo 的参考 token 路径**不做 z-score**（直接把 `torchaudio.load` 的
> float 波形喂 HubertModel；归一化只在 Wav2Vec2FeatureExtractor 里，而那条路径绕过了它）。
> 旧的 T=49 对拍 golden 用的是"手工 z-score 后喂模型"的口径（脚本自洽）——引擎本身与口径无关，
> **是否 z-score 由调用方决定**，管线按 repo 走"原始波形"。
