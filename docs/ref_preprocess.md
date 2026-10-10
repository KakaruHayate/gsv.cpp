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
