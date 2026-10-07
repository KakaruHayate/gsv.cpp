# V5 vocoder → ONNX 导出（fp32，严格不量化）

日期：2026-10-07 · 机器同前（RTX 2070 / onnxruntime 1.23.0，providers: DmlExecutionProvider + CPUExecutionProvider）

## 0. 结论

| 项 | 结果 |
|---|---|
| 模型 | `Generator`（HiFi-GAN 风格，48 kHz，14.4 M 参数） |
| 导出物 | `models/vocoder_fp32.onnx`，**57,751,020 B**（= 14.4M × 4 B，无膨胀） |
| sha256 | `13f95a88f2c1cf9c619f921038d19cfc4f38d40aaf5c9ce78c2937c61b998386` |
| 精度 | 与 torch fp32 对拍 max\|Δ\| = **2.8e-05 ~ 1.1e-04**，corr = 1.000000000，无 NaN |
| 量化 | **无**：图内 0 个 `Quantize*/QLinear*` 算子，全部 initializer 为 float32（已程序校验） |
| 图 | opset 17，249 节点：Conv 92 / LeakyRelu 86 / Add 55 / ConvTranspose 5 / Tanh 1（即 HiFi-GAN 原结构，无额外改写） |
| 接口 | `mel [B,100,T]` → `wav [B,1,T×480]`（两维都动态） |

复现：

```bash
python tools/export_vocoder_onnx.py            # 导出 + 对拍 + 基准
# 或只导出:  --out <path> --opset 17 --skip-bench
```

## 1. 导出要点（踩坑记录）

1. **必须 `remove_weight_norm()` 先于 `load_state_dict()`**。官方 `vocoder.pth` 是"weight_norm 已物化"的
   平铺 `weight/bias` 版本；若先加载再 remove，`weight_g/weight_v` 缺失会被 `strict=False` 静默吞掉，
   得到的是一半随机权重的模型。脚本里 `load_state_dict` 返回 `<All keys matched successfully>` 即为证据。
2. 配置取自 `TTS_infer_pack/TTS.py::init_vocoder` 的 **v4/v5 分支**（`initial_channel=100`,
   kernels `[3,7,11]`, dilations `[[1,3,5]]×3`, `upsample_rates=[10,6,2,2,2]`,
   `upsample_kernel_sizes=[20,12,4,4,4]`, `upsample_initial_channel=512`, `is_bias=True`, `gin_channels=0`）。
3. 输入是 **`denorm_spec()` 之后的 mel**（值域 ≈ [-12, 2]），不是归一化后的 [-1,1]；输出末端是 `tanh`
   （严格 |wav| ≤ 1）。
4. 导出用 TorchScript exporter（`dynamo=False`）+ `do_constant_folding=True`，opset 17。

## 2. 稳健性对拍

| 用例 | max\|Δ\| | corr | 输出范围 |
|---|---|---|---|
| 随机 T=100 / 257 / 500 / 1000（N(0,1) 输入） | 2.8e-5 ~ 3.9e-5 | 1.000000000 | [-1, 1] |
| **denorm 真实量程 U[-12,2]，T=300** | 8.6e-07 | 1.000000000 | [-0.071, 0.066] |
| 全 -12（下界常数输入） | 3.3e-09 | 0.999999283 | ~0 |
| 全 +2（上界常数输入） | 1.1e-04 | 1.000000000 | [-1, 1]（tanh 饱和） |
| T=1（极短） | 4.7e-07 | 1.000000000 | — |

无 NaN/Inf；常数极值输入下也只是 fp32 归约差异量级。

## 3. 基准（每次调用 = 一段 mel → 音频）

| mel 帧数 | 音频时长 | torch CPU | torch CUDA (fp32) | **ORT CPU** | **ORT DML** |
|---|---|---|---|---|---|
| 200 | 2.0 s | 1256 ms | 72.7 ms | 558 ms | 47.9 ms |
| 500 | 5.0 s | 2933 ms | 134.9 ms | 1398 ms | 223 ms |
| 1000 | 10.0 s | 5245 ms | 299.3 ms | 2840 ms | 988 ms / **180–296 ms*** |

\* DML 存在**形状切换陷阱**：同一个 session 内换 mel 长度会退化（10 s 音频 296 ms），
而"每个形状一个全新 session"是线性的（10 s ≈ 180 ms，≈0.09 ms/帧）。但 **session 创建成本 DML 435 ms /
CPU 183 ms** → 对"一句一个长度"的 TTS 负载，按形状建 session 不划算（4 次同形状调用才回本）。

**集成结论：用单个动态 session**。RTF ≈ 0.03（10 s 音频 296 ms），vocoder 不是瓶颈；
若将来批量任务中同一长度反复出现（例如固定 640 帧 chunk 的批处理），可加一个 LRU 的形状-session 缓存。

CPU 路径：ORT 比 torch 快 **1.85×**（2840 vs 5245 ms）。

## 4. C++ 引擎集成方案

- **依赖**：onnxruntime C/C++ API——与 G2PW 的 ONNX 复用同一份 ORT（依赖合并，不再引入新的推理栈）。
- **调用形状**：每句一次（V5 的 vocoder 在 `synthesize_v5_mel` 之后整句调用；chunk 上限 1000 帧 → 10 s →
  480 000 samples ≈ 1.9 MB），跨 runtime 的拷贝一次即可忽略；追求零拷贝可用 IOBinding 把已有 buffer
  绑给 session 的输入/输出。
- **provider 选择**：有 GPU 时用 DML（≈ torch CUDA 水平，且不占 torch 显存池）；纯 CPU 时 ORT 仍比 torch 快 1.85×。
- **严格不量化**：导出即 fp32；**不要**对这份 ONNX 跑 ORT 量化工具（`quantize_dynamic` 等），
  也不要在 vocoder 路上引入 int8/fp16 权重——量化误差在波形末端直接可听，这也是最初决策
  （报告 §6-3）明确的约束。仓库里已用程序断言"图内无量化算子"，可作为回归检查。
- **边界情况**：T=1、极值输入已验；DML 对动态形状会重新 plan，属正常代价。

## 5. 与 DiffSinger 做法的对应

DiffSinger 的 `nsf_hifigan` 也是"backbone trace + 整体（含 LPC/激励）导出"；我们的 V5 vocoder
**不含 NSF 激励源**（纯 HiFi-GAN，无 pitch/phase 输入），因此导出更简单——只需 mel 一个输入，
这也意味着将来若要换 `pc-nsf-hifigan` 类声码器，接口上要额外引入 f0 通道。
