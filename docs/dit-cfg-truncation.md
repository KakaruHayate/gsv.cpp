# DiT CFG 截断与 Uncond 复用：实验报告

本文量化验证 0.1.2 中唯一保留的 DiT 级加速开关：**CFG 截断**（`dit_cfg_off_t`）与
**Uncond 复用**（`dit_uncond_reuse`）。所有数据在**修好 ggml 构建之后**采集。

- 硬件/后端：RTX 2070 8GB / Vulkan（`ggml-vulkan`，coopmat2 开启）
- DiT 权重：`gsv-v5dev-dit-f16all.gguf`，cfg=1.30，32 步
- 测量工具：`tests/test_dit_opt.cpp`（确定性 DiT 级 A/B，mel 域）、
  `tests/probe_dit_share.cpp`（成本结构探测器）
- **质量指标统一用 DiT mel 输出**：`relL1 = Σ|Δ|/Σ|base|`、`corr`（Pearson）

## 1. 方法与测量口径

### 1.1 为什么必须在 DiT 级做 A/B

端到端（HTTP → AR → DiT → vocoder）有两个不可控源：

1. **AR 输出长度不稳定**：同一文本、同一 seed 下，参考音频与 prompt 文本不匹配时 AR 会
   提前输出 EOS。本机实测某段 7.87s 参考音频配不匹配的 prompt 文本时，2 字与 12 字文本
   都只出 0.48s，长度完全不随文本变化。
2. **自举参考会发散**：用模型合成一段音频再当参考合成下一段，长度出现 1.36s → 10.32s →
   0.56s 的抖动，无法作为可控基准。

因此统一用仓库既有 fixture 直接驱动 `gsv_dit::sample()`：固定
`mu / prompt / noise / steps / cfg`，只改算法开关，输出逐位可比。

### 1.2 fixture 规模

| fixture | T（总帧） | P（参考帧） | 生成帧 | 对应语音 | steps | cfg |
|---|---|---|---|---|---|---|
| `s32c13` | 96 | 32 | 64 | ≈0.64s | 32 | 1.30 |
| `s32c13_big` | 1000 | 500 | 500 | ≈5.0s | 32 | 1.30 |

帧率 100 Hz（10ms/帧）。`s32c13_big` 对应本机最常见的真实工况（5 秒句 + 5 秒参考）。

### 1.3 基线定义与确定性

"全关档"= `cfg_off_t=0, uncond_reuse=0`，即未加速的上游口径。所有变体的 `relL1/corr`
都是**对全关档**在生成段上计算的。

注意：`gsv_dit_alg` 的默认值是生产值（0.7），所以"全关档"必须在代码里显式清零，
否则会静默带上默认开关。

同参数连跑两次，`corr=1.000000, max|Δ|=0`，逐位一致。所以下面所有差异都只来自被改的开关。

### 1.4 golden 绝对对齐的适用边界（重要限制）

修好 ggml 之后，Vulkan 在 `s4c0` fixture 上与 torch golden 对齐到
`max|Δ|=6.636e-3, relL1=1.145e-3, corr=0.999999`——后端本身是可信的。

但两个 `s32c13*` fixture（v5dev 权重、32 步）对 golden 的偏差大得多：

| fixture | vs golden max\|Δ\| | relL1 | corr |
|---|---|---|---|
| `s4c0`（turbo 权重, 4 步） | 6.636e-3 | 1.145e-3 | 0.999999 |
| `s32c13` | 1.028 | 0.182 | 0.9829 |
| `s32c13_big` | 0.929 | 0.136 | 0.9892 |

这**不影响本文结论**：判定全部基于"变体 vs 全关档"的内部 A/B（同一进程、同一 fixture、
同一后端），这类比较不受 golden 口径影响。但它意味着 `s32c13*` 的 golden 文件与当前
v5dev gguf 的对应关系尚未确立，**不能**用它做绝对数值断言。

## 2. 机制

每步 CFG 要跑两次网络（cond + uncond）：

```
v = v_cond + cfg * (v_cond - v_uncond)
```

- **截断**：`t >= dit_cfg_off_t` 的步不再算 uncond，等价 `cfg_scale = 1.0`。
  采样循环里 `t = j/steps` 递增（`t≈1` 是低噪声端），所以 `t >= cfg_off_t` 截的是
  **轨迹末段**的 Uncond。跳过比率 = `1 − dit_cfg_off_t`。
- **复用**：偶数步实算 uncond；奇数步 `u1` 直接沿用上一步，`u2` 线性外推
  `2·neg[k−1] − neg[k−2]`。
- `v5turbo`（cfg=0）没有 Uncond 分支，两者在其上都是**逐位无操作**。

为什么末段可以截：`x_t` 在 `t→1` 已接近干净数据，cond 与 uncond 的速度场趋同，
去掉 CFG 放大的影响有限。因此误差应随跳过比率**平缓增长**，而不是阈值型崩塌——这正是
§3 实测到的形状。

## 3. 实验结果（`s32c13_big`，T=1000, P=500, gen=500）

| 配置 | 跳过比率 | mel relL1 | mel corr | 热时间(ms) | 加速 |
|---|---|---|---|---|---|
| 全关 | 0% | — | 1.00000 | 4863 | 1.00x |
| cfg0.95 | 5% | 2.454e-4 | 1.00000 | 4998 | 1.01x |
| cfg0.90 | 10% | 6.507e-4 | 1.00000 | 4949 | 1.02x |
| cfg0.85 | 15% | 8.274e-4 | 1.00000 | 4912 | 1.02x |
| cfg0.80 | 20% | 1.150e-3 | 1.00000 | 4767 | 1.06x |
| **cfg0.70（默认）** | 30% | 1.582e-3 | 0.99999 | 4516 | 1.11x |
| cfg0.60 | 40% | 1.971e-3 | 0.99999 | 4195 | 1.20x |
| cfg0.50 | 50% | 2.421e-3 | 0.99998 | 3780 | 1.33x |
| cfg0.70 + u1 | 30% | 4.166e-3 | 0.99999 | 3471 | **1.45x** |
| cfg0.70 + u2 | 30% | 2.643e-3 | 0.99999 | 3609 | 1.39x |
| cfg0.60 + u1 | 40% | 4.052e-3 | 0.99998 | 3264 | **1.54x** |

短 fixture（`s32c13`，gen=64）趋势相同、绝对值更大（同样的跳过比率影响更集中）：

| 配置 | mel relL1 | 加速 |
|---|---|---|
| cfg0.90 | 1.526e-3 | 1.05x |
| cfg0.70 | 3.767e-3 | 1.12x |
| cfg0.50 | 5.706e-3 | 1.34x |
| cfg0.70 + u1 | 4.768e-3 | 1.45x |
| cfg0.60 + u1 | 5.197e-3 | 1.53x |

**噪声地板**：`s32c13_big` 上全关档与自身比较读作 1.03x（run-to-run ±3%），所以上表
1.06x 以下的行不构成可测加速；1.20x 及以上显著。短 fixture 单次前向仅 82–100ms，自比可差
22%，计时只以 `s32c13_big` 为准。

## 4. 质量/速度曲线的解读

- **误差随跳过比率近似线性**：2.5e-4 → 6.5e-4 → 1.2e-3 → 1.6e-3 → 2.0e-3 → 2.4e-3
  （5%→50%）。没有阈值型崩塌，说明"末段 Uncond 贡献小"的假设成立。
- **加速比符合前向次数模型**：跳过比率 r 砍掉 `r/2` 的前向，理论上限 `1/(1−r/2)`；
  r=30% → 1.18x、r=50% → 1.33x，与实测 1.11x / 1.33x 吻合（差值来自 DiT 之外的固定开销）。
- **复用比纯截断更划算**：`cfg0.70+u1` 用 1.45x 换来 4.2e-3，而纯截断要拿到同等加速需
  r≈60%（外推会掉到 2.0e-3 一档之外）。
- **`u2` 优于 `u1`**：在 cfg0.70 上 `u2`（线性外推）误差 2.6e-3 < `u1` 4.2e-3，因为外推
  补偿了速度场随步的漂移。但两者都比纯截断大一个量级，这是"省掉网络调用"的必然代价。

## 5. 选点建议

| 目标 | 配置 | mel relL1 | 加速 |
|---|---|---|---|
| 保真优先（几乎无口感变化） | `dit_cfg_off_t=0.90` | 6.5e-4 | 1.02x |
| 平衡（默认） | `dit_cfg_off_t=0.70` | 1.6e-3 | 1.11x |
| 速度优先 | `dit_cfg_off_t=0.70` + `dit_uncond_reuse=1` | 4.2e-3 | 1.45x |

三者都与 cache-dit 叠加，且在 v5turbo 上都是精确无操作。

## 6. 测量陷阱（为什么早期的数字不可用）

早期一轮给出的"波形 corr 1.00000 / relL1 1e-3"是无效的，原因有三，记录在此以免重犯：

1. **随包分发的 ggml 构建曾有一个坏版本**（早于 llama.cpp 的 vocoder 卷积内核修复），
   输出带巨大直流分量。该缺陷会**压缩所有变体之间的差异**，使任何开关都显得"近无损"。
   现已重建 ggml，`s4c0` 上与 torch golden 对齐到 `corr=0.999999`。
2. **用了波形相关度当保真度指标**。声码器相位敏感：mel 上 0.34% 的相对差异在波形上会
   放大到 15%（同一对配置：mel relL1 3.4e-3 / corr 0.99999 vs 波形 relL1 0.155 /
   corr 0.969）。波形指标既会掩盖真缺陷，也会夸大无害差异。本文一律用 mel 域指标。
3. **测量脚本自身的问题**：
   - Windows PowerShell 5.1 的 `Invoke-WebRequest -Body <string>` 按 ANSI 编码，中文被
     发成 `?`（曾因此误判"不同文本产出相同音频"）。改用显式 `UTF8.GetBytes`。
   - `text_split_method` 默认 `cut5` 会按标点切段，`mel_out` dump 只保留最后一段，
     造成"文本越长帧数越短"的假象。基准必须传 `cut0`。
   - 端口曾被两个侧车进程共用，请求可能被另一个前端损坏的进程应答。基准需固定唯一端口。

## 7. 复现方式

```powershell
# DiT 级开关扫描（确定性，含确定性自检与 golden 参考）
scripts\build-dit-opt.bat
tests\relcpu\test_dit_opt.exe --device vulkan --models models --golden tests\golden --family s32c13_big
tests\relcpu\test_dit_opt.exe --device vulkan --models models --golden tests\golden --family s32c13

# DiT 成本结构（参考段边际成本 + 二次项拟合，用于核对"注意力/FFN 各占多少"）
tests\relcpu\probe_dit_share.exe --models models --device vulkan --gen 200 --steps 32
tests\relcpu\probe_dit_share.exe --models models --device vulkan --gen 200 --steps 32 --quad --reps 1

# 端到端（HTTP）：确认开关在真实服务里生效、且 504 超时可触发
powershell -ExecutionPolicy Bypass -File tests\e2e_dit_alg.ps1
```

## 8. 未完成项

1. 与 cache-dit 叠加的联合曲线（当前两者各自独立测；前端 `auto` 档对 v5dev 会自动开
   F8，届时交互需要实测）。
2. `s32c13*` fixture 与 golden 的偏差（relL1 0.14–0.18）来源需查清，否则这些 fixture
   只能用于内部 A/B，不能用于绝对数值断言。
