// vocoder (HiFi-GAN Generator) ONNX 推理 —— onnxruntime C API, DML/WebGPU/CPU EP
// 语义对照 repo/GPT_SoVITS/module/models_onnx.py::Generator + TTS.py::init_vocoder 的 v4/v5 分支
//   (initial_channel=100, kernels [3,7,11], dilations [[1,3,5]]x3, upsample_rates [10,6,2,2,2],
//    upsample_kernel_sizes [20,12,4,4,4], upsample_initial_channel=512, is_bias=True, gin_channels=0)
//
// 关键约束 (docs/vocoder_onnx.md):
//   - **严格 fp32, 不量化**: 导出时图内 0 个量化算子; 不要对它跑 ORT 量化工具
//   - 输入 = denorm_spec() 之后的 mel [B,100,T] (值域约 [-12,2]), 不是 [-1,1] 的归一化 mel
//   - 输出 = wav [B,1,T*480] (两维都动态); 末端 tanh -> 严格 |wav| <= 1
//   - 单动态 session: DML/WebGPU 换形状会重新 plan, 按形状建 session 创建成本高, 不划算
//
// 加载方式: **运行时动态加载** (LoadLibraryA + GetProcAddress("OrtGetApiBase"))。
//   理由: onnxruntime.dll 只导出这一个符号, pip wheel 不带 import lib, 而 NuGet 只有 DML 包
//   (WebGPU EP 只发 pip wheel)。动态加载让我们在**同一个可执行文件**里按目录找 DLL,
//   并根据 DLL 实际支持的 EP 列表挑选最优 EP (见 resolve_ep)。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_vocoder_cfg {
    std::string device;        // "" = 自动 (GPU EP 优先, 回退 CPU); "cpu"; "gpu"/"dml"/"webgpu" 强制
    std::string ort_dll;       // onnxruntime.dll 的路径或目录 ("" = 系统搜索路径)
    int  n_threads = 0;        // CPU EP 的 intra-op 线程数 (0 = ORT 默认)
    bool verbose    = false;
};

class gsv_vocoder {
public:
    static gsv_vocoder * load(const std::string & onnx_path, const gsv_vocoder_cfg & cfg);
    ~gsv_vocoder();

    int  mel_dim()   const;    // 100
    int  upsample()  const;    // 480 (T 帧 -> T*480 采样 = prod(upsample_rates) 10*6*2*2*2)
    const char * ep_name() const;   // 实际生效的 EP (运行时决定)

    // mel: [100, T] torch 布局 (idx = m*T + t) —— **必须是 denorm_spec 之后的值**
    // wav: [T*480] (mono)
    bool run(const float * mel, int T, std::vector<float> & wav);

    // 同 run, 但 mel 为 ggml 布局 (idx = m + t*100); 内部转置后再喂 ORT
    bool run_ggml_mel(const float * mel, int T, std::vector<float> & wav);

private:
    gsv_vocoder();
    struct impl;
    impl * p;
};
