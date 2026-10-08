// ref_enc (V5 MelStyleEncoder: 704 -> 512 的全局风格向量 ge) ggml impl.
// 语义对照 repo/GPT_SoVITS/module/modules.py 的 MelStyleEncoder (v5:
//   self.ref_enc = MelStyleEncoder(704, style_vector_dim=512)), golden 见 tests/golden/refenc.*
// 链路: refer[:, :704] * refer_mask  ->  spectral(Linear+Mish x2)
//       -> 2 x Conv1dGLU(k=5,pad=2: conv(128->256), split, a*sigmoid(b), +residual)
//       -> 帧 mask -> 2 头自注意力(d_k=d_v=64, 温度 sqrt(128), mask=-inf) + fc + 残差
//       -> Linear(128->512) -> 有效帧均值池化 -> ge[512]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_refenc_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备
};

class gsv_refenc {
public:
    static gsv_refenc * load(const std::string & gguf_path, const gsv_refenc_cfg & cfg);
    ~gsv_refenc();

    int in_dim()  const;      // 704
    int out_dim() const;      // 512

    // spec: [in_dim, T] ggml 布局 (idx = c + t*in_dim) —— 即 refer[:, :704] 的逐帧通道向量
    // mask: [T], 1 = 有效帧, 0 = padding (与 refer_mask 同义; 参考实现在入口做 spec*mask)
    // ge  : [out_dim]
    bool encode(const float * spec, const float * frame_mask, int T, std::vector<float> & ge);

private:
    gsv_refenc();
    struct impl;
    impl * p;
};
