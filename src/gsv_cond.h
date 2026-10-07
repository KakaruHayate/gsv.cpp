// 条件段 (V5 decode_encp 链) 的 ggml 实现
//   当前: RVQ (quantizer.decode, n_q=1 -> 纯 codebook gather) + 25hz->50hz 的 x2 nearest
//   后续: bridge / ref_enc / wns1 / enc_p 陆续加入同一模块
// 张量布局: 与 ggml 原生一致 —— 量化结果 [ne0=dim, ne1=T], 展平 idx = d + t*dim
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_cond_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备
};

class gsv_cond {
public:
    static gsv_cond * load(const std::string & gguf_path, const gsv_cond_cfg & cfg);
    ~gsv_cond();

    int rvq_dim()  const;     // 768
    int rvq_bins() const;     // 1024

    // codes: [T] 语义 token (0..bins-1, 来自 AR)
    // out  : [dim, T] 行主序 (idx = d + t*dim); upsample_x2=true 时 [dim, 2T] (nearest)
    void rvq_decode(const int32_t * codes, int T, bool upsample_x2, std::vector<float> & out);

private:
    gsv_cond();
    struct impl;
    impl * p;
};
