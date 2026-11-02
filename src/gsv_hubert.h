// HuBERT (chinese-hubert-base) ggml 实现 —— 接手自另一条线的 host 版本
// 结构: z-score 音频 -> 7 层 CNN 前端 (group norm 只在 L0) -> feat_proj(LN+Linear)
//       -> pos_conv(k=128 groups=16, host, grouped conv 不适配 ggml conv_1d)
//       -> enc LN -> 12 层 post-LN transformer (D=768 NH=12 HD=64 FFN 3072 gelu_erf)
// 布局: CNN 段用 ggml_conv_1d 原生 [T, C, B]; 进入 feat_proj 后转 [C, T] (mul_mat 友好);
//       transformer 与 BERT 同构, 复用 flash_attn_ext 与融合 LN (layernorm_affine)。
// 对拍 golden: tests/golden/hubert.* (torch fp32, 来自 dump_golden_hubert.py)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_hubert_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备
};

class gsv_hubert {
public:
    static gsv_hubert * load(const std::string & gguf_path, const gsv_hubert_cfg & cfg);
    ~gsv_hubert();

    // audio: 16kHz 单声道波形 (变长; >= 720 采样, 帧数 = (n-400)/320+1)。
    //        是否 z-score 由调用方决定: repo 的参考 token 路径喂原始波形 (不做归一化);
    //        旧 T=49 golden 用的是"手工 z-score 后喂模型"的口径。
    // out        : [768, T] 行主序 (idx = d + t*768), 即 encoder last_hidden_state^T
    //              帧数或采样数变化时自动重建图 (与 wns1 同款按需重建)
    bool encode(const float * audio_norm, int n_samples, std::vector<float> & out);

    int  hidden() const;      // 768
    int  n_frames() const;    // 49

private:
    gsv_hubert();
    struct impl;
    impl * p;
};
