// GPT-SoVITS v2 / v2Pro / v2ProPlus 的 S2 解码器 (VITS 结构, 与 v5 的 cond+DiT 完全不同)。
//
// 数据流（repo module/models.py::SynthesizerTrn，version="v2"）:
//
//   sem codes (25 Hz, 1024 码本)
//     └─ quantizer.vq.decode(codebook)              → [768, Tc]
//     └─ F.interpolate(nearest, ×2)                 → [768, 2Tc]        (50 Hz)
//     └─ enc_p(ssl_proj + encoder_ssl×3 + text_embedding/encoder_text×6
//              + MRTE(cross-attn) + encoder2×3 + proj)  → m_p, logs_p   (由 gsv_encp 复用)
//   z_p = m_p + randn*exp(logs_p)*noise_scale        （noise_scale 默认 0.5）
//   z   = flow(z_p, g=ge, reverse=True)              ← 4 × ResidualCouplingLayer(mean_only)
//   y   = dec(z, g=ge)                               ← HiFi-GAN, 5 段上采样 (10,8,2,2,2)
//
// 参考音频的全局嵌入 ge（[512]）由 ref_enc (MelStyleEncoder) 给出 —— 与 v5 的 ref_enc 完全
// 同一个模块，因此这里直接复用 gsv_refenc（同一个 GGUF 里的 ref_enc.* 前缀）。
//
// 与 v5 的差别（本文件存在的理由）:
//   * 没有 DiT / CFM，也没有 cond 段的 bridge/wns1；解码由 flow+dec 一步到位给出波形。
//   * 码本只有一层 (n_q=1, 1024 bins)，语义码本身就是最终码；v5 走 RVQ + bridge。
//   * flow 的耦合层是 mean_only=True（无 log_scale 参数），10 层 WaveNet ≠ v5 的 cond WN。
//   * dec 的 gin_channels 是全通道条件 (cond: 512→512, 1×1)，结构同 v5 vocoder 但权重是
//     torch weight_norm (weight_g/weight_v)，必须在加载时折叠成稠密核。
//
// 权重布局: GGUF 里张量名就是 checkpoint 的键名（enc_p.* / flow.* / dec.* / ref_enc.* /
// quantizer.*）。weight_v / weight_g 未折叠、未重排（torch [OC,IC,K]）；
// 普通 conv 已被导出器重排成 ggml 的 [K,IC,OC]。本模块加载时把两者统一折叠成 **F32**
// 的 [K,IC,OC]（ggml_conv_direct_1d 要求 F32 权重），ConvTranspose1d 折叠成
// 「相位主序」的子像素卷积核 [M, IC, OC*stride]（见 gsv_v2sovits.cpp 顶部推导）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_v2sovits_cfg {
    int n_threads = 0;
    bool verbose = false;
    std::string device;      // 同其它模块: "" = CPU, "vulkan"/"gpu"/"N"
    // enc_p 所在的 GGUF（空 = 与 s2 同一个文件；v2 家族两者同文件）
    std::string encp_file;
};

class gsv_v2sovits {
public:
    static gsv_v2sovits * load(const std::string & gguf_path, const gsv_v2sovits_cfg & cfg);
    ~gsv_v2sovits();

    // ---- 元数据（来自 GGUF KV + 权重形状校验）----
    const std::string & version() const;   // "v2" / "v2pro" / "v2proplus"
    int sample_rate() const;               // 32000 / 32000 / 44100
    int upsample() const;                  // Π ups = sr / 50  (v2 家族恒为 640)
    int codebook_size() const;             // 1024
    int hidden() const;                    // 192 (inter_channels)
    int gin_channels() const;              // 512
    int frame_rate() const;                // 50 (25 Hz 语义码 ×2)

    // ---- S1: 参考侧 ssl → 语义码（repo SynthesizerTrn.extract_latent）----
    // ssl: [768, T] torch 行主 (flat[c*T + t])；codes: [T/2]  (ssl_proj k=2 stride=2 + VQ)
    bool codes_from_ssl(const float * ssl, int T, std::vector<int32_t> & codes);

    // 语义码 → quantized [768, 2*Tc] torch 行主（码本 + nearest×2）
    void decode_codes(const int32_t * codes, int Tc, std::vector<float> & q768) const;

    // ---- S2: enc_p ----
    // codes [Tc] + phones [n_phones] + ge [512] → m/logs [192, T] torch 行主
    // speed != 1 时按 repo 的 F.interpolate(mode="linear") 口径改变帧数 T
    bool encode_p(const int32_t * codes, int Tc, const int32_t * phones, int n_phones,
                  const float * ge, std::vector<float> & m, std::vector<float> & logs, int & T,
                  double speed = 1.0);

    // ---- S3: flow (reverse) ----
    // zp/z: [192, T] torch 行主；ge [512]
    bool flow_reverse(const float * zp, int T, const float * ge, std::vector<float> & z);

    // ---- S4: dec (HiFi-GAN) ----
    // z: [192, T] torch 行主 → wav [T * upsample()]
    bool decode_wav(const float * z, int T, const float * ge, std::vector<float> & wav);

    // ---- 全链 ----
    // codes + phones + ge → wav。seed 决定 z_p 的噪声（gsv_rng，noise_scale 默认 0.5）。
    // zp_override != NULL 时跳过 enc_p 的采样，直接用它当 z_p（对拍用，长度必须 = 192*T）。
    bool synthesize(const int32_t * codes, int Tc, const int32_t * phones, int n_phones,
                    const float * ge, std::vector<float> & wav,
                    double speed = 1.0, uint64_t seed = 0, float noise_scale = 0.5f,
                    const float * zp_override = nullptr, int T_override = 0);

    // 分段耗时（最近一次 synthesize 的毫秒数，诊断/benchmark 用）
    void last_stage_ms(double & encp_ms, double & flow_ms, double & dec_ms) const;

private:
    gsv_v2sovits();
    struct impl;
    impl * p;
};
