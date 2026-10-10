// 参考音频 -> prompt semantic tokens（v5 的 _set_prompt_semantic 链的后半段）:
//   ssl [768,T] (CNHubert last_hidden_state 转置后, 50Hz) -> ssl_proj(Conv1d 768->768 k=2 s=2)
//   -> RVQ 编码 (EuclideanCodebook.quantize: argmin(x² - 2x·Eᵀ + E²)) -> codes [T/2] (25Hz)
// 另含 match_librosa 重采样（repo 的 16k 路径用 resampy kaiser_best 多相插值, 非 torchaudio）:
//   tools/export_resampy_filter.py -> models/resampy_kaiser_best.bin, 本模块读表复刻 TTS.py::resample。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_refcode_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;          // "" = CPU; "vulkan"/"gpu" = GPU
};

class gsv_refcode {
public:
    static gsv_refcode * load(const std::string & gguf_path, const gsv_refcode_cfg & cfg);
    ~gsv_refcode();

    int ssl_dim() const;         // 768
    int bins()    const;         // 1024

    // ssl: [768, T] torch 布局 (idx = d*T + t) -> codes [T/2] (int32), T 需为偶数(或末列忽略)
    // z_out 非空时输出 ssl_proj 的 [768, T/2] (调试/分段对拍用)
    bool codes_from_ssl(const float * ssl, int T, std::vector<int32_t> & codes,
                        std::vector<float> * z_out = nullptr);

    // ---- match_librosa 重采样 (resampy kaiser_best; repo TTS.py::resample 的等价复刻) ----
    static bool load_filter_table(const std::string & bin_path);
    static void resample_librosa(const float * x, int64_t n, int sr_in, int sr_out,
                                 std::vector<float> & out);

private:
    gsv_refcode();
    struct impl;
    impl * p;
};
