#pragma once

#include <string>
#include <vector>

struct gsv_vocoder_ggml_cfg {
    std::string device;   // same syntax as the engine device ("", "cpu", "vulkan", "vulkan:N", "N")
    int  n_threads = 0;
    bool verbose = false;
};

// HiFi-GAN generator (GPT-SoVITS V5 vocoder) in ggml: 100-bin mel -> 48 kHz waveform.
// Replaces the ONNX Runtime vocoder; weights come from tools/export_vocoder_gguf.py.
class gsv_vocoder_ggml {
public:
    static gsv_vocoder_ggml * load(const std::string & gguf_path, const gsv_vocoder_ggml_cfg & cfg);
    ~gsv_vocoder_ggml();
    gsv_vocoder_ggml(const gsv_vocoder_ggml &) = delete;
    gsv_vocoder_ggml & operator=(const gsv_vocoder_ggml &) = delete;

    int mel_dim() const;
    int upsample() const;
    int sample_rate() const;
    const char * device_name() const;

    // mel: torch layout [mel_dim, T] (idx = c*T + t), already de-normalized
    bool forward(const float * mel, int T, std::vector<float> & wav);

private:
    gsv_vocoder_ggml();
    struct impl;
    impl * p;
};
