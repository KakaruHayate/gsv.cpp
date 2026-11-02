// Standalone smoke test for the ggml HiFi-GAN vocoder.
//   test_vocoder_ggml <gsv-vocoder.gguf> [T] [--cpu]
#include "gsv_vocoder_ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s <vocoder.gguf> [T] [--cpu]\n", argv[0]); return 2; }
    int T = 40;
    gsv_vocoder_ggml_cfg cfg;
    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--cpu") == 0) cfg.device = "cpu";
        else if (std::strcmp(argv[i], "--vulkan") == 0) cfg.device = "vulkan";
        else if (argv[i][0] == '-' && argv[i][1] == '-') { /* ignore */ }
        else T = std::atoi(argv[i]);
    }
    gsv_vocoder_ggml * v = gsv_vocoder_ggml::load(argv[1], cfg);
    if (!v) { printf("load failed\n"); return 1; }
    printf("device=%s mel=%d ups=%d sr=%d T=%d\n", v->device_name(), v->mel_dim(), v->upsample(),
           v->sample_rate(), T);

    std::vector<float> mel((size_t) v->mel_dim() * T);
    for (size_t i = 0; i < mel.size(); i++)
        mel[i] = -2.0f + 3.0f * (float) ((i * 37) % 97) / 97.0f;

    std::vector<float> wav;
    if (!v->forward(mel.data(), T, wav)) { printf("forward failed\n"); return 1; }
    double sum = 0.0, mx = 0.0;
    for (float x : wav) { sum += x; mx = fmax(mx, fabs(x)); }
    printf("wav=%zu expected=%d mean=%.5f max=%.5f\n", wav.size(), T * v->upsample(),
           sum / (double) wav.size(), mx);
    delete v;
    return 0;
}
