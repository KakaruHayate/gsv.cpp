// Minimal repro: gsv_dit sample() vs golden, no alg API. Used to bisect the
// CPU/Vulkan divergence. Build: see scripts/build-dit-opt.bat (same flags).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include "gsv_dit.h"

static bool read_bin(const std::string & p, std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    const size_t rd = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return rd == v.size();
}

int main(int argc, char ** argv) {
    std::string device = argc > 1 ? argv[1] : "cpu";
    std::string models = argc > 2 ? argv[2] : "models";
    std::string golden = argc > 3 ? argv[3] : "tests/golden";
    const std::string base = golden + "/cfm.s4c0";
    std::vector<float> mu, prompt, noise, out;
    if (!read_bin(base + ".mu.bin", mu) || !read_bin(base + ".prompt.bin", prompt) ||
        !read_bin(base + ".noise.bin", noise) || !read_bin(base + ".out.bin", out)) {
        printf("fixture read failed\n");
        return 2;
    }
    const int T = (int) (mu.size() / 512), P = (int) (prompt.size() / 100);
    std::vector<float> x0 = noise;
    for (float & v : x0) v *= 0.875f;

    gsv_dit_cfg cfg;
    cfg.device = device;
    gsv_dit * dit = gsv_dit::load(models + "/gsv-dit-f16all.gguf", cfg);
    if (!dit) { printf("load failed\n"); return 2; }
    std::vector<float> mel;
    if (!dit->sample(mu.data(), prompt.data(), P, T, 4, 0.0f, 1, x0.data(), mel)) {
        printf("sample failed\n");
        return 2;
    }
    const int G = T - P;
    double mx = 0, sa = 0, na = 0;
    for (int c = 0; c < 100; c++)
        for (int t = 0; t < G; t++) {
            const double a = out[(size_t) c * T + P + t];
            const double b = mel[(size_t) c * G + t];
            const double d = fabs(a - b);
            if (d > mx) mx = d;
            sa += d; na += fabs(a);
        }
    printf("[repro %s] T=%d P=%d max|d|=%.4e relL1=%.3e\n", device.c_str(), T, P, mx, sa / (na + 1e-30));
    delete dit;
    return 0;
}
