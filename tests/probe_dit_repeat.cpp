// Probe: is gsv_dit::synthesize repeatable for identical inputs in one process?
//   probe_dit_repeat <dit.gguf> <dump_dir> [device] [runs]
// dump_dir must contain fea_ref.f32 / fea_todo.f32 / mel2.f32 (written by GSV_ENGINE_DUMP).
#include "gsv_dit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> rd(const std::string & p) {
    std::vector<float> v;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) (n / 4));
    if (fread(v.data(), 4, v.size(), f) != v.size()) v.clear();
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s <dit.gguf> <dump_dir> [device] [runs]\n", argv[0]); return 2; }
    const std::string gguf = argv[1], dir = argv[2];
    const char * dev = argc > 3 ? argv[3] : "";
    const int runs = argc > 4 ? atoi(argv[4]) : 3;

    gsv_dit_cfg cfg;
    cfg.device = dev ? dev : "";
    gsv_dit * dit = gsv_dit::load(gguf, cfg);
    if (!dit) { printf("dit load failed\n"); return 1; }

    const auto fea_ref = rd(dir + "/fea_ref.f32");
    const auto fea_todo = rd(dir + "/fea_todo.f32");
    const auto mel2 = rd(dir + "/mel2.f32");
    const int R = (int) (fea_ref.size() / 512);
    const int N = (int) (fea_todo.size() / 512);
    const int Tm = (int) (mel2.size() / 100);
    printf("device=%s  R=%d N=%d mel2_T=%d\n", cfg.device.empty() ? "cpu" : cfg.device.c_str(), R, N, Tm);
    if (R <= 0 || N <= 0 || Tm <= 0) { printf("bad dump\n"); return 1; }

    uint64_t prev = 0;
    for (int i = 0; i < runs; i++) {
        std::vector<float> mel;
        const bool ok = dit->synthesize(fea_ref.data(), R, fea_todo.data(), N, mel2.data(), Tm,
                                        32, 0.0f, 12345ull, 0, nullptr, mel, 0);
        if (!ok) { printf("run%d: synthesize failed\n", i + 1); return 1; }
        uint64_t h = 1469598103934665603ull;    // FNV-1a over the mel bytes
        for (float v : mel) {
            unsigned char b[4];
            std::memcpy(b, &v, 4);
            for (int k = 0; k < 4; k++) { h ^= b[k]; h *= 1099511628211ull; }
        }
        float mx = 0.0f, mean = 0.0f;
        for (float v : mel) { mx = std::max(mx, std::fabs(v)); mean += v; }
        printf("run%d: mel=%zu hash=%016llx  mean=%.6f max=%.4f %s\n", i + 1, mel.size() / 100,
               (unsigned long long) h, mean / (double) mel.size(), mx,
               (i == 0 || h == prev) ? "" : "<-- DIFFERS from previous run");
        prev = h;
    }
    delete dit;
    return 0;
}

