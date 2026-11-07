// Single-shot DiT forward on a golden fixture, for attention-map statistics.
//   probe_attn_scan --device vulkan --models models --golden tests\golden --family s32c13_big
//                   [--steps N] [--cfg F] [--tag NAME]
// The dump itself is driven by gsv_dit's env vars:
//   GSV_DIT_ATTN_LEGACY=1  GSV_ATTN_DUMP=<dir>  GSV_ATTN_MODE=summary
//   GSV_ATTN_TS=0,0.25,0.5,0.75   GSV_ATTN_LAYERS / GSV_ATTN_HEADS (empty = all)
#include "gsv_dit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool read_bin(const std::string & p, std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) (n / 4));
    const bool ok = fread(v.data(), 4, v.size(), f) == v.size();
    fclose(f);
    return ok;
}

static bool read_scalar(const std::string & p, float & f, int & i) {
    std::vector<float> v;
    if (!read_bin(p, v) || v.empty()) return false;
    f = v[0];
    std::memcpy(&i, &v[0], 4);
    return true;
}

int main(int argc, char ** argv) {
    std::string device = "vulkan", models = "models", golden = "tests/golden", fam = "s32c13_big", tag;
    int steps_override = 0;
    int t_crop = 0, p_crop = 0;
    float cfg_override = -1.0f;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "--device") device = next();
        else if (a == "--models") models = next();
        else if (a == "--golden") golden = next();
        else if (a == "--family") fam    = next();
        else if (a == "--steps")  steps_override = atoi(next().c_str());
        else if (a == "--cfg")    cfg_override   = (float) atof(next().c_str());
        else if (a == "--t-crop") t_crop = atoi(next().c_str());
        else if (a == "--p-crop") p_crop = atoi(next().c_str());
        else if (a == "--tag")    tag    = next();
        else { printf("unknown arg %s\n", a.c_str()); return 2; }
    }

    const std::string base = golden + "/cfm." + fam;
    std::vector<float> mu, prompt, noise, out;
    if (!read_bin(base + ".mu.bin", mu) || !read_bin(base + ".prompt.bin", prompt) ||
        !read_bin(base + ".noise.bin", noise)) {
        printf("fixture %s not found under %s\n", fam.c_str(), golden.c_str());
        return 1;
    }
    int T = (int) (mu.size() / 512);
    int P = (int) (prompt.size() / 100);
    // 几何仿真: 裁掉开头的帧 (只保留最后 T 帧) / 把 prompt 缩到最后 P 帧。
    // 内容不再自洽 (参考与 mu 不匹配), 只用来观察生产工况 (T=672/P=32) 的注意力几何。
    if (t_crop > 0 && t_crop < T) {
        mu.erase(mu.begin(), mu.begin() + (size_t) 512 * (T - t_crop));
        noise.erase(noise.begin(), noise.begin() + (size_t) 100 * (T - t_crop));
        T = t_crop;
    }
    if (p_crop > 0 && p_crop < P) {
        prompt.erase(prompt.begin(), prompt.begin() + (size_t) 100 * (P - p_crop));
        P = p_crop;
    }
    printf("[scan] geometry: T=%d P=%d (crop t=%d p=%d)\n", T, P, t_crop, p_crop);
    float cf = 0.0f, sf = 0.0f;
    int st = 0, sti = 0;
    read_scalar(base + ".cfg.bin", cf, sti);
    read_scalar(base + ".steps.bin", sf, st);
    if (st < 1 || st > 1000) st = (int) (sf + 0.5f);
    const int   steps = steps_override > 0 ? steps_override : st;
    const float cfg   = cfg_override >= 0.0f ? cfg_override : cf;

    const char * ditname = (fam.compare(0, 3, "s32") == 0) ? "gsv-v5dev-dit-f16all.gguf"
                                                           : "gsv-dit-f16all.gguf";
    const std::string dpath = models + "/" + ditname;
    printf("[scan] %s tag=%s T=%d P=%d steps=%d cfg=%.2f dit=%s\n",
           fam.c_str(), tag.c_str(), T, P, steps, cfg, ditname);

    gsv_dit_cfg c;
    c.device = device;
    gsv_dit * dit = gsv_dit::load(dpath, c);
    if (!dit) { printf("dit load failed\n"); return 1; }

    std::vector<float> x0 = noise;
    for (float & v : x0) v *= 0.875f;
    std::vector<float> mel;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = dit->sample(mu.data(), prompt.data(), P, T, steps, cfg, 12345ull,
                                x0.data(), mel);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    if (!ok) { printf("sample failed\n"); return 1; }
    float mx = 0.0f;
    double mean = 0.0;
    for (float v : mel) { mx = std::max(mx, std::fabs(v)); mean += v; }
    printf("[scan] ok mel=%zu frames mean=%.6f max=%.4f  %.0f ms\n",
           mel.size() / 100, mean / (double) mel.size(), mx, ms);
    delete dit;
    return 0;
}
