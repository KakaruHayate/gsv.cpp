// Probe how the DiT forward cost splits between the reference (prompt) segment and the
// generated segment. Direction 2 (precompute the reference rows once per chunk) can only pay
// off in proportion to the reference share, so measure it instead of assuming it.
//
// Method: hold the generated length fixed and sweep the prompt length P. The slope of
// time-vs-P is the marginal cost of one reference frame; the intercept is everything else.
// Inputs are deterministic noise (timing does not depend on content).
//
//   probe_dit_share --models models --device vulkan [--dit PATH] [--gen 200] [--steps 32]
#include "gsv_dit.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fill(std::vector<float> & v, unsigned seed) {
    unsigned s = seed;
    for (auto & x : v) { s = s * 1664525u + 1013904223u; x = ((float) (s >> 8) / 16777216.0f - 0.5f) * 2.0f; }
}

int main(int argc, char ** argv) {
    std::string models = "models", device = "vulkan", dit_path;
    int gen = 200, steps = 32, reps = 2;
    bool quad = false;
    bool winsweep = false;
    int P0 = 0;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--models") models = nx();
        else if (a == "--device") device = nx();
        else if (a == "--dit") dit_path = nx();
        else if (a == "--gen") gen = atoi(nx().c_str());
        else if (a == "--steps") steps = atoi(nx().c_str());
        else if (a == "--reps") reps = atoi(nx().c_str());
        else if (a == "--quad") quad = true;
        else if (a == "--winsweep") winsweep = true;
        else if (a == "--P0") P0 = atoi(nx().c_str());
    }
    if (dit_path.empty()) dit_path = models + "/gsv-dev-placeholder.gguf";
    if (dit_path.find("placeholder") != std::string::npos)
        dit_path = models + "/gsv-v5dev-dit-f16all.gguf";

    gsv_dit_cfg cfg; cfg.device = device;
    gsv_dit * dit = gsv_dit::load(dit_path, cfg);
    if (!dit) { printf("[fail] load %s\n", dit_path.c_str()); return 1; }
    printf("dit=%s device=%s gen(frames)=%d steps=%d\n", dit_path.c_str(), device.c_str(), gen, steps);

    gsv_dit_alg alg;                       // all-off: no window, no cfg truncation
    alg.window = 0; alg.cfg_off_t = 0.0f; alg.uncond_reuse = 0; alg.ref_iso = 0;
    dit->set_alg(alg);

    if (winsweep) {
        // 窗口掩码能否被 Vulkan FA 的 tile-skip 兑现成时间:
        // 固定 P/gen, 只改 window. T 必须 >= block_cols*16 (~1024) 才会启用 mask_opt.
        const int P = P0 > 0 ? P0 : P0 == 0 ? 1024 : 1024;
        const int T = P + gen;
        std::vector<float> mu((size_t) 512 * T), prompt((size_t) 100 * P), noise((size_t) 100 * T), mel;
        fill(mu, 1u); fill(prompt, 2u); fill(noise, 3u);
        printf("\nwinsweep: P=%d gen=%d T=%d steps=%d\n", P, gen, T, steps);
        printf("%8s %12s %12s %10s\n", "window", "ms(warm)", "vs w=0", "speedup");
        double base = 0;
        std::vector<int> ws = { 0, 32, 64, 128, 256, 512, 0 };   // 末尾再测一次 w=0 作噪声对照
        for (int W : ws) {
            alg.window = W;
            dit->set_alg(alg);
            double warm = 0;
            for (int r = 0; r < reps + 1; r++) {
                const auto t0 = std::chrono::steady_clock::now();
                const bool ok = dit->sample(mu.data(), prompt.data(), P, T, steps, 1.3f, 0,
                                            noise.data(), mel);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count();
                if (!ok) { printf("[fail] W=%d\n", W); return 1; }
                if (r > 0) warm = ms;
            }
            if (W == 0) base = warm;
            printf("%8d %12.0f %12s %10s\n", W, warm, W == 0 ? "-" : "",
                   W == 0 ? "-" : (std::to_string(base / warm).substr(0, 5) + "x").c_str());
        }
        delete dit;
        return 0;
    }

    printf("\n%6s %6s %10s %10s %10s\n", "P", "T", "ms(cold)", "ms(warm)", "ms/frame(P)");
    std::vector<std::pair<int, double>> pts;   // (P, warm ms)
    const std::vector<int> plist = quad ? std::vector<int>{ 32, 200, 500, 800, 1200, 1800, 2800 }
                                        : std::vector<int>{ 32, 100, 200, 400, 600, 800 };
    for (int P : plist) {
        const int T = P + gen;
        std::vector<float> mu((size_t) 512 * T), prompt((size_t) 100 * P), noise((size_t) 100 * T), mel;
        fill(mu, 1u); fill(prompt, 2u); fill(noise, 3u);
        double cold = 0, warm = 0;
        for (int r = 0; r < reps + 1; r++) {          // first pass = cold (graph build)
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = dit->sample(mu.data(), prompt.data(), P, T, steps, 1.3f, 0,
                                        noise.data(), mel);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();
            if (!ok) { printf("[fail] sample P=%d T=%d\n", P, T); return 1; }
            if (r == 0) cold = ms; else warm = ms;
        }
        pts.push_back({ P, warm });
        printf("%6d %6d %10.0f %10.0f %10.3f\n", P, T, cold, warm, warm / (double) (T));
    }

    // least squares over the sweep: warm_ms = a + b*P  -> b is the marginal reference cost
    const size_t n = pts.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (auto & p : pts) { sx += p.first; sy += p.second; sxx += (double) p.first * p.first; sxy += p.first * p.second; }
    const double b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double a = (sy - b * sx) / n;
    double base = 0;
    for (auto & p : pts) { if (p.first == 32) base = p.second; }
    printf("\nfit: warm_ms = %.1f + %.3f * P\n", a, b);
    printf("A 32-frame reference segment costs ~%.0f ms of a %.0f ms forward (%.1f%%)\n",
           b * 32, base, 100.0 * (b * 32) / (base > 0 ? base : 1));
    printf("a 500-frame reference segment costs ~%.0f ms of a %.0f ms forward\n", b * 500, base);
    printf("=> freezing the reference rows once per chunk can save at most ~%.0f ms per step\n", b * 500);

    // 二次拟合: 只有 attention 项是 O(T^2). 用三个系数解 T^2 项的占比, 判断稀疏注意力有没有
    // 可榨取的算力 (若 c 接近 0, 任何 tile 级跳过都无从获益).
    if (n >= 3) {
        // 正规方程 3x3 (高斯消元)
        double S[3][4] = { {0} };
        for (auto & p : pts) {
            const double T = (double) (p.first + gen), y = p.second;
            double pw[3] = { 1.0, T, T * T };
            for (int r = 0; r < 3; r++) {
                for (int c2 = 0; c2 < 3; c2++) S[r][c2] += pw[r] * pw[c2];
                S[r][3] += pw[r] * y;
            }
        }
        for (int i = 0; i < 3; i++) {
            int piv = i;
            for (int r = i + 1; r < 3; r++) if (fabs(S[r][i]) > fabs(S[piv][i])) piv = r;
            if (fabs(S[piv][i]) < 1e-12) { printf("quad fit singular\n"); break; }
            if (piv != i) for (int c2 = 0; c2 < 4; c2++) std::swap(S[i][c2], S[piv][c2]);
            for (int r = 0; r < 3; r++) {
                if (r == i) continue;
                const double f = S[r][i] / S[i][i];
                for (int c2 = i; c2 < 4; c2++) S[r][c2] -= f * S[i][c2];
            }
        }
        const double c0 = S[0][3] / S[0][0], c1 = S[1][3] / S[1][1], c2 = S[2][3] / S[2][2];
        printf("\nquad fit: warm_ms = %.1f + %.4f * T + %.3e * T^2\n", c0, c1, c2);
        for (double T : { 1000.0, 2000.0, 4000.0, 8000.0 }) {
            const double tot = c0 + c1 * T + c2 * T * T;
            const double att = c2 * T * T;
            printf("  T=%5.0f: total %8.0f ms, quadratic part %8.0f ms (%.1f%%)\n",
                   T, tot, att, 100.0 * att / (tot > 0 ? tot : 1));
        }
    }
    delete dit;
    return 0;
}
