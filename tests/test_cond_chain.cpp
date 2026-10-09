// 条件链端到端对拍 (v5turbo): codes -> RVQ.decode -> x2 nearest -> enc_p (y) ->
//   bridge (k=1 + LeakyReLU) -> x2 nearest -> wns1 -> fea[512, 4Tc]
// golden: tests/golden/chain.* (Tc=40 -> T=80 -> 160; 由 tools/dump_golden_chain.py 生成)
// 用法: test_cond_chain [cond_gguf(rvq+bridge+wns1)] [encp_gguf] [golden_dir]
//   GSV_CHAIN_DEVICE=vulkan 统一三个模块的后端; GSV_CHAIN_BENCH=N 基准
#include "../src/gsv_cond.h"
#include "../src/gsv_encp.h"
#include "../src/gsv_wns1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

static double maxdiff_nan(const std::vector<float> & a, const std::vector<float> & b, size_t & n_nan) {
    if (a.size() != b.size()) return 1e30;
    double md = 0; n_nan = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite((double) a[i])) { n_nan++; continue; }
        double d = std::fabs((double) a[i] - b[i]);
        if (!(d <= md)) md = d;
    }
    return md;
}

// 相对判据: 参考张量自身的幅度 (Vulkan 后端 mul_mat 走 F16, 相对 ~1e-3 是已知底噪)
static double relmax(const std::vector<float> & a) {
    double m = 1.0;
    for (float v : a) m = std::max(m, (double) std::fabs(v));
    return m;
}

int main(int argc, char ** argv) {
    const char * cond_g = argc > 1 ? argv[1] : "models/gsv-cond-f32.gguf";
    const char * encp_g = argc > 2 ? argv[2] : "models/gsv-cond-encp.gguf";
    const char * gdir   = argc > 3 ? argv[3] : "tests/golden";

    const char * dev = getenv("GSV_CHAIN_DEVICE");
    int n_threads = getenv("GSV_CHAIN_THREADS") ? atoi(getenv("GSV_CHAIN_THREADS")) : 0;

    std::vector<float> fcodes = read_bin(std::string(gdir) + "/chain.codes.bin");
    std::vector<float> ftext  = read_bin(std::string(gdir) + "/chain.text.bin");
    std::vector<float> ge     = read_bin(std::string(gdir) + "/chain.ge.bin");
    std::vector<float> ref_y  = read_bin(std::string(gdir) + "/chain.y.bin");
    std::vector<float> ref_br = read_bin(std::string(gdir) + "/chain.bridge.bin");
    std::vector<float> ref_fea= read_bin(std::string(gdir) + "/chain.fea.bin");
    if (fcodes.empty() || ftext.empty() || ge.empty() || ref_fea.empty()) {
        fprintf(stderr, "missing chain golden\n"); return 1;
    }
    const int Tc = (int) fcodes.size();
    const int NT = (int) ftext.size();
    std::vector<int32_t> codes(Tc), text(NT);
    for (int i = 0; i < Tc; i++) codes[i] = (int32_t) fcodes[i];
    for (int i = 0; i < NT; i++) text[i] = (int32_t) ftext[i];

    gsv_cond_cfg ccfg;  ccfg.verbose = true; ccfg.device = dev ? dev : ""; ccfg.n_threads = n_threads;
    gsv_encp_cfg ecfg;  ecfg.verbose = true; ecfg.device = dev ? dev : ""; ecfg.n_threads = n_threads;
    gsv_wns1_cfg wcfg;  wcfg.verbose = true; wcfg.device = dev ? dev : ""; wcfg.n_threads = n_threads;
    gsv_cond * cond = gsv_cond::load(cond_g, ccfg);
    gsv_encp * encp = gsv_encp::load(encp_g, ecfg);
    gsv_wns1 * wns1 = gsv_wns1::load(cond_g, wcfg);
    if (!cond || !encp || !wns1) { fprintf(stderr, "load failed\n"); return 1; }

    const int T  = 2 * Tc;        // RVQ ×2 (25hz -> 50hz)
    const int T2 = 2 * T;         // bridge 后 ×2
    std::vector<float> q, y_tc, xb, m, logs, yenc, br, fea;

    auto run = [&]() {
        // 1. RVQ decode + ×2 nearest -> [768, T] (idx = d + t*768)
        cond->rvq_decode(codes.data(), Tc, true, q);
        // 2. -> enc_p 的 torch 布局 [768, T] (idx = c*T + t)
        y_tc.assign((size_t) 768 * T, 0.0f);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 768; c++)
                y_tc[(size_t) c * T + t] = q[(size_t) c + (size_t) t * 768];
        // 3. enc_p (y = proj 前的 encoder2 输出)
        encp->encode(y_tc.data(), T, text.data(), NT, ge.data(), m, logs, &yenc);
        // 4. -> bridge ggml 布局 [192, T] (idx = c + t*192)
        xb.assign((size_t) 192 * T, 0.0f);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 192; c++)
                xb[(size_t) c + (size_t) t * 192] = yenc[(size_t) c * T + t];
        // 5. bridge + ×2 -> [T2, 512] (idx = c*T2 + t == wns1 fea 布局)
        cond->bridge_run(xb.data(), T, true, br);
        // 6. wns1 (len = 全长, mask 全 1) -> fea [T2, 512] (idx = c*T2 + t)
        wns1->encode(br.data(), ge.data(), T2, T2, fea);
    };

    run();

    size_t ny = 0, nb = 0, nf = 0;
    double dy = ref_y.empty()  ? 0 : maxdiff_nan(yenc, ref_y, ny);
    double db = ref_br.empty() ? 0 : maxdiff_nan(br, ref_br, nb);
    double df = maxdiff_nan(fea, ref_fea, nf);
    const double thr_y = 2e-2 * relmax(ref_y);
    const double thr_b = 2e-2 * relmax(ref_br);
    const double thr_f = 2e-2 * relmax(ref_fea);
    if (!ref_y.empty())
        printf("  chain.y      max|d|=%.3e rel=%.2e %s\n", dy, dy / relmax(ref_y), dy < thr_y ? "PASS" : "FAIL");
    if (!ref_br.empty())
        printf("  chain.bridge max|d|=%.3e rel=%.2e %s\n", db, db / relmax(ref_br), db < thr_b ? "PASS" : "FAIL");
    printf("  chain.fea    max|d|=%.3e rel=%.2e %s%s\n", df, df / relmax(ref_fea),
           df < thr_f ? "PASS" : "FAIL", nf ? " (nan!)" : "");

    const char * be = getenv("GSV_CHAIN_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bn + 3; i++) {
            auto t0 = std::chrono::steady_clock::now();
            run();
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 3) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml chain (%s): avg %.2f ms min %.2f ms n=%d (Tc=%d)\n",
               dev ? dev : "cpu", sum / ts.size(), ts.front(), (int) ts.size(), Tc);
    }

    const bool ok = (df < thr_f && nf == 0 && (ref_y.empty() || (dy < thr_y && ny == 0)) &&
                     (ref_br.empty() || (db < thr_b && nb == 0)));
    printf("\n%s\n", ok ? "CHAIN PARITY PASS" : "CHAIN PARITY FAIL");
    delete cond; delete encp; delete wns1;
    return ok ? 0 : 1;
}
