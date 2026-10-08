// ref_enc (MelStyleEncoder) 对拍 + 基准
// golden: tests/golden/refenc.* (T=200, len=180); 输入 [704,T] t 主序 (torch [1,704,T])
// 用法: test_refenc [gguf] [golden_dir]; GSV_REFENC_DEVICE=vulkan, GSV_REFENC_BENCH=N
#include "../src/gsv_refenc.h"

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

int main(int argc, char ** argv) {
    const char * mdl  = argc > 1 ? argv[1] : "models/gsv-cond-f32.gguf";
    const char * gdir = argc > 2 ? argv[2] : "tests/golden";

    gsv_refenc_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_REFENC_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_REFENC_THREADS")) cfg.n_threads = atoi(nt);

    gsv_refenc * m = gsv_refenc::load(mdl, cfg);
    if (!m) return 1;

    std::vector<float> xr = read_bin(std::string(gdir) + "/refenc.input.bin");   // torch [1,704,T]
    std::vector<float> mr = read_bin(std::string(gdir) + "/refenc.mask.bin");    // torch [1,1,T]
    std::vector<float> gr = read_bin(std::string(gdir) + "/refenc.out.bin");     // torch [1,512,1]
    if (xr.empty() || mr.empty() || gr.empty()) { fprintf(stderr, "missing ref_enc golden\n"); delete m; return 1; }

    const int C = m->in_dim();
    const int T = (int) (xr.size() / C);
    std::vector<float> x((size_t) C * T), fm((size_t) T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) x[(size_t) c + (size_t) t * C] = xr[(size_t) c * T + t];
    for (int t = 0; t < T; t++) fm[t] = mr[(size_t) t];

    std::vector<float> ge;
    if (!m->encode(x.data(), fm.data(), T, ge)) { delete m; return 1; }
    size_t n_nan = 0;
    double md = maxdiff_nan(ge, gr, n_nan);
    printf("  ref_enc ge [%d] max|d|=%.3e %s%s\n", m->out_dim(), md,
           md < 2e-2 ? "PASS" : "FAIL", n_nan ? " (含非有限值!!)" : "");

    const char * be = getenv("GSV_REFENC_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bn + 3; i++) {
            auto t0 = std::chrono::steady_clock::now();
            m->encode(x.data(), fm.data(), T, ge);
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 3) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml ref_enc (%s): avg %.3f ms min %.3f ms n=%d (T=%d)\n",
               cfg.device.empty() ? "cpu" : "vulkan", sum / ts.size(), ts.front(), (int) ts.size(), T);
    }

    const bool ok = (md < 2e-2) && n_nan == 0;
    printf("\n%s\n", ok ? "REF_ENC PARITY PASS" : "REF_ENC PARITY FAIL");
    delete m;
    return ok ? 0 : 1;
}
