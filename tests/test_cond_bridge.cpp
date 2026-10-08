// bridge (Conv1d 192->512 k=1 + LeakyReLU 0.01) 对拍 + 基准
// golden: tests/golden/bridge.input.bin [1,192,T] / bridge.out.bin [1,512,T] (torch, t 主序)
// 用法: test_cond_bridge [gguf] [golden_dir]; GSV_COND_DEVICE=vulkan 切后端, GSV_COND_BENCH=N 基准
#include "../src/gsv_cond.h"

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

// NaN 感知 (fabs(NaN) > md 恒 false, 会把全 NaN 输出误报 PASS)
static double maxdiff(const std::vector<float> & a, const std::vector<float> & b, size_t & n_nan) {
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

    gsv_cond_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_COND_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_COND_THREADS")) cfg.n_threads = atoi(nt);

    gsv_cond * m = gsv_cond::load(mdl, cfg);
    if (!m) return 1;

    std::vector<float> raw_in  = read_bin(std::string(gdir) + "/bridge.input.bin");
    std::vector<float> raw_ref = read_bin(std::string(gdir) + "/bridge.out.bin");
    if (raw_in.empty() || raw_ref.empty()) { fprintf(stderr, "missing bridge golden\n"); delete m; return 1; }

    const int CIN  = m->bridge_in_dim();    // 192
    const int COUT = m->bridge_out_dim();   // 512
    const int T    = (int) (raw_ref.size() / COUT);
    if ((size_t) T * CIN != raw_in.size()) {
        fprintf(stderr, "golden 尺寸不符: in %zu, ref %zu (CIN=%d COUT=%d T=%d)\n",
                raw_in.size(), raw_ref.size(), CIN, COUT, T);
        delete m; return 1;
    }

    // golden 输入是 torch [1,192,T] (t 主序, idx = c*T + t); 接口要 ggml 布局 [192,T] (idx = c + t*192)
    std::vector<float> x((size_t) CIN * T);
    for (int c = 0; c < CIN; c++)
        for (int t = 0; t < T; t++)
            x[(size_t) c + (size_t) t * CIN] = raw_in[(size_t) c * T + t];

    std::vector<float> out;
    m->bridge_run(x.data(), T, /*upsample_x2=*/false, out);
    size_t n_nan = 0;
    double md = maxdiff(out, raw_ref, n_nan);
    printf("  bridge_out [%d,%d] max|d|=%.3e %s%s\n", T, COUT, md,
           md < 2e-2 ? "PASS" : "FAIL", n_nan ? " (含非有限值!!)" : "");

    // x2 nearest 结构检查: out2[2t] == out2[2t+1] == out[t]
    std::vector<float> out2;
    m->bridge_run(x.data(), T, /*upsample_x2=*/true, out2);
    int bad = 0; size_t n_nan2 = 0;
    if (out2.size() != (size_t) 2 * T * COUT) { bad = 1; }
    else {
        // 布局: t 主序 (idx = t + c*T / t2 + c*2T)
        for (int t = 0; t < T && !bad; t++)
            for (int c = 0; c < COUT; c++) {
                const float a = out2[(size_t) (2 * t)     + (size_t) c * 2 * T];
                const float b = out2[(size_t) (2 * t + 1) + (size_t) c * 2 * T];
                const float r = out[(size_t) t + (size_t) c * T];
                if (!std::isfinite(a) || !std::isfinite(b)) { n_nan2++; continue; }
                if (std::fabs(a - r) > 1e-6 || std::fabs(b - r) > 1e-6) { bad++; break; }
            }
    }
    printf("  x2 nearest 结构: %s%s\n", bad ? "FAIL" : "PASS", n_nan2 ? " (含非有限值!!)" : "");

    const char * be = getenv("GSV_COND_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bn + 3; i++) {
            auto t0 = std::chrono::steady_clock::now();
            m->bridge_run(x.data(), T, true, out2);
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 3) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml bridge (%s): avg %.4f ms min %.4f ms n=%d\n",
               cfg.device.empty() ? "cpu" : "vulkan", sum / ts.size(), ts.front(), (int) ts.size());
    }

    const bool ok = (md < 2e-2) && (n_nan == 0) && !bad && n_nan2 == 0;
    printf("\n%s\n", ok ? "COND BRIDGE PARITY PASS" : "COND BRIDGE PARITY FAIL");
    delete m;
    return ok ? 0 : 1;
}
