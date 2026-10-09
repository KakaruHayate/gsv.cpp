// enc_p (TextEncoder + MRTE) 对拍 + 基准
// golden: tests/golden/encp.* (T=120, ntext=50)
// 用法: test_encp [gguf] [golden_dir]; GSV_ENCP_DEVICE=vulkan, GSV_ENCP_BENCH=N
#include "../src/gsv_encp.h"

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
    const char * mdl  = argc > 1 ? argv[1] : "models/gsv-cond-encp.gguf";
    const char * gdir = argc > 2 ? argv[2] : "tests/golden";

    gsv_encp_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_ENCP_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_ENCP_THREADS")) cfg.n_threads = atoi(nt);

    gsv_encp * m = gsv_encp::load(mdl, cfg);
    if (!m) return 1;

    const int T = 120, NT = 50, COUT = 192;
    std::vector<float> y = read_bin(std::string(gdir) + "/encp.input_y.bin");   // torch [1,768,T] → 我们要 [768, T] 行主 (c + t*768) — 相同字节序 ✓
    std::vector<float> ref_m = read_bin(std::string(gdir) + "/encp.m.bin");     // torch [1,192,T] → [192, T] ✓
    std::vector<float> ref_logs = read_bin(std::string(gdir) + "/encp.logs.bin");
    std::vector<float> ge = read_bin(std::string(gdir) + "/encp.ge.bin");       // [512]
    std::vector<float> tf = read_bin(std::string(gdir) + "/encp.text.bin");   // int32 值以 float 存储
    std::vector<int32_t> text(tf.size());
    for (size_t i = 0; i < tf.size(); i++) text[i] = (int32_t) tf[i];

    if (y.empty() || ref_m.empty() || ref_logs.empty() || ge.empty()) { fprintf(stderr, "missing golden\n"); delete m; return 1; }

    std::vector<float> om, ologs, oy;
    if (!m->encode(y.data(), T, text.data(), NT, ge.data(), om, ologs, &oy)) { delete m; return 1; }
    size_t n1 = 0, n2 = 0;
    double dm = maxdiff_nan(om, ref_m, n1);
    double dl = maxdiff_nan(ologs, ref_logs, n2);
    printf("  encp_m    max|d|=%.3e %s%s\n", dm, dm < 2e-2 ? "PASS" : "FAIL", n1 ? " (nan!)" : "");
    printf("  encp_logs max|d|=%.3e %s%s\n", dl, dl < 2e-2 ? "PASS" : "FAIL", n2 ? " (nan!)" : "");
    // y (proj 前, 条件链里喂 bridge) 与 golden y_enc 对拍
    size_t n3 = 0;
    double dy = 0;
    std::vector<float> ref_y = read_bin(std::string(gdir) + "/encp.y_enc.bin");
    if (!ref_y.empty()) {
        dy = maxdiff_nan(oy, ref_y, n3);
        printf("  encp_y    max|d|=%.3e %s%s\n", dy, dy < 2e-2 ? "PASS" : "FAIL", n3 ? " (nan!)" : "");
    }

    const char * be = getenv("GSV_ENCP_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        std::vector<double> ts;
        for (int i = 0; i < bn + 3; i++) {
            auto t0 = std::chrono::steady_clock::now();
            m->encode(y.data(), T, text.data(), NT, ge.data(), om, ologs);
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 3) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml encp (%s): avg %.2f ms min %.2f ms n=%d\n",
               cfg.device.empty() ? "cpu" : "vulkan", sum / ts.size(), ts.front(), (int) ts.size());
    }

    const bool ok = (dm < 2e-2 && dl < 2e-2 && n1 == 0 && n2 == 0) &&
                    (ref_y.empty() || (dy < 2e-2 && n3 == 0));
    printf("\n%s\n", ok ? "ENCP PARITY PASS" : "ENCP PARITY FAIL");
    delete m;
    return ok ? 0 : 1;
}
