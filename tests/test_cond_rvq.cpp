// RVQ (条件段入口) 对拍: codes -> codebook gather -> x2 nearest vs torch golden
// golden 由 tools/dump_golden_rvq.py 生成: rvq.{i}.codes [T], rvq.{i}.quant [T,768], rvq.{i}.up [2T,768]
// 用例含 AR golden 的真实 token 流 (rvq.3.ar40) —— 与 AR 引擎的输出直接对接
#include "../src/gsv_cond.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

static int cmp_exact(const std::vector<float> & a, const std::vector<float> & b, const char * tag) {
    if (a.size() != b.size()) { printf("  [%s] 尺寸不符 %zu vs %zu FAIL\n", tag, a.size(), b.size()); return 1; }
    size_t bad = 0;
    double md = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) bad++;
        const double d = std::fabs((double) a[i] - b[i]);
        if (d > md) md = d;
    }
    // gather + nearest 是纯搬运, 期望逐位一致
    const bool ok = (bad == 0);
    printf("  [%s] n=%zu 不等元素=%zu max|d|=%.3g %s\n", tag, a.size(), bad, md, ok ? "PASS(bit-exact)" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * gdir = argc > 1 ? argv[1] : "tests/golden";
    const char * mdl  = getenv("GSV_COND_MODEL") ? getenv("GSV_COND_MODEL") : "models/gsv-cond-f32.gguf";

    gsv_cond_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_COND_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_COND_THREADS")) cfg.n_threads = atoi(nt);

    gsv_cond * m = gsv_cond::load(mdl, cfg);
    if (!m) { fprintf(stderr, "load failed: %s\n", mdl); return 1; }

    std::vector<float> nc = read_bin(std::string(gdir) + "/rvq.n_cases.bin");
    const int n_cases = nc.empty() ? 0 : (int) (nc[0] + 0.5f);
    if (n_cases == 0) { fprintf(stderr, "no golden in %s\n", gdir); delete m; return 1; }

    int n_bad = 0;
    for (int i = 0; i < n_cases; i++) {
        std::vector<float> cf = read_bin(std::string(gdir) + "/rvq." + std::to_string(i) + ".codes.bin");
        std::vector<float> qref = read_bin(std::string(gdir) + "/rvq." + std::to_string(i) + ".quant.bin");
        std::vector<float> uref = read_bin(std::string(gdir) + "/rvq." + std::to_string(i) + ".up.bin");
        if (cf.empty() || qref.empty() || uref.empty()) { fprintf(stderr, "case%d: golden 缺失\n", i); n_bad++; continue; }
        const int T = (int) cf.size();
        std::vector<int32_t> codes(T);
        for (int k = 0; k < T; k++) codes[k] = (int32_t) (cf[k] + 0.5f);

        std::vector<float> got_q, got_u;
        m->rvq_decode(codes.data(), T, false, got_q);
        m->rvq_decode(codes.data(), T, true,  got_u);
        n_bad += cmp_exact(got_q, qref, (std::string("case") + std::to_string(i) + " quant[" + std::to_string(T) + ",768]").c_str());
        n_bad += cmp_exact(got_u, uref, (std::string("case") + std::to_string(i) + " up[" + std::to_string(2*T) + ",768]").c_str());
    }

    printf("\nRVQ %s (%d/%d cases failed)\n", n_bad == 0 ? "ALL PASSED (bit-exact)" : "FAILED", n_bad, n_cases);
    delete m;
    return n_bad == 0 ? 0 : 1;
}
