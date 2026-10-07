// 采样链对拍: C++ gsv_logits_to_probs vs torch AR/models/utils.py logits_to_probs
// golden: tools/dump_golden_ar_sampling.py -> sample.{a_rep,b_full,c_topp}.{probs,q,idx}
#include "gsv_sampler.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & dir, const char * name, long * n_out = nullptr) {
    const std::string path = dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
    fclose(f);
    if (n_out) *n_out = sz / 4;
    return v;
}

int main(int argc, char ** argv) {
    const std::string gd = argc > 1 ? argv[1] : "tests/golden";

    auto logits0 = read_bin(gd, "sample.logits");
    auto prev_f  = read_bin(gd, "sample.prev_tokens");
    const int V = (int)logits0.size();
    std::vector<int32_t> prev(prev_f.size());
    for (size_t i = 0; i < prev_f.size(); i++) prev[i] = (int32_t) prev_f[i];
    printf("[sampler] V=%d n_prev=%zu\n", V, prev.size());

    struct case_cfg { const char * name; gsv_sampler_cfg cfg; };
    const case_cfg cases[] = {
        { "a_rep",  { 15,   1.0f, 1.0f, 1.35f } },
        { "b_full", { 15,   0.9f, 0.8f, 1.35f } },
        { "c_topp", { 0,    0.95f, 1.0f, 1.0f } },
    };

    double worst = 0;
    for (const auto & c : cases) {
        std::vector<float> logits = logits0;                 // 链会原地改动
        std::vector<float> probs(V);
        gsv_logits_to_probs(c.cfg, logits.data(), V, prev.data(), (int)prev.size(), probs.data());

        auto ref = read_bin(gd, (std::string("sample.") + c.name + ".probs").c_str());
        double md = 0, mrel = 0;
        for (int i = 0; i < V; i++) {
            const double d = std::fabs(probs[i] - ref[i]);
            md = std::max(md, d);
            if (ref[i] > 1e-6) mrel = std::max(mrel, d / ref[i]);
        }
        // 支撑集一致性: 非零位置集合
        int support_diff = 0;
        for (int i = 0; i < V; i++) {
            if ((probs[i] > 0) != (ref[i] > 0)) support_diff++;
        }

        // exp-trick 采样规则: 注入 torch 的 q, argmax(probs/q) 必须与 torch 相同
        auto q   = read_bin(gd, (std::string("sample.") + c.name + ".q").c_str());
        auto idx_ref = read_bin(gd, (std::string("sample.") + c.name + ".idx").c_str());
        const int idx_cpp = gsv_argmax_probs_over_q(probs.data(), q.data(), V);

        printf("[%s] prob max|Δ|=%.3g  max rel=%.3g  support_diff=%d  idx cpp=%d ref=%d %s\n",
               c.name, md, mrel, support_diff, idx_cpp, (int)idx_ref[0],
               ((int)idx_ref[0] == idx_cpp && support_diff == 0) ? "OK" : "MISMATCH");
        worst = std::max(worst, md);
        if (support_diff != 0 || (int)idx_ref[0] != idx_cpp) worst = 1e9;
    }

    printf("%s (worst prob |Δ| = %.3g)\n", worst < 1e-6 ? "PASS" : "FAIL", worst);
    return worst < 1e-6 ? 0 : 2;
}
