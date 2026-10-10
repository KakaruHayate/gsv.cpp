// 临时探针: HuBERT 分段对拍 (g1out vs torch feature_projection; g2 隔离: 用 torch 的 LN 输出作 g2 输入)
#include "../src/gsv_hubert.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> rb(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb"); if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f); return v;
}
static double md(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs((double) a[i] - b[i]));
    return m;
}
// [768,T] (d + 768*t) <-> [T,768] (t*768 + d)
static std::vector<float> td2dt(const std::vector<float> & x, int T) {
    std::vector<float> y(x.size());
    for (int t = 0; t < T; t++) for (int d = 0; d < 768; d++) y[(size_t) d * T + t] = x[(size_t) t * 768 + d];
    return y;
}
int main(int argc, char ** argv) {
    const std::string g = argc > 1 ? argv[1] : "tests/golden";
    const std::string st = argc > 2 ? argv[2] : "_dbg/stages";
    gsv_hubert_cfg cfg;
    if (const char * dv = getenv("GSV_HUBERT_DEVICE")) cfg.device = dv;
    gsv_hubert * h = gsv_hubert::load("models/gsv-hubert-f32.gguf", cfg);
    if (!h) return 1;

    std::vector<float> fed = rb(g + "/ref.zhs.wav16k_fed.bin");
    if (fed.empty()) { fprintf(stderr, "no fed wav\n"); return 1; }
    std::vector<float> zin(fed.size());
    {
        double m = 0; for (float v : fed) m += v; m /= (double) fed.size();
        double var = 0; for (float v : fed) { const double d = (double) v - m; var += d * d; } var /= (double) fed.size();
        const double inv = 1.0 / std::sqrt(var + 1e-7);
        for (size_t i = 0; i < fed.size(); i++) zin[i] = (float) (((double) fed[i] - m) * inv);
    }
    const int T = (int) ((fed.size() - 400) / 320 + 1);
    printf("fed n=%zu T=%d\n", fed.size(), T);

    // A) 全链 (g1 dump 开着, 引擎写 tests/golden/hubert.g1out.dbg.bin = [T,768] 行主)
    _putenv_s("GSV_HUBERT_G1DUMP", "1");
    std::vector<float> out_full;
    if (!h->encode(zin.data(), (int) zin.size(), out_full)) { fprintf(stderr, "encode failed\n"); return 1; }
    std::vector<float> g1o = rb("tests/golden/hubert.g1out.dbg.bin");     // [T,768]
    std::vector<float> feat = rb(st + "/feat_TD.bin");                    // [T,768] (torch feature_projection)
    printf("[A] g1out vs torch feat(TD):  max|d| = %.3e (refmax %.3g)\n", md(g1o, feat),
           feat.empty() ? 0.0 : [&]{ double m = 0; for (float v : feat) m = std::max(m, std::fabs((double) v)); return m; }());
    std::vector<float> sslf = rb(st + "/ssl_final.bin");                  // [768,T]
    {
        std::vector<float> ours = td2dt(out_full, T);                     // out_full 是 [768,T] (d+768t)?? 见下
        // encode 输出布局 = [768,T] flat d + 768*t -> 已经是 torch [768,T] 的 d*T+t?? 不: 我们的 out 是 ggml [D,T] (d + D*t)
        // 与 golden 的 [768,T] (d*T + t) 不同 -> 需按 T 重排
        std::vector<float> o2((size_t) 768 * T);
        for (int d = 0; d < 768; d++) for (int t = 0; t < T; t++) o2[(size_t) d * T + t] = out_full[(size_t) d + (size_t) 768 * t];
        printf("[A] ssl 全链 vs torch:       max|d| = %.3e\n", md(o2, sslf));
    }

    // B) g2 隔离: 用 torch 的 LN 输出 (需 [T,768] 行主文件) 作 g2 输入
    std::vector<float> ln_td = rb(st + "/ln_TD.bin");
    if (!ln_td.empty()) {
        _putenv_s("GSV_HUBERT_G1DUMP", "");          // 关掉 (省一次写盘)
        _putenv_s("GSV_HUBERT_ENCIN", (st + "/ln_TD.bin").c_str());
        std::vector<float> out_g2;
        if (!h->encode(zin.data(), (int) zin.size(), out_g2)) { fprintf(stderr, "encode(g2-only) failed\n"); return 1; }
        std::vector<float> o2((size_t) 768 * T);
        for (int d = 0; d < 768; d++) for (int t = 0; t < T; t++) o2[(size_t) d * T + t] = out_g2[(size_t) d + (size_t) 768 * t];
        printf("[B] g2(隔离, 用 torch LN):   max|d| = %.3e\n", md(o2, sslf));
        _putenv_s("GSV_HUBERT_ENCIN", "");
    }
    delete h;
    return 0;
}
