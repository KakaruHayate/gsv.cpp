// 参考音频 -> prompt semante tokens 对拍: match_librosa 重采样 / ssl_proj+RVQ / 全链(HuBERT)
// golden: tools/dump_golden_refcode.py
// 用法: test_refcode [golden_dir] [audio_dir] [hubert_gguf] [refcode_gguf]
//   GSV_REFCODE_DEVICE=vulkan, GSV_HUBERT_DEVICE=vulkan, GSV_MEL_THREADS=N
#include "../src/gsv_refcode.h"
#include "../src/gsv_mel.h"
#include "../src/gsv_hubert.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}
static std::vector<int32_t> read_i32(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}
static int n_fail = 0;
static void cmp(const char * what, const std::vector<float> & a, const std::vector<float> & b, double tol) {
    if (a.size() != b.size()) { printf("  %-18s SIZE %zu vs %zu  FAIL\n", what, a.size(), b.size()); n_fail++; return; }
    double md = 0, refmax = 0;
    for (size_t i = 0; i < a.size(); i++) {
        md = std::max(md, std::fabs((double) a[i] - b[i]));
        refmax = std::max(refmax, (double) std::fabs(b[i]));
    }
    const bool ok = md < tol;
    if (!ok) n_fail++;
    printf("  %-18s max|d|=%.3e refmax=%.3g  %s\n", what, md, refmax, ok ? "PASS" : "FAIL");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // 崩溃时也能看到进度
    setvbuf(stderr, NULL, _IONBF, 0);
    const std::string gdir = argc > 1 ? argv[1] : "tests/golden";
    const std::string adir = argc > 2 ? argv[2] : "tests/audio";
    const std::string hubg = argc > 3 ? argv[3] : "models/gsv-hubert-f32.gguf";
    const std::string rcg  = argc > 4 ? argv[4] : "models/gsv-refcode.gguf";

    if (!gsv_refcode::load_filter_table("models/resampy_kaiser_best.bin")) {
        fprintf(stderr, "filter table 缺失 (先跑 tools/export_resampy_filter.py)\n"); return 1;
    }
    gsv_refcode_cfg rcfg;
    if (const char * dv = getenv("GSV_REFCODE_DEVICE")) rcfg.device = dv;
    if (const char * nt = getenv("GSV_REFCODE_THREADS")) rcfg.n_threads = atoi(nt);
    gsv_refcode * rc = gsv_refcode::load(rcg, rcfg);
    if (!rc) { fprintf(stderr, "load %s failed\n", rcg.c_str()); return 1; }

    gsv_hubert_cfg hcfg;
    if (const char * dv = getenv("GSV_HUBERT_DEVICE")) hcfg.device = dv;
    if (const char * nt = getenv("GSV_HUBERT_NTHREADS")) hcfg.n_threads = atoi(nt);
    else if (const char * nt = getenv("GSV_HUBERT_THREADS")) hcfg.n_threads = atoi(nt);
    gsv_hubert * hub = nullptr;

    for (auto & spec : { std::pair<const char *, const char *>{ "zhs", "ref_zh_short_16k.wav" },
                         std::pair<const char *, const char *>{ "en", "ref_en_16k.wav" },
                         std::pair<const char *, const char *>{ "zh32", "ref_zh_32k.wav" } }) {
        const char * tag = spec.first;
        std::vector<float> raw;
        int sr = 0;
        if (!gsv_wav_load(adir + "/" + spec.second, raw, sr)) { printf("[%s] wav 缺失, skip\n", tag); continue; }
        const std::string p = gdir + "/ref." + tag + ".";
        printf("[%s] %zu samples @ %d Hz\n", tag, raw.size(), sr);

        // 1) match_librosa 重采样 -> 16k
        std::vector<float> r16;
        gsv_refcode::resample_librosa(raw.data(), (int64_t) raw.size(), sr, 16000, r16);
        cmp("resample_librosa", r16, read_bin(p + "wav16k_res.bin"), 1e-5);

        // 2) 由 golden 的 ssl 出 codes (隔离 ssl_proj+RVQ)
        const std::vector<float> ssl_g = read_bin(p + "hubert_ssl.bin");
        const std::vector<int32_t> codes_g = read_i32(p + "codes.bin");
        if (ssl_g.empty() || codes_g.empty()) { printf("  [%s] golden ssl/codes 缺失\n", tag); continue; }
        const int T = (int) (ssl_g.size() / 768);
        std::vector<int32_t> codes_m, z_m;
        std::vector<float> z_dbg;
        if (!rc->codes_from_ssl(ssl_g.data(), T, codes_m, &z_dbg)) { n_fail++; continue; }
        {   // z_dbg 是 ggml [768,T2] (flat oc + 768*i); golden 是 torch [768,T2] (flat oc*T2 + i)
            const int T2 = T / 2;
            std::vector<float> zc((size_t) 768 * T2);
            for (int oc = 0; oc < 768; oc++)
                for (int i = 0; i < T2; i++) zc[(size_t) oc * T2 + i] = z_dbg[(size_t) oc + (size_t) 768 * i];
            cmp("sslproj_out", zc, read_bin(p + "sslproj_out.bin"), 1e-3);
            if (const char * dz = getenv("GSV_REFCODE_DUMP_Z")) {   // 调试: 落盘我们的 z (torch 布局)
                FILE * fp = fopen((std::string(dz) + "/mine." + tag + ".z.bin").c_str(), "wb");
                if (fp) { fwrite(zc.data(), 4, zc.size(), fp); fclose(fp); }
            }
        }
        {
            size_t same = 0;
            const size_t n = std::min(codes_m.size(), codes_g.size());
            for (size_t i = 0; i < n; i++) if (codes_m[i] == codes_g[i]) same++;
            const bool ok = (n == codes_g.size()) && same == n;
            if (!ok) n_fail++;
            printf("  %-18s %zu/%zu 一致  %s\n", "codes(from ssl)", same, n, ok ? "PASS" : "FAIL");
        }

        // 3) 全链: golden 的 16k(含零尾) -> 我们的 HuBERT -> ssl_proj+RVQ
        const std::vector<float> fed = read_bin(p + "wav16k_fed.bin");
        if (!fed.empty()) {
            if (!hub) hub = gsv_hubert::load(hubg, hcfg);
            if (!hub) { fprintf(stderr, "load %s failed\n", hubg.c_str()); return 1; }
            // 输入口径 = 原始 16k 波形 (repo 的生产路径 `cnhuhbert_model.model(wav16k)` 直接喂波形,
            // 不经过 Wav2Vec2FeatureExtractor 的 z-score; golden 也是这个口径)
            std::vector<float> hout;
            if (!hub->encode(fed.data(), (int) fed.size(), hout)) { n_fail++; continue; }
            // hubert 布局 [d + 768*t] -> ssl 需要 [d*T + t]
            const int Th = (int) (hout.size() / 768);
            std::vector<float> ssl_ours((size_t) 768 * Th);
            for (int d = 0; d < 768; d++)
                for (int t = 0; t < Th; t++) ssl_ours[(size_t) d * Th + t] = hout[(size_t) d + (size_t) 768 * t];
            {   // 我们的 HuBERT 与 torch 的偏差 (解释全链 codes 匹配率)
                const std::vector<float> sg = read_bin(p + "hubert_ssl.bin");
                if (sg.size() == ssl_ours.size()) {
                    double md = 0, refmax = 0;
                    for (size_t i = 0; i < sg.size(); i++) {
                        md = std::max(md, std::fabs((double) ssl_ours[i] - sg[i]));
                        refmax = std::max(refmax, (double) std::fabs(sg[i]));
                    }
                    const bool okh = md < 1e-3 * refmax + 1e-4;
                    if (!okh) n_fail++;
                    printf("  %-18s max|d|=%.3e refmax=%.3g  %s\n", "hubert_ssl(ours)", md, refmax, okh ? "PASS" : "FAIL");
                    if (const char * ds = getenv("GSV_REFCODE_DUMP_Z")) {
                        FILE * fp = fopen((std::string(ds) + "/mine." + tag + ".ssl.bin").c_str(), "wb");
                        if (fp) { fwrite(ssl_ours.data(), 4, ssl_ours.size(), fp); fclose(fp); }
                    }
                } else {
                    printf("  %-18s T 不一致: %zu vs %zu\n", "hubert_ssl(ours)", ssl_ours.size() / 768, sg.size() / 768);
                }
            }
            std::vector<int32_t> codes_c;
            if (!rc->codes_from_ssl(ssl_ours.data(), Th, codes_c)) { n_fail++; continue; }
            {
                size_t same = 0;
                const size_t n = std::min(codes_c.size(), codes_g.size());
                for (size_t i = 0; i < n; i++) if (codes_c[i] == codes_g[i]) same++;
                const double rate = 100.0 * (double) same / (double) codes_g.size();
                const bool okc = (n == codes_g.size()) && rate >= 99.0;
                if (!okc) n_fail++;
                printf("  %-18s %zu/%zu 一致 (%.2f%%)  %s\n", "codes(全链)",
                       same, codes_g.size(), rate, okc ? "PASS" : "FAIL");
            }
        }
    }
    delete hub;
    delete rc;
    printf("\n%s\n", n_fail == 0 ? "REFCODE ALL PASSED" : "REFCODE FAILED");
    return n_fail == 0 ? 0 : 1;
}
