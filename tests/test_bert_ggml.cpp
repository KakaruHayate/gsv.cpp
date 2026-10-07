// chinese-roberta-wwm-ext-large (BERT-large) ggml 对拍
//   golden 由 tools/dump_golden_bert.py 生成 (HF fp32):
//     bert.{i}.ids    [1,T]        token ids (float 存)
//     bert.{i}.hidden [1,T,1024]   hidden_states[-3] (第 21 层输出)
//     bert.{i}.feat   [1024,T-2]   去 [CLS]/[SEP] 转置后的管线特征
//     bert.0.hs{k}.bin [1,T,1024]  逐层 (k=0 为 embedding+LN, k>=1 为第 k-1 层输出)
// 用法: test_bert_ggml [--bench]   环境: GSV_BERT_DEVICE=(""|vulkan) GSV_BERT_THREADS GSV_BERT_LAYERS=1
#include "../src/gsv_bert.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static std::vector<float> read_bin(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

struct stats_t { double max_abs = 0, mean_abs = 0, cos = 0, rel = 0; };

static stats_t cmp_vec(const std::vector<float> & a, const std::vector<float> & b) {
    stats_t s;
    if (a.size() != b.size() || a.empty()) { s.max_abs = 1e30; return s; }
    double sa = 0, sb = 0, sab = 0, sum = 0, ref2 = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double) a[i] - b[i]);
        if (d > s.max_abs) s.max_abs = d;
        sum += d;
        sa += (double) a[i] * a[i];
        sb += (double) b[i] * b[i];
        sab += (double) a[i] * b[i];
        ref2 += (double) b[i] * b[i];
    }
    s.mean_abs = sum / a.size();
    s.cos = sab / (std::sqrt(sa) * std::sqrt(sb) + 1e-30);
    s.rel = std::sqrt(sum * sum / a.size()) / (std::sqrt(ref2 / a.size()) + 1e-30);
    return s;
}

// [1,T,D] (t 主序) -> [D,T] (d 主序)
static std::vector<float> transpose_td(const std::vector<float> & in, int T, int D) {
    std::vector<float> out(in.size(), 0.0f);
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) out[(size_t) d * T + t] = in[(size_t) t * D + d];
    return out;
}

int main(int argc, char ** argv) {
    const bool bench = argc > 1 && strcmp(argv[1], "--bench") == 0;
    const char * gdir = getenv("GSV_BERT_GOLDEN") ? getenv("GSV_BERT_GOLDEN") : "tests/golden";
    const char * mdl  = getenv("GSV_BERT_MODEL")  ? getenv("GSV_BERT_MODEL")  : "models/gsv-bert-f32.gguf";

    gsv_bert_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_BERT_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_BERT_THREADS")) cfg.n_threads = atoi(nt);

    gsv_bert * m = gsv_bert::load(mdl, cfg);
    if (!m) { fprintf(stderr, "load failed: %s\n", mdl); return 1; }
    const int D = m->hidden();

    std::vector<float> nvec = read_bin(std::string(gdir) + "/bert.n_texts.bin");
    const int n_texts = nvec.empty() ? 0 : (int) (nvec[0] + 0.5f);
    if (n_texts == 0) { fprintf(stderr, "no golden in %s\n", gdir); delete m; return 1; }

    int n_bad = 0;
    for (int i = 0; i < n_texts; i++) {
        std::vector<float> idsf = read_bin(std::string(gdir) + "/bert." + std::to_string(i) + ".ids.bin");
        if (idsf.empty()) { fprintf(stderr, "missing bert.%d.ids\n", i); n_bad++; continue; }
        const int T = (int) idsf.size();
        std::vector<int32_t> ids(T);
        for (int k = 0; k < T; k++) ids[k] = (int32_t) (idsf[k] + 0.5f);

        std::vector<float> full;
        m->encode(ids.data(), T, full);
        if ((int) full.size() != D * T) { fprintf(stderr, "text%d: encode returned %zu\n", i, full.size()); n_bad++; continue; }

        std::vector<float> hid = read_bin(std::string(gdir) + "/bert." + std::to_string(i) + ".hidden.bin");
        std::vector<float> ref = transpose_td(hid, T, D);
        stats_t s1 = cmp_vec(full, ref);

        std::vector<float> feat, feat_ref = read_bin(std::string(gdir) + "/bert." + std::to_string(i) + ".feat.bin");
        m->encode_feat(ids.data(), T, feat);
        stats_t s2 = cmp_vec(feat, feat_ref);

        const bool ok = s1.max_abs < 1e-3 && s2.max_abs < 1e-3;
        if (!ok) n_bad++;
        printf("text%d T=%2d: hidden[-3] max|d|=%.3e mean=%.3e cos=%.8f | feat max|d|=%.3e cos=%.8f  %s\n",
               i, T, s1.max_abs, s1.mean_abs, s1.cos, s2.max_abs, s2.cos, ok ? "OK" : "FAIL");
    }

    if (getenv("GSV_BERT_LAYERS")) {
        std::vector<float> idsf = read_bin(std::string(gdir) + "/bert.0.ids.bin");
        const int T = (int) idsf.size();
        std::vector<int32_t> ids(T);
        for (int k = 0; k < T; k++) ids[k] = (int32_t) (idsf[k] + 0.5f);
        std::vector<float> full, f0 = read_bin(std::string(gdir) + "/bert.0.feat.bin");
        std::vector<std::vector<float>> per;
        m->encode_layers(ids.data(), T, full, per);
        printf("\n逐层对拍 (k=0: embedding+LN, k: 第 k-1 层输出):\n");
        for (int k = 0; k < (int) per.size(); k++) {
            std::vector<float> h = read_bin(std::string(gdir) + "/bert.0.hs" + std::to_string(k) + ".bin");
            if (h.empty()) { printf("  k=%2d: 无 golden\n", k); continue; }
            stats_t s = cmp_vec(per[k], transpose_td(h, T, D));
            printf("  k=%2d: max|d|=%.3e mean=%.3e cos=%.8f\n", k, s.max_abs, s.mean_abs, s.cos);
        }
    }

    if (bench) {
        std::vector<float> idsf = read_bin(std::string(gdir) + "/bert.0.ids.bin");
        const int T0 = (int) idsf.size();
        std::vector<int32_t> ids(T0);
        for (int k = 0; k < T0; k++) ids[k] = (int32_t) (idsf[k] + 0.5f);
        std::vector<float> out;
        for (int T : {T0, 64, 128, 256, 512}) {
            std::vector<int32_t> ii(T);
            for (int k = 0; k < T; k++) ii[k] = 1000 + (k * 37) % 20000;
            ii[0] = 101; ii[T - 1] = 102;   // [CLS]/[SEP]
            m->encode(ii.data(), T, out);   // warmup
            const int n = 20;
            const double t0 = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            for (int r = 0; r < n; r++) m->encode(ii.data(), T, out);
            const double t1 = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            printf("bench T=%3d: %.2f ms/次\n", T, (t1 - t0) / n);
        }
    }

    printf("\n%s (%d/%d texts failed)\n", n_bad == 0 ? "BERT ALL PASSED" : "BERT FAILED", n_bad, n_texts);
    delete m;
    return n_bad == 0 ? 0 : 1;
}
