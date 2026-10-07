// AR 引擎对拍: src/gsv_ar 的前端 (phones/bert/prompt -> xy_pos) + 生成循环 vs torch golden
// - 前端: 引擎内部构造输入, greedy 生成必须与 batch.greedy.* 完全一致
// - first_logits: 与 batch.step0.logits 对拍
#include "gsv_ar.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & dir, const char * name) {
    const std::string path = dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/gsv-ar-f32.gguf";
    const std::string gd = argc > 2 ? argv[2] : "tests/golden";

    gsv_ar_cfg cfg;
    cfg.n_threads = 8;
    cfg.verbose = true;
    if (const char * dev = getenv("GSV_AR_DEVICE")) cfg.device = dev;
    gsv_ar * m = gsv_ar::load(model_path, cfg);
    if (!m) return 1;

    auto xlens = read_bin(gd, "batch.x_len");
    const int B = (int) xlens.size();
    auto prompt_f = read_bin(gd, "ar.prompt");
    std::vector<int32_t> prompt(prompt_f.size());
    for (size_t i = 0; i < prompt_f.size(); i++) prompt[i] = (int32_t) prompt_f[i];

    std::vector<gsv_ar_request> reqs(B);
    for (int b = 0; b < B; b++) {
        char nb[64];
        snprintf(nb, sizeof(nb), "batch.phones%d", b);
        auto ph = read_bin(gd, nb);
        snprintf(nb, sizeof(nb), "batch.bert%d", b);
        auto be = read_bin(gd, nb);
        const int T = (int) ph.size();
        reqs[b].phones.resize(T);
        for (int i = 0; i < T; i++) reqs[b].phones[i] = (int32_t) ph[i];
        reqs[b].bert = be;                     // [1024, T] 行主序
        if ((int) be.size() != m->bert_dim() * T) {
            fprintf(stderr, "bert size mismatch: %zu vs %d*%d\n", be.size(), m->bert_dim(), T);
            return 1;
        }
        reqs[b].prompt = prompt;
    }
    printf("[engine] B=%d lengths=[%d,%d,%d] bert_dim=%d\n", B,
           (int) reqs[0].phones.size(), (int) reqs[1].phones.size(), (int) reqs[2].phones.size(), m->bert_dim());

    // bert 特征噪声注入探针: 估计前端 (BERT) 误差对 token 流的影响
    // GSV_AR_BERT_NOISE=<s>: s>=0 为绝对 sigma, s<0 为相对 sigma (|s| * |feat|)
    if (const char * ns = getenv("GSV_AR_BERT_NOISE")) {
        const float s = (float) atof(ns);
        uint64_t st = 0x243F6A8885A308D3ull;
        auto rnd = [&]() {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            return (float) ((st >> 40) / 16777216.0);
        };
        double sum = 0; size_t n = 0;
        for (int b = 0; b < B; b++)
            for (size_t i = 0; i < reqs[b].bert.size(); i++) {
                const float u1 = std::max(rnd(), 1e-7f), u2 = rnd();
                const float g = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
                const float sig = s >= 0 ? s : -s * std::fabs(reqs[b].bert[i]);
                reqs[b].bert[i] += sig * g;
                sum += std::fabs(reqs[b].bert[i]); n++;
            }
        printf("[engine] bert 噪声注入 sigma=%g (%s), 注入后均值|feat| = %.4g\n",
               s, s >= 0 ? "绝对" : "相对", sum / n);
    }

    double worst = 0;

    // ---- 1) 前端 + 首步 logits ----
    {
        auto ref = read_bin(gd, "batch.step0.logits");
        std::vector<float> lg;
        m->first_logits(reqs, lg);
        printf("      (debug) lg.size=%zu ref.size=%zu lg[0..2]=%.4g %.4g %.4g ref[0..2]=%.4g %.4g %.4g\n",
               lg.size(), ref.size(), lg.empty()?0:lg[0], lg.empty()?0:lg[1], lg.empty()?0:lg[2],
               ref[0], ref[1], ref[2]);
        double md = 0;
        for (size_t i = 0; i < lg.size() && i < ref.size(); i++) md = std::max(md, (double) std::fabs(lg[i] - ref[i]));
        printf("[1] 引擎前端 + 首步 logits max|Δ| = %.3g %s\n", md, md < 1e-3 ? "PASS" : "FAIL");
        worst = std::max(worst, md);
    }

    // ---- 2) greedy 生成 (top_k=1 + rep=1.0 == argmax) ----
    {
        gsv_sampler_cfg sc; sc.top_k = 1; sc.top_p = 1.0f; sc.temperature = 1.0f; sc.repetition_penalty = 1.0f;
        auto res = m->generate(reqs, sc, 12345, 6, 1500);
        auto ref_idx = read_bin(gd, "batch.greedy.idx");
        int idx_bad = 0, tok_bad = 0;
        for (int b = 0; b < B; b++) {
            if ((int) ref_idx[b] != res.lens[b]) idx_bad++;
            char nb[64];
            snprintf(nb, sizeof(nb), "batch.greedy.seq%d.tokens", b);
            auto ref_tok = read_bin(gd, nb);
            if (ref_tok.size() != res.tokens[b].size()) { tok_bad++; continue; }
            for (size_t i = 0; i < ref_tok.size(); i++)
                if ((int32_t) ref_tok[i] != res.tokens[b][i]) { tok_bad++; break; }
        }
        printf("[2] 引擎 greedy 生成: idx 差异 %d/%d, token 差异 %d/%d  %s\n",
               idx_bad, B, tok_bad, B, (idx_bad == 0 && tok_bad == 0) ? "PASS" : "FAIL");
        if (idx_bad || tok_bad) worst = 1e9;
    }

    // ---- 3) early-stop 路径 ----
    {
        gsv_sampler_cfg sc; sc.top_k = 1; sc.top_p = 1.0f; sc.temperature = 1.0f; sc.repetition_penalty = 1.0f;
        auto res = m->generate(reqs, sc, 12345, 3, 1500);
        auto ref_idx = read_bin(gd, "batch.earlystop.idx");
        int bad = 0;
        for (int b = 0; b < B; b++) {
            char nb[64];
            snprintf(nb, sizeof(nb), "batch.earlystop.seq%d.tokens", b);
            auto ref_tok = read_bin(gd, nb);
            if ((int) ref_idx[b] != res.lens[b] || ref_tok.size() != res.tokens[b].size()) { bad++; continue; }
            for (size_t i = 0; i < ref_tok.size(); i++)
                if ((int32_t) ref_tok[i] != res.tokens[b][i]) { bad++; break; }
        }
        printf("[3] 引擎 early-stop 生成: 差异 %d/%d  %s\n", bad, B, bad == 0 ? "PASS" : "FAIL");
        if (bad) worst = 1e9;
    }

    // ---- 4) 量化验收: 100 token greedy vs f32 参考 (GSV_AR_ACC=1) ----
    if (getenv("GSV_AR_ACC") || getenv("GSV_AR_TF")) {
        gsv_sampler_cfg sc; sc.top_k = 1; sc.top_p = 1.0f; sc.temperature = 1.0f; sc.repetition_penalty = 1.0f;
        std::vector<int32_t> oracle;
        const bool tf = getenv("GSV_AR_TF") != nullptr;
        if (tf) {
            // 教师强制探针: 用 f32 参考 token 流固定上下文, 采样器换成真实分布 (关 top-k)
            auto ref_tok0 = read_bin(gd, "batch.greedy100.seq0.tokens");
            // 参考流 = prompt + 生成; 只取生成段作为 oracle
            for (size_t i = prompt.size(); i < ref_tok0.size(); i++) oracle.push_back((int32_t) ref_tok0[i]);
            sc.top_k = 0; sc.top_p = 1.0f; sc.temperature = 1.0f; sc.repetition_penalty = 1.0f;
        }
        auto res = m->generate(reqs, sc, 12345, 99, 1500, tf ? &oracle : nullptr);
        if (tf) { printf("[5] TF 探针完成 (token 由参考流固定, 概率分布已按 GSV_AR_PROB_DUMP 转储)"); }
        auto ref_idx = read_bin(gd, "batch.greedy100.idx");
        int bad = 0, first_bad = -1, len_bad = 0;
        for (int b = 0; b < B && !tf; b++) {
            char nb[64];
            snprintf(nb, sizeof(nb), "batch.greedy100.seq%d.tokens", b);
            auto ref_tok = read_bin(gd, nb);
            const auto & got = res.tokens[b];
            if (ref_tok.size() != got.size()) len_bad++;
            const size_t n = ref_tok.size() < got.size() ? ref_tok.size() : got.size();
            bool ok = true;
            for (size_t i = 0; i < n; i++)
                if ((int32_t) ref_tok[i] != got[i]) { ok = false; if (first_bad < 0) first_bad = (int) i; break; }
            if (!ok) bad++;
        }
        if (!tf) printf("[4] ACC(100 token): token 差异 %d/%d (首次分歧 @%d), len 差异 %d, idx=[%d,%d,%d] ref=[%d,%d,%d]",
               bad, B, first_bad, len_bad,
               res.lens.size() > 0 ? res.lens[0] : -1, res.lens.size() > 1 ? res.lens[1] : -1, res.lens.size() > 2 ? res.lens[2] : -1,
               (int) ref_idx[0], (int) ref_idx[1], (int) ref_idx[2]);
        if (!tf) { printf("  %s", (bad || len_bad) ? "FAIL" : "PASS"); if (bad || len_bad) worst = 1e9; }
    }

    printf("%s (worst = %.3g)\n", worst < 1e-3 ? "ALL PASS" : "FAIL", worst);
    delete m;
    return worst < 1e-3 ? 0 : 2;
}
