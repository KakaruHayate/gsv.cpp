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

    printf("%s (worst = %.3g)\n", worst < 1e-3 ? "ALL PASS" : "FAIL", worst);
    delete m;
    return worst < 1e-3 ? 0 : 2;
}
