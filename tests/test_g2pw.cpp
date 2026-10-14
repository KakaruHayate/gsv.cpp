// G2PW (多音字消歧) 对拍 —— 输入构造 / 输出 pred / 端到端 predict_one
//   golden: tools/dump_golden_g2pw.py
// 用法: test_g2pw [golden_dir] [model_dir] [bert_vocab]
//   GSV_G2PW_DEVICE=cpu|gpu|webgpu|dml, GSV_G2PW_ORT=<dll>, GSV_G2PW_BENCH=N
#include "../src/gsv_g2pw.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// golden 的 int64 张量按 **raw 字节** 落盘 (numpy .tofile), 故按字节读再转 int64
static std::vector<int64_t> read_i64(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int64_t> v((size_t) sz / 8);
    if (fread(v.data(), 8, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}
static std::vector<int64_t> as_i64(const std::vector<float> & f) {
    std::vector<int64_t> v(f.size());
    for (size_t i = 0; i < f.size(); i++) v[i] = (int64_t) f[i];
    return v;
}
static std::vector<std::vector<int64_t>> as_i64_rows(const std::vector<int64_t> & f, size_t N, size_t L) {
    std::vector<std::vector<int64_t>> v(N, std::vector<int64_t>(L));
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < L; j++) v[i][j] = f[i * L + j];
    return v;
}

// 每个 case 的 (texts, query_ids) —— 必须与 dump_golden_g2pw.py 的 cases 完全一致
struct g2pw_case { const char * name; std::vector<std::string> texts; std::vector<int> qids; };

// 从 code point 数组造 UTF-8 (避免源码里写长转义串, MSVC 的 2147 字符上限 + 可读性)
static std::string u8(std::initializer_list<uint32_t> cps) {
    std::string s;
    for (uint32_t c : cps) {
        if (c < 0x80) s.push_back((char) c);
        else if (c < 0x800) { s.push_back((char) (0xC0 | (c >> 6))); s.push_back((char) (0x80 | (c & 0x3F))); }
        else { s.push_back((char) (0xE0 | (c >> 12))); s.push_back((char) (0x80 | ((c >> 6) & 0x3F))); s.push_back((char) (0x80 | (c & 0x3F))); }
    }
    return s;
}

static int n_fail = 0;

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const std::string gdir = argc > 1 ? argv[1] : "tests/golden";
    const std::string mdir = argc > 2 ? argv[2] : "models/G2PWModel";
    const std::string vtxt = argc > 3 ? argv[3] : "models/chinese-roberta-wwm-ext-large/vocab.txt";

    gsv_g2pw_cfg cfg;
    cfg.model_dir = mdir;
    cfg.vocab_txt = vtxt;
    cfg.verbose   = true;
    if (const char * dv = getenv("GSV_G2PW_DEVICE")) cfg.device = dv;
    if (const char * od = getenv("GSV_G2PW_ORT"))    cfg.ort_dll = od;
    if (const char * nt = getenv("GSV_G2PW_THREADS")) cfg.n_threads = atoi(nt);

    gsv_g2pw * g = gsv_g2pw::load(cfg);
    if (!g) { fprintf(stderr, "load failed\n"); return 1; }
    printf("g2pw: labels=%d EP=%s\n\n", g->n_labels(), g->ep_name());

    // 句子 (code point 形式, 与 dump_golden_g2pw.py 逐字一致):
    const std::string s_single = u8({0x6211,0x559c,0x6b22,0x65c5,0x884c});                     // 我喜欢旅行
    const std::string s_chong  = u8({0x91cd,0x5e86,0x7684,0x8def,0x5f88,0x957f});              // 重庆的路很长
    const std::string s_dedao  = u8({0x5f97,0x5230,0x4ec0,0x4e48,0x4e1c,0x897f});              // 得到什么东西
    const std::string s_yinhang= u8({0x94f6,0x884c,0x91cc,0x53ef,0x4ee5,0x884c,0x52a8,0x5417}); // 银行里可以行动吗
    const std::string s_mixed  = u8({0x6211,0x5403,0x4e86,0x33,0x4e2a,0x82f9,0x679c,0x4f1a,0x4e0d,0x4f1a}); // 我吃了3个苹果会不会
    const std::string s_head   = u8({0x957f,0x6c5f,0x7684,0x6c34,0x5f88,0x6e05});              // 长江的水很清
    const std::string s_long   = u8({0x8fd9,0x91cc,0x7684,0x98ce,0x666f,0x771f,0x7684,0x5f88,0x6f02,0x4eae,0xff0c,
                                     0x6211,0x4eec,0x8d70,0x4e86,0x5f88,0x4e45,0x5f88,0x4e45,0xff0c,
                                     0x5927,0x5bb6,0x90fd,0x8fd8,0x4e0d,0x60f3,0x56de,0x5bb6,0x5462}); // 这里的风景真的很漂亮，我们走了很久很久，大家都还不想回家呢

    const std::vector<g2pw_case> cases = {
        { "single", { s_single }, { 4 } },
        { "batch",  { s_single, s_chong, s_dedao }, { 4, 0, 0 } },
        { "ctx",    { s_yinhang, s_yinhang }, { 1, 5 } },
        { "mixed",  { s_mixed, s_mixed }, { 2, 8 } },
        { "head",   { s_head }, { 2 } },
        { "long",   { s_long, s_long, s_long }, { 2, 4, 6 } },
        { "dup",    { s_single, s_single }, { 4, 4 } },
    };

    for (const auto & c : cases) {
        const std::string p = gdir + "/g2pw." + c.name + ".";
        // int64 张量按字节读; f32 张量 (phoneme_masks/probs) 按 float 读
        const std::vector<int64_t> ids = read_i64(p + "input_ids.bin");
        const std::vector<int64_t> tts = read_i64(p + "token_type_ids.bin");
        const std::vector<int64_t> ams = read_i64(p + "attention_masks.bin");
        const std::vector<float>   pms = read_bin(p + "phoneme_masks.bin");
        const std::vector<int64_t> cid = read_i64(p + "char_ids.bin");
        const std::vector<int64_t> pid = read_i64(p + "position_ids.bin");
        const std::vector<float>   prb = read_bin(p + "probs.bin");
        const std::vector<float>   prd = read_bin(p + "pred.bin");
        if (ids.empty() || tts.empty() || ams.empty() || pms.empty() || cid.empty() ||
            pid.empty() || prb.empty() || prd.empty()) {
            printf("[%s] golden 缺失, skip\n", c.name);
            continue;
        }
        const size_t N = cid.size();                 // char_ids 是 [N] int64
        if (N == 0) { printf("[%s] golden 空\n", c.name); n_fail++; continue; }
        const size_t L = ids.size() / N;             // input_ids 是 [N, L] int64
        if (L == 0 || L * N != ids.size() ||
            tts.size() != ids.size() || ams.size() != ids.size() ||
            pid.size() != N ||
            pms.size() != N * (size_t) g->n_labels() ||
            prb.size() != N * (size_t) g->n_labels() ||
            prd.size() != N) {
            printf("[%s] golden 尺寸不自洽 (N=%zu L=%zu ids=%zu tts=%zu ams=%zu pms=%zu pid=%zu prb=%zu prd=%zu)\n",
                   c.name, N, L, ids.size(), tts.size(), ams.size(), pms.size(), pid.size(), prb.size(), prd.size());
            n_fail++; continue;
        }
        printf("[%s] N=%zu L=%zu\n", c.name, N, L);

        // 1) 用 golden 的输入原样喂 (隔离"输入构造"与"ONNX 计算")
        std::vector<std::vector<float>> pms_rows(N, std::vector<float>((size_t) g->n_labels()));
        for (size_t i = 0; i < N; i++)
            for (int j = 0; j < g->n_labels(); j++) pms_rows[i][(size_t) j] = pms[i * (size_t) g->n_labels() + (size_t) j];
        std::vector<int> pred;
        std::vector<std::vector<float>> probs;
        if (!g->run_raw(as_i64_rows(ids, N, L), as_i64_rows(tts, N, L), as_i64_rows(ams, N, L),
                        pms_rows, std::vector<int64_t>(cid), std::vector<int64_t>(pid), pred, &probs)) {
            printf("  run_raw FAIL\n"); n_fail++; continue;
        }
        {   // pred 必须与 repo 的 argmax 完全一致
            size_t same = 0;
            for (size_t i = 0; i < N; i++) if (pred[i] == (int) prd[i]) same++;
            const bool ok = same == N;
            if (!ok) n_fail++;
            printf("  %-14s %zu/%zu 一致  %s\n", "pred", same, N, ok ? "PASS" : "FAIL");
            for (size_t i = 0; i < N; i++)
                if (pred[i] != (int) prd[i])
                    printf("      行 %zu: ours=%d(%s) ref=%d(%s)\n", i, pred[i],
                           g->label(pred[i]) ? g->label(pred[i]) : "?", (int) prd[i],
                           g->label((int) prd[i]) ? g->label((int) prd[i]) : "?");
        }
        {   // probs 数值
            double md = 0, mean = 0;
            for (size_t i = 0; i < N * (size_t) g->n_labels(); i++) {
                const double d = std::fabs((double) probs[i / (size_t) g->n_labels()][i % (size_t) g->n_labels()] - (double) prb[i]);
                md = std::max(md, d); mean += d;
            }
            mean /= (double) (N * (size_t) g->n_labels());
            const bool ok = md < 1e-4;
            if (!ok) n_fail++;
            printf("  %-14s max|d|=%.3e mean=%.3e  %s\n", "probs", md, mean, ok ? "PASS" : "FAIL");
        }

        // 2) 端到端: 只给 text + query_id, 让 C++ 自己 tokenize + 构造
        std::vector<int> qids(c.qids);
        std::vector<int> pred2;
        if (!g->predict_batch(c.texts, qids, pred2, nullptr)) {
            printf("  %-14s FAIL (predict_batch)\n", "e2e"); n_fail++; continue;
        }
        {
            size_t same = 0;
            for (size_t i = 0; i < N; i++) if (pred2[i] == (int) prd[i]) same++;
            const bool ok = same == N;
            if (!ok) n_fail++;
            printf("  %-14s %zu/%zu 一致  %s\n", "e2e(tokenize)", same, N, ok ? "PASS" : "FAIL");
        }
    }

    // 3) 单条接口
    {
        const int qid = 4;
        int lab = -1;
        float conf = 0.0f;
        if (g->predict_one(u8({0x6211,0x559c,0x6b22,0x65c5,0x884c}), qid, lab, &conf)) {
            const std::vector<float> prd = read_bin(gdir + "/g2pw.single.pred.bin");
            const bool ok = !prd.empty() && lab == (int) prd[0];
            if (!ok) n_fail++;
            printf("[predict_one] label=%d (%s) conf=%.4f  %s\n", lab,
                   g->label(lab) ? g->label(lab) : "?", conf, ok ? "PASS" : "FAIL");
        } else { printf("[predict_one] FAIL\n"); n_fail++; }
    }

    // 4) 基准
    if (const char * be = getenv("GSV_G2PW_BENCH")) {
        const int bn = atoi(be);
        if (bn > 0) {
            std::vector<std::string> texts;
            std::vector<int> qids;
            for (int i = 0; i < 16; i++) {
                texts.push_back(u8({0x94f6,0x884c,0x91cc,0x53ef,0x4ee5,0x884c,0x52a8,0x5417}));
                qids.push_back(1);
            }
            std::vector<int> out;
            for (int i = 0; i < 3; i++) g->predict_batch(texts, qids, out, nullptr);
            std::vector<double> ts;
            for (int i = 0; i < bn; i++) {
                const auto t0 = std::chrono::steady_clock::now();
                g->predict_batch(texts, qids, out, nullptr);
                const auto t1 = std::chrono::steady_clock::now();
                ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ts.begin(), ts.end());
            printf("bench N=%zu (batch): avg %.2f ms min %.2f ms  [%s]\n",
                   texts.size(), ts[ts.size()/2], ts.front(), g->ep_name());
        }
    }
    delete g;
    printf("\n%s\n", n_fail == 0 ? "G2PW ALL PASSED" : "G2PW FAILED");
    return n_fail == 0 ? 0 : 1;
}
