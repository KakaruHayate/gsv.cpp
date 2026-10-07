// AR 引擎基准: 首步 + N 步 greedy 解码 (bs=1 / bs=3 / bs=8)
// 与 tools/bench_ar.py 使用同一 workload (相同的 phones/bert/prompt golden)
#include "gsv_ar.h"

#include <chrono>
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
    const int n_gen = argc > 3 ? atoi(argv[3]) : 100;    // 生成 token 数
    const int n_threads = argc > 4 ? atoi(argv[4]) : 8;

    gsv_ar_cfg cfg; cfg.n_threads = n_threads;
    gsv_ar * m = gsv_ar::load(model_path, cfg);
    if (!m) return 1;
    if (n_threads > 0) printf("[bench] threads=%d\n", n_threads);

    auto prompt_f = read_bin(gd, "ar.prompt");
    std::vector<int32_t> prompt(prompt_f.size());
    for (size_t i = 0; i < prompt_f.size(); i++) prompt[i] = (int32_t) prompt_f[i];

    // 读取 3 条 golden 请求 (长度 32/26/20), bs=8 时循环复用
    std::vector<gsv_ar_request> base(3);
    for (int b = 0; b < 3; b++) {
        char nb[64];
        snprintf(nb, sizeof(nb), "batch.phones%d", b);
        auto ph = read_bin(gd, nb);
        snprintf(nb, sizeof(nb), "batch.bert%d", b);
        auto be = read_bin(gd, nb);
        base[b].phones.resize(ph.size());
        for (size_t i = 0; i < ph.size(); i++) base[b].phones[i] = (int32_t) ph[i];
        base[b].bert = be;
        base[b].prompt = prompt;
    }

    gsv_sampler_cfg sc; sc.top_k = 1; sc.top_p = 1.0f; sc.temperature = 1.0f; sc.repetition_penalty = 1.0f;  // greedy == torch 参考

    printf("[bench] generate %d tokens (greedy), warmup ...\n", n_gen);
    for (int bs : {1, 3, 8}) {
        std::vector<gsv_ar_request> reqs;
        for (int i = 0; i < bs; i++) reqs.push_back(base[i % 3]);

        // 预热 (1 次完整)
        auto warm = m->generate(reqs, sc, 1, n_gen - 1, 1500);
        (void) warm;

        // 首步计时
        std::vector<float> lg;
        auto t0 = std::chrono::high_resolution_clock::now();
        m->first_logits(reqs, lg);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 完整生成计时 (含首步)
        auto t2 = std::chrono::high_resolution_clock::now();
        auto res = m->generate(reqs, sc, 1, n_gen - 1, 1500);
        auto t3 = std::chrono::high_resolution_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        int steps = 0;
        for (int v : res.lens) steps = std::max(steps, v);
        steps += 1;   // 各序列中止步数的最大值 (避免只取 seq0 导致 per-step 失真)
        printf("[bench] bs=%d: first=%.1fms total=%.1fms steps=%d per_step=%.2fms %.1f tok/s(单序列) %.1f tok/s(总)\n",
               bs, first_ms, total_ms, steps, (total_ms - first_ms) / std::max(1, steps),
               steps / ((total_ms - first_ms) / 1000.0),
               bs * steps / ((total_ms - first_ms) / 1000.0));
    }
    delete m;
    return 0;
}
