// Width-scaling benchmark for the AR decoder.
//
// All speculation-style speedups (prompt lookup, lookahead, self-speculative layer skipping) rest
// on one property: putting W tokens into a single decode pass must cost less than W single-token
// passes. This measures that directly, without any draft logic: every sequence is teacher-forced
// with the same oracle token list, so each run does exactly max_steps steps, and B is varied.
//
//   bench_ar_width <ar.gguf> [device] [max_steps]
//
// Per-token decode cost at B=1 is the baseline; the ratio at larger B is the ceiling that a
// speculation scheme with a perfect draft could reach (they widen the same matmul dimension).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gsv_ar.h"

static double now_ms() {
    using namespace std::chrono;
    return (double) duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <ar.gguf> [device] [max_steps]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    gsv_ar_cfg cfg;
    cfg.device = argc > 2 ? argv[2] : "vulkan";
    const int steps = argc > 3 ? atoi(argv[3]) : 160;

    gsv_ar * ar = gsv_ar::load(path, cfg);
    if (!ar) { fprintf(stderr, "load failed\n"); return 1; }
    const int D = ar->hidden_dim(), V = ar->vocab_size(), BD = ar->bert_dim();
    printf("ar: D=%d V=%d bert=%d eos=%d device=%s\n", D, V, BD, ar->eos(), cfg.device.c_str());

    // A realistic shape: 60 phones + a 35-code reference prompt (the shapes of a short sentence).
    const int T = 60, P = 35;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> phone_id(0, 731), code_id(0, 1023);

    gsv_ar_request proto;
    proto.phones.resize(T);
    for (auto & p : proto.phones) p = phone_id(rng);
    proto.bert.resize((size_t) BD * T);
    for (auto & b : proto.bert) b = uf(rng);
    proto.prompt.resize(P);
    for (auto & c : proto.prompt) c = code_id(rng);

    // Teacher forcing: a fixed token list, no EOS, so every run does exactly `steps` steps.
    std::vector<int32_t> oracle(steps, 0);
    for (auto & t : oracle) t = code_id(rng);

    gsv_sampler_cfg sampler;   // TTS defaults: top_k 15, top_p 1.0, temp 1.0, rep 1.35

    printf("\n%-3s %9s %10s %10s %10s %12s\n",
           "B", "total ms", "ms/step", "ms/token", "vs B=1", "tok/s");
    double base = 0.0;
    for (int B : { 1, 2, 4, 8 }) {
        std::vector<gsv_ar_request> reqs(B, proto);
        std::vector<uint64_t> seeds(B);
        for (int b = 0; b < B; b++) seeds[b] = 1000 + b;

        // warm up (KV alloc, graph build, shader/pipeline warmup for this B)
        ar->generate(reqs, sampler, 1, -1, steps, &oracle, &seeds);
        const double t0 = now_ms();
        gsv_ar_result r = ar->generate(reqs, sampler, 1, -1, steps, &oracle, &seeds);
        const double ms = now_ms() - t0;
        const double per_step = ms / (double) steps;
        const double per_token = per_step / (double) B;
        if (B == 1) base = per_token;
        printf("%-3d %9.1f %10.3f %10.3f %9.2fx %12.1f\n", B, ms, per_step, per_token,
               base / per_token, (double) B * steps * 1000.0 / ms);
        fflush(stdout);
        if (r.lens.size() != (size_t) B) { fprintf(stderr, "unexpected lens\n"); }
    }
    ar->release_cache();
    delete ar;
    return 0;
}
