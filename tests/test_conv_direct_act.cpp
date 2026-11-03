// conv_direct_1d activation test: the fused epilogue (leaky / tanh) must match the unfused graph.
// CPU always; GPU when present (CI Linux runs it under lavapipe).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int T = 64, IC = 32, OC = 32, K = 3;

static ggml_tensor * conv_direct(ggml_context * c, ggml_tensor * w, ggml_tensor * x,
                                 ggml_tensor * b, float slope) {
    return ggml_conv_direct_1d(c, w, x, b, K / 2, 1, slope);
}

static bool run_on(ggml_backend_t be, const std::vector<float> & x, const std::vector<float> & w,
                   const std::vector<float> & b, int mode, std::vector<float> & out,
                   const char * tag) {
    // mode 0: leaky 0.1 fused; 1: tanh fused; 2: no act
    ggml_init_params ip = { 8 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
    ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
    ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
    ggml_tensor * y = nullptr;
    if (mode == 0) y = conv_direct(ctx, tw, tx, tb, 0.1f);
    else if (mode == 1) y = conv_direct(ctx, tw, tx, tb, -1.0f);
    else y = conv_direct(ctx, tw, tx, tb, 0.0f);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t all = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(all, gf)) { printf("[%s] alloc failed\n", tag); return false; }
    ggml_backend_tensor_set(tw, w.data(), 0, w.size() * 4);
    ggml_backend_tensor_set(tx, x.data(), 0, x.size() * 4);
    ggml_backend_tensor_set(tb, b.data(), 0, b.size() * 4);
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    out.assign((size_t) ggml_nelements(y), 0.0f);
    if (ok) ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
    ggml_gallocr_free(all);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
    const bool require_gpu = argc > 1 && std::strcmp(argv[1], "--require-gpu") == 0;
    ggml_backend_load_all();

    std::vector<float> x((size_t) T * IC), w((size_t) K * IC * OC), b((size_t) OC);
    for (size_t i = 0; i < x.size(); i++) x[i] = 0.02f * (float) ((int) (i * 37 % 101) - 50);
    for (size_t i = 0; i < w.size(); i++) w[i] = 0.05f * (float) ((int) (i * 53 % 97) - 48);
    for (int i = 0; i < OC; i++) b[i] = 0.01f * (float) (i - 8);

    std::vector<ggml_backend_t> backends{ ggml_backend_cpu_init() };
    std::vector<std::string> names{ "CPU" };
    if (!cpu_only) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_t be = ggml_backend_dev_init(d, nullptr);
            if (be) { backends.push_back(be); names.push_back(ggml_backend_dev_name(d)); }
        }
    }

    int failures = 0;
    for (size_t bi = 0; bi < backends.size(); bi++) {
        std::vector<float> raw;
        if (!run_on(backends[bi], x, w, b, 2, raw, names[bi].c_str())) { failures++; continue; }
        std::vector<float> act;
        if (!run_on(backends[bi], x, w, b, 0, act, names[bi].c_str())) { failures++; continue; }
        double maxd = 0;
        for (size_t i = 0; i < raw.size(); i++) {
            const float want = raw[i] > 0 ? raw[i] : 0.1f * raw[i];
            maxd = std::fmax(maxd, (double) std::fabs(act[i] - want));
        }
        printf("%-10s leaky max|d|=%.3e %s\n", names[bi].c_str(), maxd, maxd < 1e-5 ? "ok" : "MISMATCH");
        if (maxd >= 1e-5) failures++;

        if (!run_on(backends[bi], x, w, b, 1, act, names[bi].c_str())) { failures++; continue; }
        maxd = 0;
        for (size_t i = 0; i < raw.size(); i++)
            maxd = std::fmax(maxd, (double) std::fabs(act[i] - std::tanh(raw[i])));
        printf("%-10s tanh  max|d|=%.3e %s\n", names[bi].c_str(), maxd, maxd < 1e-6 ? "ok" : "MISMATCH");
        if (maxd >= 1e-6) failures++;
    }
    for (auto be : backends) ggml_backend_free(be);
    if (require_gpu && backends.size() < 2) { printf("CONV_DIRECT_ACT_TESTS_FAILED: --require-gpu but no GPU backend was registered\n"); return 1; }
    if (failures) { printf("CONV_DIRECT_ACT_TESTS_FAILED (%d)\n", failures); return 1; }
    printf("CONV_DIRECT_ACT_TESTS_OK (%zu backend(s))\n", backends.size());
    return 0;
}
