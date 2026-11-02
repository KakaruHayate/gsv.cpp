// add_act kernel test: verifies (a + b) followed by each activation on every available backend.
// CPU is always checked; a GPU backend is checked too when one is present unless --cpu is given.
//
// Built for CI (Linux runners exercise the Vulkan shader through lavapipe).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int NE0 = 64;
static const int NR  = 7;

static float ref_act(float v, int act) {
    switch (act) {
        case 1: return v > 0.0f ? v : 0.0f;
        case 2: return 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
        case 3: {
            const float sp = v > 20.0f ? v : log1pf(expf(v));
            return v * tanhf(sp);
        }
        default: return v;
    }
}

static bool run_on(ggml_backend_t be, int act, const std::vector<float> & a, const std::vector<float> & b,
                   std::vector<float> & out, const char * tag) {
    ggml_init_params ip = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NE0, NR);
    ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NE0);
    ggml_tensor * ty = ggml_add_act(ctx, ta, tb, act);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ty);
    ggml_gallocr_t all = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(all, gf)) { printf("[%s] alloc failed\n", tag); return false; }
    ggml_backend_tensor_set(ta, a.data(), 0, a.size() * 4);
    ggml_backend_tensor_set(tb, b.data(), 0, b.size() * 4);
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    out.resize(a.size());
    if (ok) ggml_backend_tensor_get(ty, out.data(), 0, out.size() * 4);
    ggml_gallocr_free(all);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const bool cpu_only = argc > 1 && std::strcmp(argv[1], "--cpu") == 0;
    const bool require_gpu = argc > 1 && std::strcmp(argv[1], "--require-gpu") == 0;

    // With shared ggml libraries the GPU backend is a separate .so that must be loaded
    // explicitly before enumerating devices (no symbol from it is referenced here).
    ggml_backend_load_all();
    printf("[test_add_act] registries=%zu devices=%zu\n", ggml_backend_reg_count(), ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("[test_add_act]   dev %zu: %s (%s) type=%d\n", i, ggml_backend_dev_name(d),
               ggml_backend_dev_description(d), (int) ggml_backend_dev_type(d));
    }

    std::vector<float> a((size_t) NE0 * NR), b((size_t) NE0);
    for (size_t i = 0; i < a.size(); i++) a[i] = 0.02f * (float) ((int) (i * 37 % 101) - 50) + 0.5f;
    for (int i = 0; i < NE0; i++) b[i] = 0.01f * (float) ((i * 13) % 29 - 14);

    std::vector<ggml_backend_t> backends;
    std::vector<std::string> names;
    backends.push_back(ggml_backend_cpu_init());
    names.push_back("CPU");
    if (!cpu_only) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (!be) continue;
            backends.push_back(be);
            names.push_back(ggml_backend_dev_name(dev));
        }
    }

    int failures = 0;
    for (int act = 0; act <= 3; act++) {
        std::vector<float> ref(a.size());
        for (size_t i = 0; i < a.size(); i++) ref[i] = ref_act(a[i] + b[i % NE0], act);
        for (size_t bi = 0; bi < backends.size(); bi++) {
            std::vector<float> out;
            if (!run_on(backends[bi], act, a, b, out, names[bi].c_str())) {
                printf("FAIL %s act=%d: compute error\n", names[bi].c_str(), act);
                failures++;
                continue;
            }
            double maxd = 0.0;
            for (size_t i = 0; i < out.size(); i++) maxd = std::fmax(maxd, (double) std::fabs(out[i] - ref[i]));
            const double tol = act == 2 ? 5e-3 : 2e-3;   // erf approximation on GPU
            const bool ok = maxd <= tol;
            printf("%-18s act=%d max|diff|=%.3e %s\n", names[bi].c_str(), act, maxd, ok ? "ok" : "MISMATCH");
            if (!ok) failures++;
        }
    }

    for (size_t i = 0; i < backends.size(); i++) ggml_backend_free(backends[i]);
    if (require_gpu && backends.size() < 2) {
        printf("ADD_ACT_TESTS_FAILED: --require-gpu but no GPU backend was registered\n");
        return 1;
    }
    if (failures) { printf("ADD_ACT_TESTS_FAILED (%d)\n", failures); return 1; }
    printf("ADD_ACT_TESTS_OK (%zu backend(s))\n", backends.size());
    return 0;
}
