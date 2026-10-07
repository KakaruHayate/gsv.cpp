// 融合算子对拍: ggml_layernorm_affine vs norm+mul+add; ggml_add_act vs add+relu/add+gelu_erf
// 覆盖多种形状 (含行内元素 > 512、多行、非 2 的幂), CPU 后端逐元素比较
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <vector>

static ggml_backend_t g_backend = nullptr;

// 跑一个图: 输入 vec 列表 -> 输出张量数据
static std::vector<float> run_graph(ggml_context * ctx, ggml_tensor * out,
                                   const std::vector<ggml_tensor *> & ins,
                                   const std::vector<std::vector<float>> & vals) {
    ggml_set_output(out);
    for (auto * t : ins) ggml_set_input(t);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    ggml_gallocr_alloc_graph(galloc, graph);
    for (size_t i = 0; i < ins.size(); i++)
        ggml_backend_tensor_set(ins[i], vals[i].data(), 0, vals[i].size() * 4);
    ggml_backend_graph_compute(g_backend, graph);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
    ggml_gallocr_free(galloc);
    return res;
}

static double cmp(const std::vector<float> & a, const std::vector<float> & b, const char * tag) {
    double md = 0, ma = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double) a[i] - b[i]);
        if (d > md) md = d;
        if (std::fabs((double) b[i]) > ma) ma = std::fabs((double) b[i]);
    }
    const char * ok = (md < 1e-4 * (ma + 1.0)) ? "PASS" : "FAIL";
    if (md >= 1e-4 * (ma + 1.0)) printf("  [%s] max|d|=%.3e (ref max %.3g) %s\n", tag, md, ma, ok);
    return md;
}

int main() {
    // 设备选择: GSV_FUSED_DEVICE=vulkan 走 GPU (验证 Vulkan shader), 默认 CPU
    const char * dv = getenv("GSV_FUSED_DEVICE");
    const bool want_gpu = dv && (!strcmp(dv, "vulkan") || !strcmp(dv, "gpu"));
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && !dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(d);
        if (want_gpu ? (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU)
                     : (ty == GGML_BACKEND_DEVICE_TYPE_CPU)) dev = d;
    }
    if (!dev) { fprintf(stderr, "no device\n"); return 1; }
    printf("[fused] device: %s\n", ggml_backend_dev_name(dev));
    g_backend = ggml_backend_dev_init(dev, nullptr);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int n_bad = 0;

    // ---- layernorm_affine ----
    const int shapes[][2] = {{64, 8}, {512, 3}, {1024, 2}, {56, 5}, {4096, 1}};
    for (const auto & sh : shapes) {
        const int D = sh[0], T = sh[1];
        std::vector<float> xv((size_t) D * T), wv(D), bv(D);
        for (auto & v : xv) v = nd(rng) * 2.0f;
        for (auto & v : wv) v = nd(rng) * 0.5f + 1.0f;
        for (auto & v : bv) v = nd(rng) * 0.3f;
        const float eps = 1e-5f;

        std::vector<float> got, ref;
        {
            ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
            ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            got = run_graph(ctx, ggml_layernorm_affine(ctx, x, w, b, eps), {x, w, b}, {xv, wv, bv});
            ggml_free(ctx);
        }
        {
            ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
            ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            ggml_tensor * r = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
            ref = run_graph(ctx, r, {x, w, b}, {xv, wv, bv});
            ggml_free(ctx);
        }
        char tag[64];
        snprintf(tag, sizeof(tag), "layernorm_affine D=%d T=%d", D, T);
        if (cmp(got, ref, tag) >= 1e-4 * 3.0) n_bad++;
        else printf("  [%s] PASS\n", tag);
    }

    // ---- add_act (relu / gelu_erf) ----
    for (int act = 1; act <= 2; ++act) {
        const int D = 128, T = 6;
        std::vector<float> av((size_t) D * T), bv(D);
        for (auto & v : av) v = nd(rng) * 1.5f;
        for (auto & v : bv) v = nd(rng) * 0.5f;
        std::vector<float> got, ref;
        {
            ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            got = run_graph(ctx, ggml_add_act(ctx, a, b, act), {a, b}, {av, bv});
            ggml_free(ctx);
        }
        {
            ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
            ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            ggml_tensor * s = ggml_add(ctx, a, b);
            ggml_tensor * r = act == 1 ? ggml_relu(ctx, s) : ggml_gelu_erf(ctx, s);
            ref = run_graph(ctx, r, {a, b}, {av, bv});
            ggml_free(ctx);
        }
        char tag[64];
        snprintf(tag, sizeof(tag), "add_act act=%d (%s)", act, act == 1 ? "relu" : "gelu_erf");
        if (cmp(got, ref, tag) >= 1e-4 * 3.0) n_bad++;
        else printf("  [%s] PASS\n", tag);
    }

    printf("\nFUSED OPS %s\n", n_bad == 0 ? "ALL PASSED" : "FAILED");
    return n_bad == 0 ? 0 : 1;
}
