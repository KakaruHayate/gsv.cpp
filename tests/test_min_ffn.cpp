// 最小对拍: x0[512] -> LN(norm1) -> FFN -> LN(norm2) -> predict -> logits[1025]
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    const char * gf_path = argc > 1 ? argv[1] : "models/gsv-ar-f32.gguf";
    const char * gd = argc > 2 ? argv[2] : "tests/golden";

    auto read_bin = [&](const char * name) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", gd, name);
        FILE * f = fopen(path, "rb");
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<float> v(sz / 4);
        if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
        fclose(f);
        return v;
    };
    auto x0 = read_bin("min_x0.bin");
    auto ref = read_bin("min_logits.bin");

    ggml_context * wctx = nullptr;
    // 直接用 gguf 读权重 (no_alloc=false: 数据读进 wctx 内存)
    gguf_init_params gip = { /*no_alloc*/ false, /*ctx*/ &wctx };
    gguf_context * gf = gguf_init_from_file(gf_path, gip);
    auto t = [&](const char * n){ return ggml_get_tensor(wctx, n); };

    ggml_init_params gip2 = { ggml_tensor_overhead()*256, NULL, true };
    ggml_context * ctx = ggml_init(gip2);

    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
    ggml_tensor * n1 = ggml_norm(ctx, x, 1e-5f);
    n1 = ggml_mul(ctx, n1, t("transformer.block0.norm1_w"));
    n1 = ggml_add(ctx, n1, t("transformer.block0.norm1_b"));
    ggml_tensor * h = ggml_mul_mat(ctx, t("transformer.block0.ffn1_w"), n1);
    h = ggml_add(ctx, h, t("transformer.block0.ffn1_b"));
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, t("transformer.block0.ffn2_w"), h);
    h = ggml_add(ctx, h, t("transformer.block0.ffn2_b"));
    ggml_tensor * s2 = ggml_add(ctx, n1, h);
    ggml_tensor * n2 = ggml_norm(ctx, s2, 1e-5f);
    n2 = ggml_mul(ctx, n2, t("transformer.block0.norm2_w"));
    n2 = ggml_add(ctx, n2, t("transformer.block0.norm2_b"));
    ggml_tensor * out = ggml_mul_mat(ctx, t("ar.predict"), n2);
    ggml_set_input(x);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, graph);
    ggml_backend_tensor_set(x, x0.data(), 0, x0.size()*4);
    ggml_backend_graph_compute(backend, graph);

    std::vector<float> got(1025);
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*4);
    float md = 0;
    for (int i = 0; i < 1025; i++) { float d = fabsf(got[i]-ref[i]); if (d>md) md=d; }
    printf("minimal LN+FFN max|diff| = %.6g\n%s\n", md, md < 1e-3 ? "PASS" : "FAIL");
    return 0;
}
