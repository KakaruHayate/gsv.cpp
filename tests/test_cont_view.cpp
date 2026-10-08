// 探针: cont(strided view) 的拷贝正确性
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "vulkan";
    ggml_backend_dev_t dev = nullptr;
    for (int i = 0; i < (int) ggml_backend_dev_count() && !dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(d);
        if (gpu && (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU)) dev = d;
        if (!gpu && ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    ggml_init_params ip = { ggml_tensor_overhead() * 256, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 200);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
    (void) b;
    std::vector<float> qv(128 * 200);
    for (int i = 0; i < 128 * 200; i++) qv[i] = (float) ((i * 7) % 97);
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * 4);

    ggml_tensor * v0 = ggml_view_2d(ctx, q, 64, 200, 128 * 4, 0);
    ggml_tensor * cc = ggml_cont(ctx, v0);
    ggml_set_output(cc);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, cc);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, g);
    ggml_backend_graph_compute(be, g);
    std::vector<float> out(64 * 200);
    ggml_backend_tensor_get(cc, out.data(), 0, out.size() * 4);
    double md = 0;
    for (int t = 0; t < 200; t++)
        for (int d = 0; d < 64; d++)
            md = std::max(md, (double) std::fabs(out[(size_t) d + (size_t) t * 64] - (float) ((d + (size_t) t * 128) * 7 % 97)));
    printf("cont(view) max|d| = %.3e\n", md);
    printf("out[0..3]   = %.1f %.1f %.1f %.1f (want 0 7 14 21)\n", out[0], out[1], out[2], out[3]);
    printf("out[64..67] = %.1f %.1f %.1f %.1f (want %d %d %d %d)\n",
           out[64], out[65], out[66], out[67],
           (128 * 7) % 97, (129 * 7) % 97, (130 * 7) % 97, (131 * 7) % 97);

}
