// 最小复现: ggml_conv_direct_1d 在 Vulkan 下的小输入行为
//   case A: x = [T=120, IC=512], w = [5,512,1024], pad=2   (wns1 in_layer 形状)
//   case B: x = [T=1,   IC=512], w = [1,512,1024], pad=0   (wns1 cond/gin 形状)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_t pick(bool gpu) {
    for (int i = 0; i < (int) ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(d);
        if (gpu && (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU))
            return ggml_backend_dev_init(d, nullptr);
        if (!gpu && ty == GGML_BACKEND_DEVICE_TYPE_CPU)
            return ggml_backend_dev_init(d, nullptr);
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    const bool gpu = argc > 1 && std::string(argv[1]) == "vulkan";
    ggml_backend_t be = pick(gpu);
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }
    printf("backend: %s\n", ggml_backend_name(be));

    static int64_t Kc[16], Tc[16], Pc[16];
    int NC = 0;
    if (argc >= 5) {   // K T pad 由命令行给出 (单例探测)
        Kc[0] = atoll(argv[2]); Tc[0] = atoll(argv[3]); Pc[0] = atoll(argv[4]); NC = 1;
    } else {
        // K=1 在 Vulkan 下会崩 (含 T=120/128), 默认序列跳过; 用 argv 单测: test_conv_direct_vk.exe vulkan 1 120 0
        for (int64_t t : {32, 8, 2, 1}) { Kc[NC] = 5; Tc[NC] = t; Pc[NC] = 2; NC++; }
    }
    for (int ci = 0; ci < NC; ci++) {
        const int64_t K = Kc[ci], T = Tc[ci], IC = 512, OC = 1024, pad = Pc[ci];
        ggml_init_params ip = { ggml_tensor_overhead() * 256, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, IC);
        ggml_tensor * y = ggml_conv_direct_1d(ctx, w, x, NULL, (int) pad, 1, 0.0f);
        ggml_set_output(y);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
        (void) b;
        { std::vector<float> z(ggml_nelements(w), 0.01f); ggml_backend_tensor_set(w, z.data(), 0, z.size() * 4); }
        { std::vector<float> z(ggml_nelements(x), 0.02f); ggml_backend_tensor_set(x, z.data(), 0, z.size() * 4); }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, g)) { printf("case %c: alloc_graph failed\n", 'A' + ci); continue; }
        printf("case %d: K=%d T=%d pad=%d -> out [%lld,%lld] computing...\n",
               ci, (int) K, (int) T, (int) pad, (long long) y->ne[0], (long long) y->ne[1]);
        fflush(stdout);
        ggml_backend_graph_compute(be, g);
        std::vector<float> o(ggml_nelements(y));
        ggml_backend_tensor_get(y, o.data(), 0, o.size() * 4);
        double s = 0; for (float v : o) s += v;
        printf("case %c: OK  sum=%.3f\n", 'A' + ci, s);
        fflush(stdout);
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
