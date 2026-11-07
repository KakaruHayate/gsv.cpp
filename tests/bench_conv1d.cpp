// Conv1d 执行策略 A/B：ggml_conv_direct_1d（现用，寄存器分块直接卷积）
//                   vs ggml_im2col_fast_1d + ggml_mul_mat（GGML 的 GEMM 路径）
//
//   ./bench_conv1d [threads] [reps]
//
// 对 dec 里三类典型形状各跑一遍，打印 MAC / ms / GFLOPS（MAC 由形状算，不猜）：
//   stage1 K=11/IC=128/OC=128/T=16480    （dec 的最大单项：34 GMAC 那层的代表）
//   stage1 K=3 /IC=128/OC=128/T=16480
//   stage0 up0  K=3/IC=512/OC=2560/T=206 （子像素上采样，Wt 15.7 MB）
//   stage2 K=7 /IC=64 /OC=64 /T=32960
//
// im2col 的产物是 [K*IC, OL] F32，所以 mul_mat 的输出是 [OC, OL]，要用还得再转置一次
// （"＋transpose" 列就是算上这次转置的成本；只有落到 torch 那种 [C,T] 布局才免）。
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct shape { const char * name; int K, IC, OC, T; };

static void run_one(ggml_backend_t be, const shape & s, int reps) {
    ggml_init_params ip = { ggml_tensor_overhead() * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.IC, s.OC);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.T, s.IC);
    ggml_set_input(w);
    ggml_set_input(x);

    // ---- A: 直接卷积 ----
    ggml_tensor * a = ggml_conv_direct_1d(ctx, w, x, nullptr, (s.K - 1) / 2, 1, 0.0f);
    ggml_set_name(a, "direct");
    ggml_set_output(a);

    // ---- B: im2col + mul_mat + transpose ----
    ggml_tensor * im   = ggml_im2col_fast_1d(ctx, w, x, 1, (s.K - 1) / 2, 1, GGML_TYPE_F32, 1);
    ggml_tensor * w2   = ggml_reshape_2d(ctx, w, (int64_t) s.K * s.IC, s.OC);
    ggml_tensor * im2  = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]);
    ggml_tensor * mm   = ggml_mul_mat(ctx, w2, im2);                       // [OC, OL]
    ggml_tensor * mmT  = ggml_cont(ctx, ggml_permute(ctx, mm, 1, 0, 2, 3)); // [OL, OC]
    ggml_set_name(mmT, "im2col+gemm");
    ggml_set_output(mmT);

    ggml_cgraph * ga = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(ga, a);
    ggml_cgraph * gb = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gb, mmT);

    ggml_gallocr_t gal = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(gal, ga) || !ggml_gallocr_alloc_graph(gal, gb)) {
        printf("%-28s alloc failed (im2col buffer = %.1f MB)\n", s.name,
               (double) s.K * s.IC * s.T * 4 / 1048576.0);
        ggml_gallocr_free(gal);
        ggml_free(ctx);
        return;
    }
    // 填随机数（值不影响计时，但避免全零触发任何捷径）
    {
        std::vector<float> buf((size_t) ggml_nelements(w));
        for (size_t i = 0; i < buf.size(); i++) buf[i] = std::sin(1e-3 * (double) i) * 0.1f;
        ggml_backend_tensor_set(w, buf.data(), 0, buf.size() * 4);
        buf.assign((size_t) ggml_nelements(x), 0.0f);
        for (size_t i = 0; i < buf.size(); i++) buf[i] = std::cos(1e-4 * (double) i);
        ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * 4);
    }

    const double mac = (double) s.T * s.OC * s.IC * s.K;
    const int64_t im_mb = (int64_t) s.K * s.IC * s.T * 4;

    for (int variant = 0; variant < 2; variant++) {
        ggml_cgraph * g = variant ? gb : ga;
        ggml_backend_graph_compute(be, g);                 // 预热
        double best = 1e30;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_ms();
            ggml_backend_graph_compute(be, g);
            const double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        printf("%-28s K=%-2d IC=%-4d OC=%-5d T=%-6d %-14s %8.2f ms %7.1f GFLOPS %8.1f %%peak\n",
               s.name, s.K, s.IC, s.OC, s.T, variant ? "im2col+gemm" : "direct-conv",
               best, 2.0 * mac / (best * 1e6), 100.0 * (2.0 * mac / (best * 1e6)) / 319.0);
    }
    if (im_mb > 8 * 1024 * 1024)
        printf("%-28s im2col 缓冲 = %.1f MB（额外的写+读流量）\n", s.name,
               (double) im_mb / 1048576.0);

    // 进程短命，不回收 ctx/gallocr（回收顺序在 gallocr+多图下有坑，且与本文无关）
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    const int reps    = argc > 2 ? std::atoi(argv[2]) : 3;

    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { printf("no CPU backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(be, threads);
    printf("threads=%d reps=%d  (peak 319 GFLOPS @4 threads, AVX2 FMA)\n\n", threads, reps);

    const shape all[4] = {
        { "stage1 k11 (最大单项)", 11, 128, 128, 16480 },
        { "stage1 k3",              3, 128, 128, 16480 },
        { "ups0 k3 (OC=2560)",      3, 512, 2560, 206 },
        { "stage2 k7",              7,  64,  64, 32960 },
    };
    // 一个进程只跑一个形状（多图共用一个 gallocr 在收尾时会被内部回收，与本题无关）
    const int which = argc > 3 ? std::atoi(argv[3]) : 0;
    if (which >= 0 && which < 4) run_one(be, all[which], reps);
    else for (const shape & s : all) { run_one(be, s, reps); break; }

    ggml_backend_free(be);
    return 0;
}
