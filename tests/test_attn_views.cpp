// 最小复现: 带步长的视图进 mul_mat (ref_enc 注意力的两步)
//   sc  = mul_mat(view(k,64,T,512,off), view(q,64,T,512,off))     [T_ki, T_qi]
//   out = mul_mat(view(vT,T,64,T*4,off), softmax(sc*scale+mask))  [64, T_qi]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
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
    const int H = 128, T = 200, DK = 64;
    ggml_backend_t be = pick(gpu);
    if (!be) { fprintf(stderr, "no backend\n"); return 1; }
    printf("backend: %s\n", ggml_backend_name(be));

    ggml_init_params ip = { ggml_tensor_overhead() * 256, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v); ggml_set_input(msk);

    const int h = (argc > 2) ? atoi(argv[2]) : 0;
    const size_t off = (size_t) h * DK * 4;
    const size_t st_q = (size_t) H * 4;       // q/k 的 T 维步长 (128*4)
    ggml_tensor * vT = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));    // [T, H]
    const size_t st_vt = (size_t) T * 4;
    ggml_tensor * qh = ggml_cont(ctx, ggml_view_2d(ctx, q, DK, T, st_q, off));
    ggml_tensor * kh = ggml_cont(ctx, ggml_view_2d(ctx, k, DK, T, st_q, off));
    ggml_tensor * vTh = ggml_view_2d(ctx, vT, T, DK, st_vt, off);
    ggml_set_output(qh); ggml_set_name(qh, "dbg_qh");
    ggml_set_output(kh); ggml_set_name(kh, "dbg_kh");
    ggml_tensor * sc = ggml_mul_mat(ctx, kh, qh);                       // [T_ki, T_qi]
    ggml_tensor * p  = ggml_soft_max_ext(ctx, sc, msk, 1.0f / std::sqrt(128.0f), 0.0f);
    ggml_tensor * o  = ggml_mul_mat(ctx, vTh, p);                       // [DK, T_qi]
    ggml_set_output(sc); ggml_set_output(o);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, sc);
    ggml_build_forward_expand(g, o);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, be);
    (void) b;

    std::vector<float> qv((size_t) H * T), kv((size_t) H * T), vv((size_t) H * T), mv((size_t) T * T);
    for (size_t i = 0; i < qv.size(); i++) {
        qv[i] = 1.0f * ((i * 7) % 97);
        kv[i] = 2.0f * ((i * 13) % 89);
        vv[i] = 3.0f * ((i * 5) % 83);
    }
    for (int ki = 0; ki < T; ki++)
        for (int qi = 0; qi < T; qi++)
            mv[(size_t) ki + (size_t) qi * T] = (ki > 150) ? -1e30f : 0.0f;
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * 4);
    ggml_backend_tensor_set(k, kv.data(), 0, kv.size() * 4);
    ggml_backend_tensor_set(v, vv.data(), 0, vv.size() * 4);
    ggml_backend_tensor_set(msk, mv.data(), 0, mv.size() * 4);
    {
        float chk[4];
        ggml_backend_tensor_get(q, chk, 0, sizeof(chk));
        printf("  readback q[0..3] = %.3f %.3f %.3f %.3f (expect %.3f %.3f %.3f %.3f\n",
               chk[0], chk[1], chk[2], chk[3], qv[0], qv[1], qv[2], qv[3]);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, g)) { printf("alloc_graph failed\n"); return 1; }
    ggml_backend_graph_compute(be, g);

    std::vector<float> scv((size_t) T * T), ov((size_t) DK * T);
    {
        float qhchk[8];
        ggml_backend_tensor_get(ggml_get_tensor(ctx, "dbg_qh"), qhchk, 0, 8 * sizeof(float));
        printf("  qh_data[0..3] = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f (cont 语义应为 q[0..3,0] = 1,2,3,4)\n",
               qhchk[0], qhchk[1], qhchk[2], qhchk[3], qhchk[4], qhchk[5], qhchk[6], qhchk[7]);
    }
    ggml_backend_tensor_get(sc, scv.data(), 0, scv.size() * 4);
    ggml_backend_tensor_get(o,  ov.data(),  0, ov.size() * 4);

    // numpy 参考 (与图同语义)
    auto at = [&](std::vector<float> & x, int r, int c) -> float & { return x[(size_t) r * c + c * 0]; };
    (void) at;
    double d_sc = 0, d_o = 0;
    const float scale = 1.0f / std::sqrt(128.0f);
    std::vector<double> pv((size_t) T * T);
    for (int ki = 0; ki < T; ki++)
        for (int qi = 0; qi < T; qi++) {
            double s = 0;
            for (int d = 0; d < DK; d++)
                s += (double) kv[(size_t) (h * DK + d) * T + ki] * qv[(size_t) (h * DK + d) * T + qi];
            s *= scale;
            s += (ki > 150) ? -1e30 : 0.0;
            pv[(size_t) ki + (size_t) qi * T] = s;
            const float got = scv[(size_t) ki + (size_t) qi * T];
            if (qi < 2 && ki < 2)
                printf("  sc[%d,%d] ref=%.4f graph=%.4f\n", ki, qi, s, got);
            d_sc = std::max(d_sc, std::fabs(s - got));
        }
    for (int qi = 0; qi < T; qi++) {
        double mx = -1e30, sum = 0;
        for (int ki = 0; ki < T; ki++) mx = std::max(mx, pv[(size_t) ki + (size_t) qi * T]);
        for (int ki = 0; ki < T; ki++) { pv[(size_t) ki + (size_t) qi * T] = std::exp(pv[(size_t) ki + (size_t) qi * T] - mx); sum += pv[(size_t) ki + (size_t) qi * T]; }
        for (int ki = 0; ki < T; ki++) pv[(size_t) ki + (size_t) qi * T] /= sum;
        for (int d = 0; d < DK; d++) {
            double o = 0;
            for (int ki = 0; ki < T; ki++) o += pv[(size_t) ki + (size_t) qi * T] * vv[(size_t) (h * DK + d) * T + ki];
            d_o = std::max(d_o, std::fabs(o - ov[(size_t) d + (size_t) qi * DK]));
        }
    }
    if (getenv("PROBE_DEBUG")) {
        for (int qi = 0; qi < 2; qi++)
            for (int ki = 0; ki < 2; ki++)
                printf("  sc[%d,%d] graph=%.4f ref=%.4f\n", ki, qi,
                       scv[(size_t) ki + (size_t) qi * T], pv[(size_t) ki + (size_t) qi * T]);
    }

    printf("sc (mul_mat of strided views) max|d| = %.3e\n", d_sc);
    printf("o  (vTh x softmax)            max|d| = %.3e\n", d_o);
    return (d_sc < 1e-3 && d_o < 1e-3) ? 0 : 2;
}
