// per-head attention 单元测试: 用主程序同款形状 (DK=64, T=200) + golden z
// 验证 cont 拷贝 -> mul_mat -> soft_max_ext(F16 mask) -> mul_mat 的数值
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<float> read_bin(const char * p) {
    FILE * f = fopen(p, "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

int main() {
    const int T = 200, DK = 64, H = 128;
    auto z = read_bin("tests/golden/_mine_z.bin");
    if (z.empty()) { printf("missing z dump\n"); return 1; }
    // 头 0 的 q/k/v: 用固定伪随机权重生成 (与主程序不同没关系, 只验证组合语义)
    std::vector<float> qv(H * T), kv(H * T), vv(H * T);
    for (size_t i = 0; i < qv.size(); i++) { qv[i] = 150.0f * (((i * 7) % 23) - 11) + 20.0f; kv[i] = 150.0f * (((i * 11) % 19) - 9) + 20.0f; vv[i] = 150.0f * (((i * 13) % 17) - 8) + 20.0f; }
    std::vector<float> am(T * T);
    auto m1 = read_bin("tests/golden/refenc.mask.bin");
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < T; ki++) {
            const bool attend = (m1[qi] != 0) ? (m1[ki] != 0) : (ki == 0);
            am[ki + qi * T] = attend ? 0.0f : -1e30f;
        }

    ggml_init_params ip = { ggml_tensor_overhead() * 1024, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * qt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * kt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * vt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    (void) buf;
    ggml_backend_tensor_set(qt, qv.data(), 0, qv.size() * 4);
    ggml_backend_tensor_set(kt, kv.data(), 0, kv.size() * 4);
    ggml_backend_tensor_set(vt, vv.data(), 0, vv.size() * 4);
    ggml_backend_tensor_set(msk, am.data(), 0, am.size() * 2);

    ggml_tensor * qh = ggml_cont(ctx, ggml_view_2d(ctx, qt, DK, T, H * 4, 0));
    ggml_tensor * kh = ggml_cont(ctx, ggml_view_2d(ctx, kt, DK, T, H * 4, 0));
    ggml_tensor * vh = ggml_cont(ctx, ggml_view_2d(ctx, vt, DK, T, H * 4, 0));
    ggml_tensor * vhT = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));
    ggml_tensor * sc = ggml_mul_mat(ctx, kh, qh);
    ggml_tensor * pm = ggml_soft_max_ext(ctx, sc, msk, 1.0f / (float) std::sqrt((double) DK), 0.0f);
    ggml_tensor * oh = ggml_mul_mat(ctx, pm, vhT);
    ggml_set_output(sc); ggml_set_name(sc, "usc");
    ggml_set_output(pm); ggml_set_name(pm, "upm");
    ggml_set_output(oh);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, oh);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, g);
    ggml_backend_graph_compute(be, g);
    std::vector<float> o(ggml_nelements(oh)), scv(ggml_nelements(sc)), pmv(ggml_nelements(pm));
    ggml_backend_tensor_get(oh, o.data(), 0, o.size() * 4);
    ggml_backend_tensor_get(sc, scv.data(), 0, scv.size() * 4);
    ggml_backend_tensor_get(pm, pmv.data(), 0, pmv.size() * 4);

    // numpy 参考
    std::vector<double> S(T * T), P(T * T), oref(T * DK);
    for (int tq = 0; tq < T; tq++)
        for (int ki = 0; ki < T; ki++) {
            double acc = 0;
            for (int d = 0; d < DK; d++) acc += (double) kv[(size_t) d + ki * H] * qv[(size_t) d + tq * H];
            S[tq * T + ki] = acc / std::sqrt((double) DK);
        }
    for (int tq = 0; tq < T; tq++) {
        double mx = -1e30;
        for (int ki = 0; ki < T; ki++) mx = std::max(mx, S[tq * T + ki]);
        double sum = 0;
        for (int ki = 0; ki < T; ki++) { P[tq * T + ki] = std::exp(S[tq * T + ki] - mx); sum += P[tq * T + ki]; }
        for (int ki = 0; ki < T; ki++) P[tq * T + ki] /= sum;
    }
    for (int d = 0; d < DK; d++)
        for (int tq = 0; tq < T; tq++) {
            double acc = 0;
            for (int ki = 0; ki < T; ki++) acc += P[tq * T + ki] * vv[(size_t) d + ki * H];
            oref[d + tq * DK] = acc;
        }
    double md = 0;
    for (size_t i = 0; i < o.size(); i++) md = std::max(md, (double) std::fabs(o[i] - oref[i]));
    printf("per-head combo (T=200 DK=64) max|d| = %.3e\n", md);
    double d_sc = 0, d_pm = 0;
    for (int tq = 0; tq < T; tq++)
        for (int ki = 0; ki < T; ki++) {
            d_sc = std::max(d_sc, std::fabs(scv[ki + tq * T] - S[tq * T + ki]));
            d_pm = std::max(d_pm, std::fabs(pmv[ki + tq * T] - P[tq * T + ki]));
            d_sc = std::max(d_sc, std::fabs(scv[tq + ki * T] - S[tq * T + ki]));
            d_pm = std::max(d_pm, std::fabs(pmv[tq + ki * T] - P[tq * T + ki]));
        }
    printf("sc vs numpy: %.3e / pm: %.3e\n", d_sc, d_pm);
    {
        double rel_max = 0, rel_sum = 0; int n = 0;
        for (int tq = 0; tq < T; tq++)
            for (int ki = 0; ki < T; ki++) {
                double a = scv[ki + tq * T], b = S[tq * T + ki];
                if (std::fabs(b) > 100) { double rr = std::fabs(a - b) / std::fabs(b); rel_sum += rr; n++; if (rr > rel_max) rel_max = rr; }
            }
        printf("sc rel-err: max %.2e avg %.2e\n", rel_max, rel_sum / n);
        double sab = 0;
        for (int q2 = 0; q2 < T * T; q2++) sab = std::max(sab, (double) std::fabs(S[q2]));
        printf("numpy S absmax = %.1f\n", sab);
    }
    {   
        double rel_max = 0, rel_sum = 0; int n = 0;
        for (int tq = 0; tq < T; tq++)
            for (int ki = 0; ki < T; ki++) {
                double a = scv[ki + tq * T], b = S[tq * T + ki];
                if (std::fabs(b) > 100) { double rr = std::fabs(a - b) / std::fabs(b); rel_sum += rr; n++; if (rr > rel_max) rel_max = rr; }
            }
        if (n) printf("sc 相对误差 (|S|>100): max %.2e avg %.2e\n", rel_max, rel_sum / n);
    }
    double per_t[200] = {0};
    for (int tq = 0; tq < T; tq++)
        for (int d = 0; d < DK; d++)
            per_t[tq] = std::max(per_t[tq], (double) std::fabs(o[d + tq * DK] - oref[d + tq * DK]));
    printf("oh per-frame [0..5]: %.2f %.2f %.2f %.2f %.2f %.2f\n",
           per_t[0], per_t[1], per_t[2], per_t[3], per_t[4], per_t[5]);
    printf("oh per-frame [179..184]: %.2f %.2f %.2f %.2f %.2f %.2f\n",
           per_t[179], per_t[180], per_t[181], per_t[182], per_t[183], per_t[184]);
    return md < 1e-2 ? 0 : 2;
}
