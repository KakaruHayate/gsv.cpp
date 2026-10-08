// 小矩阵枚举: 找到 softmax 后 oh 的正确 mul_mat 组合
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

int main() {
    const int T = 6, DK = 2, NH = 2;
    ggml_init_params ip = { ggml_tensor_overhead() * 512, NULL, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NH*DK, T);
    ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NH*DK, T);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, NH*DK, T);
    std::vector<float> qv(NH*DK*T), kv(NH*DK*T), vv(NH*DK*T);
    for (int i = 0; i < NH*DK*T; i++) { qv[i] = (float)((i*3)%11)*0.5f; kv[i] = (float)((i*5)%13)*0.4f; vv[i] = (float)((i*7)%17)*0.3f; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, ggml_backend_cpu_init());
    ggml_backend_t be = ggml_backend_cpu_init();
    (void) buf;
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size()*4);
    ggml_backend_tensor_set(k, kv.data(), 0, kv.size()*4);
    ggml_backend_tensor_set(v, vv.data(), 0, vv.size()*4);

    // 每头: qh/kh/vh 用 cont(view) [DK, T]
    // 组合枚举: sc = mul_mat(kh, qh) → [T, T]; pm = soft_max(sc); 
    // oh 候选:
    //   C1: mul_mat(cont(permute(vh)), pm)          (当前, 输出 ne=(pm.ne1?..))
    //   C2: mul_mat(pm, cont(permute(vh)))          
    //   C3: mul_mat(vh, pm)
    // 期望 (torch): oh[d, tq] = sum_ki P[tq, ki] v[d, ki]; P[tq, ki] = softmax(sc)[tq, ki]
    //   torch sc[tq, ki] = sum_d q[d,tq] k[d,ki]; ggml sc = mul_mat(kh, qh) out ne=(kh.ne1, qh.ne1)
    ggml_tensor * n1 = ggml_view_3d(ctx, q, DK, T, NH, DK*4, NH*DK*4, 0);
    // 简化: 只做头 0, DK=2: qh = view(q, 2, 6, nb1=4?..) 
    // 直接用 2D view: qh = view_2d(q, DK, T, NH*DK*4, 0) — strided!
    // 我们要测的是"cont 拷贝后的正确组合", 先 cont:
    ggml_tensor * qh = ggml_cont(ctx, ggml_view_2d(ctx, q, DK, T, NH*DK*4, 0));        // [DK, T]
    ggml_tensor * kh = ggml_cont(ctx, ggml_view_2d(ctx, k, DK, T, NH*DK*4, 0));
    ggml_tensor * vh = ggml_cont(ctx, ggml_view_2d(ctx, v, DK, T, NH*DK*4, 0));
    ggml_tensor * sc = ggml_mul_mat(ctx, kh, qh);                                       // [T, T]
    // 加性 F16 mask: m[ki, tq] = 0 / -inf
    std::vector<ggml_fp16_t> mv(T * T);
    for (int qi = 0; qi < T; qi++)
        for (int ki = 0; ki < T; ki++)
            mv[ki + qi*T] = ggml_fp32_to_fp16((ki % 2 == 1) ? -1e30f : 0.0f);   // 奇数 key 被 mask
    ggml_backend_tensor_set(msk, mv.data(), 0, mv.size() * 2);
    ggml_tensor * pm = ggml_soft_max_ext(ctx, sc, msk, 1.0f/ (float)std::sqrt(DK), 0.0f);
    ggml_tensor * vhT = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3));              // [T, DK]
    ggml_tensor * c1 = ggml_mul_mat(ctx, vhT, pm);                                       // 候选 1
    ggml_tensor * c2 = ggml_mul_mat(ctx, pm, vhT);                                       // 候选 2 (ne?)
    ggml_set_output(c1); ggml_set_output(c2);
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, c1);
    ggml_build_forward_expand(g, c2);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(ga, g)) { printf("alloc fail\n"); return 1; }
    ggml_backend_graph_compute(be, g);

    // numpy 期望: qh[d,t]=qv[h*DK+d + t*NH*DK]... h=0: qh[d,t] = qv[d + t*NH*DK]
    //   sc[tq,ki] = sum_d qh[d,tq] kh[d,ki]; P = softmax rows; oh[d,tq] = sum_ki P[tq,ki] vh[d,ki]
    double S[6][6], P[6][6], oh[2][6];
    for (int tq = 0; tq < T; tq++)
        for (int ki = 0; ki < T; ki++) {
            double s = 0;
            for (int d = 0; d < DK; d++) s += qv[(size_t)d + tq*NH*DK] * kv[(size_t)d + ki*NH*DK];
            S[tq][ki] = s / std::sqrt((double)DK);
        }
    for (int tq = 0; tq < T; tq++) {
        double mx = -1e30; for (int ki = 0; ki < T; ki++) { S[tq][ki] += (ki % 2 == 1) ? -1e30 : 0.0; mx = std::max(mx, S[tq][ki]); }
        double sum = 0;
        for (int ki = 0; ki < T; ki++) { P[tq][ki] = std::exp(S[tq][ki]-mx); sum += P[tq][ki]; }
        for (int ki = 0; ki < T; ki++) P[tq][ki] /= sum;
    }
    for (int d = 0; d < DK; d++)
        for (int tq = 0; tq < T; tq++) {
            double s = 0;
            for (int ki = 0; ki < T; ki++) s += P[tq][ki] * vv[(size_t)d + ki*NH*DK];
            oh[d][tq] = s;
        }

    std::vector<float> c1v(ggml_nelements(c1)), c2v(ggml_nelements(c2));
    // mask 掉 key 的缓冲也置 0 (否则会混入)
    for (int i = 0; i < NH*DK*T; i++) { kv[i] = kv[i]; }
    ggml_backend_tensor_get(c1, c1v.data(), 0, c1v.size()*4);
    ggml_backend_tensor_get(c2, c2v.data(), 0, c2v.size()*4);
    // 期望 oh [d, tq] flat[d + tq*DK] (若 ne=(DK, T))
    double e1 = 0, e2 = 0;
    if (c1->ne[0] == DK && c1->ne[1] == T)
        for (int d = 0; d < DK; d++) for (int tq = 0; tq < T; tq++)
            e1 = std::max(e1, std::fabs(c1v[d + tq*DK] - oh[d][tq]));
    if (c2->ne[0] == DK && c2->ne[1] == T)
        for (int d = 0; d < DK; d++) for (int tq = 0; tq < T; tq++)
            e2 = std::max(e2, std::fabs(c2v[d + tq*DK] - oh[d][tq]));
    printf("C1 err %.3e | C2 err %.3e\n", e1, e2);
    return 0;
}
