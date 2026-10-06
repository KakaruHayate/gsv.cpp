// GSV AR 增量 decode 对拍（分层图实现，正确性优先）
// step0 图: 全前向 56 token, 每层 K/V 拷回 host cache
// decode 图: 每层一张图 — 单 token q + 该层 cache 输入 → 新 K/V 列 + 层输出
// golden: ar.dec{n}.logits (官方 decode_next_token 路径)
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int D = 512, NH = 16, HD = 32, NL = 24, VOCAB = 1025;

struct weights {
    ggml_tensor * qkv_w, * qkv_b, * out_w, * out_b, * n1w, * n1b, * f1w, * f1b, * f2w, * f2b, * n2w, * n2b;
};

int main(int argc, char ** argv) {
    std::string model_path = argc > 1 ? argv[1] : "models/gsv-ar-f32.gguf";
    std::string gd = argc > 2 ? argv[2] : "tests/golden";
    auto read_bin = [&](const char * name) {
        std::string path = gd + "/" + name + ".bin";
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<float> v(sz/4);
        if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
        fclose(f);
        return v;
    };
    auto xy0 = read_bin("ar.xy_pos");
    auto mask0 = read_bin("ar.xy_attn_mask");
    auto ref0 = read_bin("ar.step0.logits");
    auto dec_tokens = read_bin("ar.dec_tokens");

    ggml_context * wctx = nullptr;
    gguf_init_params gip = { false, &wctx };
    gguf_context * gf = gguf_init_from_file(model_path.c_str(), gip);
    auto t = [&](const char * n){ return ggml_get_tensor(wctx, n); };
    std::vector<weights> ws(NL);
    char buf[128];
    for (int li = 0; li < NL; li++) {
        #define GW(f, nm) snprintf(buf, sizeof(buf), "transformer.block%d." nm, li); ws[li].f = t(buf)
        GW(qkv_w, "qkv_w"); GW(qkv_b, "qkv_b"); GW(out_w, "out_w"); GW(out_b, "out_b");
        GW(n1w, "norm1_w"); GW(n1b, "norm1_b"); GW(n2w, "norm2_w"); GW(n2b, "norm2_b");
        GW(f1w, "ffn1_w"); GW(f1b, "ffn1_b"); GW(f2w, "ffn2_w"); GW(f2b, "ffn2_b");
        #undef GW
    }
    ggml_tensor * predict = t("ar.predict");

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    std::vector<std::vector<float>> kcache(NL), vcache(NL);  // [D, S] 行主 (ne0=D fastest)

    // ============ step0 ============
    {
        const int S = 56;
        ggml_init_params ip = { ggml_tensor_overhead()*8192, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * xy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S);
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);
        ggml_set_input(xy); ggml_set_input(mask);
        ggml_tensor * cur = xy;
        std::vector<ggml_tensor *> kcache_t(NL), vcache_t(NL);
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv3 = ggml_reshape_3d(ctx, qkv, D, 3, S);
            ggml_tensor * q = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], 0);
            ggml_tensor * k = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], qkv3->nb[1]);
            ggml_tensor * v = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], 2*qkv3->nb[1]);
            q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, S), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, S), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, NH, S), 0, 2, 1, 3));
            ggml_tensor * mask16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kh, vh, mask16, 1.0f/std::sqrt((float)HD), 0, 0);
            attn_out = ggml_reshape_2d(ctx, attn_out, D, S);
            ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].out_w, attn_out), ws[li].out_b);
            cur = ggml_add(ctx, cur, o);
            ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur, 1e-5f), ws[li].n1w), ws[li].n1b);
            ggml_tensor * h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b));
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            cur = ggml_add(ctx, n1, h);
            cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur, 1e-5f), ws[li].n2w), ws[li].n2b);
            // K/V cache: kh (HD,S,NH) → reshape (D,S); v 同
            kcache_t[li] = ggml_reshape_2d(ctx, kh, D, S);
            vcache_t[li] = ggml_reshape_2d(ctx, vh, D, S);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_tensor * last = ggml_view_1d(ctx, logits, VOCAB, 55*logits->nb[1]);
        ggml_set_output(last);
        for (int li = 0; li < NL; li++) { ggml_set_output(kcache_t[li]); ggml_set_output(vcache_t[li]); }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, last);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, kcache_t[li]); ggml_build_forward_expand(graph, vcache_t[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(xy, xy0.data(), 0, xy0.size()*4);
        ggml_backend_tensor_set(mask, mask0.data(), 0, mask0.size()*4);
        ggml_backend_graph_compute(backend, graph);
        kcache.resize(NL); vcache.resize(NL);
        for (int li = 0; li < NL; li++) {
            kcache[li].resize((size_t)D*S); vcache[li].resize((size_t)D*S);
            ggml_backend_tensor_get(kcache_t[li], kcache[li].data(), 0, (size_t)D*S*4);
            ggml_backend_tensor_get(vcache_t[li], vcache[li].data(), 0, (size_t)D*S*4);
        }
        std::vector<float> got(VOCAB);
        ggml_backend_tensor_get(last, got.data(), 0, got.size()*4);
        double md = 0;
        for (int i = 0; i < VOCAB; i++) md = fmax(md, fabs(got[i]-ref0[i]));
        printf("[step0] max|diff| = %.6g %s\n", md, md < 1e-3 ? "PASS" : "FAIL");
        ggml_free(ctx);
    }

    // ============ decode 步 (固定 stride cache, 每步构造完整 K/V 输入) ============
    // cache 布局: 每层 HD*S_max*NH 浮点, 元素 (d,s,h) 位于 h*HD*S_max + s*HD + d
    const int S_max = 64;
    std::vector<std::vector<float>> kc(NL), vc(NL);
    for (int li = 0; li < NL; li++) {
        kc[li].assign((size_t)HD * S_max * NH, 0.0f);
        vc[li].assign((size_t)HD * S_max * NH, 0.0f);
        for (int h = 0; h < NH; h++)
            for (int s = 0; s < 56; s++)
                for (int d = 0; d < HD; d++) {
                    kc[li][(size_t)h*HD*S_max + s*HD + d] = kcache[li][(size_t)h*HD*56 + s*HD + d];
                    vc[li][(size_t)h*HD*S_max + s*HD + d] = vcache[li][(size_t)h*HD*56 + s*HD + d];
                }
    }

    int S_total = 56;
    double max_all = 0;
    for (int di = 0; di < 3; di++) {
        char pb[64];
        snprintf(pb, sizeof(pb), "ar.dec%d.x_in", di+1);
        auto x_in = read_bin(pb);
        snprintf(pb, sizeof(pb), "ar.dec%d.logits", di+1);
        auto ref = read_bin(pb);
        S_total += 1;
        const int S_kv = S_total;

        ggml_init_params ip = { ggml_tensor_overhead()*512*NL*4, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
        ggml_set_input(x);
        std::vector<float> cur_host(D, 0.0f);
        std::vector<ggml_tensor *> kin_t(NL), vin_t(NL), knew_t(NL), vnew_t(NL);
        ggml_tensor * cur_t = x;
        for (int li = 0; li < NL; li++) {
            // cache 输入: 仅历史 S_kv-1 列; 新列由 concat 加入 (与官方 decode 语义一致)
            ggml_tensor * kin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, S_kv-1, NH);
            ggml_tensor * vin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, S_kv-1, NH);
            ggml_set_input(kin); ggml_set_input(vin);
            kin_t[li] = kin; vin_t[li] = vin;
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur_t), ws[li].qkv_b);
            ggml_tensor * qkv3 = ggml_reshape_3d(ctx, qkv, D, 3, 1);
            ggml_tensor * q = ggml_view_2d(ctx, qkv3, D, 1, qkv3->nb[2], 0);
            ggml_tensor * k = ggml_view_2d(ctx, qkv3, D, 1, qkv3->nb[2], qkv3->nb[1]);
            ggml_tensor * v = ggml_view_2d(ctx, qkv3, D, 1, qkv3->nb[2], 2*qkv3->nb[1]);
            q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, 1), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, NH, 1), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, NH, 1), 0, 2, 1, 3));
            ggml_tensor * kfull = ggml_concat(ctx, kin, kh, 1);
            ggml_tensor * vfull = ggml_concat(ctx, vin, vh, 1);
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kfull, vfull, nullptr, 1.0f/std::sqrt((float)HD), 0, 0);
            attn_out = ggml_reshape_2d(ctx, attn_out, D, 1);
            ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].out_w, attn_out), ws[li].out_b);
            ggml_tensor * cur2 = ggml_add(ctx, cur_t, o);
            ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur2, 1e-5f), ws[li].n1w), ws[li].n1b);
            ggml_tensor * h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b));
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            ggml_tensor * cur3 = ggml_add(ctx, n1, h);
            cur_t = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, cur3, 1e-5f), ws[li].n2w), ws[li].n2b);
            knew_t[li] = kh; vnew_t[li] = vh;
        }
        ggml_set_output(cur_t);
        for (int li = 0; li < NL; li++) { ggml_set_output(knew_t[li]); ggml_set_output(vnew_t[li]); }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512*NL*2, false);
        ggml_build_forward_expand(graph, cur_t);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, knew_t[li]); ggml_build_forward_expand(graph, vnew_t[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(x, x_in.data(), 0, x_in.size()*4);
        for (int li = 0; li < NL; li++) {
            std::vector<float> kflat((size_t)HD*(S_kv-1)*NH), vflat((size_t)HD*(S_kv-1)*NH);
            for (int h = 0; h < NH; h++)
                for (int s = 0; s < S_kv-1; s++)
                    for (int d = 0; d < HD; d++) {
                        kflat[(size_t)(h*(S_kv-1) + s)*HD + d] = kc[li][(size_t)h*HD*S_max + s*HD + d];
                        vflat[(size_t)(h*(S_kv-1) + s)*HD + d] = vc[li][(size_t)h*HD*S_max + s*HD + d];
                    }
            ggml_backend_tensor_set(kin_t[li], kflat.data(), 0, kflat.size()*4);
            ggml_backend_tensor_set(vin_t[li], vflat.data(), 0, vflat.size()*4);
        }
        ggml_backend_graph_compute(backend, graph);
        ggml_backend_tensor_get(cur_t, cur_host.data(), 0, D*4);
        for (int li = 0; li < NL; li++) {
            std::vector<float> knew(HD*NH), vnew(HD*NH);
            ggml_backend_tensor_get(knew_t[li], knew.data(), 0, knew.size()*4);
            ggml_backend_tensor_get(vnew_t[li], vnew.data(), 0, vnew.size()*4);
            for (int h = 0; h < NH; h++)
                for (int d = 0; d < HD; d++) {
                    kc[li][(size_t)h*HD*S_max + (S_kv-1)*HD + d] = knew[h*HD + d];
                    vc[li][(size_t)h*HD*S_max + (S_kv-1)*HD + d] = vnew[h*HD + d];
                }
        }
        std::vector<float> logits(VOCAB, 0.0f);
        for (int i = 0; i < VOCAB; i++) {
            const float * w = (const float *)((const char *)predict->data + (size_t)i*predict->nb[1]);
            float s = 0;
            for (int d = 0; d < D; d++) s += w[d] * cur_host[d];
            logits[i] = s;
        }
        double md = 0;
        for (int i = 0; i < VOCAB; i++) md = fmax(md, fabs(logits[i]-ref[i]));
        printf("[decode %d] max|diff| = %.6g %s\n", di+1, md, md < 1e-3 ? "PASS" : "FAIL");
        max_all = fmax(max_all, md);
        ggml_free(ctx);
    }
    printf("%s\n", max_all < 1e-3 ? "ALL PASS" : "FAIL");
    return max_all < 1e-3 ? 0 : 2;
}
