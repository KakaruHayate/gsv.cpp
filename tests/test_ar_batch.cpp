// GSV AR batch 能力 + 采样链集成
//  0) mask 解析构造校验 (vs python dump)
//  1) batched 首步 (B 条不同长度序列, 左 padding) vs torch process_prompt
//  2) batched 解码步 (KV cache [hd,L,nh,B] + concat 新列) vs torch decode_next_token
//  3) greedy 生成全循环 (argmax + EOS/early-stop 规则 + PE 位置) vs torch 参考
// golden: tools/dump_golden_ar_sampling.py
#include "gsv_sampler.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static const int D = 512, NH = 16, HD = 32, NL = 24, VOCAB = 1025, EOS = 1024;
static const int Y_LEN = 24;

static std::vector<float> read_bin(const std::string & dir, const char * name) {
    const std::string path = dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
    fclose(f);
    return v;
}

struct weights {
    ggml_tensor * qkv_w, * qkv_b, * out_w, * out_b, * n1w, * n1b, * f1w, * f1b, * f2w, * f2b, * n2w, * n2b;
};

// cache 布局: idx = d + l*HD + h*HD*Lmax + b*HD*Lmax*NH
struct batch_cache {
    int Lmax = 0, B = 0, len = 0;
    std::vector<std::vector<float>> k, v;
    void init(int n_layers, int Lmax_, int B_) {
        Lmax = Lmax_; B = B_;
        k.assign(n_layers, std::vector<float>((size_t)HD * Lmax * NH * B, 0.0f));
        v.assign(n_layers, std::vector<float>((size_t)HD * Lmax * NH * B, 0.0f));
        len = 0;
    }
    size_t idx(int d, int l, int h, int b) const {
        return (size_t)d + (size_t)l * HD + (size_t)h * HD * Lmax + (size_t)b * HD * Lmax * NH;
    }
};

struct runner {
    std::vector<weights> ws;
    ggml_tensor * predict = nullptr;
    ggml_tensor * audio_emb = nullptr;
    float alpha = 1.0f;
    std::vector<float> pe_tab;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    int B = 0, max_len = 0, S = 0;
    std::vector<int> pad_len;
    batch_cache cache;

    void init(const std::string & model_path, int B_, int max_len_) {
        B = B_; max_len = max_len_; S = max_len + Y_LEN;
        pad_len.assign(B, 0);
        ggml_context * wctx = nullptr;
        gguf_init_params gip = { false, &wctx };
        gguf_context * gf = gguf_init_from_file(model_path.c_str(), gip);
        if (!gf) { fprintf(stderr, "failed to open %s\n", model_path.c_str()); exit(1); }
        auto t = [&](const char * n) {
            ggml_tensor * tt = ggml_get_tensor(wctx, n);
            if (!tt) { fprintf(stderr, "missing tensor %s\n", n); exit(1); }
            return tt;
        };
        ws.resize(NL);
        char buf[128];
        for (int li = 0; li < NL; li++) {
            #define GW(f, nm) snprintf(buf, sizeof(buf), "transformer.block%d." nm, li); ws[li].f = t(buf)
            GW(qkv_w, "qkv_w"); GW(qkv_b, "qkv_b"); GW(out_w, "out_w"); GW(out_b, "out_b");
            GW(n1w, "norm1_w"); GW(n1b, "norm1_b"); GW(n2w, "norm2_w"); GW(n2b, "norm2_b");
            GW(f1w, "ffn1_w"); GW(f1b, "ffn1_b"); GW(f2w, "ffn2_w"); GW(f2b, "ffn2_b");
            #undef GW
        }
        predict = t("ar.predict");
        audio_emb = t("ar.audio_emb");
        alpha = *(const float *) t("ar.audio_pe_alpha")->data;
        pe_tab.assign((size_t)4096 * D, 0.0f);
        for (int pos = 0; pos < 4096; pos++)
            for (int i = 0; i < D; i += 2) {
                const float div = std::exp(i * -(std::log(10000.0f) / (float)D));
                pe_tab[(size_t)pos * D + i]     = std::sin(pos * div);
                pe_tab[(size_t)pos * D + i + 1] = std::cos(pos * div);
            }
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        backend = ggml_backend_dev_init(dev, nullptr);
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    void emb_lookup(int tok, float * out) const {
        const float * base = (const float *) audio_emb->data;
        for (int d = 0; d < D; d++) out[d] = base[(size_t)tok * D + d];
    }

    // 首步: x [B,S,D] 内存, mask [B,S,S] (-inf/0)
    void first(const std::vector<float> & x_host, const std::vector<float> & mask_host, std::vector<float> & logits_out) {
        ggml_init_params ip = { ggml_tensor_overhead() * 16384, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, S, B);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, 1, B);
        ggml_set_input(x); ggml_set_input(mask);
        ggml_tensor * mask16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
        ggml_tensor * cur = x;
        std::vector<ggml_tensor *> kout(NL), vout(NL);
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, S, B);
            ggml_tensor * q = ggml_view_3d(ctx, qkv4, D, S, B, qkv4->nb[2], qkv4->nb[3], 0);
            ggml_tensor * k = ggml_view_3d(ctx, qkv4, D, S, B, qkv4->nb[2], qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * v = ggml_view_3d(ctx, qkv4, D, S, B, qkv4->nb[2], qkv4->nb[3], 2 * qkv4->nb[1]);
            q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, HD, NH, S, B), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, HD, NH, S, B), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, HD, NH, S, B), 0, 2, 1, 3));
            kout[li] = kh; vout[li] = vh;
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kh, vh, mask16, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_4d(ctx, attn_out, D, S, B, 1);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, S, B);
            ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].out_w, attn_out), ws[li].out_b);
            ggml_tensor * c2 = ggml_add(ctx, cur, o);
            ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, c2, 1e-5f), ws[li].n1w), ws[li].n1b);
            ggml_tensor * h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b));
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            ggml_tensor * c3 = ggml_add(ctx, n1, h);
            cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, c3, 1e-5f), ws[li].n2w), ws[li].n2b);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);              // [V,S,B]
        ggml_tensor * last_view = ggml_view_3d(ctx, logits, VOCAB, 1, B, logits->nb[1], logits->nb[2], (int64_t)(S - 1) * logits->nb[1]);
        ggml_tensor * last = ggml_cont(ctx, last_view);   // tensor_get 是裸 memcpy, 需连续
        ggml_set_output(last);
        for (int li = 0; li < NL; li++) { ggml_set_output(kout[li]); ggml_set_output(vout[li]); }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, last);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, kout[li]); ggml_build_forward_expand(graph, vout[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 4);
        ggml_backend_graph_compute(backend, graph);
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(last, logits_out.data(), 0, logits_out.size() * 4);
        if (cache.Lmax == 0) cache.init(NL, S + 64, B);
        cache.len = S;
        for (int li = 0; li < NL; li++) {
            std::vector<float> kk((size_t)HD * S * NH * B), vv((size_t)HD * S * NH * B);
            ggml_backend_tensor_get(kout[li], kk.data(), 0, kk.size() * 4);
            ggml_backend_tensor_get(vout[li], vv.data(), 0, vv.size() * 4);
            for (int b = 0; b < B; b++)
                for (int h = 0; h < NH; h++)
                    for (int l = 0; l < S; l++)
                        for (int d = 0; d < HD; d++) {
                            const size_t src = (size_t)d + (size_t)l * HD + (size_t)h * HD * S + (size_t)b * HD * S * NH;
                            cache.k[li][cache.idx(d, l, h, b)] = kk[src];
                            cache.v[li][cache.idx(d, l, h, b)] = vv[src];
                        }
        }
        ggml_free(ctx);
    }

    // 解码步: x [B,1,D] 内存; mask 由 pad_len 解析构造 (k < pad_len[b] 屏蔽)
    void decode(const std::vector<float> & x_host, std::vector<float> & logits_out) {
        const int L = cache.len + 1;
        std::vector<float> mask_host((size_t)L * B, 0.0f);
        for (int b = 0; b < B; b++)
            for (int k = 0; k < L; k++)
                mask_host[(size_t)k + (size_t)b * L] = (k < pad_len[b]) ? -INFINITY : 0.0f;

        ggml_init_params ip = { ggml_tensor_overhead() * 16384, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, B);
        ggml_set_input(x);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, L, 1, 1, B);
        ggml_set_input(mask);
        ggml_tensor * mask16 = ggml_cast(ctx, mask, GGML_TYPE_F16);
        ggml_tensor * cur = x;
        std::vector<ggml_tensor *> kin_t(NL), vin_t(NL), kcol_t(NL), vcol_t(NL);
        for (int li = 0; li < NL; li++) {
            ggml_tensor * kin = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, cache.len, NH, B);
            ggml_tensor * vin = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HD, cache.len, NH, B);
            ggml_set_input(kin); ggml_set_input(vin);
            kin_t[li] = kin; vin_t[li] = vin;
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, 1, B);
            ggml_tensor * q = ggml_view_3d(ctx, qkv4, D, 1, B, qkv4->nb[2], qkv4->nb[3], 0);
            ggml_tensor * k = ggml_view_3d(ctx, qkv4, D, 1, B, qkv4->nb[2], qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * v = ggml_view_3d(ctx, qkv4, D, 1, B, qkv4->nb[2], qkv4->nb[3], 2 * qkv4->nb[1]);
            q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, HD, NH, 1, B), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, HD, NH, 1, B), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, HD, NH, 1, B), 0, 2, 1, 3));
            ggml_tensor * kfull = ggml_concat(ctx, kin, kh, 1);
            ggml_tensor * vfull = ggml_concat(ctx, vin, vh, 1);
            kcol_t[li] = ggml_cont(ctx, ggml_view_4d(ctx, kfull, HD, 1, NH, B, kfull->nb[1], kfull->nb[2], kfull->nb[3], (int64_t)cache.len * kfull->nb[1]));
            vcol_t[li] = ggml_cont(ctx, ggml_view_4d(ctx, vfull, HD, 1, NH, B, vfull->nb[1], vfull->nb[2], vfull->nb[3], (int64_t)cache.len * vfull->nb[1]));
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kfull, vfull, mask16, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, 1, B);
            ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].out_w, attn_out), ws[li].out_b);
            ggml_tensor * c2 = ggml_add(ctx, cur, o);
            ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, c2, 1e-5f), ws[li].n1w), ws[li].n1b);
            ggml_tensor * h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b));
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            ggml_tensor * c3 = ggml_add(ctx, n1, h);
            cur = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, c3, 1e-5f), ws[li].n2w), ws[li].n2b);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_set_output(logits);
        for (int li = 0; li < NL; li++) { ggml_set_output(kcol_t[li]); ggml_set_output(vcol_t[li]); }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, logits);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, kcol_t[li]); ggml_build_forward_expand(graph, vcol_t[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 4);
        for (int li = 0; li < NL; li++) {
            std::vector<float> kb((size_t)HD * cache.len * NH * B), vb((size_t)HD * cache.len * NH * B);
            for (int b = 0; b < B; b++)
                for (int h = 0; h < NH; h++)
                    for (int l = 0; l < cache.len; l++)
                        for (int d = 0; d < HD; d++) {
                            const size_t dst = (size_t)d + (size_t)l * HD + (size_t)h * HD * cache.len + (size_t)b * HD * cache.len * NH;
                            kb[dst] = cache.k[li][cache.idx(d, l, h, b)];
                            vb[dst] = cache.v[li][cache.idx(d, l, h, b)];
                        }
            ggml_backend_tensor_set(kin_t[li], kb.data(), 0, kb.size() * 4);
            ggml_backend_tensor_set(vin_t[li], vb.data(), 0, vb.size() * 4);
        }
        ggml_backend_graph_compute(backend, graph);
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, logits_out.size() * 4);
        for (int li = 0; li < NL; li++) {
            std::vector<float> kc((size_t)HD * NH * B), vc((size_t)HD * NH * B);
            ggml_backend_tensor_get(kcol_t[li], kc.data(), 0, kc.size() * 4);
            ggml_backend_tensor_get(vcol_t[li], vc.data(), 0, vc.size() * 4);
            for (int b = 0; b < B; b++)
                for (int h = 0; h < NH; h++)
                    for (int d = 0; d < HD; d++) {
                        const size_t src = (size_t)d + (size_t)h * HD + (size_t)b * HD * NH;
                        cache.k[li][cache.idx(d, cache.len, h, b)] = kc[src];
                        cache.v[li][cache.idx(d, cache.len, h, b)] = vc[src];
                    }
        }
        cache.len = L;
        ggml_free(ctx);
    }
};

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "models/gsv-ar-f32.gguf";
    const std::string gd = argc > 2 ? argv[2] : "tests/golden";

    auto xlens_f = read_bin(gd, "batch.x_len");
    const int B = (int)xlens_f.size();
    const int max_len = (int)(xlens_f[0]);
    const int S = max_len + Y_LEN;
    auto xy_pos_g = read_bin(gd, "batch.xy_pos");
    auto mask_g   = read_bin(gd, "batch.attn_mask");
    auto ref0     = read_bin(gd, "batch.step0.logits");
    auto prompt_f = read_bin(gd, "ar.prompt");
    std::vector<int32_t> prompt(Y_LEN);
    for (int i = 0; i < Y_LEN; i++) prompt[i] = (int32_t) prompt_f[i];

    runner R;
    R.init(model_path, B, max_len);
    for (int b = 0; b < B; b++) R.pad_len[b] = max_len - (int) xlens_f[b];
    printf("[batch] B=%d max_len=%d S=%d pad_len=[%d,%d,%d]\n", B, max_len, S, R.pad_len[0], R.pad_len[1], R.pad_len[2]);

    double worst = 0;

    // ---- 0) mask 解析构造校验 ----
    {
        std::vector<float> mask_ana((size_t)B * S * S, 0.0f);
        for (int b = 0; b < B; b++)
            for (int qq = 0; qq < S; qq++)
                for (int kk = 0; kk < S; kk++) {
                    bool m;
                    if (kk < R.pad_len[b]) m = true;
                    else if (qq < max_len) m = (kk >= max_len);
                    else m = (kk > qq);
                    mask_ana[(size_t)b * S * S + (size_t)qq * S + kk] = m ? -INFINITY : 0.0f;
                }
        double mm = 0;
        for (size_t i = 0; i < mask_ana.size(); i++) mm = std::max(mm, (double) std::fabs(mask_ana[i] - mask_g[i]));
        printf("[0] mask 解析构造 vs python dump max|Δ| = %.3g %s\n", mm, mm == 0 ? "OK" : "MISMATCH");
        if (mm != 0) worst = 1e9;
    }

    // ---- 1) batched 首步 ----
    std::vector<float> logits;
    R.first(xy_pos_g, mask_g, logits);
    {
        double md = 0;
        for (size_t i = 0; i < logits.size(); i++) md = std::max(md, (double) std::fabs(logits[i] - ref0[i]));
        printf("[1] batched 首步 logits max|Δ| = %.3g %s\n", md, md < 1e-3 ? "PASS" : "FAIL");
        worst = std::max(worst, md);
    }

    // ---- 2) batched 解码步 (固定 token 100/200/300) ----
    for (int step = 1; step <= 3; step++) {
        char pb[64];
        snprintf(pb, sizeof(pb), "batch.dec%d.x_in", step);
        auto x_in = read_bin(gd, pb);
        snprintf(pb, sizeof(pb), "batch.dec%d.logits", step);
        auto ref = read_bin(gd, pb);
        R.decode(x_in, logits);
        double md = 0;
        for (size_t i = 0; i < logits.size(); i++) md = std::max(md, (double) std::fabs(logits[i] - ref[i]));
        printf("[2] batched 解码步 %d logits max|Δ| = %.3g %s\n", step, md, md < 1e-3 ? "PASS" : "FAIL");
        worst = std::max(worst, md);
    }

    // ---- 3) greedy 生成 (argmax + EOS/early-stop 语义 + PE) ----
    auto run_greedy = [&](int early_stop, const char * tag) {
        R.cache = batch_cache{};
        std::vector<float> lg;
        R.first(xy_pos_g, mask_g, lg);
        std::vector<std::vector<int32_t>> y(B, prompt);
        std::vector<int32_t> idx_list(B, -1);
        std::vector<std::vector<int32_t>> y_out(B);
        for (int idx = 0; idx < 1500; idx++) {
            const int V = (idx < 11) ? VOCAB - 1 : VOCAB;    // idx<11: 排除 EOS
            for (int b = 0; b < B; b++) {
                const float * row = lg.data() + (size_t)b * VOCAB;
                int best = 0; float bv = -INFINITY;
                for (int v = 0; v < V; v++) if (row[v] > bv) { bv = row[v]; best = v; }
                y[b].push_back(best);
                if (idx_list[b] < 0 && best == EOS) {
                    idx_list[b] = idx;
                    y_out[b].assign(y[b].begin(), y[b].end() - 1);
                }
            }
            if (early_stop != -1 && ((int)y[0].size() - Y_LEN) > early_stop) {
                for (int b = 0; b < B; b++)
                    if (idx_list[b] < 0) {
                        idx_list[b] = idx;
                        y_out[b].assign(y[b].begin(), y[b].end() - 1);
                    }
            }
            bool any_unfinished = false;
            for (int b = 0; b < B; b++) if (idx_list[b] < 0) any_unfinished = true;
            if (!any_unfinished) break;
            std::vector<float> x_in((size_t)D * B, 0.0f);
            for (int b = 0; b < B; b++) {
                float e[512];
                R.emb_lookup(y[b].back(), e);
                for (int d = 0; d < D; d++)
                    x_in[(size_t)d + (size_t)b * D] = e[d] + R.alpha * R.pe_tab[(size_t)(Y_LEN + idx) * D + d];
            }
            R.decode(x_in, lg);
        }
        char nb[128];
        snprintf(nb, sizeof(nb), "batch.%s.idx", tag);
        auto ref_idx = read_bin(gd, nb);
        int idx_bad = 0;
        for (int b = 0; b < B; b++) if ((int)ref_idx[b] != idx_list[b]) idx_bad++;
        int tok_bad = 0;
        for (int b = 0; b < B; b++) {
            snprintf(nb, sizeof(nb), "batch.%s.seq%d.tokens", tag, b);
            auto ref_tok = read_bin(gd, nb);
            if (ref_tok.size() != y_out[b].size()) { tok_bad++; continue; }
            for (size_t i = 0; i < ref_tok.size(); i++)
                if ((int32_t) ref_tok[i] != y_out[b][i]) { tok_bad++; break; }
        }
        printf("[3] greedy(%s, early_stop=%d): idx 差异 %d/%d, token 序列差异 %d/%d  %s\n",
               tag, early_stop, idx_bad, B, tok_bad, B, (idx_bad == 0 && tok_bad == 0) ? "PASS" : "FAIL");
        if (idx_bad || tok_bad) worst = 1e9;
    };
    run_greedy(6, "greedy");
    run_greedy(3, "earlystop");

    // ---- 4) 采样链接入生成循环: top_k=1 必须与 greedy 完全一致; 固定种子可复现 ----
    auto run_sampled = [&](const gsv_sampler_cfg & cfg, uint64_t seed, int early_stop, std::vector<std::vector<int32_t>> & out) {
        R.cache = batch_cache{};
        std::vector<float> lg;
        R.first(xy_pos_g, mask_g, lg);
        std::vector<std::vector<int32_t>> y(B, prompt);
        std::vector<gsv_rng> rng;
        rng.reserve(B);
        for (int b = 0; b < B; b++) rng.emplace_back(seed + 7919ull * (uint64_t) b);
        std::vector<int32_t> idx_list(B, -1);
        out.assign(B, {});
        for (int idx = 0; idx < 1500; idx++) {
            for (int b = 0; b < B; b++) {
                float row[VOCAB];
                memcpy(row, lg.data() + (size_t)b * VOCAB, sizeof(row));
                if (idx < 11) row[EOS] = -INFINITY;              // 排除 EOS (等价 torch 的 [:, :-1])
                int32_t prev[2048];
                const int n_prev = (int) y[b].size() < 2048 ? (int) y[b].size() : 2048;
                for (int i = 0; i < n_prev; i++) prev[i] = y[b][y[b].size() - n_prev + i];
                float probs[VOCAB];
                gsv_logits_to_probs(cfg, row, VOCAB, prev, n_prev, probs);
                const int tok = gsv_sample(probs, VOCAB, rng[b], nullptr);
                y[b].push_back(tok);
                if (idx_list[b] < 0 && tok == EOS) {
                    idx_list[b] = idx;
                    out[b].assign(y[b].begin(), y[b].end() - 1);
                }
            }
            if (early_stop != -1 && ((int)y[0].size() - Y_LEN) > early_stop) {
                for (int b = 0; b < B; b++)
                    if (idx_list[b] < 0) {
                        idx_list[b] = idx;
                        out[b].assign(y[b].begin(), y[b].end() - 1);
                    }
            }
            bool any_unfinished = false;
            for (int b = 0; b < B; b++) if (idx_list[b] < 0) any_unfinished = true;
            if (!any_unfinished) break;
            std::vector<float> x_in((size_t)D * B, 0.0f);
            for (int b = 0; b < B; b++) {
                float e[512];
                R.emb_lookup(y[b].back(), e);
                for (int d = 0; d < D; d++)
                    x_in[(size_t)d + (size_t)b * D] = e[d] + R.alpha * R.pe_tab[(size_t)(Y_LEN + idx) * D + d];
            }
            R.decode(x_in, lg);
        }
    };

    {
        // top_k=1 -> probs 为 one-hot -> 采样等价 argmax; rep=1.0 与 greedy 参考一致
        gsv_sampler_cfg cfg; cfg.top_k = 1; cfg.top_p = 1.0f; cfg.temperature = 1.0f; cfg.repetition_penalty = 1.0f;
        std::vector<std::vector<int32_t>> out;
        run_sampled(cfg, 12345, 6, out);
        int bad = 0;
        for (int b = 0; b < B; b++) {
            char nb[128];
            snprintf(nb, sizeof(nb), "batch.greedy.seq%d.tokens", b);
            auto ref_tok = read_bin(gd, nb);
            if (ref_tok.size() != out[b].size()) { bad++; continue; }
            for (size_t i = 0; i < ref_tok.size(); i++)
                if ((int32_t) ref_tok[i] != out[b][i]) { bad++; break; }
        }
        printf("[4] 采样链(top_k=1) vs greedy 参考: 差异 %d/%d  %s\n", bad, B, bad == 0 ? "PASS" : "FAIL");
        if (bad) worst = 1e9;

        // 固定种子可复现
        gsv_sampler_cfg cfg2; cfg2.top_k = 15; cfg2.top_p = 0.9f; cfg2.temperature = 0.8f; cfg2.repetition_penalty = 1.35f;
        std::vector<std::vector<int32_t>> a, b2;
        run_sampled(cfg2, 777, 8, a);
        run_sampled(cfg2, 777, 8, b2);
        printf("[5] 采样(固定种子)可复现: %s\n", (a == b2) ? "PASS" : "FAIL");
        if (!(a == b2)) worst = 1e9;
    }

    printf("%s (worst = %.3g)\n", worst < 1e-3 ? "ALL PASS" : "FAIL", worst);
    return worst < 1e-3 ? 0 : 2;
}
