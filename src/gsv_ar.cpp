#include "gsv_ar.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include <chrono>

static bool g_prof = false;
static double g_ms_build = 0, g_ms_copy = 0, g_ms_compute = 0, g_ms_io = 0;
static int    g_steps = 0;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
static void prof_reset() { g_ms_build = g_ms_copy = g_ms_compute = g_ms_io = 0; g_steps = 0; }
static void prof_report(const char * tag) {
    if (!g_prof) return;
    printf("[prof:%s] steps=%d build=%.1fms copy=%.1fms compute=%.1fms other=%.1fms (per-step: %.2f/%.2f/%.2f)\n",
           tag, g_steps, g_ms_build, g_ms_copy, g_ms_compute, g_ms_io,
           g_ms_build / std::max(1, g_steps), g_ms_copy / std::max(1, g_steps), g_ms_compute / std::max(1, g_steps));
}

static const int Y_PE_MAX = 4096;

struct ar_weights {
    ggml_tensor * qkv_w, * qkv_b, * out_w, * out_b, * n1w, * n1b, * f1w, * f1b, * f2w, * f2b, * n2w, * n2b;
};

// cache 布局: idx = d + l*HD + h*HD*Lmax + b*HD*Lmax*NH
struct ar_cache {
    int Lmax = 0, B = 0, len = 0, hd = 0, nh = 0;
    std::vector<std::vector<float>> k, v;
    void init(int n_layers, int Lmax_, int B_, int hd_, int nh_) {
        Lmax = Lmax_; B = B_; hd = hd_; nh = nh_; len = 0;
        k.assign(n_layers, std::vector<float>((size_t)hd * Lmax * nh * B, 0.0f));
        v.assign(n_layers, std::vector<float>((size_t)hd * Lmax * nh * B, 0.0f));
    }
    size_t idx(int d, int l, int h, int b) const {
        return (size_t)d + (size_t)l * hd + (size_t)h * hd * Lmax + (size_t)b * hd * Lmax * nh;
    }
};

struct gsv_ar::impl {
    // hparams
    int D = 512, NH = 16, HD = 32, NL = 24, VOCAB = 1025, EOS = 1024, PHONES_VOCAB = 732, BERT_DIM = 1024;
    // weights
    std::vector<ar_weights> ws;
    ggml_tensor * predict = nullptr;
    ggml_tensor * text_emb = nullptr;
    ggml_tensor * audio_emb = nullptr;
    ggml_tensor * bert_proj = nullptr;
    ggml_tensor * bert_proj_b = nullptr;
    float text_pe_alpha = 1.0f, audio_pe_alpha = 1.0f;
    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    // 前端在 host 侧使用的权重副本 (GPU 后端时 tensor->data 不可直读)
    std::vector<float> h_text_emb, h_audio_emb, h_bert_proj, h_bert_proj_b;
    float h_text_alpha = 1.0f, h_audio_alpha = 1.0f;
    // runtime
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<float> pe_tab;   // [Y_PE_MAX, D] 行主序 (位置在前的正弦表, 无 alpha)
    ar_cache cache;          // host 镜像 (仅用于 Lmax 记录/兼容)
    int prompt_len = 0;
    // 常驻后端的 KV cache (避免每步回传整个 cache)
    ggml_context * cctx = nullptr;
    ggml_backend_buffer_t cbuf = nullptr;
    std::vector<ggml_tensor *> kc_t, vc_t;   // 各 [HD, Lmax, NH, B]
    int Lmax_alloc = 0;

    void alloc_cache(int Lmax, int B) {
        if (cctx && Lmax_alloc >= Lmax && cache.B == B) return;
        if (cbuf) { ggml_backend_buffer_free(cbuf); cbuf = nullptr; }
        if (cctx) { ggml_free(cctx); cctx = nullptr; }
        ggml_init_params ip = { ggml_tensor_overhead() * (2 * NL + 8), NULL, true };
        cctx = ggml_init(ip);
        kc_t.resize(NL); vc_t.resize(NL);
        for (int li = 0; li < NL; li++) {
            kc_t[li] = ggml_new_tensor_4d(cctx, GGML_TYPE_F32, HD, Lmax, NH, B);
            vc_t[li] = ggml_new_tensor_4d(cctx, GGML_TYPE_F32, HD, Lmax, NH, B);
        }
        cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
        if (!cbuf) { fprintf(stderr, "[gsv_ar] cache alloc failed\n"); exit(1); }
        for (int li = 0; li < NL; li++) {
            ggml_backend_tensor_memset(kc_t[li], 0, 0, ggml_nbytes(kc_t[li]));
            ggml_backend_tensor_memset(vc_t[li], 0, 0, ggml_nbytes(vc_t[li]));
        }
        Lmax_alloc = Lmax;
        cache.B = B;
    }

    // 从常驻 cache 取 [HD, L, NH, B] 视图
    ggml_tensor * k_view(ggml_context * ctx, int li, int L) const {
        ggml_tensor * t = kc_t[li];
        return ggml_view_4d(ctx, t, HD, L, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], 0);
    }
    ggml_tensor * v_view(ggml_context * ctx, int li, int L) const {
        ggml_tensor * t = vc_t[li];
        return ggml_view_4d(ctx, t, HD, L, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], 0);
    }
    ggml_tensor * k_col_view(ggml_context * ctx, int li, int col) const {
        ggml_tensor * t = kc_t[li];
        return ggml_view_4d(ctx, t, HD, 1, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], (int64_t) col * t->nb[1]);
    }
    ggml_tensor * v_col_view(ggml_context * ctx, int li, int col) const {
        ggml_tensor * t = vc_t[li];
        return ggml_view_4d(ctx, t, HD, 1, NH, cache.B, t->nb[1], t->nb[2], t->nb[3], (int64_t) col * t->nb[1]);
    }

    bool fuse_ln = true, fuse_act = true;   // 融合算子开关 (GSV_NO_FUSE / GSV_NO_FUSE_LN / GSV_NO_FUSE_ACT)

    // LN(x + r)*w + b: 融合算子不可用时退回 add + norm + mul + add
    ggml_tensor * ln_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * r,
                            ggml_tensor * w, ggml_tensor * b, float eps) {
        if (fuse_ln) return ggml_layernorm_affine(ctx, x, r, w, b, eps);
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, ggml_add(ctx, x, r), eps), w), b);
    }
    // act(a + b): 融合算子不可用时退回 add + 激活
    ggml_tensor * add_act(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int act) {
        if (fuse_act) return ggml_add_act(ctx, a, b, act);
        ggml_tensor * t = ggml_add(ctx, a, b);
        return act == GGML_ACT_RELU ? ggml_relu(ctx, t) : ggml_gelu_erf(ctx, t);
    }

    void mk_pe() {
        pe_tab.assign((size_t)Y_PE_MAX * D, 0.0f);
        for (int pos = 0; pos < Y_PE_MAX; pos++)
            for (int i = 0; i < D; i += 2) {
                const float div = std::exp(i * -(std::log(10000.0f) / (float)D));
                pe_tab[(size_t)pos * D + i]     = std::sin(pos * div);
                pe_tab[(size_t)pos * D + i + 1] = std::cos(pos * div);
            }
    }

    // 构造 batched 输入: xy_pos [B,S,D] 与 mask [B,S,S] (-inf/0); S = max_len + prompt_len
    void build_inputs(const std::vector<gsv_ar_request> & reqs, std::vector<float> & xy,
                      std::vector<float> & mask, int & S, int & max_len) const {
        const int B = (int) reqs.size();
        max_len = 0;
        for (const auto & r : reqs) max_len = std::max(max_len, (int) r.phones.size());
        S = max_len + prompt_len;
        xy.assign((size_t)B * S * D, 0.0f);
        mask.assign((size_t)B * S * S, 0.0f);
        const float * te  = h_text_emb.data();
        const float * bp  = h_bert_proj.data();      // ne=(BERT_DIM, D)
        const float * bpb = h_bert_proj_b.data();
        const float * ae  = h_audio_emb.data();
        for (int b = 0; b < B; b++) {
            const gsv_ar_request & r = reqs[b];
            const int T = (int) r.phones.size();
            const int pad = max_len - T;                          // 左 padding
            for (int t = 0; t < T; t++) {
                float * dst = xy.data() + ((size_t)b * S + pad + t) * D;
                const float * tok = te + (size_t) r.phones[t] * D;
                // bert_proj: y[d] = sum_k W[d*BERT_DIM + k] * bert[k*T + t] + bias[d]
                for (int d = 0; d < D; d++) {
                    float acc = bpb[d];
                    const float * wr = bp + (size_t)d * BERT_DIM;
                    for (int k = 0; k < BERT_DIM; k++) acc += wr[k] * r.bert[(size_t)k * T + t];
                    dst[d] = tok[d] + acc + h_text_alpha * pe_tab[(size_t)t * D + d];
                }
            }
            for (int t = 0; t < prompt_len; t++) {
                float * dst = xy.data() + ((size_t)b * S + max_len + t) * D;
                const float * emb = ae + (size_t) r.prompt[t] * D;
                for (int d = 0; d < D; d++)
                    dst[d] = emb[d] + h_audio_alpha * pe_tab[(size_t)t * D + d];
            }
            // mask: 左 padding key 屏蔽; x query 看不到 y key; y query causal
            for (int q = 0; q < S; q++)
                for (int k = 0; k < S; k++) {
                    bool m;
                    if (k < pad) m = true;
                    else if (q < max_len) m = (k >= max_len);
                    else m = (k > q);
                    mask[(size_t)b * S * S + (size_t)q * S + k] = m ? -INFINITY : 0.0f;
                }
        }
    }

    // 首步: 全前向, 输出 logits [B,V] 与各层 K/V cache
    void run_first(const std::vector<float> & xy_host, const std::vector<float> & mask_host, int S,
                   std::vector<float> & logits_out) {
        const int B = cache.B;
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
            ggml_tensor * n1 = ln_affine(ctx, cur, o, ws[li].n1w, ws[li].n1b, 1e-5f);
            ggml_tensor * h = add_act(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b, GGML_ACT_RELU);
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            cur = ln_affine(ctx, n1, h, ws[li].n2w, ws[li].n2b, 1e-5f);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_tensor * last_view = ggml_view_3d(ctx, logits, VOCAB, 1, B, logits->nb[1], logits->nb[2], (int64_t)(S - 1) * logits->nb[1]);
        ggml_tensor * last = ggml_cont(ctx, last_view);   // tensor_get 是裸 memcpy, 需连续
        ggml_set_output(last);
        // K/V 直接写入常驻 cache (无 host 往返)
        std::vector<ggml_tensor *> kcpy(NL), vcpy(NL);
        for (int li = 0; li < NL; li++) {
            kcpy[li] = ggml_cpy(ctx, kout[li], ggml_view_4d(ctx, kc_t[li], HD, S, NH, B, kc_t[li]->nb[1], kc_t[li]->nb[2], kc_t[li]->nb[3], 0));
            vcpy[li] = ggml_cpy(ctx, vout[li], ggml_view_4d(ctx, vc_t[li], HD, S, NH, B, vc_t[li]->nb[1], vc_t[li]->nb[2], vc_t[li]->nb[3], 0));
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, last);
        for (int li = 0; li < NL; li++) { ggml_build_forward_expand(graph, kcpy[li]); ggml_build_forward_expand(graph, vcpy[li]); }
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(x, xy_host.data(), 0, xy_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 4);
        ggml_backend_graph_compute(backend, graph);
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(last, logits_out.data(), 0, logits_out.size() * 4);
        cache.len = S;
        ggml_free(ctx);
    }

    // 解码步: x [B,1,D]; 每序列 mask = (k < pad_len[b])
    void run_decode(const std::vector<float> & x_host, const std::vector<int> & pad_len, std::vector<float> & logits_out) {
        const double t_a = now_ms();
        const int B = cache.B;
        const int L = cache.len + 1;
        std::vector<ggml_fp16_t> mask_host((size_t)L * B);
        for (int b = 0; b < B; b++)
            for (int k = 0; k < L; k++)
                mask_host[(size_t)k + (size_t)b * L] = ggml_fp32_to_fp16((k < pad_len[b]) ? -INFINITY : 0.0f);

        ggml_init_params ip = { ggml_tensor_overhead() * 2048, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, B);
        ggml_set_input(x);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, L, 1, 1, B);
        ggml_set_input(mask);
        ggml_tensor * cur = x;
        for (int li = 0; li < NL; li++) {
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].qkv_w, cur), ws[li].qkv_b);
            ggml_tensor * qkv4 = ggml_reshape_4d(ctx, qkv, D, 3, 1, B);
            // S=1: q/k/v 在 qkv 里的内存序已是 (hd, nh)，可直接用 (HD,1,NH,B) 视图送 flash / 送 KV cpy
            // —— 省掉每层 6 次 ggml_cont (3 次切 q/k/v + 3 次 permute)，数值完全相同
            ggml_tensor * qh      = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 0);
            ggml_tensor * ksrc    = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], qkv4->nb[1]);
            ggml_tensor * vsrc    = ggml_view_4d(ctx, qkv4, HD, 1, NH, B, 4 * HD, 4 * HD, qkv4->nb[3], 2 * qkv4->nb[1]);
            // 新列写入常驻 cache (cpy 副作用); flash 读 concat(旧列视图, cpy输出) 以显式化 "写 -> 读" 依赖
            ggml_tensor * kcol_w = ggml_cpy(ctx, ksrc, k_col_view(ctx, li, cache.len));
            ggml_tensor * vcol_w = ggml_cpy(ctx, vsrc, v_col_view(ctx, li, cache.len));
            ggml_tensor * kfull = ggml_concat(ctx, k_view(ctx, li, cache.len), kcol_w, 1);
            ggml_tensor * vfull = ggml_concat(ctx, v_view(ctx, li, cache.len), vcol_w, 1);
            ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, qh, kfull, vfull, mask, 1.0f / std::sqrt((float)HD), 0.0f, 0.0f);
            attn_out = ggml_reshape_3d(ctx, attn_out, D, 1, B);
            ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].out_w, attn_out), ws[li].out_b);
            ggml_tensor * n1 = ln_affine(ctx, cur, o, ws[li].n1w, ws[li].n1b, 1e-5f);
            ggml_tensor * h = add_act(ctx, ggml_mul_mat(ctx, ws[li].f1w, n1), ws[li].f1b, GGML_ACT_RELU);
            h = ggml_add(ctx, ggml_mul_mat(ctx, ws[li].f2w, h), ws[li].f2b);
            cur = ln_affine(ctx, n1, h, ws[li].n2w, ws[li].n2b, 1e-5f);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, predict, cur);
        ggml_set_output(logits);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, logits);   // KV cpy 经 concat 连到 flash, 已是 logits 的祖先
        ggml_gallocr_alloc_graph(galloc, graph);
        const double t_b = now_ms();
        ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * 4);
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * 2);
        const double t_c = now_ms();
        ggml_backend_graph_compute(backend, graph);
        const double t_d = now_ms();
        logits_out.assign((size_t)VOCAB * B, 0.0f);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, logits_out.size() * 4);
        const double t_e = now_ms();
        g_ms_build += t_b - t_a; g_ms_copy += t_c - t_b; g_ms_compute += t_d - t_c; g_ms_io += t_e - t_d;
        g_steps++;
        cache.len = L;
        ggml_free(ctx);
    }

    void emb_lookup(int tok, float * out) const {
        const float * base = h_audio_emb.data();
        for (int d = 0; d < D; d++) out[d] = base[(size_t)tok * D + d];
    }
};

gsv_ar::gsv_ar() : p(new impl()) {}
gsv_ar::~gsv_ar() {
    if (p->galloc) ggml_gallocr_free(p->galloc);
    if (p->backend) ggml_backend_free(p->backend);
    if (p->wbuf) ggml_backend_buffer_free(p->wbuf);
    if (p->gf) gguf_free(p->gf);
    delete p;
}

int gsv_ar::hidden_dim() const { return p->D; }
int gsv_ar::vocab_size() const { return p->VOCAB; }
int gsv_ar::eos() const { return p->EOS; }
int gsv_ar::bert_dim() const { return p->BERT_DIM; }
int gsv_ar::prompt_len_expected() const { return p->prompt_len; }

gsv_ar * gsv_ar::load(const std::string & gguf_path, const gsv_ar_cfg & cfg) {
    gsv_ar * m = new gsv_ar();
    impl & s = *m->p;
    // 注意: ggml-vulkan 在**静态初始化**阶段就读取 GGML_VK_DISABLE_COOPMAT2 (早于 main),
    // 因此内部设置无效, 只能提示调用方在启动前设置 (实测: 外部设置 logits Δ 0.0035, 未设置 0.0201)

    // 设备选择 (cfg.device: "" = CPU, "vulkan"/"gpu" = 第一个 GPU 设备)
    ggml_backend_dev_t dev = nullptr;
    {
        const bool want_gpu = cfg.device == "gpu" || cfg.device == "vulkan" || cfg.device == "cuda" ||
                               cfg.device == "GPU"  || cfg.device == "Vulkan" || cfg.device == "CUDA";
        const int n_dev = (int) ggml_backend_dev_count();
        for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            if (want_gpu) { if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = d; }
            else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (!dev) {
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        }
        if (!dev) { fprintf(stderr, "[gsv_ar] no usable backend device"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_ar] device: %s (%s)", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        if (want_gpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            getenv("GGML_VK_DISABLE_COOPMAT2") == nullptr) {
            fprintf(stderr, "[gsv_ar] warning: Vulkan coopmat2 未禁用, AR 精度会下降 (logits Δ 0.0035 -> 0.0201); "
                            "请在进程启动前设置 GGML_VK_DISABLE_COOPMAT2=1");
        }
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_ar] backend init failed"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_ar] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    // 后端 buffer 分配 + 从文件上传 (no_alloc 模式下 tensor 没有数据)
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_ar] weight buffer alloc failed"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_ar] reopen failed"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tname);
            if (!tt) continue;
            const size_t nbytes = ggml_nbytes(tt);
            std::vector<char> buf(nbytes);
            if (fseek(fp, (long)(data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_ar] read tensor %s failed", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    auto kv_u32 = [&](const char * key, int def) {
        const int64_t id = gguf_find_key(s.gf, key);
        return id < 0 ? def : (int) gguf_get_val_u32(s.gf, id);
    };
    s.D            = kv_u32("ar.hidden_dim", 512);
    s.NH           = kv_u32("ar.head", 16);
    s.NL           = kv_u32("ar.n_layer", 24);
    s.VOCAB        = kv_u32("ar.vocab_size", 1025);
    s.PHONES_VOCAB = kv_u32("ar.phoneme_vocab_size", 732);
    s.EOS          = kv_u32("ar.eos", 1024);
    s.BERT_DIM     = 1024;
    s.HD           = s.D / s.NH;

    auto t = [&](const char * n) {
        ggml_tensor * tt = ggml_get_tensor(s.wctx, n);
        if (!tt) { fprintf(stderr, "[gsv_ar] missing tensor %s\n", n); return (ggml_tensor *) nullptr; }
        return tt;
    };
    s.ws.resize(s.NL);
    char buf[128];
    for (int li = 0; li < s.NL; li++) {
        #define GW_L(f, nm) snprintf(buf, sizeof(buf), "transformer.block%d." nm, li); if (!(s.ws[li].f = t(buf))) { delete m; return nullptr; }
        GW_L(qkv_w, "qkv_w"); GW_L(qkv_b, "qkv_b"); GW_L(out_w, "out_w"); GW_L(out_b, "out_b");
        GW_L(n1w, "norm1_w"); GW_L(n1b, "norm1_b"); GW_L(n2w, "norm2_w"); GW_L(n2b, "norm2_b");
        GW_L(f1w, "ffn1_w"); GW_L(f1b, "ffn1_b"); GW_L(f2w, "ffn2_w"); GW_L(f2b, "ffn2_b");
        #undef GW_L
    }
    if (!(s.predict      = t("ar.predict")))      { delete m; return nullptr; }
    if (!(s.text_emb     = t("ar.text_emb")))     { delete m; return nullptr; }
    if (!(s.audio_emb    = t("ar.audio_emb")))    { delete m; return nullptr; }
    if (!(s.bert_proj    = t("ar.bert_proj")))    { delete m; return nullptr; }
    if (!(s.bert_proj_b  = t("ar.bert_proj_b")))  { delete m; return nullptr; }
    // host 侧依赖的可直读权重必须是 F32 (bert_proj 当前在 host 侧投影; embedding 查表同理)
    if (s.bert_proj->type != GGML_TYPE_F32 || s.text_emb->type != GGML_TYPE_F32 || s.audio_emb->type != GGML_TYPE_F32) {
        fprintf(stderr, "[gsv_ar] text_emb/audio_emb/bert_proj 必须为 F32 (host 侧直读); 当前为 %d/%d/%d\n",
                s.text_emb->type, s.audio_emb->type, s.bert_proj->type);
        delete m; return nullptr;
    }
    // host 侧副本 (CPU/GPU 后端统一)
    auto to_host = [&](ggml_tensor * tt, std::vector<float> & out) {
        out.assign(ggml_nelements(tt), 0.0f);
        ggml_backend_tensor_get(tt, out.data(), 0, out.size() * 4);
    };
    to_host(s.text_emb, s.h_text_emb);
    to_host(s.audio_emb, s.h_audio_emb);
    to_host(s.bert_proj, s.h_bert_proj);
    to_host(s.bert_proj_b, s.h_bert_proj_b);
    {
        float a1 = 0, a2 = 0;
        ggml_backend_tensor_get(t("ar.text_pe_alpha"), &a1, 0, 4);
        ggml_backend_tensor_get(t("ar.audio_pe_alpha"), &a2, 0, 4);
        s.h_text_alpha = a1; s.h_audio_alpha = a2;
    }

    g_prof = getenv("GSV_AR_PROFILE") != nullptr;
    // 融合算子支持探测: 用代表性形状建一次节点问后端 (CPU 恒支持; Vulkan 需 shader 在册)
    {
        ggml_init_params fp = { ggml_tensor_overhead() * 32, NULL, true };
        ggml_context * fctx = ggml_init(fp);
        ggml_tensor * fx = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * fw = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 64);
        ggml_tensor * fb = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 64);
        ggml_tensor * fln = ggml_layernorm_affine(fctx, fx, nullptr, fw, fb, 1e-5f);
        ggml_tensor * fac = ggml_add_act(fctx, fx, fw, GGML_ACT_RELU);
        const bool sup = ggml_backend_supports_op(s.backend, fln) && ggml_backend_supports_op(s.backend, fac);
        ggml_free(fctx);
        s.fuse_ln = s.fuse_act = sup;
    }
    if (getenv("GSV_NO_FUSE"))     s.fuse_ln = s.fuse_act = false;
    if (getenv("GSV_NO_FUSE_LN"))  s.fuse_ln = false;
    if (getenv("GSV_NO_FUSE_ACT")) s.fuse_act = false;
    s.mk_pe();
    if (cfg.verbose)
        printf("[gsv_ar] loaded %s: D=%d head=%d layers=%d vocab=%d EOS=%d bert=%d alpha=(%.4f, %.4f)\n",
               gguf_path.c_str(), s.D, s.NH, s.NL, s.VOCAB, s.EOS, s.BERT_DIM, s.text_pe_alpha, s.audio_pe_alpha);
    if (cfg.verbose) printf("[gsv_ar] fused ops: ln=%s act=%s\n", s.fuse_ln ? "on" : "off", s.fuse_act ? "on" : "off");
    return m;
}

void gsv_ar::first_logits(const std::vector<gsv_ar_request> & reqs, std::vector<float> & logits_out) {
    impl & s = *p;
    if (reqs.empty()) return;
    if (s.prompt_len == 0) s.prompt_len = (int) reqs[0].prompt.size();
    std::vector<float> xy, mask;
    int S = 0, max_len = 0;
    s.build_inputs(reqs, xy, mask, S, max_len);
    s.alloc_cache(S + 8, (int) reqs.size());
    s.cache.len = 0;
    s.run_first(xy, mask, S, logits_out);
}

gsv_ar_result gsv_ar::generate(const std::vector<gsv_ar_request> & reqs,
                               const gsv_sampler_cfg & sampler,
                               uint64_t seed_base,
                               int early_stop_num,
                               int max_steps,
                               const std::vector<int32_t> * oracle_tokens) {
    impl & s = *p;
    gsv_ar_result res;
    const int B = (int) reqs.size();
    if (B == 0) return res;
    for (const auto & r : reqs)
        if (r.prompt.size() != reqs[0].prompt.size())
            fprintf(stderr, "[gsv_ar] warning: 批内 prompt 长度不一致\n");
    s.prompt_len = (int) reqs[0].prompt.size();

    std::vector<float> xy, mask, lg;
    int S = 0, max_len = 0;
    s.build_inputs(reqs, xy, mask, S, max_len);
    std::vector<int> pad_len(B, 0);
    for (int b = 0; b < B; b++) pad_len[b] = max_len - (int) reqs[b].phones.size();

    s.alloc_cache(S + max_steps + 8, B);
    s.cache.len = 0;
    s.run_first(xy, mask, S, lg);
    prof_reset();

    std::vector<std::vector<int32_t>> y(B);
    for (int b = 0; b < B; b++) y[b] = reqs[b].prompt;
    std::vector<int32_t> idx_list(B, -1);
    res.tokens.assign(B, {});
    std::vector<gsv_rng> rng;
    rng.reserve(B);
    for (int b = 0; b < B; b++) rng.emplace_back(seed_base + 0x9E3779B9ull * (uint64_t) b);
    std::vector<float> prob_dump;   // [steps][V] (seq0), 由 GSV_AR_PROB_DUMP 触发

    for (int idx = 0; idx < max_steps; idx++) {
        for (int b = 0; b < B; b++) {
            if (idx_list[b] >= 0) continue;                  // 已完成
            float row[1025];
            memcpy(row, lg.data() + (size_t) b * s.VOCAB, sizeof(float) * s.VOCAB);
            if (idx < 11) row[s.EOS] = -INFINITY;            // 至少 10 token 不停止 (等价 torch [:, :-1])
            int32_t prev[4096];
            const int np = (int) y[b].size() < 4096 ? (int) y[b].size() : 4096;
            for (int i = 0; i < np; i++) prev[i] = y[b][y[b].size() - np + i];
            float probs[1025];
            gsv_logits_to_probs(sampler, row, s.VOCAB, prev, np, probs);
            if (b == 0 && getenv("GSV_AR_PROB_DUMP")) prob_dump.insert(prob_dump.end(), probs, probs + s.VOCAB);
            int tok = gsv_sample(probs, s.VOCAB, rng[b], nullptr);
            if (oracle_tokens != nullptr && idx < (int) oracle_tokens->size()) tok = (*oracle_tokens)[idx];
            y[b].push_back(tok);
            if (tok == s.EOS) {
                idx_list[b] = idx;
                res.tokens[b].assign(y[b].begin(), y[b].end() - 1);
                res.lens.push_back(idx);
            }
        }
        if (early_stop_num != -1 && ((int) y[0].size() - s.prompt_len) > early_stop_num) {
            for (int b = 0; b < B; b++)
                if (idx_list[b] < 0) {
                    idx_list[b] = idx;
                    res.tokens[b].assign(y[b].begin(), y[b].end() - 1);
                    res.lens.push_back(idx);
                }
        }
        bool any_unfinished = false;
        for (int b = 0; b < B; b++) if (idx_list[b] < 0) any_unfinished = true;
        if (!any_unfinished) break;

        std::vector<float> x_in((size_t) s.D * B, 0.0f);
        for (int b = 0; b < B; b++) {
            float e[512];
            s.emb_lookup(y[b].back(), e);
            const int pos = s.prompt_len + idx;
            for (int d = 0; d < s.D; d++)
                x_in[(size_t) d + (size_t) b * s.D] = e[d] + s.h_audio_alpha * s.pe_tab[(size_t) pos * s.D + d];
        }
        s.run_decode(x_in, pad_len, lg);
    }
    for (int b = 0; b < B; b++)
        if (idx_list[b] < 0) {
            res.tokens[b] = y[b];
            res.lens.push_back(max_steps);
        }
    prof_report("decode");
    const char * pd = getenv("GSV_AR_PROB_DUMP");
    if (pd != nullptr && !prob_dump.empty()) {
        FILE * f = fopen(pd, "wb");
        if (f) { fwrite(prob_dump.data(), 4, prob_dump.size(), f); fclose(f);
                 fprintf(stderr, "[gsv_ar] prob dump: %zu steps -> %s", prob_dump.size() / s.VOCAB, pd); }
    }
    return res;
}
