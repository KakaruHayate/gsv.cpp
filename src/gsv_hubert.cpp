#include "gsv_hubert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 与 convert_hubert.py 对应的常量
static const int  TRAW      = 16000;
static const int  T_FRAMES  = 49;
static const int  D_HID     = 768;
static const int  NH        = 12;
static const int  HD        = D_HID / NH;   // 64
static const int  C_CONV    = 512;
static const int  N_CONV    = 7;
static const int  CK[7]     = {10, 3, 3, 3, 3, 2, 2};
static const int  CS[7]     = {5, 2, 2, 2, 2, 2, 2};
static const int  CONV_OL[7] = {3199, 1599, 799, 399, 199, 99, 49};
static const int  NL        = 12;
static const float LN_EPS   = 1e-5f;
static const float GN_EPS   = 1e-5f;
static const int  GN_GROUPS = 512;  // transformers: GroupNorm(num_groups=512) = per-channel

struct gsv_hubert::impl {
    ggml_backend_t s_back = nullptr;   // 供 build_xxx 取 buffer type
    int D = D_HID, T = T_FRAMES;
    int n_threads = 0;
    bool verbose = false;

    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr;
    // 每图独立 galloc: 共用一个 galloc 时, 后一个图的 alloc 触发扩容会 free 旧缓冲,
    // 前一个图的所有节点指针悬空 (新缓冲复用同一段虚拟内存), 计算时互相涂写
    ggml_gallocr_t g1_galloc = nullptr;
    ggml_gallocr_t g2_galloc = nullptr;

    // 图 1: CNN 前端 (7 层 conv_1d + GN + gelu) + feat_proj
    ggml_context * g1_ctx = nullptr;
    ggml_cgraph  * g1 = nullptr;
    ggml_tensor  * g1_in = nullptr;
    ggml_tensor  * g1_out = nullptr;     // [768, T]

    // 图 2: 12 层 post-LN transformer
    ggml_context * g2_ctx = nullptr;
    ggml_cgraph  * g2 = nullptr;
    ggml_tensor  * g2_in = nullptr;
    ggml_tensor  * g2_out = nullptr;     // [768, T]

    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) { fprintf(stderr, "[gsv_hubert] missing weight %s\n", n); abort(); }
        return t;
    }

    void build_g1() {
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] ctx init...\n");
        ggml_init_params ip = { ggml_tensor_overhead() * 8192, NULL, true };
        g1_ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(g1_ctx, GGML_TYPE_F32, TRAW);
        ggml_set_input(x);
        g1_in = x;
        char nm[160];
        for (int i = 0; i < N_CONV; i++) {
            if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] conv %d...\n", i);
            snprintf(nm, sizeof(nm), "hubert.feat_conv.%d.w", i);
            // GGUF 字节: k 最外/oc 最内 ([K][IC][OC] 行主) —— host conv 语义.
            // ggml im2col 要求 k 最内 (列主 [OC,IC,K]) —— load() 已把字节转置为 k 最内,
            // 此处 reshape 元数据为 [K, IC, OC] (列主) 即可。
            ggml_tensor * w = ggml_reshape_3d(g1_ctx, need(nm), CK[i], i == 0 ? 1 : C_CONV, C_CONV);
            ggml_tensor * y = ggml_conv_1d(g1_ctx, w, x, CS[i], 0, 1);   // [OL, C, 1]
            if (i == 0) {
                // GroupNorm(512=per-channel): ggml_group_norm needs channels at ne2.
                // conv out [OL, C, 1] -> [OL, 1, C, 1] view; GN+affine; reshape back.
                y = ggml_reshape_4d(g1_ctx, y, CONV_OL[i], 1, C_CONV, 1);
                y = ggml_group_norm(g1_ctx, y, GN_GROUPS, GN_EPS);
                // GroupNorm(affine=True): per-channel gamma/beta (channels at ne2)
                ggml_tensor * gw = ggml_reshape_4d(g1_ctx, need("hubert.feat_conv.0.norm_w"), 1, 1, C_CONV, 1);
                ggml_tensor * gb = ggml_reshape_4d(g1_ctx, need("hubert.feat_conv.0.norm_b"), 1, 1, C_CONV, 1);
                y = ggml_add(g1_ctx, ggml_mul(g1_ctx, y, gw), gb);
                y = ggml_reshape_3d(g1_ctx, y, CONV_OL[i], C_CONV, 1);
            }
            y = ggml_gelu_erf(g1_ctx, y);
            x = y;
        }
        // [49, 512, 1] -> [512, 49] (transpose copy, 转 mul_mat 友好布局)
        x = ggml_cont(g1_ctx, ggml_permute(g1_ctx, x, 1, 0, 2, 3));
        x = ggml_add(g1_ctx,
                     ggml_mul(g1_ctx, ggml_norm(g1_ctx, x, LN_EPS), need("hubert.feat_proj.norm_w")),
                     need("hubert.feat_proj.norm_b"));
        g1_out = ggml_add(g1_ctx, ggml_mul_mat(g1_ctx, need("hubert.feat_proj.w"), x),
                          need("hubert.feat_proj.b"));
        ggml_set_output(g1_out);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] graph...\n");
        g1 = ggml_new_graph_custom(g1_ctx, 4096, false);
        ggml_build_forward_expand(g1, g1_out);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] galloc...\n");
        g1_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
        ggml_gallocr_alloc_graph(g1_galloc, g1);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] alloc ok\n");
    }

    void build_g2() {
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] ctx init...\n");
        ggml_init_params ip = { ggml_tensor_overhead() * 16384, NULL, true };
        g2_ctx = ggml_init(ip);
        ggml_tensor * cur = ggml_new_tensor_2d(g2_ctx, GGML_TYPE_F32, D, T);
        ggml_set_input(cur);
        g2_in = cur;
        // 全零 F16 mask (FA 要求 F16 mask; 无 padding, 等价无 mask —— 与 BERT 同款)
        ggml_tensor * msk = ggml_new_tensor_4d(g2_ctx, GGML_TYPE_F16, T, T, 1, 1);
        ggml_set_input(msk);

        char nm[192], base[128];
        const bool dbg = getenv("GSV_HUBERT_DEBUG") != nullptr;
        const int nl_eff = getenv("GSV_HUBERT_LAYERS") ? atoi(getenv("GSV_HUBERT_LAYERS")) : NL;
        const float scale = 1.0f / sqrtf((float) HD);
        // W(): "格式化名字 + 查表" 在单次调用内完成。切勿写成
        //   ggml_add(ctx, ggml_mul_mat(ctx, need(nm), x), need((snprintf(nm,...), nm)))
        // —— 同一全表达式内共享 nm 时, 实参求值顺序未指定 (MSVC 从右往左), 右半会先覆盖 nm,
        //    左半的 need(nm) 便取到偏置当权重。
        auto W = [&](const char * sfx) -> ggml_tensor * {
            snprintf(nm, sizeof(nm), "%s.%s", base, sfx);
            return need(nm);
        };
        for (int li = 0; li < nl_eff; li++) {
            if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] layer %d...\n", li);
            snprintf(base, sizeof(base), "hubert.layer.%d", li);
            ggml_tensor * q = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("q_w"), cur), W("q_b"));
            if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] L%d q ok\n", li);
            ggml_tensor * k = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("k_w"), cur), W("k_b"));
            ggml_tensor * v = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("v_w"), cur), W("v_b"));
            ggml_tensor * qh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, q, HD, NH, T, 1), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, k, HD, NH, T, 1), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, v, HD, NH, T, 1), 0, 2, 1, 3));
            if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] L%d fa prep\n", li);
            ggml_tensor * attn = ggml_flash_attn_ext(g2_ctx, qh, kh, vh, msk, scale, 0.0f, 0.0f);
            if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] L%d fa ok\n", li);
            // (HD, NH, T, 1) -> [D, T]: 数据序 = hd + h*HD (h*HD+hd, torch concat 语义), 折叠前两维即可
            attn = ggml_reshape_2d(g2_ctx, ggml_reshape_4d(g2_ctx, attn, D, T, 1, 1), D, T);
            ggml_tensor * ao = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("out_w"), attn), W("out_b"));
            // post-LN 1: LN(cur + ao) * w + b
            cur = ggml_add(g2_ctx,
                           ggml_mul(g2_ctx, ggml_norm(g2_ctx, ggml_add(g2_ctx, cur, ao), LN_EPS), W("ln1_w")),
                           W("ln1_b"));
            // FFN
            ggml_tensor * h = ggml_gelu_erf(g2_ctx,
                                            ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("ffn1_w"), cur), W("ffn1_b")));
            ggml_tensor * fo = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("ffn2_w"), h), W("ffn2_b"));
            // post-LN 2
            cur = ggml_add(g2_ctx,
                           ggml_mul(g2_ctx, ggml_norm(g2_ctx, ggml_add(g2_ctx, cur, fo), LN_EPS), W("ln2_w")),
                           W("ln2_b"));
            if (dbg) fprintf(stderr, "  g2 L%d: cur=[%lld,%lld] attn=[%lld,%lld] h=[%lld,%lld]\n",
                             li, cur->ne[0], cur->ne[1], attn->ne[0], attn->ne[1], h->ne[0], h->ne[1]);
        }
        g2_out = cur;
        ggml_set_output(g2_out);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] graph...\n");
        g2 = ggml_new_graph_custom(g2_ctx, 4096, false);
        ggml_build_forward_expand(g2, g2_out);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] galloc...\n");
        g2_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
        ggml_gallocr_alloc_graph(g2_galloc, g2);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g2] alloc ok\n");
        // mask 是 input 张量, galloc 内存不清零 —— 必须显式写 0 (全零 = 无 padding 的自注意)
        {
            const size_t nb = ggml_nbytes(msk);
            std::vector<char> z(nb, 0);
            ggml_backend_tensor_set(msk, z.data(), 0, nb);
        }
    }

};

gsv_hubert::gsv_hubert() : p(new impl) {}
gsv_hubert::~gsv_hubert() {
    impl & s = *p;
    if (s.g1_ctx) ggml_free(s.g1_ctx);
    if (s.g2_ctx) ggml_free(s.g2_ctx);
    if (s.g1_galloc) ggml_gallocr_free(s.g1_galloc);
    if (s.g2_galloc) ggml_gallocr_free(s.g2_galloc);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.wctx) ggml_free(s.wctx);
    if (s.gf) gguf_free(s.gf);
    delete p;
}

int gsv_hubert::hidden()   const { return p->D; }
int gsv_hubert::n_frames() const { return p->T; }

gsv_hubert * gsv_hubert::load(const std::string & gguf_path, const gsv_hubert_cfg & cfg) {
    gsv_hubert * m = new gsv_hubert();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;

    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] begin\n");
    ggml_backend_dev_t dev = nullptr;
    {
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] dev select, n_dev=%d\n", (int) ggml_backend_dev_count());
        const bool want_gpu = cfg.device == "gpu" || cfg.device == "vulkan" || cfg.device == "GPU" || cfg.device == "Vulkan";
        const int n_dev = (int) ggml_backend_dev_count();
        for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            if (want_gpu) { if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = d; }
            else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (!dev)
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        if (!dev) { fprintf(stderr, "[gsv_hubert] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_hubert] device: %s\n", ggml_backend_dev_name(dev));
    }
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] dev=%s\n", ggml_backend_dev_name(dev));
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_hubert] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.s_back = s.backend;

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] gguf open...\n");
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_hubert] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] alloc wbuf...\n");
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_hubert] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_hubert] reopen failed\n"); delete m; return nullptr; }
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
                fprintf(stderr, "[gsv_hubert] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] upload done, transpose conv w...\n");
    // ggml im2col 的 conv 权重要求 k 最内; GGUF (host 语义) 是 k 最外 —— 一次性转置回写
    {
        const int ICs[7] = {1, 512, 512, 512, 512, 512, 512};
        for (int i = 0; i < N_CONV; i++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "hubert.feat_conv.%d.w", i);
            ggml_tensor * t = ggml_get_tensor(s.wctx, nm);
            const int OC = (int) t->ne[0], IC = ICs[i], K = (int) t->ne[2];
            std::vector<float> src((size_t) OC * IC * K), dst((size_t) OC * IC * K);
            ggml_backend_tensor_get(t, src.data(), 0, src.size() * 4);
            // src 字节: [K][IC][OC] 行主 (k 最外) -> dst: [OC][IC][K] 行主 (k 最内)
            for (int k = 0; k < K; k++)
                for (int ic = 0; ic < IC; ic++)
                    for (int oc = 0; oc < OC; oc++)
                        dst[(size_t) oc * IC * K + ic * K + k] = src[(size_t) k * IC * OC + ic * OC + oc];
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size() * 4);
        }
        if (cfg.verbose) printf("[gsv_hubert] conv weights transposed to k-inner layout\n");
    }
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] build_g1...\n");
    s.build_g1();
    s.build_g2();
    if (cfg.verbose) printf("[gsv_hubert] loaded %s: D=%d layers=%d frames=%d\n",
                            gguf_path.c_str(), s.D, NL, s.T);
    return m;
}

// host: pos_conv (grouped conv k=128 same-pad + bias + gelu_erf) —— 沿用已验证的 host 实现
// (groups=16 的 grouped conv1d 不适配 ggml_conv_1d; 49 帧 ~230M MAC, OpenMP 下 ~10ms)
static void hubert_pos_conv(const std::vector<float> & x, int T,
        const float * w, const float * bias, std::vector<float> & y) {
    const int K = 128, G = 16, CG = D_HID / 16, C = D_HID;
    y.assign((size_t)T * C, 0.0f);
    #pragma omp parallel for schedule(static)
    for(int t = 0; t < T; t++){
        for(int g = 0; g < G; g++){
            for(int k = 0; k < K; k++){
                int ti = t + k - 64;
                if(ti < 0 || ti >= T) continue;
                for(int ic = 0; ic < CG; ic++){
                    float xv = x[(size_t)ti * C + g * CG + ic];
                    const float * wr = w + ((size_t)k * CG + ic) * C + g * CG;
                    float * yo = &y[(size_t)t * C + g * CG];
                    for(int oc = 0; oc < CG; oc++) yo[oc] += xv * wr[oc];
                }
            }
        }
    }
    for(int t = 0; t < T; t++)
        for(int c = 0; c < C; c++){
            const float v = y[(size_t)t * C + c] + bias[c];
            y[(size_t)t * C + c] = 0.5f * v * (1.0f + erff(v / sqrtf(2.0f)));
        }
}

bool gsv_hubert::encode(const float * audio_norm, int n_samples, std::vector<float> & out) {
    impl & s = *p;
    if (n_samples != TRAW) { fprintf(stderr, "[gsv_hubert] expect %d samples, got %d\n", TRAW, n_samples); return false; }

    // ---- 图 1: CNN 前端 + feat_proj -> feat [768, T] ----
    ggml_backend_tensor_set(s.g1_in, audio_norm, 0, (size_t) TRAW * 4);
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [run] g1 compute...\n");
    ggml_backend_graph_compute(s.backend, s.g1);
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [run] g1 done\n");
    std::vector<float> feat((size_t) T_FRAMES * D_HID);   // host [T, C] 布局 (pos_conv 用)
    {
        // g1_out 是 ggml [D, T] 列主: g_data[d + t*D] = F[d, t]
        // host feat [T, D] 行主: feat[t*D + d] = F[d, t] —— 两者字节序恒等, 直接拷贝
        ggml_backend_tensor_get(s.g1_out, feat.data(), 0, feat.size() * 4);
    }
    if (getenv("GSV_HUBERT_DEBUG")) {
        size_t bad = 0;
        for (float v : feat) if (!std::isfinite(v)) bad++;
        fprintf(stderr, "  after g1: bad=%zu / %zu\n", bad, feat.size());
    }

    // ---- host: pos_conv + 残差 + enc LN ----
    {
        // 权重可能在 GPU buffer 上 —— 统一 tensor_get 拷回 host 再算 (仅 3.4MB pos_conv.w)
        auto fetch = [&](const char * n, std::vector<float> & dst) {
            ggml_tensor * t = ggml_get_tensor(s.wctx, n);
            dst.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * 4);
        };
        std::vector<float> pwv, pbv, nwv, nbv;
        fetch("hubert.pos_conv.w", pwv);
        fetch("hubert.pos_conv.b", pbv);
        fetch("hubert.enc_norm_w", nwv);
        fetch("hubert.enc_norm_b", nbv);
        std::vector<float> pos;
        hubert_pos_conv(feat, T_FRAMES, pwv.data(), pbv.data(), pos);
        for (size_t i = 0; i < feat.size(); i++) feat[i] += pos[i];
        const float * wd = nwv.data();
        const float * bd = nbv.data();
        for (int t = 0; t < T_FRAMES; t++) {
            float * xr = &feat[(size_t) t * D_HID];
            double m = 0; for (int d = 0; d < D_HID; d++) m += xr[d]; m /= D_HID;
            double v = 0; for (int d = 0; d < D_HID; d++) { double e = xr[d] - m; v += e * e; }
            float inv = 1.0f / sqrtf((float)(v / D_HID) + LN_EPS);
            for (int d = 0; d < D_HID; d++) xr[d] = (xr[d] - (float) m) * inv * wd[d] + bd[d];
        }
    }
    // 调试: 直接用 golden enc_in 作为 g2 输入 (隔离 host 段)
    if (const char * gf2 = getenv("GSV_HUBERT_ENCIN")) {
        std::vector<float> ref((size_t) T_FRAMES * D_HID);
        FILE * f = fopen(gf2, "rb");
        if (!f || fread(ref.data(), 4, ref.size(), f) != ref.size()) { fprintf(stderr, "encin read fail\n"); return false; }
        fclose(f);
        feat.swap(ref);   // [T, D] host 布局
        fprintf(stderr, "  [g2-only] using external enc_in: %s\n", gf2);
    }

    // ---- 图 2: 12 层 transformer -> [768, T] ----
    {
        // g2_in 是 ggml [D, T] 列主; feat [T, D] 行主 —— 字节序恒等 (nn.Linear 语义:
        // mul_mat 的 src1 按行主 [T, in] 读取, 即文件/特征帧的自然顺序), 直接整块拷贝
        ggml_backend_tensor_set(s.g2_in, feat.data(), 0, feat.size() * 4);
        if (getenv("GSV_HUBERT_DEBUG")) {
            FILE * f = fopen("tests/golden/hubert.g2in.ggml.bin", "wb");
            if (f) { fwrite(feat.data(), 4, feat.size(), f); fclose(f); }
        }
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [run] g2 compute...\n");
        ggml_backend_graph_compute(s.backend, s.g2);
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [run] g2 done\n");
    }
    out.assign((size_t) D_HID * T_FRAMES, 0.0f);
    ggml_backend_tensor_get(s.g2_out, out.data(), 0, out.size() * 4);
    return true;
}
