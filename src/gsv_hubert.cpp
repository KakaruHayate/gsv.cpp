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
#include <chrono>
#include <immintrin.h>
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

    // pos_conv / enc LN 的权重由 host 使用 —— load 时拷回一份常驻, 避免每次 encode 传输 18.9MB
    std::vector<float> pos_w, pos_b, enc_nw, enc_nb;
    std::vector<float> pos_w_perm;   // [g][ic][k][ocg] —— k/ocg 连续, 供 host GEMM 顺序访问
    bool diet = true;                // g2 图瘦身: 融合 QKV + layernorm_affine + add_act
    ggml_context * dctx = nullptr;                                     // diet 融合张量专用 ctx
    ggml_backend_buffer_t dbuf = nullptr;
    std::vector<ggml_tensor *> t_ln1_wb, t_ln2_wb;                     // LN (w|b) 打包
    bool pos_gpu = false;             // GPU 后端: pos_conv 放进 g1 图 (省去 host 段)
    ggml_tensor * t_pos_w = nullptr;  // [K, IC_g, OC] k-inner (conv_1d 布局)
    ggml_tensor * t_enc_wb = nullptr; // enc LN 的 (w|b) 打包
    std::vector<float> pos_xt, pos_y;                                  // pos_conv 常驻工作区

    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) { fprintf(stderr, "[gsv_hubert] missing weight %s\n", n); abort(); }
        return t;
    }

    void build_g1() {
        const bool cpu_be = ggml_backend_get_device(s_back) &&
                            ggml_backend_dev_type(ggml_backend_get_device(s_back)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const bool im2col_f32 = getenv("GSV_HUBERT_IM2COL_F32") ? atoi(getenv("GSV_HUBERT_IM2COL_F32")) != 0 : cpu_be;
        const bool pos_gpu = this->pos_gpu;
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
            // audio patch 的 1D 专用 im2col (逐行 memcpy/零填充), CPU 上比通用 im2col 快得多;
            // Vulkan 侧与 IM2COL 共用同一批 pipeline。
            // F32 版 (im2col_f32) 可省掉 CPU 侧 F16->F32 的 wdata 转换 pass, 并让 CNN 全程 F32。
            ggml_tensor * y = nullptr;
            if (im2col_f32) {
                ggml_tensor * im = ggml_im2col_fast_1d(g1_ctx, w, x, CS[i], 0, 1, GGML_TYPE_F32, 1);
                ggml_tensor * mm = ggml_mul_mat(g1_ctx,
                        ggml_reshape_2d(g1_ctx, im, im->ne[0], im->ne[1] * im->ne[2]),
                        ggml_reshape_2d(g1_ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
                y = ggml_reshape_3d(g1_ctx, mm, im->ne[1], w->ne[2], im->ne[2]);   // [OL, C, 1]
            } else {
                y = ggml_conv_1d_fast_1d_im2col(g1_ctx, w, x, CS[i], 0, 1);        // [OL, C, 1]
            }
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
        ggml_tensor * feat = ggml_add(g1_ctx, ggml_mul_mat(g1_ctx, need("hubert.feat_proj.w"), x),
                                      need("hubert.feat_proj.b"));   // [768, 49] = [C, T]
        if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [g1] pos_gpu=%d\n", (int) pos_gpu);
        if (pos_gpu) {
            // pos_conv 进图: 16 个分组 conv_1d (k-inner 权重 slab 连续) + 转置/拼接组装 + LN
            // 输入视图: feat 的字节序 == [T, C] 行主 -> 以 ne=(T, C) 解读 (T 步长 4, C 步长 49*4)
            // feat 的 ne=(C,T); conv_1d 需要 b 为 ne=(W, IC, N) 且 W 步长为 1
            // -> cont(permute) 成 [T, C] 连续视图 (一次性 150KB 拷贝)
            ggml_tensor * ftc = ggml_cont(g1_ctx, ggml_permute(g1_ctx, feat, 1, 0, 2, 3));   // [T, C]
            ggml_tensor * pparts[16];
            for (int g = 0; g < 16; g++) {
                ggml_tensor * xg = ggml_view_3d(g1_ctx, ftc, T, 48, 1,
                                                (size_t) T * 4, (size_t) T * 48 * 4, (size_t) g * 48 * T * 4);
                ggml_tensor * wg = ggml_view_3d(g1_ctx, t_pos_w, 128, 48, 48,
                                                128 * 4, 128 * 48 * 4, (size_t) g * 48 * 128 * 48 * 4);
                ggml_tensor * cg = ggml_conv_1d(g1_ctx, wg, xg, 1, 64, 1);       // [T+1, 48, 1]
                cg = ggml_view_3d(g1_ctx, cg, T, 48, 1, cg->nb[1], cg->nb[2], 0); // 裁掉末帧
                cg = ggml_cont(g1_ctx, ggml_permute(g1_ctx, cg, 1, 0, 2, 3));      // [48, T, 1]
                pparts[g] = cg;
            }
            ggml_tensor * pos = pparts[0];
            for (int g = 1; g < 16; g++) pos = ggml_concat(g1_ctx, pos, pparts[g], 0);   // [768, T, 1]
            pos = ggml_reshape_2d(g1_ctx, pos, D_HID, T);
            pos = ggml_gelu_erf(g1_ctx, ggml_add(g1_ctx, pos, need("hubert.pos_conv.b")));
            g1_out = ggml_layernorm_affine(g1_ctx, feat, pos, nullptr, t_enc_wb, LN_EPS);
        } else {
            g1_out = feat;
        }
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
        const bool diet = this->diet;
        const bool pos_gpu = this->pos_gpu;
        const bool cpu_be = s_back && ggml_backend_dev_type(ggml_backend_get_device(s_back)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const bool im2col_f32 = getenv("GSV_HUBERT_IM2COL_F32") ? atoi(getenv("GSV_HUBERT_IM2COL_F32")) != 0 : cpu_be;
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
            ggml_tensor * attn = nullptr;
            if (diet) {
                // 瘦身版: QKV 融合为一次 matmul, 视图直送 FA (省 3 次 cont 拷贝);
                // bias 吸收进 layernorm_affine; ffn1 的 bias+gelu 用 add_act 合一
                ggml_tensor * q = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("q_w"), cur), W("q_b"));
                ggml_tensor * k = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("k_w"), cur), W("k_b"));
                ggml_tensor * v = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("v_w"), cur), W("v_b"));
                // 视图直送 FA (不 cont): permute 只是改步长, FA 两端都按传入步长寻址
                ggml_tensor * qh = ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, q, HD, NH, T, 1), 0, 2, 1, 3);
                ggml_tensor * kh = ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, k, HD, NH, T, 1), 0, 2, 1, 3);
                ggml_tensor * vh = ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, v, HD, NH, T, 1), 0, 2, 1, 3);
                attn = ggml_flash_attn_ext(g2_ctx, qh, kh, vh, msk, scale, 0.0f, 0.0f);
                attn = ggml_reshape_2d(g2_ctx, ggml_reshape_4d(g2_ctx, attn, D, T, 1, 1), D, T);
                ggml_tensor * ao = ggml_mul_mat(g2_ctx, W("out_w"), attn);
                cur = ggml_layernorm_affine(g2_ctx, cur, ao, W("out_b"), t_ln1_wb[li], LN_EPS);
                ggml_tensor * h = ggml_add_act(g2_ctx,
                                               ggml_mul_mat(g2_ctx, W("ffn1_w"), cur), W("ffn1_b"), GGML_ACT_GELU_ERF);
                ggml_tensor * fo = ggml_mul_mat(g2_ctx, W("ffn2_w"), h);
                cur = ggml_layernorm_affine(g2_ctx, cur, fo, W("ffn2_b"), t_ln2_wb[li], LN_EPS);
                if (dbg) fprintf(stderr, "  g2 L%d: cur=[%lld,%lld] attn=[%lld,%lld] h=[%lld,%lld] (diet)\n",
                                 li, cur->ne[0], cur->ne[1], attn->ne[0], attn->ne[1], h->ne[0], h->ne[1]);
                continue;
            }
            ggml_tensor * q = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("q_w"), cur), W("q_b"));
            ggml_tensor * k = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("k_w"), cur), W("k_b"));
            ggml_tensor * v = ggml_add(g2_ctx, ggml_mul_mat(g2_ctx, W("v_w"), cur), W("v_b"));
            ggml_tensor * qh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, q, HD, NH, T, 1), 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, k, HD, NH, T, 1), 0, 2, 1, 3));
            ggml_tensor * vh = ggml_cont(g2_ctx, ggml_permute(g2_ctx, ggml_reshape_4d(g2_ctx, v, HD, NH, T, 1), 0, 2, 1, 3));
            attn = ggml_flash_attn_ext(g2_ctx, qh, kh, vh, msk, scale, 0.0f, 0.0f);
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
        const bool is_gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU;
        s.pos_gpu = getenv("GSV_HUBERT_POS_GPU") ? atoi(getenv("GSV_HUBERT_POS_GPU")) != 0 : is_gpu;
        if (cfg.verbose) printf("[gsv_hubert] device: %s\n", ggml_backend_dev_name(dev));
    }
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] dev=%s\n", ggml_backend_dev_name(dev));
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_hubert] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.s_back = s.backend;
    s.diet = getenv("GSV_HUBERT_DIET") ? atoi(getenv("GSV_HUBERT_DIET")) != 0 : true;

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    if (getenv("GSV_HUBERT_DEBUG")) fprintf(stderr, "  [load] gguf open...\n");
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_hubert] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    // g2 瘦身用的 LN (w|b) 打包张量: 放独立 ctx (wctx 的内存池按 GGUF 张量数分配, 无余量)
    if (s.diet || s.pos_gpu) {
        ggml_init_params dip = { ggml_tensor_overhead() * (2 * NL + 16), NULL, true };
        s.dctx = ggml_init(dip);
        if (s.diet) {
            for (int li = 0; li < NL; li++) {
                s.t_ln1_wb.push_back(ggml_new_tensor_1d(s.dctx, GGML_TYPE_F32, 2 * D_HID));
                s.t_ln2_wb.push_back(ggml_new_tensor_1d(s.dctx, GGML_TYPE_F32, 2 * D_HID));
            }
        }
        if (s.pos_gpu) {
            s.t_pos_w = ggml_new_tensor_3d(s.dctx, GGML_TYPE_F32, 128, 48, D_HID);   // [K, IC_g, OC]
            s.t_enc_wb = ggml_new_tensor_1d(s.dctx, GGML_TYPE_F32, 2 * D_HID);
        }
        // (CPU 路径仍走 host pos_conv, 不需要以上两个张量)
        s.dbuf = ggml_backend_alloc_ctx_tensors(s.dctx, s.backend);
        if (!s.dbuf) { fprintf(stderr, "[gsv_hubert] diet buffer alloc failed\n"); delete m; return nullptr; }
    }
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
            if (s.pos_gpu && strcmp(tname, "hubert.pos_conv.w") == 0) continue;   // 图内用 t_pos_w
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
    // 填充 g2 瘦身的融合张量 (从已上传的 GGUF 张量取数拼接)
    if (s.diet) {
        auto getv = [&](const char * n, std::vector<float> & dst) {
            ggml_tensor * t = ggml_get_tensor(s.wctx, n);
            dst.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * 4);
        };
        char nm[96];
        for (int li = 0; li < NL; li++) {
            std::vector<float> l1w, l1b, l2w, l2b;
            snprintf(nm, sizeof(nm), "hubert.layer.%d.ln1_w", li); getv(nm, l1w);
            snprintf(nm, sizeof(nm), "hubert.layer.%d.ln1_b", li); getv(nm, l1b);
            snprintf(nm, sizeof(nm), "hubert.layer.%d.ln2_w", li); getv(nm, l2w);
            snprintf(nm, sizeof(nm), "hubert.layer.%d.ln2_b", li); getv(nm, l2b);
            std::vector<float> cat;
            cat.reserve(2 * D_HID);
            for (int k = 1; k <= 2; k++) {
                const std::vector<float> & w = (k == 1) ? l1w : l2w;
                const std::vector<float> & b = (k == 1) ? l1b : l2b;
                cat.clear();
                cat.insert(cat.end(), w.begin(), w.end());
                cat.insert(cat.end(), b.begin(), b.end());
                ggml_backend_tensor_set(k == 1 ? s.t_ln1_wb[li] : s.t_ln2_wb[li], cat.data(), 0, cat.size() * 4);
            }
        }
    }
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
    // pos_conv / enc LN 的权重由 host 使用 —— 拷回一份常驻 (避免每次 encode 传输 18.9MB)
    {
        auto fetch0 = [&](const char * n, std::vector<float> & dst) {
            ggml_tensor * t = ggml_get_tensor(s.wctx, n);
            dst.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * 4);
        };
        if (s.pos_gpu) {
            // GPU 路径跳过了 pos_conv.w 的上传 (省 18.9MB VRAM) -> 直接从 GGUF 文件读原始字节
            ggml_tensor * t = ggml_get_tensor(s.wctx, "hubert.pos_conv.w");
            s.pos_w.resize(ggml_nelements(t));
            FILE * fp = fopen(gguf_path.c_str(), "rb");
            bool ok = false;
            if (fp) {
                const int64_t ti = gguf_find_tensor(s.gf, "hubert.pos_conv.w");
                if (ti >= 0 &&
                    fseek(fp, (long)(gguf_get_data_offset(s.gf) + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) == 0 &&
                    fread(s.pos_w.data(), 4, s.pos_w.size(), fp) == s.pos_w.size()) ok = true;
                fclose(fp);
            }
            if (!ok) { fprintf(stderr, "[gsv_hubert] read pos_conv.w from file failed\n"); delete m; return nullptr; }
        } else {
            fetch0("hubert.pos_conv.w", s.pos_w);
        }
        // 重排为 [g][ocg][ic][k] (k 最内): 原布局 [k][ic][g*48+ocg] 下 w 访问跨 3KB 步长
        // (每 32B 用一条 line, 预取失效 —— 实测 10.5ms); 重排后每个输出通道的
        // (ic,k) 平面是一段连续 24KB, 内核可按顺序流广播读取, 每个 w 元素只读一遍
        {
            const int K = 128, CG = D_HID / 16, C = D_HID, G = 16;
            s.pos_w_perm.assign(s.pos_w.size(), 0.0f);
            for (int g = 0; g < G; g++)
                for (int ocg = 0; ocg < CG; ocg++)
                    for (int ic = 0; ic < CG; ic++)
                        for (int k = 0; k < K; k++)
                            s.pos_w_perm[((size_t)(g * CG + ocg) * CG + ic) * K + k] =
                                s.pos_w[((size_t)(k * CG + ic)) * C + g * CG + ocg];
        }
        fetch0("hubert.pos_conv.b", s.pos_b);
        fetch0("hubert.enc_norm_w",  s.enc_nw);
        fetch0("hubert.enc_norm_b",  s.enc_nb);
    }
    if (s.pos_gpu) {
        // t_pos_w: 直接吃 pos_w_perm ([oc][ic][k] 行主 == ne=(K,IC,OC) 列主)
        ggml_backend_tensor_set(s.t_pos_w, s.pos_w_perm.data(), 0, s.pos_w_perm.size() * 4);
        // 图内路径不再需要 host 侧两份 (pos_w / pos_w_perm): 释放 ~38MB
        {
            std::vector<float>().swap(s.pos_w);
            std::vector<float>().swap(s.pos_w_perm);
        }
        // enc LN 的 (w|b) 打包
        {
            std::vector<float> wb;
            wb.reserve(2 * D_HID);
            wb.insert(wb.end(), s.enc_nw.begin(), s.enc_nw.end());
            wb.insert(wb.end(), s.enc_nb.begin(), s.enc_nb.end());
            ggml_backend_tensor_set(s.t_enc_wb, wb.data(), 0, wb.size() * 4);
        }
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
        const float * w, const float * bias,
        std::vector<float> & xt, std::vector<float> & y) {
    const int K = 128, G = 16, CG = D_HID / 16, C = D_HID;
    y.resize((size_t)T * C);          // 内核完整覆写, 无需清零
    // 旧实现的瓶颈: 权重 18.9MB 被每帧重读一遍 (49 x 1.18MB/组 ≈ 923MB, 纯带宽受限 ~11ms)。
    // 新结构: 以 (组, 输出通道块 OB) 为任务; k/ic 外层 -> 权重块 (OB floats) 常驻 L1,
    // 输出帧全量累加在 acc[T][OB] (~T*OB*4B) 里, 内层跨 o 向量化。
    // 权重总读取降到一遍 (18.9MB), 累加顺序与旧实现一致 (k, ic 顺序不变) => 数值逐位相同。
    // xt 为通道主序 [C][Tpad]: 读 x 沿 t 连续 (帧主序时每 (k,ic) 要跳 49 条 cache line, 16x 放大)
    const int TPAD = T + K - 1 + 8;   // +8 供尾块 8 宽越界读 (读到零填充)
    if ((int) xt.size() != C * TPAD) {
        xt.assign((size_t) C * TPAD, 0.0f);   // 首次: 全零 (padding 区永远保持零)
    }
    (void) 0;
    const bool dbg_p = getenv("GSV_HUBERT_TIMING") != nullptr;
    const auto ppc = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double p0 = dbg_p ? ppc() : 0;
    // 中间区每帧覆写; 上下各 64/63 帧的补零区首建时为零, 之后不再触碰
#pragma omp parallel for schedule(static)
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            xt[(size_t) c * TPAD + 64 + t] = x[(size_t) t * C + c];
    if (dbg_p) fprintf(stderr, "      [pos] transpose %.2f ms\n", ppc() - p0);

    // 结构: 任务 = (组 g, 输出通道块 ob 8 个); o 外层, (ic,k) 内层;
    //   acc 为**全部帧**的 8 宽向量 (7 个 YMM), w 以广播方式从顺序流读取
    //   => 每个 w 元素只读一遍 (18.9MB), x 留在 L1/L2
    // 累加顺序 (ic, k 递增) 与初版 (k, ic 递增) 不同 —— 见下方 N 顺序说明
    const int NTB = (T + 7) / 8;
#pragma omp parallel for schedule(static) collapse(2)
    for (int g = 0; g < G; g++) {
        for (int ob = 0; ob < CG; ob += 8) {
            for (int o = 0; o < 8; o++) {
                const int c = g * CG + ob + o;
                const float * wp = w + (size_t)c * (CG * K);
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                __m256 a4 = a0, a5 = a0, a6 = a0;
                for (int ic = 0; ic < CG; ic++) {
                    const float * xp = &xt[(size_t)(g * CG + ic) * TPAD];
                    const float * wrow = wp + (size_t)ic * K;
                    for (int k = 0; k < K; k++) {
                        const __m256 wv = _mm256_broadcast_ss(wrow + k);
                        const float * xb = xp + k;
                        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb),      a0);
                        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 8),  a1);
                        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 16), a2);
                        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 24), a3);
                        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 32), a4);
                        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 40), a5);
                        a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xb + 48), a6);
                    }
                }
                float tmp[64];
                _mm256_storeu_ps(tmp,      a0); _mm256_storeu_ps(tmp + 8,  a1);
                _mm256_storeu_ps(tmp + 16, a2); _mm256_storeu_ps(tmp + 24, a3);
                _mm256_storeu_ps(tmp + 32, a4); _mm256_storeu_ps(tmp + 40, a5);
                _mm256_storeu_ps(tmp + 48, a6);
                float * yc = &y[(size_t)c];
                for (int t = 0; t < T; t++) {
                    const float v = tmp[t] + bias[c];
                    yc[(size_t)t * C] = 0.5f * v * (1.0f + erff(v * 0.70710678118f));
                }
            }
        }
    }
    if (dbg_p) fprintf(stderr, "      [pos] gemm %.2f ms\n", ppc() - p0);
}

bool gsv_hubert::encode(const float * audio_norm, int n_samples, std::vector<float> & out) {
    impl & s = *p;
    const bool dbg_t = getenv("GSV_HUBERT_TIMING") != nullptr;
    const auto tpc = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t_a = 0, t_b = 0, t_c = 0, t_d = 0;
    if (dbg_t) t_a = tpc();
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
    if (getenv("GSV_HUBERT_G1DUMP")) {
        FILE * df = fopen("tests/golden/hubert.g1out.dbg.bin", "wb");
        if (df) { fwrite(feat.data(), 4, feat.size(), df); fclose(df); }
    }
    if (getenv("GSV_HUBERT_DEBUG")) {
        size_t bad = 0;
        for (float v : feat) if (!std::isfinite(v)) bad++;
        fprintf(stderr, "  after g1: bad=%zu / %zu\n", bad, feat.size());
    }

    if (dbg_t) { t_b = tpc(); }
    // ---- host: pos_conv + 残差 + enc LN (仅 CPU 路径; GPU 路径已在图内完成) ----
    if (!s.pos_gpu) {
        std::vector<float> & pos = s.pos_y;
        const double t_p0 = dbg_t ? tpc() : 0;
        hubert_pos_conv(feat, T_FRAMES, s.pos_w_perm.data(), s.pos_b.data(), s.pos_xt, pos);
        if (dbg_t) fprintf(stderr, "    [pos_conv] %.2f ms\n", tpc() - t_p0);
        for (size_t i = 0; i < feat.size(); i++) feat[i] += pos[i];
        const float * wd = s.enc_nw.data();
        const float * bd = s.enc_nb.data();
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T_FRAMES; t++) {
            float * xr = &feat[(size_t) t * D_HID];
            double m = 0; for (int d = 0; d < D_HID; d++) m += xr[d]; m /= D_HID;
            double v = 0; for (int d = 0; d < D_HID; d++) { double e = xr[d] - m; v += e * e; }
            float inv = 1.0f / sqrtf((float)(v / D_HID) + LN_EPS);
            for (int d = 0; d < D_HID; d++) xr[d] = (xr[d] - (float) m) * inv * wd[d] + bd[d];
        }
    }
    if (dbg_t) { t_c = tpc(); }
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
    if (dbg_t) {
        t_d = tpc();
        fprintf(stderr, "  [time] g1 %.2f ms | host(pos_conv+LN) %.2f ms | g2 %.2f ms\n",
                t_b - t_a, t_c - t_b, t_d - t_c);
    }
    return true;
}
