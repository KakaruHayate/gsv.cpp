// DiT/CFM ggml 实现 — 见 gsv_dit.h 的语义对照说明.
// 图结构:
//   [cache 图]  prompt_x + mu -> text_embed(ConvNeXt x4) + concat + proj -> condition / negative_condition
//               (每次 prepare 算一次; 结果拷进持久张量 p_static/p_neg)
//   [step 图]   x_mel + t -> linear(proj[:100]) + static -> conv_pos(分组 conv, 2 层 + Mish)
//               -> 22 x DiTBlock(AdaLN + 16 头注意力 + RoPE(head0) + FFN) -> norm_out -> proj_out -> vel
#include "gsv_dit.h"
#include "gsv_sampler.h"      // gsv_rng

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int   DIM   = 1024;
static const int   DEPTH = 22;
static const int   HEADS = 16;
static const int   DH    = 64;
static const int   MEL   = 100;
static const int   TEXTD = 512;
static const int   FF    = 2048;
static const int   ADLN  = 6144;
static const int   K_CP  = 31;      // conv_pos 核长
static const int   G_CP  = 16;      // conv_pos 分组
static const int   ICG   = 64;      // conv_pos 组内输入通道
static const int   OCG   = 64;      // conv_pos 组内输出通道
static const float NORM_EPS = 1e-6f;
static const float ATTN_SCALE = 0.125f;   // 1/sqrt(64)

struct gsv_dit::impl {
    ggml_context * wctx = nullptr;
    gguf_context  * gf   = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr, s_back = nullptr;
    int n_threads = 0;
    bool verbose = false;
    std::vector<float> inv_freq;          // [32] (checkpoint fp16 存储值 upcast)

    ggml_context * kctx = nullptr;        // 常量 (ones/eps) 独立 ctx + buffer
    ggml_backend_buffer_t kbuf = nullptr;
    ggml_tensor * c_ones = nullptr;       // [1024,1] 常量 1 (AdaLN 的 1+scale)
    ggml_tensor * c_eps  = nullptr;       // [1] 1e-6 (GRN)

    ggml_context * pctx = nullptr;        // 持久张量 (T 相关, prepare 时重建)
    ggml_backend_buffer_t pbuf = nullptr;
    ggml_tensor * p_static = nullptr, * p_neg = nullptr, * p_maskrow = nullptr,
                * p_amask = nullptr, * p_cos = nullptr, * p_sin = nullptr;

    ggml_context * gctx = nullptr;        // cache + step 图共用 (张量元数据)
    // 三张图 (cache/pos/neg) 各自独立 allotter:
    //   共用一个时, 后一张图的 (auto-)reserve 会 free/重建整块 vbuffer 并复位 tallocr,
    //   先建图的张量指针全部悬空 (踩内存 / 调到 NULL iface). 代价是多两份激活显存, 可接受.
    ggml_gallocr_t galloc = nullptr;      // cache 图专用
    ggml_cgraph * cg = nullptr;
    ggml_tensor * c_in_prompt = nullptr, * c_in_mu = nullptr, * c_out = nullptr, * c_out_neg = nullptr;
    struct step_graph {
        ggml_cgraph * g = nullptr;
        ggml_gallocr_t galloc = nullptr;   // 每张图独立 allotter (切图重规划会改写共享张量地址)
        ggml_tensor * in_x = nullptr, * in_t = nullptr, * out = nullptr;
    };
    step_graph sp, sn;

    // cache-dit (DBCache): vipshop/cache-dit 的三段式 — front = 前 Fn 块 (每步必算, 出 Fn 残差判 fd);
    // middle = 其余块 (命中时用上次全算的 Mn 残差顶替); back 段本模型未启用 (Bn=0)。
    // 复用缓存 = Mn 残差 (x_mid - x_front), 判据 = Fn 残差 R 的相对 L1 diff。
    struct db_config {
        int   fn = 0;               // Fn_compute_blocks (0 = 关闭)
        float thr = 0.08f;          // residual_diff_threshold
        int   warmup = 8;           // max_warmup_steps
        int   warmup_interval = 1;  // warmup_interval
        int   max_cached = -1;      // max_cached_steps (per 分支)
        int   max_cont = -1;        // max_continuous_cached_steps (per 分支)
        float max_accum = 0.0f;     // max_accumulated_residual_diff_threshold (0 = 关)
        int   ds = 4;               // Fn 残差缓冲的时间下采样因子 (上游 downsample_factor; 1 = 关闭)
    } dbc;
    struct db_graph {
        ggml_cgraph * g = nullptr;
        ggml_gallocr_t galloc = nullptr;   // 每张图独立 allotter
        ggml_tensor * in_x = nullptr, * in_t = nullptr, * out = nullptr;
    };
    struct db_branch {
        db_graph front, mid, head, head_hit;   // head_hit 把命中步的 add 融进来 (省一次 submit + x_mid 往返)
        ggml_tensor * fd_t = nullptr;      // front 图的 fd 标量 (device, 读回 1 float)
        int   cached_steps = 0, cont = 0, hits = 0, misses = 0;
        bool  valid = false;               // 首次全算 (miss) 前不可命中
        float acc = 0.0f, last_fd = 0.0f;
    };
    db_branch dbb[2];                      // [0] = pos, [1] = neg (cfg 分支独立记账)
    ggml_context * dbctx = nullptr;
    ggml_backend_buffer_t dbbuf = nullptr;
    ggml_tensor * db_x_front[2] = {};      // [DIM,T] 当前 front 输出 (= 进 middle 的 hidden)
    ggml_tensor * db_x_mid[2] = {};        // [DIM,T] 进 head 的输入 (命中 = front + Mn 残差)
    ggml_tensor * db_r_prev[2] = {};       // [DIM,T] Fn 残差 (上次 miss 存档, 判据基准)
    ggml_tensor * db_r_cur[2] = {};        // [DIM,T] Fn 残差 (本次, miss 时拷入 prev)
    ggml_tensor * db_mr_prev[2] = {};      // [DIM,T] Mn 残差 (上次 miss 存档, 命中的复用增量)
    int db_T = 0;
    int db_ds = 1;                         // 本次构图的 Fn 残差下采样因子 (T 不整除时退回 1)
    int db_step = 0;                       // 当前 diffusion step (调用方经 db_set_step 设置)
    int db_total_steps = 0;                // (预留: 复用窗口; 0 = 不限制)
    bool db_prof = false;                  // GSV_DIT_DBCACHE_TRACE: 每步 HIT/MISS/fd 打印

    int T = 0, x_lens = 0, prompt_T = 0;
    bool debug = false;                      // GSV_DIT_DEBUG=1: 把中间张量标为 output 供读回
    bool mark_on = false;                    // 仅对 pos 图开启标记 (避免 neg 图重名)
    std::vector<float> cos_host, sin_host;   // [32*T]

    void mark(ggml_tensor * t, const char * n) {
        if (!debug || !mark_on) return;
        ggml_set_output(t);
        ggml_set_name(t, n);
    }

    // ---------------- 图内算子助手 (Domain B: ggml ne{C,T}) ----------------

    // mul_mat 诊断包装: can_mul_mat 失败时打印形状 (构建期断言难以定位)
    static ggml_tensor * mm(ggml_context * c, ggml_tensor * a, ggml_tensor * b, const char * tag) {
        if (!(a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0)) {
            printf("[dit mm FAIL] %s: a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n", tag,
                   (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                   (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3]);
            fflush(stdout);
        }
        return ggml_mul_mat(c, a, b);
    }

    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) { fprintf(stderr, "[gsv_dit] missing %s\n", n); abort(); }
        return t;
    }
    // 2D linear: y = W x + b; W: [in, out], x: [in, T], b: [out] (广播到 T 列)
    static ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
        ggml_tensor * y = mm(c, w, x, "L98");
        return b ? ggml_add(c, y, b) : y;
    }
    // Domain A <-> B: [T,C] <-> [C,T]
    static ggml_tensor * to_a(ggml_context * c, ggml_tensor * t) { return ggml_cont(c, ggml_permute(c, t, 1, 0, 2, 3)); }
    static ggml_tensor * to_b(ggml_context * c, ggml_tensor * t) { return ggml_cont(c, ggml_permute(c, t, 1, 0, 2, 3)); }
    static ggml_tensor * mish(ggml_context * c, ggml_tensor * x) {
        return ggml_mul(c, x, ggml_tanh(c, ggml_softplus(c, x)));
    }
    // x*(1+scale)+shift (scale/shift: [C,1])
    static ggml_tensor * modulate(ggml_context * c, ggml_tensor * x, ggml_tensor * scale,
                                  ggml_tensor * shift, ggml_tensor * ones) {
        return ggml_add(c, ggml_mul(c, x, ggml_add(c, scale, ones)), shift);
    }
    // RoPE (只作用于 head 0 = q/k 的前 64 维): q [64, T] 稠密, cos/sin [32, T] -> [64, T]
    // torch: t*cos + rotate_half(t)*sin (交错配对), 角度 = n * inv_freq[i] 且同对相邻维共享
    static ggml_tensor * rope64(ggml_context * c, ggml_tensor * q, ggml_tensor * cos_t,
                                ggml_tensor * sin_t, int Tt) {
        ggml_tensor * q3 = ggml_reshape_3d(c, q, 2, 32, Tt);
        ggml_tensor * xe = ggml_view_3d(c, q3, 1, 32, Tt, q3->nb[1], q3->nb[2], 0);
        ggml_tensor * xo = ggml_view_3d(c, q3, 1, 32, Tt, q3->nb[1], q3->nb[2], q3->nb[0]);
        ggml_tensor * c3 = ggml_reshape_3d(c, cos_t, 1, 32, Tt);
        ggml_tensor * s3 = ggml_reshape_3d(c, sin_t, 1, 32, Tt);
        ggml_tensor * e = ggml_sub(c, ggml_mul(c, xe, c3), ggml_mul(c, xo, s3));
        ggml_tensor * o = ggml_add(c, ggml_mul(c, xe, s3), ggml_mul(c, xo, c3));
        return ggml_reshape_2d(c, ggml_concat(c, e, o, 0), 64, Tt);
    }
    // 分组 conv1d (k=31, groups=16, pad=15): Domain B 输入/输出.
    //   每组一次 im2col (x 的通道切片在 Domain A 下是连续 [T, 64] view, 零拷贝;
    //   输出的行序 r = ic*K + kw = 权重 [ICG*K, OCG] 块的展平序, 与 wns1 的 [K,IC,OC] 一致).
    //   几何张量用权重的 view (ne0=K, ne1=ICG 满足 im2col 断言; 该 view 数据不会被读).
    ggml_tensor * conv_pos(ggml_context * c, ggml_tensor * x, const char * wname,
                           const char * bname, int Tt) {
        ggml_tensor * w = need(wname);
        ggml_tensor * xt = to_a(c, x);                                   // [T, 1024] 连续
        ggml_tensor * y = nullptr;
        for (int g = 0; g < G_CP; g++) {                       // 组块字节偏移 = g × nb[2] (F32/F16 通用)
            const size_t g_off = (size_t) g * w->nb[2];
            ggml_tensor * geom_g = ggml_view_3d(c, w, K_CP, ICG, 1, w->nb[1], w->nb[2], g_off);
            ggml_tensor * x_g = ggml_view_2d(c, xt, Tt, ICG, xt->nb[1], (size_t) g * ICG * xt->nb[1]);
            ggml_tensor * im = ggml_im2col_fast_1d(c, geom_g, x_g, 1, K_CP / 2, 1, GGML_TYPE_F32, 1);
            ggml_tensor * wg = ggml_reshape_2d(c,
                    ggml_view_3d(c, w, ICG * K_CP, OCG, 1, w->nb[1], w->nb[2], g_off),
                    ICG * K_CP, OCG);
            ggml_tensor * hg = mm(c, wg, im, "conv_pos_g");              // [64, T]
            y = y ? ggml_concat(c, y, hg, 0) : hg;
        }
        return ggml_add(c, y, need(bname));                              // [1024, T]
    }
    // GRN (ConvNeXtV2): 沿时间轴求 L2 范数, 通道均值归一, gamma*(x*Nx)+beta+x
    ggml_tensor * grn(ggml_context * c, ggml_tensor * x, int bi) {
        char nm[160];
        ggml_tensor * t_ = to_a(c, ggml_sqr(c, x));                       // [T, 1024]
        ggml_tensor * gx = ggml_sqrt(c, ggml_sum_rows(c, t_));            // [1, 1024]
        // torch: Gx.mean(dim=-1) 是全通道标量均值; ggml_mean 只归约 ne0 而 gx 的 ne0=1 → 会退化成恒等,
        // 必须用全元素 sum * 1/DIM (1/1024 为 2 的幂, 与 torch 除法逐位一致)
        ggml_tensor * mean = ggml_scale(c, ggml_sum(c, gx), 1.0f / (float) DIM);
        ggml_tensor * nx = ggml_div(c, gx, ggml_add(c, mean, c_eps));
        ggml_tensor * g = ggml_mul(c, x, ggml_reshape_2d(c, nx, DIM, 1));
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.grn.gamma", bi);
        ggml_tensor * gamma = need(nm);
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.grn.beta", bi);
        ggml_tensor * beta = need(nm);
        return ggml_add(c, ggml_add(c, ggml_mul(c, g, gamma), beta), x);
    }
    // ConvNeXtV2Block: dwconv(k=7, groups=512, pad=3) -> LN -> pwconv1 -> GELU(erf) -> GRN -> pwconv2 (+res)
    ggml_tensor * convnext_block(ggml_context * c, ggml_tensor * x, int bi, int Tt) {
        char nm[160];
        ggml_tensor * res = x;
        ggml_tensor * xt = to_a(c, x);                                    // [T, 512]
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.dwconv.weight", bi);
        ggml_tensor * w = need(nm);                                       // ne{512, 7}
        ggml_tensor * z = ggml_scale(c, ggml_cont(c, ggml_view_2d(c, xt, 3, TEXTD, xt->nb[1], 0)), 0.0f);
        ggml_tensor * xp = ggml_concat(c, ggml_concat(c, z, xt, 0), z, 0);       // [T+6, 512] (pad=3)
        ggml_tensor * y = nullptr;
        for (int kw = 0; kw < 7; kw++) {
            ggml_tensor * xw = ggml_view_2d(c, xp, Tt, TEXTD, xp->nb[1], (size_t) kw * 4);
            ggml_tensor * wc = ggml_reshape_2d(c, ggml_view_1d(c, w, TEXTD, (size_t) kw * TEXTD * 4), 1, TEXTD);
            ggml_tensor * term = ggml_mul(c, xw, wc);
            y = y ? ggml_add(c, y, term) : term;
        }
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.dwconv.bias", bi);
        y = ggml_add(c, y, ggml_reshape_2d(c, need(nm), 1, TEXTD));
        x = to_b(c, y);                                                   // [512, T]
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.norm.weight", bi);
        ggml_tensor * nw = need(nm);
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.norm.bias", bi);
        x = ggml_add(c, ggml_mul(c, ggml_norm(c, x, NORM_EPS), nw), need(nm));
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.pwconv1.weight", bi);
        ggml_tensor * pw1 = need(nm);
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.pwconv1.bias", bi);
        ggml_tensor * h = lin(c, pw1, need(nm), x);                        // [1024, T]
        h = ggml_gelu_erf(c, h);
        h = grn(c, h, bi);
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.pwconv2.weight", bi);
        ggml_tensor * pw2 = need(nm);
        snprintf(nm, sizeof(nm), "dit.text_embed.text_blocks.%d.pwconv2.bias", bi);
        h = lin(c, pw2, need(nm), h);                                      // [512, T]
        return ggml_add(c, res, h);
    }
    // time_embed MLP: 正弦 [256] -> [1024, 1]
    ggml_tensor * time_embed(ggml_context * c, ggml_tensor * t_sin) {
        ggml_tensor * h = lin(c, need("dit.time_embed.time_mlp.0.weight"),
                                 need("dit.time_embed.time_mlp.0.bias"), t_sin);
        h = ggml_silu(c, h);
        return lin(c, need("dit.time_embed.time_mlp.2.weight"),
                      need("dit.time_embed.time_mlp.2.bias"), h);
    }
    // 16 头注意力 (每头独立投影, 权重按输出行块切稠密 view) + head0 的 RoPE
    ggml_tensor * attention(ggml_context * c, ggml_tensor * x, int li,
                            ggml_tensor * amask, ggml_tensor * cos_t, ggml_tensor * sin_t, int Tt) {
        char nm[160];
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_q.weight", li);
        ggml_tensor * wq = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_q.bias", li);
        ggml_tensor * bq = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_k.weight", li);
        ggml_tensor * wk = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_k.bias", li);
        ggml_tensor * bk = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_v.weight", li);
        ggml_tensor * wv = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_v.bias", li);
        ggml_tensor * bv = need(nm);
        ggml_tensor * cat = nullptr;
        static const bool attn_legacy = getenv("GSV_DIT_ATTN_LEGACY") != nullptr;
        if (attn_legacy) {
            // [legacy] 每头独立投影 (权重按输出行块切稠密 view), 逐头 softmax
            // 输出行块 [h*64, h*64+64) 的字节偏移 = 行数 × 行步长 (nb[1] 对 F32/F16/量化块类型统一成立;
            // 量化张量的行是整块存储, 不能用 元素数×type_size)
            for (int h = 0; h < HEADS; h++) {
                const size_t wo = (size_t) h * DH * wq->nb[1];
                ggml_tensor * qh = lin(c, ggml_view_2d(c, wq, DIM, DH, wq->nb[1], wo),
                                          ggml_view_1d(c, bq, DH, (size_t) h * DH * 4), x);
                ggml_tensor * kh = lin(c, ggml_view_2d(c, wk, DIM, DH, wk->nb[1], wo),
                                          ggml_view_1d(c, bk, DH, (size_t) h * DH * 4), x);
                ggml_tensor * vh = lin(c, ggml_view_2d(c, wv, DIM, DH, wv->nb[1], wo),
                                          ggml_view_1d(c, bv, DH, (size_t) h * DH * 4), x);
                if (h == 0) {                                     // RoPE 只在第 0 号头
                    if (li == 0 && mark_on) { mark(qh, "dbg_q0_raw"); }
                    qh = rope64(c, qh, cos_t, sin_t, Tt);
                    kh = rope64(c, kh, cos_t, sin_t, Tt);
                    if (li == 0 && mark_on) { mark(qh, "dbg_q0_roped"); mark(kh, "dbg_k0_roped"); }
                }
                ggml_tensor * vT = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));   // [T_k, 64]
                ggml_tensor * sc = mm(c, kh, qh, "L234");                         // [T_k, T_q]
                ggml_tensor * pm = ggml_soft_max_ext(c, sc, amask, ATTN_SCALE, 0.0f);
                ggml_tensor * oh = mm(c, vT, pm, "L236");                         // [64, T_q]
                cat = cat ? ggml_concat(c, cat, oh, 0) : oh;
            }
        } else {
            // [fused] 整块 QKV (3 次大 matmul) + 批量 flash_attn_ext
            //  - head0 单独 RoPE: 前 64 行切片 cont 后 rope64, 再与其余 960 行 concat
            //    (rope64 内部的 reshape 要求连续, 直接切 strided view 会断言)
            //  - q/k/v 以 (hd, T, nh) 步长视图直送 FA (与 AR decode 同款: 两端都按视图步长寻址)
            //  - FA 输出 (hd, nh, T) 的元素序 = hd + 64*nh + 1024*t, 与 [DIM, T] 行主序一致
            ggml_tensor * q = lin(c, wq, bq, x);
            ggml_tensor * k = lin(c, wk, bk, x);
            ggml_tensor * v = lin(c, wv, bv, x);
            ggml_tensor * q0 = ggml_cont(c, ggml_view_2d(c, q, DH, Tt, q->nb[1], 0));
            ggml_tensor * k0 = ggml_cont(c, ggml_view_2d(c, k, DH, Tt, k->nb[1], 0));
            if (li == 0 && mark_on) { mark(q0, "dbg_q0_raw"); }
            ggml_tensor * q0r = rope64(c, q0, cos_t, sin_t, Tt);
            ggml_tensor * k0r = rope64(c, k0, cos_t, sin_t, Tt);
            if (li == 0 && mark_on) { mark(q0r, "dbg_q0_roped"); mark(k0r, "dbg_k0_roped"); }
            ggml_tensor * qf = ggml_concat(c, q0r,
                    ggml_cont(c, ggml_view_2d(c, q, DIM - DH, Tt, q->nb[1], (size_t) DH * 4)), 0);
            ggml_tensor * kf = ggml_concat(c, k0r,
                    ggml_cont(c, ggml_view_2d(c, k, DIM - DH, Tt, k->nb[1], (size_t) DH * 4)), 0);
            auto fa_view = [&](ggml_tensor * t) {
                return ggml_view_4d(c, t, DH, Tt, HEADS, 1, (size_t) DIM * 4, (size_t) DH * 4, 0, 0);
            };
            ggml_tensor * o = ggml_flash_attn_ext(c, fa_view(qf), fa_view(kf), fa_view(v),
                                                  ggml_reshape_4d(c, amask, Tt, Tt, 1, 1),
                                                  ATTN_SCALE, 0.0f, 0.0f);
            cat = ggml_reshape_2d(c, ggml_reshape_4d(c, o, DIM, Tt, 1, 1), DIM, Tt);
        }
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_out.0.weight", li);
        ggml_tensor * wo2 = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn.to_out.0.bias", li);
        // torch AttnProcessor: to_out 之后还有一次 x.masked_fill(~mask, 0) (pad 行清零)
        return ggml_mul(c, lin(c, wo2, need(nm), cat), p_maskrow);          // [1024, T]
    }
    // FFN: Linear(1024->2048) -> GELU(tanh 近似) -> Linear(2048->1024)
    ggml_tensor * ffn(ggml_context * c, ggml_tensor * x, int li) {
        char nm[160];
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.ff.ff.0.0.weight", li);
        ggml_tensor * w0 = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.ff.ff.0.0.bias", li);
        ggml_tensor * h = lin(c, w0, need(nm), x);                          // [2048, T]
        h = ggml_gelu(c, h);                                               // tanh 近似 (approximate="tanh")
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.ff.ff.2.weight", li);
        ggml_tensor * w2 = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.ff.ff.2.bias", li);
        return lin(c, w2, need(nm), h);                                     // [1024, T]
    }
    // DiTBlock: AdaLN(x, t) -> attn -> +gate_msa*attn -> AdaLN -> ffn -> +gate_mlp*ff
    ggml_tensor * dit_block(ggml_context * c, ggml_tensor * x, ggml_tensor * t_emb, int li,
                            ggml_tensor * amask, ggml_tensor * cos_t, ggml_tensor * sin_t, int Tt) {
        char nm[160];
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn_norm.linear.weight", li);
        ggml_tensor * wa = need(nm);
        snprintf(nm, sizeof(nm), "dit.transformer_blocks.%d.attn_norm.linear.bias", li);
        ggml_tensor * emb = lin(c, wa, need(nm), ggml_silu(c, t_emb));      // [6144, 1]
        // chunk 顺序: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        ggml_tensor * m[6];
        for (int i = 0; i < 6; i++)
            m[i] = ggml_view_2d(c, emb, DIM, 1, emb->nb[1], (size_t) i * DIM * 4);
        ggml_tensor * xn = modulate(c, ggml_norm(c, x, NORM_EPS), m[1], m[0], c_ones);
        if (li == 0 && mark_on) { mark(xn, "dbg_b0_norm"); mark(m[1], "dbg_b0_scale_msa"); mark(m[0], "dbg_b0_shift_msa");
                       mark(m[2], "dbg_b0_gate_msa"); mark(m[3], "dbg_b0_shift_mlp");
                       mark(m[4], "dbg_b0_scale_mlp"); mark(m[5], "dbg_b0_gate_mlp"); }
        ggml_tensor * a = attention(c, xn, li, amask, cos_t, sin_t, Tt);
        if (li == 0 && mark_on) { mark(a, "dbg_b0_attn"); }
        x = ggml_add(c, x, ggml_mul(c, a, m[2]));   // 广播要求大张量在前 (gate: [1024,1])
        ggml_tensor * xn2 = modulate(c, ggml_norm(c, x, NORM_EPS), m[4], m[3], c_ones);
        ggml_tensor * f = ffn(c, xn2, li);
        if (li == 0 && mark_on) { mark(f, "dbg_b0_ff"); }
        ggml_tensor * bo = ggml_add(c, x, ggml_mul(c, f, m[5]));
        if (li == 0)  { mark(bo, "dbg_b0_out"); }
        if (li == 21) { mark(bo, "dbg_b21_out"); }
        return bo;
    }

    // ---------------- 图构建/运行 ----------------
    bool make_persistents();
    void free_graphs();
    bool build_prepare();
    ggml_tensor * build_trunk(ggml_context * c, ggml_tensor * x_in, bool negative, int Tt);
    ggml_tensor * build_blocks(ggml_context * c, ggml_tensor * x, ggml_tensor * t_emb, int l0, int l1, int Tt);
    ggml_tensor * build_head(ggml_context * c, ggml_tensor * x, ggml_tensor * t_emb, bool negative);
    void build_step(step_graph & sg, bool negative);
    bool run_step(step_graph & sg, const float * x_mel, float t, float * vel);

    // ---------------- cache-dit (DBCache 三段式; vipshop/cache-dit 语义) ----------------
    bool db_ensure();                       // 按当前 T 建 device 张量 + 每分支 front/mid/head/head_hit 图
    void db_release();                      // 释放 db 图 + 张量 (+ 状态复位)
    void db_reset_state();                  // 只复位计数/门控状态 (每次采样开始时调用)
    bool run_step_cached(int bi, const float * x_mel, float t, float * vel);
    bool db_in_warmup() const;              // 当前 db_step 是否在 warmup 强制全算集内
};

// ============================ 图构建 ============================

void gsv_dit::impl::free_graphs() {
    db_release();
    if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
    for (step_graph * sg : { &sp, &sn }) {
        if (sg->galloc) { ggml_gallocr_free(sg->galloc); sg->galloc = nullptr; }
    }
    if (gctx)   { ggml_free(gctx); gctx = nullptr; }
    cg = nullptr; c_in_prompt = c_in_mu = c_out = c_out_neg = nullptr;
    sp = step_graph(); sn = step_graph();
}

// 持久张量: mask/rope 表 = host 填充; condition = cache 图算出后拷入
bool gsv_dit::impl::make_persistents() {
    impl * s = this;
    const int T = s->T, x_lens = s->x_lens;
    if (s->pctx) {
        ggml_backend_buffer_free(s->pbuf); s->pbuf = nullptr;
        ggml_free(s->pctx); s->pctx = nullptr;
    }
    ggml_init_params pip = { ggml_tensor_overhead() * 32, NULL, true };
    s->pctx = ggml_init(pip);
    ggml_context * p = s->pctx;
    s->p_static  = ggml_new_tensor_2d(p, GGML_TYPE_F32, DIM, T);
    s->p_neg     = ggml_new_tensor_2d(p, GGML_TYPE_F32, DIM, T);
    s->p_maskrow = ggml_new_tensor_2d(p, GGML_TYPE_F32, 1, T);
    s->p_amask   = ggml_new_tensor_2d(p, GGML_TYPE_F16, T, T);
    s->p_cos     = ggml_new_tensor_2d(p, GGML_TYPE_F32, 32, T);
    s->p_sin     = ggml_new_tensor_2d(p, GGML_TYPE_F32, 32, T);
    s->pbuf = ggml_backend_alloc_ctx_tensors(s->pctx, s->backend);
    if (!s->pbuf) { fprintf(stderr, "[gsv_dit] persistent buffer alloc failed\n"); return false; }

    std::vector<float> maskrow((size_t) T);
    for (int t = 0; t < T; t++) maskrow[t] = (t < x_lens) ? 1.0f : 0.0f;
    ggml_backend_tensor_set(s->p_maskrow, maskrow.data(), 0, maskrow.size() * 4);

    std::vector<ggml_fp16_t> am((size_t) T * T);
    const ggml_fp16_t f_keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f_mask = ggml_fp32_to_fp16(-1e30f);          // -> -inf
    for (int q = 0; q < T; q++)
        for (int k = 0; k < T; k++)
            am[(size_t) k + (size_t) T * q] = (k < x_lens) ? f_keep : f_mask;
    ggml_backend_tensor_set(s->p_amask, am.data(), 0, am.size() * 2);

    ggml_backend_tensor_set(s->p_cos, s->cos_host.data(), 0, s->cos_host.size() * 4);
    ggml_backend_tensor_set(s->p_sin, s->sin_host.data(), 0, s->sin_host.size() * 4);
    return true;
}

// cache 图: prompt_x + mu -> condition / negative_condition
bool gsv_dit::impl::build_prepare() {
    if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] build_prepare\n"); fflush(stdout); }
    free_graphs();
    // cache + pos/neg 两张 step 图共用一个 ctx, 张量数 ~9000 (每块 16 头 x 约 10 张 + rope/conv)
    ggml_init_params ip = { ggml_tensor_overhead() * 65536, NULL, true };
    gctx = ggml_init(ip);
    ggml_context * c = gctx;

    c_in_prompt = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, MEL);     // torch [1,100,T] 字节
    ggml_set_input(c_in_prompt); ggml_set_name(c_in_prompt, "c_prompt_x");
    c_in_mu = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, TEXTD);       // torch [1,512,T] 字节
    ggml_set_input(c_in_mu); ggml_set_name(c_in_mu, "c_mu");

    // text 路: mu.transpose + pos emb -> 4 x ConvNeXtV2
    ggml_tensor * tx = to_b(c, c_in_mu);                            // [512, T]
    ggml_tensor * cis = need("dit.text_embed.freqs_cis");           // [512, 4096]
    tx = ggml_add(c, tx, ggml_view_2d(c, cis, TEXTD, T, (size_t) TEXTD * 4, 0));
    if (debug) { ggml_set_output(tx); ggml_set_name(tx, "dbg_te_pos"); }   // 对拍 golden te_pos (mu + pos emb)
    for (int i = 0; i < 4; i++) {
        if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] convnext %d\n", i); fflush(stdout); }
        tx = convnext_block(c, tx, i, T);
        if (debug && i < 3) { char tn[32]; snprintf(tn, sizeof(tn), "dbg_te_cn%d", i); ggml_set_output(tx); ggml_set_name(tx, tn); }
    }
    if (debug) { ggml_set_output(tx); ggml_set_name(tx, "dbg_text_embed"); }   // cn3 输出 = text_embed

    // cond 路 + cat + proj(ctx 段)
    ggml_tensor * cond = to_b(c, c_in_prompt);                      // [100, T]
    ggml_tensor * zcat = ggml_concat(c, ggml_scale(c, cond, 0.0f), tx, 0);
    ggml_tensor * cat  = ggml_concat(c, cond, tx, 0);               // [612, T]
    ggml_tensor * wp = need("dit.input_embed.proj.weight_ctx");     // [612, 1024]
    ggml_tensor * pb = need("dit.input_embed.proj.bias");
    c_out     = ggml_add(c, mm(c, wp, cat, "cache_cond"),  pb);
    c_out_neg = ggml_add(c, mm(c, wp, zcat, "cache_neg"), pb);
    ggml_set_output(c_out);     ggml_set_name(c_out, "condition");
    ggml_set_output(c_out_neg); ggml_set_name(c_out_neg, "negative_condition");

    cg = ggml_new_graph_custom(c, 4096, false);
    ggml_build_forward_expand(cg, c_out);
    ggml_build_forward_expand(cg, c_out_neg);
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
    if (!ggml_gallocr_alloc_graph(galloc, cg)) {
        fprintf(stderr, "[gsv_dit] cache graph alloc failed (T=%d)\n", T);
        return false;
    }
    if (verbose) printf("[gsv_dit] cache graph built: T=%d x_lens=%d\n", T, x_lens);
    return true;
}

// ---- 图段构建 (full 图与 dbcache 的 front/mid/head 图共用同一套节点) ----

// x_lin = linear(x, proj[:, :100]) + static;  conv_pos_embed(x, mask) + x   -> [1024, T]
ggml_tensor * gsv_dit::impl::build_trunk(ggml_context * c, ggml_tensor * x_in, bool negative, int Tt) {
    ggml_tensor * xb = to_b(c, x_in);                                   // [100, T]
    ggml_tensor * x = ggml_add(c, mm(c, need("dit.input_embed.proj.weight_mel"), xb, "L392"),
                                  negative ? p_neg : p_static);         // [1024, T]
    mark(x, "dbg_x_lin");
    ggml_tensor * xm = ggml_mul(c, x, p_maskrow);
    ggml_tensor * cp = conv_pos(c, xm, "dit.input_embed.conv_pos_embed.conv1d.0.weight",
                                "dit.input_embed.conv_pos_embed.conv1d.0.bias", Tt);
    cp = mish(c, cp);
    cp = conv_pos(c, cp, "dit.input_embed.conv_pos_embed.conv1d.2.weight",
                  "dit.input_embed.conv_pos_embed.conv1d.2.bias", Tt);
    cp = mish(c, cp);                                                   // torch: Sequential 末尾还有一次 Mish, 后才 masked_fill
    cp = ggml_mul(c, cp, p_maskrow);
    mark(cp, "dbg_convpos");
    x = ggml_add(c, x, cp);
    mark(x, "dbg_trunk_in");
    return x;
}

ggml_tensor * gsv_dit::impl::build_blocks(ggml_context * c, ggml_tensor * x, ggml_tensor * t_emb,
                                          int l0, int l1, int Tt) {
    for (int li = l0; li < l1; li++) {
        if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] block %d\n", li); fflush(stdout); }
        x = dit_block(c, x, t_emb, li, p_amask, p_cos, p_sin, Tt);
    }
    return x;
}

// norm_out (AdaLayerNormZero_Final) + proj_out -> vel (torch [1,100,T] 字节序)
ggml_tensor * gsv_dit::impl::build_head(ggml_context * c, ggml_tensor * x, ggml_tensor * t_emb, bool negative) {
    ggml_tensor * emb2 = lin(c, need("dit.norm_out.linear.weight"), need("dit.norm_out.linear.bias"),
                                ggml_silu(c, t_emb));                   // [2048, 1]
    ggml_tensor * scale = ggml_view_2d(c, emb2, DIM, 1, emb2->nb[1], 0);
    ggml_tensor * shift = ggml_view_2d(c, emb2, DIM, 1, emb2->nb[1], (size_t) DIM * 4);
    x = modulate(c, ggml_norm(c, x, NORM_EPS), scale, shift, c_ones);
    mark(x, "dbg_norm_out");
    ggml_tensor * out = lin(c, need("dit.proj_out.weight"), need("dit.proj_out.bias"), x);   // [100, T]
    ggml_tensor * out_t = to_a(c, out);                                 // [T, 100] = torch [1,100,T] 字节
    ggml_set_output(out_t); ggml_set_name(out_t, negative ? "vel_neg" : "vel");
    return out_t;
}

void gsv_dit::impl::build_step(step_graph & sg, bool negative) {
    if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] build_step neg=%d\n", (int) negative); fflush(stdout); }
    ggml_context * c = gctx;
    ggml_tensor * x_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, MEL);
    ggml_set_input(x_in); ggml_set_name(x_in, negative ? "in_x_neg" : "in_x");
    ggml_tensor * t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
    ggml_set_input(t_in); ggml_set_name(t_in, "in_t");

    ggml_tensor * x = build_trunk(c, x_in, negative, T);
    ggml_tensor * t_emb = time_embed(c, t_in);                          // [1024, 1]
    x = build_blocks(c, x, t_emb, 0, DEPTH, T);
    ggml_tensor * out_t = build_head(c, x, t_emb, negative);

    sg.g = ggml_new_graph_custom(c, 65536, false);
    if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] graph created\n"); fflush(stdout); }
    ggml_build_forward_expand(sg.g, out_t);
    if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] expand ok, nodes=%d\n", (int) ggml_graph_n_nodes(sg.g)); fflush(stdout); }
    sg.in_x = x_in; sg.in_t = t_in; sg.out = out_t;
    if (!sg.galloc) sg.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
    if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] galloc begin\n"); fflush(stdout); }
    if (!ggml_gallocr_alloc_graph(sg.galloc, sg.g)) {
        fprintf(stderr, "[gsv_dit] step graph alloc failed (T=%d)\n", T);
        sg.g = nullptr;
        return;
    }
    if (verbose) printf("[gsv_dit] step graph built (%s): T=%d, nodes=%d\n",
                        negative ? "neg" : "pos", T, (int) ggml_graph_n_nodes(sg.g));
}

// ============================ 生命周期 ============================

gsv_dit::gsv_dit() : p(new impl) {}
gsv_dit::~gsv_dit() {
    impl & s = *p;
    s.free_graphs();
    if (s.pctx) { ggml_backend_buffer_free(s.pbuf); ggml_free(s.pctx); }
    if (s.kbuf) ggml_backend_buffer_free(s.kbuf);
    if (s.kctx) ggml_free(s.kctx);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.gf) gguf_free(s.gf);
    if (s.wctx) ggml_free(s.wctx);
    delete p;
}

int gsv_dit::mel_dim() const { return MEL; }
int gsv_dit::dim()     const { return DIM; }
int gsv_dit::depth()   const { return DEPTH; }
int gsv_dit::cur_T() const { return p->T; }
int gsv_dit::cur_prompt_T() const { return p->prompt_T; }

gsv_dit * gsv_dit::load(const std::string & gguf_path, const gsv_dit_cfg & cfg) {
    gsv_dit * m = new gsv_dit();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;
    s.debug = getenv("GSV_DIT_DEBUG") != nullptr;
    s.dbc.fn              = cfg.dbcache_fn;
    s.dbc.thr             = cfg.dbcache_threshold;
    s.dbc.warmup          = cfg.dbcache_warmup;
    s.dbc.warmup_interval = cfg.dbcache_warmup_interval;
    s.dbc.max_cached      = cfg.dbcache_max_cached;
    s.dbc.max_cont        = cfg.dbcache_max_cont;
    s.dbc.max_accum       = cfg.dbcache_max_accum;
    s.dbc.ds              = cfg.dbcache_downsample >= 1 ? cfg.dbcache_downsample : 1;

    ggml_backend_dev_t dev = nullptr;
    {
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
        if (!dev) { fprintf(stderr, "[gsv_dit] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_dit] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_dit] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.s_back = s.backend;

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_dit] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }

    // 常量张量: gguf 的 ctx 池按张量数精确分配(无余量), 必须另开一个 ctx
    {
        ggml_init_params kip = { ggml_tensor_overhead() * 8, NULL, true };
        s.kctx = ggml_init(kip);
        s.c_ones = ggml_new_tensor_2d(s.kctx, GGML_TYPE_F32, DIM, 1);
        s.c_eps  = ggml_new_tensor_1d(s.kctx, GGML_TYPE_F32, 1);
        s.kbuf = ggml_backend_alloc_ctx_tensors(s.kctx, s.backend);
        if (!s.kbuf) { fprintf(stderr, "[gsv_dit] const buffer alloc failed\n"); delete m; return nullptr; }
    }

    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_dit] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_dit] reopen failed\n"); delete m; return nullptr; }
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
                fprintf(stderr, "[gsv_dit] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    {
        std::vector<float> ones(DIM, 1.0f), eps{ 1e-6f };
        ggml_backend_tensor_set(s.c_ones, ones.data(), 0, DIM * 4);
        ggml_backend_tensor_set(s.c_eps, eps.data(), 0, 4);
        ggml_tensor * inv = ggml_get_tensor(s.wctx, "dit.rotary_embed.inv_freq");
        if (!inv) { fprintf(stderr, "[gsv_dit] missing dit.rotary_embed.inv_freq\n"); delete m; return nullptr; }
        s.inv_freq.resize(inv->ne[0]);
        ggml_backend_tensor_get(inv, s.inv_freq.data(), 0, s.inv_freq.size() * 4);
    }
    {   // 形状自检
        struct { const char * n; int64_t n0, n1; } chk[] = {
            { "dit.transformer_blocks.0.attn.to_q.weight", DIM, DIM },
            { "dit.transformer_blocks.0.ff.ff.0.0.weight", DIM, FF },
            { "dit.transformer_blocks.0.attn_norm.linear.weight", DIM, ADLN },
            { "dit.input_embed.proj.weight_mel", MEL, DIM },
            { "dit.input_embed.proj.weight_ctx", MEL + TEXTD, DIM },
            { "dit.input_embed.conv_pos_embed.conv1d.0.weight", (int64_t) ICG * K_CP, OCG },
            { "dit.text_embed.text_blocks.0.dwconv.weight", TEXTD, 7 },
            { "dit.text_embed.freqs_cis", TEXTD, 4096 },
            { "dit.proj_out.weight", DIM, MEL },
        };
        for (auto & e : chk) {
            ggml_tensor * t = ggml_get_tensor(s.wctx, e.n);
            if (!t || t->ne[0] != e.n0 || t->ne[1] != e.n1) {
                fprintf(stderr, "[gsv_dit] shape check failed: %s (ne=[%d,%d,%d,%d], want [%d,%d])\n", e.n,
                        t ? (int) t->ne[0] : -1, t ? (int) t->ne[1] : -1, t ? (int) t->ne[2] : -1,
                        t ? (int) t->ne[3] : -1, (int) e.n0, (int) e.n1);
                delete m; return nullptr;
            }
        }
    }
    if (cfg.verbose) printf("[gsv_dit] loaded %s (dim=%d depth=%d heads=%d, inv_freq %d)\n",
                            gguf_path.c_str(), DIM, DEPTH, HEADS, (int) s.inv_freq.size());
    return m;
}

// ============================ 前置 (static cache) ============================

void gsv_dit::time_sin(float t, float * out256) {
    const int F = 128;
    static float freq[F];
    static bool init = false;
    if (!init) {
        const float e = (float) (log(10000.0) / (double) (F - 1));
        for (int i = 0; i < F; i++) freq[i] = expf(-e * (float) i);
        init = true;
    }
    for (int i = 0; i < F; i++) {
        const float a = 1000.0f * t * freq[i];
        out256[i]     = sinf(a);
        out256[F + i] = cosf(a);
    }
}

bool gsv_dit::prepare(const float * prompt_mel, const float * mu, int T, int x_lens) {
    impl & s = *p;
    if (T <= 0) { fprintf(stderr, "[gsv_dit] bad T=%d\n", T); return false; }
    if (x_lens <= 0 || x_lens > T) x_lens = T;

    // rope 角度表: angle = n * inv_freq[i] (fp32, 与 torch einsum 同式)
    s.cos_host.resize((size_t) 32 * T);
    s.sin_host.resize((size_t) 32 * T);
    for (int n = 0; n < T; n++)
        for (int i = 0; i < 32; i++) {
            const float a = (float) n * s.inv_freq[i];
            s.cos_host[(size_t) i + 32 * (size_t) n] = cosf(a);
            s.sin_host[(size_t) i + 32 * (size_t) n] = sinf(a);
        }

    const bool first = (s.T != T || s.x_lens != x_lens);
    if (first) {
        s.T = T; s.x_lens = x_lens;
        if (!s.make_persistents()) return false;
        if (!s.build_prepare()) return false;
    }
    // 先跑 cache 图再构建 step 图 (顺序探针)
    {
        if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] cache compute (early)\n"); }
        ggml_backend_tensor_set(s.c_in_prompt, prompt_mel, 0, (size_t) MEL * T * 4);
        ggml_backend_tensor_set(s.c_in_mu, mu, 0, (size_t) TEXTD * T * 4);
        if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] cache compute (early) go\n"); }
        if (ggml_backend_graph_compute(s.s_back, s.cg) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gsv_dit] cache graph compute failed\n"); return false;
        }
        ggml_backend_tensor_copy(s.c_out, s.p_static);
        ggml_backend_tensor_copy(s.c_out_neg, s.p_neg);
        if (getenv("GSV_DIT_BUILD_TRACE")) { printf("[bt] cache compute (early) done\n"); }
    }
    if (first) {
        if (s.dbc.fn > 0) {
            // cache-dit: 不建 full step 图 (省两份激活), 建分段图 (front/mid/head ×2 分支)
            if (!s.db_ensure()) return false;
        } else {
            s.mark_on = true;
            s.build_step(s.sp, false);
            s.mark_on = false;
            if (!s.sp.g) return false;
            s.build_step(s.sn, true);
            if (!s.sn.g) return false;
        }
    }
    if (!first) {
        // 同 T: 图与持久张量复用, 只重算 condition
        ggml_backend_tensor_set(s.c_in_prompt, prompt_mel, 0, (size_t) MEL * T * 4);
        ggml_backend_tensor_set(s.c_in_mu, mu, 0, (size_t) TEXTD * T * 4);
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(s.s_back, s.cg) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gsv_dit] cache graph compute failed\n"); return false;
        }
        ggml_backend_tensor_copy(s.c_out, s.p_static);
        ggml_backend_tensor_copy(s.c_out_neg, s.p_neg);
    }
    if (s.verbose) {
        printf("[gsv_dit] prepare: T=%d x_lens=%d\n", T, x_lens);
    }
    return true;
}

bool gsv_dit::impl::run_step(step_graph & sg, const float * x_mel, float t, float * vel) {
    // 三张图各自独立 galloc, 已在 build_* 时规划一次; 运行期不再重规划 (spec: 见 gctx 注释)
    std::vector<float> ts(256);
    gsv_dit::time_sin(t, ts.data());
    ggml_backend_tensor_set(sg.in_x, x_mel, 0, (size_t) MEL * T * 4);
    ggml_backend_tensor_set(sg.in_t, ts.data(), 0, 256 * 4);
    if (ggml_backend_graph_compute(s_back, sg.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_dit] step graph compute failed\n");
        return false;
    }
    ggml_backend_tensor_get(sg.out, vel, 0, (size_t) MEL * T * 4);
    return true;
}

// ============================ cache-dit (DBCache) ============================
//
// 段结构 (vipshop/cache-dit 语义, 参考 CachedBlocks_Pattern_Base.forward):
//   front 图:  trunk(x_in) -> 前 Fn 块 -> x_F; 存档 x_F、Fn 残差 R = x_F - trunk;
//              fd = mean|R - R_prev| / mean|R_prev| (device 侧归约, 回读 1 float)
//   head_hit 图 (命中): x_mid = x_F + Mn残差_prev 后直接 norm_out/proj_out [add 融合, 省一次 submit]
//   mid 图 (未命中): x_F -> 第 Fn..DEPTH 块 -> x_M; 存档 Mn残差 = x_M - x_F、R -> R_prev
//   head 图:  x_mid -> norm_out -> proj_out -> vel (每步都算)
// 命中前提 (can_cache): 非 warmup、fd < threshold、max_cached/max_cont/max_accum 三道闸。
// cfg 分支独立记账 (enable_separate_cfg + cfg_diff_compute_separate), 与本引擎 pos/neg 两次调用对应。

void gsv_dit::impl::db_release() {
    for (int b = 0; b < 2; b++) {
        db_branch & d = dbb[b];
        if (d.front.galloc) { ggml_gallocr_free(d.front.galloc); d.front.galloc = nullptr; }
        if (d.mid.galloc)   { ggml_gallocr_free(d.mid.galloc);   d.mid.galloc   = nullptr; }
        if (d.head.galloc)  { ggml_gallocr_free(d.head.galloc);  d.head.galloc  = nullptr; }
        if (d.head_hit.galloc) { ggml_gallocr_free(d.head_hit.galloc); d.head_hit.galloc = nullptr; }
        d.front = db_graph(); d.mid = db_graph(); d.head = db_graph(); d.head_hit = db_graph();
        d.fd_t = nullptr;
    }
    if (dbbuf) { ggml_backend_buffer_free(dbbuf); dbbuf = nullptr; }
    if (dbctx) { ggml_free(dbctx); dbctx = nullptr; }
    for (int b = 0; b < 2; b++)
        db_x_front[b] = db_x_mid[b] = db_r_prev[b] = db_r_cur[b] = db_mr_prev[b] = nullptr;
    db_T = 0;
    db_reset_state();
}

void gsv_dit::impl::db_reset_state() {
    for (int b = 0; b < 2; b++) {
        db_branch & d = dbb[b];
        d.cached_steps = d.cont = d.hits = d.misses = 0;
        d.valid = false; d.acc = 0.0f; d.last_fd = 0.0f;
    }
    db_step = 0;
}

bool gsv_dit::impl::db_ensure() {
    if (dbc.fn <= 0 || dbc.fn >= DEPTH) return false;
    const int Tt = T;
    if (db_T == Tt && dbb[0].front.g) return true;
    db_release();

    db_ds = (dbc.ds >= 1 && (Tt % dbc.ds) == 0) ? dbc.ds : 1;
    const int Tds = Tt / db_ds;
    {   // device 缓存张量 (x_front/x_mid/Mn残差为 [DIM,T]; Fn 残差为 [DIM,T/ds])
        ggml_init_params ip = { ggml_tensor_overhead() * 24, NULL, true };
        dbctx = ggml_init(ip);
        for (int b = 0; b < 2; b++) {
            db_x_front[b] = ggml_new_tensor_2d(dbctx, GGML_TYPE_F32, DIM, Tt);
            db_x_mid[b]   = ggml_new_tensor_2d(dbctx, GGML_TYPE_F32, DIM, Tt);
            db_r_prev[b]  = ggml_new_tensor_2d(dbctx, GGML_TYPE_F32, DIM, Tds);
            db_r_cur[b]   = ggml_new_tensor_2d(dbctx, GGML_TYPE_F32, DIM, Tds);
            db_mr_prev[b] = ggml_new_tensor_2d(dbctx, GGML_TYPE_F32, DIM, Tt);
            ggml_set_name(db_x_front[b], b ? "db_x_front_neg" : "db_x_front");
            ggml_set_name(db_r_prev[b],  b ? "db_r_prev_neg"  : "db_r_prev");
            ggml_set_name(db_mr_prev[b], b ? "db_mr_prev_neg" : "db_mr_prev");
        }
        dbbuf = ggml_backend_alloc_ctx_tensors(dbctx, s_back);
        if (!dbbuf) { fprintf(stderr, "[gsv_dit] dbcache buffer alloc failed\n"); return false; }
        std::vector<float> zeros((size_t) DIM * Tds, 0.0f);     // R_prev 清零 (valid=false 前不参与判定)
        for (int b = 0; b < 2; b++)
            ggml_backend_tensor_set(db_r_prev[b], zeros.data(), 0, zeros.size() * 4);
    }

    db_prof = getenv("GSV_DIT_DBCACHE_TRACE") != nullptr;
    for (int b = 0; b < 2; b++) {
        db_branch & d = dbb[b];
        const bool negative = (b == 1);
        ggml_context * c = gctx;
        {   // front: trunk + 前 Fn 块 -> x_F;  cpy x_F / R;  fd 归约
            ggml_tensor * x_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, Tt, MEL);
            ggml_set_input(x_in); ggml_set_name(x_in, negative ? "db_in_x_neg" : "db_in_x");
            ggml_tensor * t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
            ggml_set_input(t_in); ggml_set_name(t_in, "db_in_t");
            ggml_tensor * trunk = build_trunk(c, x_in, negative, Tt);
            ggml_tensor * t_emb = time_embed(c, t_in);
            ggml_tensor * xf = build_blocks(c, trunk, t_emb, 0, dbc.fn, Tt);
            ggml_tensor * c_xf = ggml_cpy(c, xf, db_x_front[b]);
            ggml_set_output(c_xf);
            ggml_tensor * r_cur = ggml_sub(c, xf, trunk);            // Fn 残差 R (cache-dit 的判据量)
            if (db_ds > 1)                                           // 时间轴下采样 (downsample_factor)
                r_cur = ggml_cont(c, ggml_view_2d(c, r_cur, DIM, Tds, (size_t) db_ds * 4, 0));
            ggml_tensor * c_rc  = ggml_cpy(c, r_cur, db_r_cur[b]);
            ggml_set_output(c_rc);
            ggml_tensor * diff = ggml_abs(c, ggml_sub(c, r_cur, db_r_prev[b]));
            ggml_tensor * fd = ggml_div(c, ggml_sum(c, diff),
                                             ggml_add(c, ggml_sum(c, ggml_abs(c, db_r_prev[b])), c_eps));
            d.front.g = ggml_new_graph_custom(c, 65536, false);
            ggml_build_forward_expand(d.front.g, c_xf);
            ggml_build_forward_expand(d.front.g, c_rc);
            ggml_build_forward_expand(d.front.g, fd);
            d.front.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
            if (!ggml_gallocr_alloc_graph(d.front.galloc, d.front.g)) {
                fprintf(stderr, "[gsv_dit] db front graph alloc failed\n"); return false;
            }
            d.front.in_x = x_in; d.front.in_t = t_in;
            d.fd_t = fd;
        }
        {   // mid (未命中): x_F -> 第 Fn..DEPTH 块 -> x_M; 存档 Mn 残差与 R
            ggml_tensor * t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
            ggml_set_input(t_in); ggml_set_name(t_in, "db_in_t_mid");
            ggml_tensor * t_emb = time_embed(c, t_in);
            ggml_tensor * xm = build_blocks(c, db_x_front[b], t_emb, dbc.fn, DEPTH, Tt);
            ggml_tensor * c_xm = ggml_cpy(c, xm, db_x_mid[b]);
            ggml_set_output(c_xm);
            ggml_tensor * mr = ggml_sub(c, xm, db_x_front[b]);       // Mn 残差 (复用增量)
            ggml_tensor * c_mr = ggml_cpy(c, mr, db_mr_prev[b]);
            ggml_set_output(c_mr);
            ggml_tensor * c_rp = ggml_cpy(c, db_r_cur[b], db_r_prev[b]);   // R -> R_prev
            ggml_set_output(c_rp);
            d.mid.g = ggml_new_graph_custom(c, 65536, false);
            ggml_build_forward_expand(d.mid.g, c_xm);
            ggml_build_forward_expand(d.mid.g, c_mr);
            ggml_build_forward_expand(d.mid.g, c_rp);
            d.mid.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
            if (!ggml_gallocr_alloc_graph(d.mid.galloc, d.mid.g)) {
                fprintf(stderr, "[gsv_dit] db mid graph alloc failed\n"); return false;
            }
            d.mid.in_t = t_in;
        }
        {   // head: x_mid -> norm_out/proj_out -> vel (每步)
            ggml_tensor * t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
            ggml_set_input(t_in); ggml_set_name(t_in, "db_in_t_head");
            ggml_tensor * t_emb = time_embed(c, t_in);
            ggml_tensor * out_t = build_head(c, db_x_mid[b], t_emb, negative);
            d.head.g = ggml_new_graph_custom(c, 4096, false);
            ggml_build_forward_expand(d.head.g, out_t);
            d.head.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
            if (!ggml_gallocr_alloc_graph(d.head.galloc, d.head.g)) {
                fprintf(stderr, "[gsv_dit] db head graph alloc failed\n"); return false;
            }
            d.head.in_t = t_in;
            d.head.out = out_t;
        }
        {   // head_hit: x_mid = x_F + Mn残差_prev (add 融合) -> norm_out/proj_out -> vel
            ggml_tensor * t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
            ggml_set_input(t_in); ggml_set_name(t_in, "db_in_t_head_hit");
            ggml_tensor * t_emb = time_embed(c, t_in);
            ggml_tensor * xm = ggml_add(c, db_x_front[b], db_mr_prev[b]);
            ggml_tensor * out_t = build_head(c, xm, t_emb, negative);
            d.head_hit.g = ggml_new_graph_custom(c, 4096, false);
            ggml_build_forward_expand(d.head_hit.g, out_t);
            d.head_hit.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
            if (!ggml_gallocr_alloc_graph(d.head_hit.galloc, d.head_hit.g)) {
                fprintf(stderr, "[gsv_dit] db head_hit graph alloc failed\n"); return false;
            }
            d.head_hit.in_t = t_in;
            d.head_hit.out = out_t;
        }
    }
    db_T = Tt;
    if (verbose) printf("[gsv_dit] dbcache graphs built: T=%d Fn=%d thr=%.3f warmup=%d\n",
                        Tt, dbc.fn, dbc.thr, dbc.warmup);
    return true;
}

bool gsv_dit::impl::db_in_warmup() const {
    if (db_step >= dbc.warmup) return false;
    const int iv = dbc.warmup_interval < 1 ? 1 : dbc.warmup_interval;
    return (db_step % iv) == 0;               // warmup_steps = range(0, max_warmup, interval)
}

bool gsv_dit::impl::run_step_cached(int bi, const float * x_mel, float t, float * vel) {
    db_branch & d = dbb[bi];
    std::vector<float> ts(256);
    gsv_dit::time_sin(t, ts.data());
    auto now = []{ return std::chrono::steady_clock::now(); };
    const auto t_all0 = now();
    double ms_front = 0, ms_mid = 0, ms_add = 0, ms_head = 0, ms_fd = 0;

    // ---- front (每步必算) + fd ----
    ggml_backend_tensor_set(d.front.in_x, x_mel, 0, (size_t) MEL * T * 4);
    ggml_backend_tensor_set(d.front.in_t, ts.data(), 0, 256 * 4);
    auto t0 = now();
    if (ggml_backend_graph_compute(s_back, d.front.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_dit] db front compute failed\n"); return false;
    }
    ms_front = std::chrono::duration<double, std::milli>(now() - t0).count();
    float fd = 0.0f;
    t0 = now();
    ggml_backend_tensor_get(d.fd_t, &fd, 0, sizeof(float));
    ms_fd = std::chrono::duration<double, std::milli>(now() - t0).count();
    d.last_fd = fd;

    // ---- can_cache (warmup / 三道闸 / fd 门) ----
    bool hit = false;
    if (d.valid && dbc.thr > 0.0f && !db_in_warmup()) {
        bool ok = fd < dbc.thr;
        if (ok && dbc.max_cached >= 0 && d.cached_steps >= dbc.max_cached) ok = false;
        if (ok && dbc.max_cont >= 0 && d.cont >= dbc.max_cont) { ok = false; d.cont = 0; }
        if (ok && dbc.max_accum > 0.0f && d.acc >= dbc.max_accum) ok = false;
        hit = ok;
    }
    if (fd > 0.0f) d.acc += fd;              // 累计残差 diff (仅正值; 对应 add_residual_diff)

    if (hit) {
        // ---- 命中: x_mid = x_F + Mn残差_prev 已融进 head_hit 图 (省一次 submit 与 4MB cpy) ----
        ggml_backend_tensor_set(d.head_hit.in_t, ts.data(), 0, 256 * 4);
        t0 = now();
        if (ggml_backend_graph_compute(s_back, d.head_hit.g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gsv_dit] db head_hit compute failed\n"); return false;
        }
        ms_add = std::chrono::duration<double, std::milli>(now() - t0).count();
        t0 = now();
        ggml_backend_tensor_get(d.head_hit.out, vel, 0, (size_t) MEL * T * 4);
        d.hits++; d.cached_steps++; d.cont++;
    } else {
        // ---- mid (未命中): 全算 middle, 存档 Mn 残差 / R ----
        ggml_backend_tensor_set(d.mid.in_t, ts.data(), 0, 256 * 4);
        t0 = now();
        if (ggml_backend_graph_compute(s_back, d.mid.g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gsv_dit] db mid compute failed\n"); return false;
        }
        ms_mid = std::chrono::duration<double, std::milli>(now() - t0).count();
        d.misses++; d.valid = true; d.cont = 0;

        // ---- head (经 x_mid) ----
        ggml_backend_tensor_set(d.head.in_t, ts.data(), 0, 256 * 4);
        t0 = now();
        if (ggml_backend_graph_compute(s_back, d.head.g) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[gsv_dit] db head compute failed\n"); return false;
        }
        ms_head = std::chrono::duration<double, std::milli>(now() - t0).count();
        t0 = now();
        ggml_backend_tensor_get(d.head.out, vel, 0, (size_t) MEL * T * 4);
    }
    const double ms_vel = std::chrono::duration<double, std::milli>(now() - t0).count();
    const double ms_all = std::chrono::duration<double, std::milli>(now() - t_all0).count();

    if (db_prof) {
        fprintf(stderr, "[dbcache] step=%d %s fd=%.5f (hits=%d misses=%d) | front=%.1f fdread=%.1f %s=%.1f head=%.1f velread=%.1f total=%.1f ms\n",
                db_step, hit ? "HIT " : "MISS", fd, d.hits, d.misses, ms_front, ms_fd,
                hit ? "add" : "mid", hit ? ms_add : ms_mid, ms_head, ms_vel, ms_all);
    }
    return true;
}

bool gsv_dit::velocity(const float * x_mel, float t, bool negative, float * vel) {
    impl & s = *p;
    if (s.dbc.fn > 0) {                       // cache-dit 路径 (front -> mid|head_hit -> head 分段图)
        if (s.T <= 0 || (!s.db_T && !s.db_ensure())) {
            fprintf(stderr, "[gsv_dit] velocity() before prepare()\n"); return false;
        }
        if (s.db_T != s.T && !s.db_ensure()) return false;
        return s.run_step_cached(negative ? 1 : 0, x_mel, t, vel);
    }
    if (s.T <= 0 || !s.sp.g) { fprintf(stderr, "[gsv_dit] velocity() before prepare()\n"); return false; }
    impl::step_graph & sg = negative ? s.sn : s.sp;
    if (!sg.g) { fprintf(stderr, "[gsv_dit] negative step graph missing\n"); return false; }
    return s.run_step(sg, x_mel, t, vel);
}

void gsv_dit::db_set_step(int j) {
    p->db_step = j;
}

void gsv_dit::db_get_stats(db_stats & st) const {
    const impl & s = *p;
    st.steps = s.db_step;
    for (int b = 0; b < 2; b++) {
        st.hits[b]    = s.dbb[b].hits;
        st.misses[b]  = s.dbb[b].misses;
        st.last_fd[b] = s.dbb[b].last_fd;
        st.acc[b]     = s.dbb[b].acc;
    }
}

void gsv_dit::debug_tables(std::vector<float> & cos_tab, std::vector<float> & sin_tab) const {
    cos_tab = p->cos_host;
    sin_tab = p->sin_host;
}

// 读回被 mark() 标记的中间张量 (需在 velocity()/prepare() 之后; GSV_DIT_DEBUG=1 时可用)
bool gsv_dit::debug_read(const char * name, std::vector<float> & out) {
    impl & s = *p;
    if (!s.gctx) return false;
    ggml_tensor * t = ggml_get_tensor(s.gctx, name);
    if (!t) return false;
    out.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * 4);
    return true;
}

// ============================ CFM host 循环 ============================

static float randn01(gsv_rng & r) {
    float u1 = r.uniform01();
    const float u2 = r.uniform01();
    if (u1 < 1e-30f) u1 = 1e-30f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

// CFM.inference: mu [text_dim, T] / prompt [mel_dim, P] (torch 布局) -> mel [mel_dim, T-P]
bool gsv_dit::sample(const float * mu, const float * prompt, int P, int T, int steps, float cfg,
                     uint64_t seed, const float * x0, std::vector<float> & mel_out) {
    impl & s = *p;
    if (T <= 0 || steps <= 0) return false;
    if (P <= 0 || P >= T) { fprintf(stderr, "[gsv_dit] sample(): bad prompt_len=%d (T=%d)\n", P, T); return false; }

    // prompt_x [mel_dim, T]: 前 P 帧 = prompt, 其余 0
    std::vector<float> prompt_x((size_t) MEL * T, 0.0f);
    for (int c = 0; c < MEL; c++)
        memcpy(prompt_x.data() + (size_t) c * T, prompt + (size_t) c * P, (size_t) P * 4);
    if (!prepare(prompt_x.data(), mu, T, T)) return false;
    s.prompt_T = P;
    s.db_reset_state();                       // 每次采样 = 一次新的推理 (对应 mark_step_begin 的复位)
    s.db_total_steps = steps;

    // x = randn * 0.875 (或注入的 x0); x[:, :P] = 0
    std::vector<float> x((size_t) MEL * T);
    if (x0) {
        memcpy(x.data(), x0, x.size() * 4);
    } else {
        gsv_rng rng(seed ? seed : 1);
        for (size_t i = 0; i < x.size(); i++) x[i] = randn01(rng) * 0.875f;
    }
    for (int c = 0; c < MEL; c++)
        memset(x.data() + (size_t) c * T, 0, (size_t) P * 4);

    const float step = 1.0f / (float) steps;
    std::vector<float> vel((size_t) MEL * T), neg((size_t) MEL * T);
    for (int j = 0; j < steps; j++) {
        const float t = (float) j * step;
        s.db_step = j;                        // dbcache 的 current_step (warmup/gates 用)
        if (!velocity(x.data(), t, false, vel.data())) return false;
        if (cfg > 1e-5f) {
            if (!velocity(x.data(), t, true, neg.data())) return false;
            for (size_t i = 0; i < vel.size(); i++) vel[i] = vel[i] + (vel[i] - neg[i]) * cfg;
        }
        for (size_t i = 0; i < vel.size(); i++) x[i] += step * vel[i];
        for (int c = 0; c < MEL; c++)
            memset(x.data() + (size_t) c * T, 0, (size_t) P * 4);
    }
    mel_out.resize((size_t) MEL * (T - P));
    for (int c = 0; c < MEL; c++)
        memcpy(mel_out.data() + (size_t) c * (T - P), x.data() + (size_t) c * T + P, (size_t) (T - P) * 4);
    return true;
}

// synthesize_v5_mel 等价: rolling prompt 分块
bool gsv_dit::synthesize(const float * fea_ref, int R, const float * fea_todo, int N,
                         const float * mel_ref, int steps, float cfg, uint64_t seed,
                         int block_override, const std::vector<const float *> * x0_chunks,
                         std::vector<float> & mel_out) {
    impl & s = *p;
    static const int REF = 500, TAIL = 32, TOTAL = 1000, MAXCHUNK = 640;
    if (R <= 0 || N <= 0) { fprintf(stderr, "[gsv_dit] synthesize(): bad R=%d N=%d\n", R, N); return false; }
    const int ref_off = (R > REF) ? (R - REF) : 0;            // [..., -REF:]
    const int ref_frames = R - ref_off;
    const int chunk_frames = std::min(TOTAL - ref_frames, block_override > 0 ? block_override : MAXCHUNK);
    if (chunk_frames <= 0) { fprintf(stderr, "[gsv_dit] synthesize(): ref too long\n"); return false; }

    // 从 [C, R] 的列偏移 ref_off 处取 ref_frames 帧 -> [C, ref_frames]
    auto slice_cols = [&](const float * src, int C, std::vector<float> & dst) {
        dst.resize((size_t) C * ref_frames);
        for (int c = 0; c < C; c++)
            memcpy(dst.data() + (size_t) c * ref_frames, src + (size_t) c * R + ref_off, (size_t) ref_frames * 4);
    };
    std::vector<float> mel_ref_s, fea_ref_s;
    slice_cols(mel_ref, MEL, mel_ref_s);
    slice_cols(fea_ref, TEXTD, fea_ref_s);

    std::vector<float> rolling_mel = mel_ref_s;      // [MEL, ref_frames]
    std::vector<float> rolling_feat = fea_ref_s;     // [TEXTD, ref_frames]
    std::vector<float> prompt((size_t) MEL * ref_frames), feat((size_t) TEXTD * ref_frames);
    std::vector<float> out;
    uint64_t chunk_seed = seed;
    int chunk_idx = 0;
    for (int start = 0; start < N; start += chunk_frames) {
        const int tlen = std::min(chunk_frames, N - start);
        if (start == 0) {
            prompt = mel_ref_s;
            feat = fea_ref_s;
        } else {
            const int rm = (int) (rolling_mel.size() / MEL);
            const int rf = (int) (rolling_feat.size() / TEXTD);
            const int tail = std::min({ TAIL, ref_frames, rm, rf });
            const int prefix = ref_frames - tail;
            for (int c = 0; c < MEL; c++) {
                memcpy(prompt.data() + (size_t) c * ref_frames, mel_ref_s.data() + (size_t) c * ref_frames,
                       (size_t) prefix * 4);
                memcpy(prompt.data() + (size_t) c * ref_frames + prefix,
                       rolling_mel.data() + (size_t) c * rm + (rm - tail), (size_t) tail * 4);
            }
            for (int c = 0; c < TEXTD; c++) {
                memcpy(feat.data() + (size_t) c * ref_frames, fea_ref_s.data() + (size_t) c * ref_frames,
                       (size_t) prefix * 4);
                memcpy(feat.data() + (size_t) c * ref_frames + prefix,
                       rolling_feat.data() + (size_t) c * rf + (rf - tail), (size_t) tail * 4);
            }
        }
        // mu = cat(features, target) [TEXTD, ref+tlen]
        const int T = ref_frames + tlen;
        std::vector<float> mu((size_t) TEXTD * T);
        for (int c = 0; c < TEXTD; c++) {
            memcpy(mu.data() + (size_t) c * T, feat.data() + (size_t) c * ref_frames, (size_t) ref_frames * 4);
            memcpy(mu.data() + (size_t) c * T + ref_frames, fea_todo + (size_t) c * N + start, (size_t) tlen * 4);
        }
        s.prompt_T = ref_frames;
        std::vector<float> gen;      // [MEL, tlen]
        const float * x0 = (x0_chunks && chunk_idx < (int) x0_chunks->size()) ? (*x0_chunks)[chunk_idx] : nullptr;
        if (!sample(mu.data(), prompt.data(), ref_frames, T, steps, cfg, chunk_seed, x0, gen)) return false;
        chunk_idx++;
        chunk_seed = chunk_seed * 6364136223846793005ull + 1442695040888963407ull;
        if (out.empty()) out.assign((size_t) MEL * N, 0.0f);
        for (int c = 0; c < MEL; c++)
            memcpy(out.data() + (size_t) c * N + start, gen.data() + (size_t) c * tlen, (size_t) tlen * 4);
        // rolling: 本次生成的末 ref 帧 (不足则全量) / 对应 target 特征
        const int keep = std::min(ref_frames, tlen);
        rolling_mel.resize((size_t) MEL * keep);
        for (int c = 0; c < MEL; c++)
            memcpy(rolling_mel.data() + (size_t) c * keep,
                   gen.data() + (size_t) c * tlen + (tlen - keep), (size_t) keep * 4);
        rolling_feat.resize((size_t) TEXTD * keep);
        for (int c = 0; c < TEXTD; c++)
            memcpy(rolling_feat.data() + (size_t) c * keep,
                   fea_todo + (size_t) c * N + start + (tlen - keep), (size_t) keep * 4);
    }
    mel_out = std::move(out);
    return true;
}
