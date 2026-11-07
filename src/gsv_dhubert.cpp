#include "gsv_dhubert.h"
#include "gsv_device.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// 与 tools/export_gguf/convert_distilled_hubert.py 对应的默认值 (实际以 GGUF KV 为准, 缺 KV 时用这些兜底)
static const int   DH_DFLT_K[7]   = {10, 3, 3, 3, 3, 2, 2};
static const int   DH_DFLT_S[7]   = {5, 2, 2, 2, 2, 2, 2};
static const int   DH_DFLT_DIL[6] = {1, 2, 4, 8, 16, 32};
static const float LN_EPS         = 1e-5f;   // nn.LayerNorm 默认 eps (GroupNorm 的 eps 来自 KV)

namespace {

// 1D 卷积 (F32 im2col, 省掉 ggml_conv_1d 默认的 F16 中间缓冲):
//   w: [K, IC, OC] 列主 (k 最内), x: [T, IC, B] -> [OL, OC, B]
// OL = (T + 2p - d*(K-1) - 1)/s + 1
ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                     ggml_tensor * x, int s, int p, int d) {
    ggml_tensor * im = ggml_im2col_fast_1d(ctx, w, x, s, p, d, GGML_TYPE_F32, 1);
    ggml_tensor * mm = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]),
            ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));
    mm = ggml_reshape_3d(ctx, mm, im->ne[1], w->ne[2], im->ne[2]);   // [OL, OC, B]
    // 偏置按输出通道广播 ([1, OC, 1]): 输入是 [OL, OC, B], 通道在 ne[1] (不是 ne[0])
    return ggml_add(ctx, mm, ggml_reshape_3d(ctx, b, 1, w->ne[2], 1));
}

// GroupNorm(ng) + 逐通道 affine。ggml 的 group_norm 以 ne[2] 为通道轴, 组内连续切分
// (== torch GroupNorm 的分组语义), 所以把 [OL, C, 1] 摊成 [OL, 1, C, 1] 再调用。
ggml_tensor * gn_affine(ggml_context * ctx, ggml_tensor * y, int OL, int C, int ng, float eps,
                        ggml_tensor * gw, ggml_tensor * gb) {
    y = ggml_reshape_4d(ctx, y, OL, 1, C, 1);
    y = ggml_group_norm(ctx, y, ng, eps);
    y = ggml_add(ctx, ggml_mul(ctx, y, ggml_reshape_4d(ctx, gw, 1, 1, C, 1)),
                        ggml_reshape_4d(ctx, gb, 1, 1, C, 1));
    return ggml_reshape_3d(ctx, y, OL, C, 1);
}

// LayerNorm (over ne0 = 特征维) + affine; a: [C, T]
ggml_tensor * ln_affine(ggml_context * ctx, ggml_tensor * a, ggml_tensor * gw, ggml_tensor * gb) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, a, LN_EPS), gw), gb);
}

// 建图期 mul_mat 的形状自检: ggml 内部只有一句 GGML_ASSERT(ggml_can_mul_mat), 对不上时
// 只会 abort 而没有任何线索 —— 这里先给出 a/b 的 ne, 方便定位是哪一处 attention/FFN。
ggml_tensor * mul_mat_checked(ggml_context * ctx, const char * what, ggml_tensor * a, ggml_tensor * b) {
    const bool ok = a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0;
    if (!ok) {
        fprintf(stderr, "[gsv_dhubert] %s: mul_mat shape mismatch\n"
                        "    a=(%lld,%lld,%lld,%lld) nb=(%zu,%zu,%zu)\n"
                        "    b=(%lld,%lld,%lld,%lld) nb=(%zu,%zu,%zu)\n", what,
                (long long) a->ne[0], (long long) a->ne[1], (long long) a->ne[2], (long long) a->ne[3],
                a->nb[0], a->nb[1], a->nb[2],
                (long long) b->ne[0], (long long) b->ne[1], (long long) b->ne[2], (long long) b->ne[3],
                b->nb[0], b->nb[1], b->nb[2]);
        abort();
    }
    return ggml_mul_mat(ctx, a, b);
}

bool ends_with(const char * s, const char * suf) {
    const size_t n = strlen(s), m = strlen(suf);
    return n >= m && strcmp(s + n - m, suf) == 0;
}

} // namespace

struct gsv_dhubert::impl {
    ggml_backend_t backend = nullptr;        // 计算 + 权重所在后端
    ggml_context * wctx = nullptr;           // GGUF 权重张量
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;

    // 图 (帧数变化时重建); galloc 与图 ctx 一一对应
    ggml_context * gctx = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor  * g_in = nullptr;           // [Traw, 1, 1] F32
    ggml_tensor  * g_out = nullptr;          // [out_dim, T] F32

    int Traw = 0;                            // 当前图对应的原始采样数
    int T = 0;                               // 当前图的帧数 (50 Hz)
    int n_threads = 0;
    bool verbose = false;
    bool ffn_gelu = false;
    std::vector<std::pair<std::string, ggml_tensor *>> dumps;   // 逐段对拍用的中间张量

    // ---- 结构参数 (GGUF KV) ----
    int n_conv = 7, n_tcn = 6, n_attn = 2;
    int D = 192, NH = 4, FFN = 768, NG = 8, OD = 768;
    float gn_eps = 1e-5f;
    std::vector<int> CK, CS, DIL;

    // ---------------- GGUF 读取辅助 ----------------
    int kv_u32(const char * k, int def) const {
        const int64_t id = gguf_find_key(gf, k);
        if (id < 0) return def;
        if (gguf_get_kv_type(gf, id) == GGUF_TYPE_UINT32) return (int) gguf_get_val_u32(gf, id);
        if (gguf_get_kv_type(gf, id) == GGUF_TYPE_INT32)  return (int) gguf_get_val_i32(gf, id);
        return def;
    }
    float kv_f32(const char * k, float def) const {
        const int64_t id = gguf_find_key(gf, k);
        if (id < 0) return def;
        if (gguf_get_kv_type(gf, id) == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gf, id);
        return def;
    }
    // int 数组 KV (导出侧写的是 python int list -> GGUF ARRAY)
    std::vector<int> kv_ints(const char * k, const int * dflt, int n) const {
        std::vector<int> v;
        const int64_t id = gguf_find_key(gf, k);
        if (id < 0) { v.assign(dflt, dflt + n); return v; }
        const int64_t cnt = gguf_get_arr_n(gf, id);
        const enum gguf_type et = gguf_get_arr_type(gf, id);
        const void * p = gguf_get_arr_data(gf, id);
        v.resize((size_t) cnt);
        for (int64_t i = 0; i < cnt; i++) {
            switch (et) {
                case GGUF_TYPE_INT32:   v[(size_t) i] = ((const int32_t  *) p)[i]; break;
                case GGUF_TYPE_UINT32:  v[(size_t) i] = (int) ((const uint32_t *) p)[i]; break;
                case GGUF_TYPE_INT64:   v[(size_t) i] = (int) ((const int64_t  *) p)[i]; break;
                case GGUF_TYPE_UINT64:  v[(size_t) i] = (int) ((const uint64_t *) p)[i]; break;
                default:
                    fprintf(stderr, "[gsv_dhubert] KV %s: unsupported array type %d\n", k, (int) et);
                    v.assign(dflt, dflt + n);
                    return v;
            }
        }
        return v;
    }

    ggml_tensor * W(const std::string & n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n.c_str());
        if (!t) { fprintf(stderr, "[gsv_dhubert] missing weight %s\n", n.c_str()); abort(); }
        return t;
    }

    // ---------------- 建图 ----------------
    void build(int n_samples) {
        if (gctx) {
            if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
            ggml_free(gctx);
            gctx = nullptr; graph = nullptr; g_in = g_out = nullptr;
            dumps.clear();
        }
        const bool dbg = getenv("GSV_DHUBERT_DEBUG") != nullptr;
        Traw = n_samples;

        // 逐层卷积输出长度 (与 torch 的 shape 推导完全一致, 卷积无 padding)
        std::vector<int> olen((size_t) n_conv + 1);
        olen[0] = n_samples;
        for (int i = 0; i < n_conv; i++) olen[(size_t) i + 1] = (olen[(size_t) i] - CK[(size_t) i]) / CS[(size_t) i] + 1;
        T = olen[(size_t) n_conv];
        if (T < 1) {
            fprintf(stderr, "[gsv_dhubert] audio too short: %d samples -> %d frames\n", n_samples, T);
            return;
        }

        ggml_init_params ip = { ggml_tensor_overhead() * 16384, NULL, true };
        gctx = ggml_init(ip);

        // ---- front: 7 x [conv1d(k,s,无 padding) -> GroupNorm(8) -> SiLU] ----
        ggml_tensor * x = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_samples, 1, 1);
        ggml_set_input(x);
        g_in = x;
        const char * dump_env = getenv("GSV_DHUBERT_DUMP");
        for (int i = 0; i < n_conv; i++) {
            const std::string base = "front." + std::to_string(i) + ".";
            const int IC = i == 0 ? 1 : D, OL = olen[(size_t) i + 1];
            ggml_tensor * wc = ggml_reshape_3d(gctx, W(base + "c_w"), CK[(size_t) i], IC, D);
            ggml_tensor * y = conv1d(gctx, wc, W(base + "c_b"), x, CS[(size_t) i], 0, 1);      // [OL, D, 1]
            y = gn_affine(gctx, y, OL, D, NG, gn_eps, W(base + "g_w"), W(base + "g_b"));
            x = ggml_silu(gctx, y);
            if (dbg) fprintf(stderr, "  front%d: [%d, %d]\n", i, OL, D);
            if (dump_env) { ggml_set_output(x); dumps.push_back({ "front" + std::to_string(i), x }); }
        }
        if (dump_env) { ggml_set_output(x); dumps.push_back({ "front", x }); }

        // ---- tcn: 6 x 残差块 (k=3, dilation d, padding d) ----
        for (int i = 0; i < n_tcn; i++) {
            const std::string base = "tcn." + std::to_string(i) + ".";
            const int d = DIL[(size_t) i];
            ggml_tensor * c1 = conv1d(gctx, ggml_reshape_3d(gctx, W(base + "c1_w"), 3, D, D),
                                      W(base + "c1_b"), x, 1, d, d);
            ggml_tensor * h = ggml_silu(gctx, gn_affine(gctx, c1, T, D, NG, gn_eps,
                                                        W(base + "n1_w"), W(base + "n1_b")));
            ggml_tensor * c2 = conv1d(gctx, ggml_reshape_3d(gctx, W(base + "c2_w"), 3, D, D),
                                      W(base + "c2_b"), h, 1, d, d);
            ggml_tensor * n2 = gn_affine(gctx, c2, T, D, NG, gn_eps,
                                         W(base + "n2_w"), W(base + "n2_b"));
            x = ggml_add(gctx, x, n2);
            if (dbg) fprintf(stderr, "  tcn%d: dilation %d -> [%d, %d]\n", i, d, T, D);
        }
        if (dump_env) { ggml_set_output(x); dumps.push_back({ "tcn", x }); }

        // ---- attn: [D, T] 布局 (每帧一行 -> 每列一帧, 与 nn.Linear 的 src1 语义一致) ----
        ggml_tensor * cur = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));   // [D, T]
        const int HD = D / NH;
        const float scale = 1.0f / sqrtf((float) HD);
        for (int li = 0; li < n_attn; li++) {
            const std::string base = "attn." + std::to_string(li) + ".";
            // pre-norm: x = x + SA(LN1(x)); x = x + FFN(LN2(x))
            ggml_tensor * h = ln_affine(gctx, cur, W(base + "ln1_w"), W(base + "ln1_b"));
            // 融合 QKV: [3D, T] -> 视图切成 3 个 [HD, NH, T], 再各自 permute 成 [HD, T, NH]
            ggml_tensor * qkv = ggml_add(gctx, mul_mat_checked(gctx, "qkv", W(base + "qkv_w"), h),
                                         W(base + "qkv_b"));
            auto part = [&](int p) -> ggml_tensor * {
                // qkv nb = (4, (3D)*4); 第 p 段起始 = p*D 个 float; 段内 (d + head*HD + t*3D)
                ggml_tensor * v = ggml_view_3d(gctx, qkv, HD, NH, T, (size_t) HD * 4,
                                               (size_t) 3 * D * 4, (size_t) p * D * 4);
                return ggml_permute(gctx, v, 0, 2, 1, 3);      // [HD, T, NH]
            };
            ggml_tensor * kq = mul_mat_checked(gctx, "scores", part(1), part(0));          // [T, T, NH]
            kq = ggml_soft_max_ext(gctx, kq, nullptr, scale, 0.0f);           // softmax(scale * K^T Q)
            // V 转置成 [T, HD, NH] (cont 一次, 让收缩维在 ne0 且连续)
            // ggml_permute(a, axis0, axis1, ...) 里 axis_i = 源轴 i 落到目标的位置 (ne[axis_i] = a->ne[i]),
            // 与 numpy.transpose 的实参顺序正好相反。源 (HD, NH, T) -> (T, HD, NH) 就是交换前两轴。
            ggml_tensor * vt = ggml_cont(gctx, ggml_permute(gctx, part(2), 1, 0, 2, 3));   // [T, HD, NH]
            ggml_tensor * av = mul_mat_checked(gctx, "attn_values", vt, kq);                    // [HD, T, NH]
            ggml_tensor * attn = ggml_reshape_2d(gctx,
                    ggml_cont(gctx, ggml_permute(gctx, av, 0, 2, 1, 3)), D, T);   // [D, T]
            ggml_tensor * ao = ggml_add(gctx, mul_mat_checked(gctx, "out_proj", W(base + "o_w"), attn),
                                        W(base + "o_b"));
            cur = ggml_add(gctx, cur, ao);
            // FFN (torch 默认 activation=relu; student_hubert.py 未传 activation)
            ggml_tensor * h2 = ln_affine(gctx, cur, W(base + "ln2_w"), W(base + "ln2_b"));
            ggml_tensor * f1 = ggml_add(gctx, mul_mat_checked(gctx, "ff1", W(base + "ff1_w"), h2),
                                        W(base + "ff1_b"));
            f1 = ffn_gelu ? ggml_gelu_erf(gctx, f1) : ggml_relu(gctx, f1);
            ggml_tensor * f2 = ggml_add(gctx, mul_mat_checked(gctx, "ff2", W(base + "ff2_w"), f1),
                                        W(base + "ff2_b"));
            cur = ggml_add(gctx, cur, f2);
            if (dbg) fprintf(stderr, "  attn%d: [%d, %d] (hd=%d scale=%.6f)\n", li, D, T, HD, (double) scale);
            if (dump_env) {
                ggml_set_output(cur);
                dumps.push_back({ "attn" + std::to_string(li), cur });
            }
        }

        // ---- head: 1x1 conv 192 -> 768 ----
        ggml_tensor * hw = ggml_reshape_2d(gctx, W("head_w"), D, OD);   // [D, OD]
        g_out = ggml_add(gctx, mul_mat_checked(gctx, "head", hw, cur), W("head_b"));   // [OD, T]
        ggml_set_output(g_out);

        graph = ggml_new_graph_custom(gctx, 8192, false);
        ggml_build_forward_expand(graph, g_out);
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            fprintf(stderr, "[gsv_dhubert] graph alloc failed (T=%d)\n", T);
            ggml_gallocr_free(galloc); galloc = nullptr; graph = nullptr;
            return;
        }
        if (dbg) fprintf(stderr, "  graph: %d nodes, T=%d, Traw=%d\n", (int) ggml_graph_n_nodes(graph), T, Traw);
    }

    void dump_tensors() const {
        for (const auto & d : dumps) {
            const char * pre = getenv("GSV_DHUBERT_DUMP");
            if (!pre || !d.second) continue;
            const std::string p = std::string(pre) + "." + d.first + ".f32";
            FILE * f = fopen(p.c_str(), "wb");
            if (!f) { fprintf(stderr, "[gsv_dhubert] dump open failed %s\n", p.c_str()); continue; }
            std::vector<float> buf((size_t) ggml_nelements(d.second));
            ggml_backend_tensor_get(d.second, buf.data(), 0, buf.size() * 4);
            fwrite(buf.data(), 4, buf.size(), f);
            fclose(f);
            fprintf(stderr, "[gsv_dhubert] dump %s: ne=(%lld,%lld,%lld,%lld)\n", p.c_str(),
                    (long long) d.second->ne[0], (long long) d.second->ne[1],
                    (long long) d.second->ne[2], (long long) d.second->ne[3]);
        }
    }
};

gsv_dhubert::gsv_dhubert() : p(new impl) {}

gsv_dhubert::~gsv_dhubert() {
    impl & s = *p;
    if (s.galloc) ggml_gallocr_free(s.galloc);
    if (s.gctx) ggml_free(s.gctx);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.wctx) ggml_free(s.wctx);
    if (s.gf) gguf_free(s.gf);
    delete p;
}

int gsv_dhubert::hidden()   const { return p->OD; }
int gsv_dhubert::n_frames() const { return p->T; }

gsv_dhubert * gsv_dhubert::load(const std::string & gguf_path, const gsv_dhubert_cfg & cfg) {
    gsv_dhubert * m = new gsv_dhubert();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;
    {
        std::string act = cfg.ffn_act;
        if (const char * e = getenv("GSV_DHUBERT_ACT")) act = e;
        s.ffn_gelu = act == "gelu" || act == "gelu_erf";
    }

    // ---- 设备选择 (与 gsv_hubert 同款: 默认 CPU, "vulkan"/"gpu"/"1" 走 GPU) ----
    ggml_backend_dev_t dev = nullptr;
    {
        int gpu_idx = 0;
        bool gpu_explicit = false;
        const bool want_gpu = gsv_device_parse(cfg.device, &gpu_idx, &gpu_explicit);
        const int n_dev = (int) ggml_backend_dev_count();
        int gpu_seen = 0;
        for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            if (want_gpu) {
                if ((ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
                    gpu_seen++ == gpu_idx) dev = d;
            } else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (want_gpu && gpu_explicit &&
            (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fprintf(stderr, "[gsv_dhubert] GPU %d not found (run --list-devices)\n", gpu_idx);
            delete m; return nullptr;
        }
        if (!dev) {
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        }
        if (!dev) { fprintf(stderr, "[gsv_dhubert] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_dhubert] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_dhubert] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);

    // ---- GGUF: 只建张量元数据 (no_alloc), 再按需上传 ----
    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_dhubert] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    {
        const int64_t aid = gguf_find_key(s.gf, "general.architecture");
        if (aid >= 0) {
            const char * arch = gguf_get_val_str(s.gf, aid);
            if (strcmp(arch, "gsv-dhubert") != 0)
                fprintf(stderr, "[gsv_dhubert] warning: arch=%s (expected gsv-dhubert)\n", arch);
        }
        s.n_conv = s.kv_u32("dhubert.n_conv", (int) (sizeof(DH_DFLT_K) / sizeof(DH_DFLT_K[0])));
        s.n_tcn  = s.kv_u32("dhubert.n_tcn",  (int) (sizeof(DH_DFLT_DIL) / sizeof(DH_DFLT_DIL[0])));
        s.n_attn = s.kv_u32("dhubert.n_attn", 2);
        s.D      = s.kv_u32("dhubert.hidden", 192);
        s.NH     = s.kv_u32("dhubert.n_head", 4);
        s.FFN    = s.kv_u32("dhubert.ffn", 768);
        s.NG     = s.kv_u32("dhubert.gn_groups", 8);
        s.OD     = s.kv_u32("dhubert.out_dim", 768);
        s.gn_eps = s.kv_f32("dhubert.gn_eps", 1e-5f);
        s.CK  = s.kv_ints("dhubert.conv_kernel", DH_DFLT_K,   s.n_conv);
        s.CS  = s.kv_ints("dhubert.conv_stride", DH_DFLT_S,   s.n_conv);
        s.DIL = s.kv_ints("dhubert.dilations",   DH_DFLT_DIL, s.n_tcn);
        if ((int) s.CK.size() != s.n_conv || (int) s.CS.size() != s.n_conv || (int) s.DIL.size() != s.n_tcn) {
            fprintf(stderr, "[gsv_dhubert] KV length mismatch (conv_kernel=%zu conv_stride=%zu dilations=%zu)\n",
                    s.CK.size(), s.CS.size(), s.DIL.size());
            delete m; return nullptr;
        }
        if (s.D % s.NH != 0 || s.D % s.NG != 0) {
            fprintf(stderr, "[gsv_dhubert] bad arch: hidden=%d n_head=%d gn_groups=%d\n", s.D, s.NH, s.NG);
            delete m; return nullptr;
        }
    }

    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_dhubert] weight buffer alloc failed\n"); delete m; return nullptr; }

    // ---- 上传权重: 卷积权重一次性转置成 ggml im2col 需要的 k 最内布局 ----
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_dhubert] reopen failed\n"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        int n_conv_w = 0;
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            ggml_tensor * t = ggml_get_tensor(s.wctx, tname);
            if (!t) continue;
            const size_t nbytes = ggml_nbytes(t);
            std::vector<char> raw(nbytes);
            if (fseek(fp, (long) (data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(raw.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_dhubert] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            const bool is_conv_w = ends_with(tname, ".c_w") || ends_with(tname, ".c1_w") ||
                                  ends_with(tname, ".c2_w") || strcmp(tname, "head_w") == 0;
            if (is_conv_w) {
                if (t->type != GGML_TYPE_F32) {
                    fprintf(stderr, "[gsv_dhubert] conv weight %s is not F32\n", tname);
                    fclose(fp); delete m; return nullptr;
                }
                // GGUF 字节 = [K][IC][OC] 行主 (k 最外, 与 torch [OC,IC,K] 的 permute 结果一致);
                // ggml im2col 要 ne=(K, IC, OC) 即 [K] 连续 -> 转成 [OC][IC][K] 行主再上传。
                const int OC = (int) t->ne[0], IC = (int) t->ne[1], K = (int) t->ne[2];
                const size_t nel = (size_t) OC * IC * K;
                const float * src = (const float *) raw.data();
                std::vector<float> dst(nel);
                for (int k = 0; k < K; k++)
                    for (int ic = 0; ic < IC; ic++)
                        for (int oc = 0; oc < OC; oc++)
                            dst[(size_t) oc * IC * K + ic * K + k] = src[(size_t) k * IC * OC + ic * OC + oc];
                ggml_backend_tensor_set(t, dst.data(), 0, nel * 4);
                n_conv_w++;
            } else {
                ggml_backend_tensor_set(t, raw.data(), 0, nbytes);
            }
        }
        fclose(fp);
        if (cfg.verbose)
            printf("[gsv_dhubert] loaded %s: conv1d weights transposed to k-inner layout (%d), tensors=%lld\n",
                   gguf_path.c_str(), n_conv_w, (long long) n_tensors);
    }

    if (cfg.verbose)
        printf("[gsv_dhubert] arch: conv=%d tcn=%d attn=%d D=%d NH=%d FFN=%d GN=%d out=%d ffn_act=%s\n",
               s.n_conv, s.n_tcn, s.n_attn, s.D, s.NH, s.FFN, s.NG, s.OD, s.ffn_gelu ? "gelu" : "relu");
    return m;
}

bool gsv_dhubert::encode(const float * audio, int n_samples, std::vector<float> & out) {
    impl & s = *p;
    if (!audio || n_samples <= 0) return false;
    // 最短可用长度: 逐层至少要能出 1 帧 (n = (L-1)*S + K 反向推)
    int min_len = 1;
    for (int i = s.n_conv - 1; i >= 0; i--) min_len = (min_len - 1) * s.CS[(size_t) i] + s.CK[(size_t) i];
    if (n_samples < min_len) {
        fprintf(stderr, "[gsv_dhubert] audio too short: %d samples (need >= %d)\n", n_samples, min_len);
        return false;
    }

    const bool timing = getenv("GSV_DHUBERT_TIMING") != nullptr;
    const auto now = [] {
        return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    const double t0 = timing ? now() : 0.0;
    if (!s.graph || n_samples != s.Traw) {
        s.build(n_samples);
        if (!s.graph) { fprintf(stderr, "[gsv_dhubert] build failed\n"); return false; }
    }
    ggml_backend_tensor_set(s.g_in, audio, 0, (size_t) n_samples * 4);
    ggml_backend_graph_compute(s.backend, s.graph);
    out.assign((size_t) s.OD * (size_t) s.T, 0.0f);
    ggml_backend_tensor_get(s.g_out, out.data(), 0, out.size() * 4);
    if (!s.dumps.empty()) s.dump_tensors();
    if (timing) fprintf(stderr, "  [dhubert] T=%d time %.2f ms\n", s.T, now() - t0);
    return true;
}
