// =====================================================================
// ref_enc (MelStyleEncoder): spectral → temporal(Conv1dGLU×2) → 2 头自注意力
//   (d_k=d_v=64, 温度 √128, 加性 mask) → fc 128→512 → 有效帧均值池化 → ge
// 注意力修复要点 (原"双后端 FAIL 87"的真正原因, 非后端问题):
//   oh = mul_mat(vhT, pm) —— 原代码 operands 反了 ([T,64] 而非 [64,T], concat 轴也错);
//   MHA 输出 = fc(cat) + 残差 z —— 原来是断链 (fc 直接吃 z, attn 成死代码).
//   权重按头预切稠密块 (load 时) → mul_mat src0 全稠密, 避开 llamafile 步进假设.
//   mask: ggml sc 布局 [ki, tq], 无效 ki 处加 -1e30; padding query 只 attend key0.
// =====================================================================
#include "gsv_refenc.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

static const int  C_IN   = 704;    // refer[:, :704]
static const int  H_HID  = 128;    // style_hidden
static const int  C_OUT  = 512;    // style_vector_dim (= gin_channels)
static const int  K_CONV = 5;
static const int  PAD    = (K_CONV - 1) / 2;
static const int  N_HEAD = 2;
static const int  D_KV   = H_HID / N_HEAD;   // 64
static const float TEMP  = 11.313708499f;    // sqrt(128)

struct gsv_refenc::impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_t s_back = nullptr;
    gguf_context * gf = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;

    ggml_context * gctx = nullptr;
    ggml_cgraph  * g = nullptr;
    ggml_gallocr_t galloc = nullptr;
    int T = 0;                        // 当前图缓存对应的 T
    ggml_tensor * t_in = nullptr;     // [704, T]
    ggml_tensor * t_mask = nullptr;   // [1, T] 帧 mask (乘在输入上)
    ggml_tensor * t_amask = nullptr;  // [T, T] 注意力加性 mask
    // 注意力权重的每头稠密块: torch [out,in] 行主下, 头 h 的 64 行天然连续,
    // 拷出后 reshape (in, OH) 即 mul_mat 可直接用的稠密 src0 (避免图内 view/cont)
    ggml_context * hctx = nullptr;
    ggml_backend_buffer_t hbuf = nullptr;
    ggml_tensor * t_wq[2] = {nullptr, nullptr};
    ggml_tensor * t_wk[2] = {nullptr, nullptr};
    ggml_tensor * t_wv[2] = {nullptr, nullptr};
    ggml_tensor * t_pool = nullptr;   // [T, 1] 池化权重 (有效帧 1 否则 0)
    // 注意力权重的每头连续块: torch [out,in] 行主下头 h 的行块天然连续
    // (flat[h*OH*IN, (h+1)*OH*IN)), 每块 reshape 成 (IN, OH) 即 mul_mat 可用的稠密 src0
    std::vector<ggml_tensor *> t_wq_h, t_wk_h, t_wv_h;   // [NH] 各 (IN=128, OH=64)
    ggml_tensor * t_out = nullptr;    // [1, 512]

    int n_threads = 0;
    bool verbose = false;

    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) { fprintf(stderr, "[gsv_refenc] missing %s\n", n); abort(); }
        return t;
    }

    // Linear: mul_mat(W [in,out], x) + b  (b 为 [out], 广播到各列)
    ggml_tensor * linear(ggml_context * c, const char * wn, const char * bn, ggml_tensor * x) {
        return ggml_add(c, ggml_mul_mat(c, need(wn), x), need(bn));
    }
    // Mish = x * tanh(softplus(x))
    ggml_tensor * mish(ggml_context * c, ggml_tensor * x) {
        return ggml_mul(c, x, ggml_tanh(c, ggml_softplus(c, x)));
    }
    // k=5 pad=2 conv (stride 1): 用 audio patch 的 1D 专用 im2col + F32 缓冲
    // (与 wns1 的默认档一致: CPU 侧省掉 F16->F32 转换, 精度与速度都更好)
    ggml_tensor * conv(ggml_context * c, const char * wn, const char * bn, ggml_tensor * x, int T) {
        ggml_tensor * w = ggml_reshape_3d(c, need(wn), K_CONV, H_HID, 2 * H_HID);
        ggml_tensor * im = ggml_im2col_fast_1d(c, w, x, 1, PAD, 1, GGML_TYPE_F32, 1);
        ggml_tensor * mm = ggml_mul_mat(c,
                ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]),
                ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]));
        ggml_tensor * y = ggml_reshape_3d(c, mm, im->ne[1], w->ne[2], im->ne[2]);   // [T, 256, 1]
        return ggml_add(c, y, ggml_reshape_3d(c, need(bn), 1, 2 * H_HID, 1));
    }

    void build_graph(int T_) {
        ggml_init_params ip = { ggml_tensor_overhead() * 2048, NULL, true };
        gctx = ggml_init(ip);
        ggml_context * c = gctx;
        T = T_;

        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, C_IN, T);
        ggml_set_input(x); t_in = x;
        ggml_tensor * m1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T);
        ggml_set_input(m1); t_mask = m1;
        ggml_tensor * am = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, T);   // soft_max_ext 常用 F16 mask
        ggml_set_input(am); t_amask = am;
        ggml_tensor * pw = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, 1);
        ggml_set_input(pw); t_pool = pw;

        // 入口: spec * mask (参考在调用侧做 refer[:, :704] * refer_mask)
        x = ggml_mul(c, x, m1);                                            // [704, T]

        // spectral: Linear(704->128)+Mish -> Linear(128->128)+Mish
        x = mish(c, linear(c, "ref_enc.spectral.0.fc.weight", "ref_enc.spectral.0.fc.bias", x));
        x = mish(c, linear(c, "ref_enc.spectral.3.fc.weight", "ref_enc.spectral.3.fc.bias", x));

        // temporal: 转 t 主序, 2 x Conv1dGLU
        ggml_tensor * y = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));     // [T, 128]
        for (int li = 0; li < 2; li++) {
            char wn[96], bn[96];
            snprintf(wn, sizeof(wn), "ref_enc.temporal.%d.conv1.conv.weight", li);
            snprintf(bn, sizeof(bn), "ref_enc.temporal.%d.conv1.conv.bias", li);
            ggml_tensor * cc = conv(c, wn, bn, y, T);                        // [T, 256, 1]
            ggml_tensor * a = ggml_reshape_3d(c, ggml_view_1d(c, cc, (int64_t) H_HID * T, 0), T, H_HID, 1);
            ggml_tensor * g = ggml_reshape_3d(c, ggml_view_1d(c, cc, (int64_t) H_HID * T, (size_t) H_HID * T * 4), T, H_HID, 1);
            y = ggml_add(c, y, ggml_mul(c, a, ggml_sigmoid(c, g)));          // 残差 + GLU
        }
        // 转回 [128, T] 后按帧 mask 置零 (参考的 masked_fill)
        ggml_tensor * z = ggml_mul(c, ggml_cont(c, ggml_permute(c, y, 1, 0, 2, 3)), m1);   // [128, T]

        // 自注意力 (2 头, d=64): 每头投影 mul_mat 用 load 时预切的稠密块
        // sc [T_ki, T_q] → soft_max_ext(加性 mask, 1/√128) → oh = vhT·p → [64, T]
        // MHA 输出 = fc(cat_heads) + 残差 z; 之后 fc.fc 128→512
        ggml_tensor * attn_cat = nullptr;
        for (int h = 0; h < N_HEAD; h++) {
            ggml_tensor * qh = ggml_add(c, ggml_mul_mat(c, t_wq[h], z),
                                        ggml_view_1d(c, need("ref_enc.slf_attn.w_qs.bias"), D_KV, (size_t) h * D_KV * 4));
            ggml_tensor * kh = ggml_add(c, ggml_mul_mat(c, t_wk[h], z),
                                        ggml_view_1d(c, need("ref_enc.slf_attn.w_ks.bias"), D_KV, (size_t) h * D_KV * 4));
            ggml_tensor * vh = ggml_add(c, ggml_mul_mat(c, t_wv[h], z),
                                        ggml_view_1d(c, need("ref_enc.slf_attn.w_vs.bias"), D_KV, (size_t) h * D_KV * 4));
            ggml_tensor * vhT = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));   // [T_ki, 64] 稠密
            ggml_tensor * sc = ggml_mul_mat(c, kh, qh);                          // [T_ki, T_q]
            ggml_tensor * pm = ggml_soft_max_ext(c, sc, am, 1.0f / TEMP, 0.0f);
            ggml_tensor * oh = ggml_mul_mat(c, vhT, pm);                         // [64, T_q]
            attn_cat = attn_cat ? ggml_concat(c, attn_cat, oh, 0) : oh;
        }
        ggml_tensor * o = ggml_add(c,
                ggml_mul_mat(c, need("ref_enc.slf_attn.fc.weight"), attn_cat),
                need("ref_enc.slf_attn.fc.bias"));                               // [128, T]
        ggml_tensor * aout = ggml_add(c, o, z);                                  // MHA 输出 (fc+残差)

        // fc 128->512, 转 t 主序后按有效帧池化 (sum/len)
        ggml_tensor * f = linear(c, "ref_enc.fc.fc.weight", "ref_enc.fc.fc.bias", aout);       // [512, T]
        ggml_tensor * ft = ggml_cont(c, ggml_permute(c, f, 1, 0, 2, 3));                        // [T, 512]
        t_out = ggml_mul_mat(c, pw, ft);                                                        // [1, 512]
        ggml_set_output(t_out);
        if (getenv("GSV_REFENC_DEBUG")) {
            ggml_set_output(x);   ggml_set_name(x, "dbg_spectral");   // [128, T]
            ggml_set_output(y);   ggml_set_name(y, "dbg_temporal");   // [T, 128]
            ggml_set_output(aout); ggml_set_name(aout, "dbg_attn");   // [128, T] MHA 输出
            ggml_set_output(z);   ggml_set_name(z, "dbg_z");          // [128, T] attention 输入
            ggml_set_output(f);   ggml_set_name(f, "dbg_fc");         // [512, T]
        }

        g = ggml_new_graph_custom(c, 2048, false);
        ggml_build_forward_expand(g, t_out);
        if (getenv("GSV_REFENC_DEBUG")) {
            ggml_graph_print(g);
        }
        if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
        if (!ggml_gallocr_alloc_graph(galloc, g)) {
            fprintf(stderr, "[gsv_refenc] galloc alloc_graph failed (T=%d)\n", T);
            ggml_free(gctx); gctx = nullptr; g = nullptr;
            return;
        }
    }
};

gsv_refenc::gsv_refenc() : p(new impl) {}
gsv_refenc::~gsv_refenc() {
    impl & s = *p;
    if (s.galloc) ggml_gallocr_free(s.galloc);
    if (s.gctx) ggml_free(s.gctx);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.gf) gguf_free(s.gf);
    if (s.wctx) ggml_free(s.wctx);
    delete p;
}

int gsv_refenc::in_dim()  const { return C_IN; }
int gsv_refenc::out_dim() const { return C_OUT; }

gsv_refenc * gsv_refenc::load(const std::string & gguf_path, const gsv_refenc_cfg & cfg) {
    gsv_refenc * m = new gsv_refenc();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;

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
        if (!dev) { fprintf(stderr, "[gsv_refenc] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_refenc] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_refenc] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.s_back = s.backend;

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_refenc] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_refenc] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        // 只上传本模块需要的张量 (ref_enc.*): 条件段 GGUF 里还有 wns1/enc_p 等 200+ 张量
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_refenc] reopen failed\n"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            if (strncmp(tname, "ref_enc.", 8) != 0) continue;
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tname);
            if (!tt) continue;
            const size_t nbytes = ggml_nbytes(tt);
            std::vector<char> buf(nbytes);
            if (fseek(fp, (long)(data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_refenc] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    // 注意力权重按头切分成稠密块 (每头 64 个输出行, 行主连续 → 拷贝后 reshape (in, OH))
    {
        ggml_init_params hip = { ggml_tensor_overhead() * 16, NULL, true };
        s.hctx = ggml_init(hip);
        for (int h = 0; h < N_HEAD; h++) {
            s.t_wq[h] = ggml_new_tensor_2d(s.hctx, GGML_TYPE_F32, H_HID, D_KV);
            s.t_wk[h] = ggml_new_tensor_2d(s.hctx, GGML_TYPE_F32, H_HID, D_KV);
            s.t_wv[h] = ggml_new_tensor_2d(s.hctx, GGML_TYPE_F32, H_HID, D_KV);
        }
        s.hbuf = ggml_backend_alloc_ctx_tensors(s.hctx, s.backend);
        if (!s.hbuf) { fprintf(stderr, "[gsv_refenc] head buffer alloc failed\n"); delete m; return nullptr; }
        for (int h = 0; h < N_HEAD; h++) {
            const char * names[3] = { "ref_enc.slf_attn.w_qs.weight", "ref_enc.slf_attn.w_ks.weight",
                                      "ref_enc.slf_attn.w_vs.weight" };
            ggml_tensor * dsts[3] = { s.t_wq[h], s.t_wk[h], s.t_wv[h] };
            for (int wi = 0; wi < 3; wi++) {
                ggml_tensor * src = ggml_get_tensor(s.wctx, names[wi]);   // ne=(in=128, out=128)
                // 每头块 = src 的 out 行 [h*64, h*64+64): 行主数据天然连续,
                // 块起点 = flat[h*64*in]; 拷到独立稠密块后 reshape (in, OH)
                // 每头 64 行 (out 维) 稠密块: 源可能是 F32/F16/Q8_0 (量化后), 统一转 F32 拷出。
                // 行字节数按类型算 (Q8_0 = 128/32*34 = 136 B/行), 否则 F16/量化会越界。
                const size_t src_row_bytes = ggml_row_size(src->type, H_HID);
                std::vector<char> raw((size_t) D_KV * src_row_bytes);
                ggml_backend_tensor_get(src, raw.data(), (size_t) h * D_KV * src_row_bytes, raw.size());
                std::vector<float> head((size_t) D_KV * H_HID);
                if (src->type == GGML_TYPE_F32) {
                    memcpy(head.data(), raw.data(), head.size() * 4);
                } else if (src->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), head.data(), (int64_t) head.size());
                } else {
                    ggml_get_type_traits(src->type)->to_float(raw.data(), head.data(), (int64_t) head.size());
                }
                ggml_backend_tensor_set(dsts[wi], head.data(), 0, head.size() * 4);
            }
        }
    }
    {
        ggml_tensor * w0 = ggml_get_tensor(s.wctx, "ref_enc.spectral.0.fc.weight");
        if (!w0) { fprintf(stderr, "[gsv_refenc] missing ref_enc weights\n"); delete m; return nullptr; }
        if (w0->ne[0] != C_IN || w0->ne[1] != H_HID) {
            fprintf(stderr, "[gsv_refenc] spectral.0.fc.weight ne=[%d,%d], 期望 [%d,%d]\n",
                    (int) w0->ne[0], (int) w0->ne[1], C_IN, H_HID);
            delete m; return nullptr;
        }
    }
    // 图按 T 编译; 首次 encode(T) 时构建, T 变化时重建
    if (cfg.verbose) printf("[gsv_refenc] loaded %s (%d -> %d)\n", gguf_path.c_str(), C_IN, C_OUT);
    return m;
}

bool gsv_refenc::encode(const float * spec, const float * frame_mask, int T, std::vector<float> & ge) {
    impl & s = *p;
    auto _te0 = std::chrono::steady_clock::now();
    if (T <= 0) { fprintf(stderr, "[gsv_refenc] bad T=%d\n", T); return false; }
    if (T != s.T) {
        if (s.galloc) { ggml_gallocr_free(s.galloc); s.galloc = nullptr; }
        if (s.gctx) { ggml_free(s.gctx); s.gctx = nullptr; }
        s.g = nullptr; s.t_in = s.t_mask = s.t_amask = s.t_pool = s.t_out = nullptr;
        s.build_graph(T);
        if (s.verbose) printf("[gsv_refenc] graph rebuilt for T=%d\n", T);
    }
    // 帧 mask / 注意力 mask / 池化权重都在 host 侧准备好
    int n_valid = 0;
    std::vector<float> m1((size_t) T);
    for (int t = 0; t < T; t++) { m1[t] = frame_mask[t] != 0.0f ? 1.0f : 0.0f; if (m1[t] != 0.0f) n_valid++; }
    if (n_valid == 0) { fprintf(stderr, "[gsv_refenc] 全部帧都被 mask 掉\n"); return false; }
    std::vector<ggml_fp16_t> am((size_t) T * T);
    std::vector<float> pw((size_t) T, 0.0f);
    for (int t = 0; t < T; t++) if (m1[t] != 0.0f) pw[t] = 1.0f;
    const bool nomask = getenv("GSV_REFENC_NOMASK") != nullptr;
    const char * mv = getenv("GSV_REFENC_MASKVAL");
    const float maskval = mv ? (float) atof(mv) : -1e30f;
    for (int qi = 0; qi < T && !nomask; qi++) {
        for (int ki = 0; ki < T; ki++) {
            // 有效 query: 只 attend 有效 key; padding query: 只 attend key 0 (避免整行 -inf 出 NaN,
            // 其结果不参与池化);
            const bool attend = (m1[qi] != 0.0f) ? (m1[ki] != 0.0f) : (ki == 0);
            am[(size_t) ki + (size_t) qi * T] = ggml_fp32_to_fp16(attend ? 0.0f : maskval);
        }
    }
    ggml_backend_tensor_set(s.t_in, spec, 0, (size_t) C_IN * T * 4);
    ggml_backend_tensor_set(s.t_mask, m1.data(), 0, m1.size() * 4);
    if (s.t_amask->buffer) ggml_backend_tensor_set(s.t_amask, am.data(), 0, am.size() * 2);
    ggml_backend_tensor_set(s.t_pool, pw.data(), 0, pw.size() * 4);
    auto _tt0 = std::chrono::steady_clock::now();
    if (getenv("GSV_REFENC_TIMING")) {
        fprintf(stderr, "[refenc timing] host %.3f ms\n",
                std::chrono::duration<double, std::milli>(_tt0 - _te0).count());
    }
    ggml_backend_graph_compute(s.backend, s.g);
    if (getenv("GSV_REFENC_TIMING")) {
        auto _tt1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[refenc timing] compute %.3f ms\n",
                std::chrono::duration<double, std::milli>(_tt1 - _tt0).count());
    }
    if (getenv("GSV_REFENC_DEBUG")) {
        const char * dumps[] = { "dbg_spectral", "dbg_temporal", "dbg_z", "dbg_attn", "dbg_fc" };
        for (const char * nm : dumps) {
            ggml_tensor * t = ggml_get_tensor(s.gctx, nm);
            if (!t) continue;
            std::vector<float> buf(ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
            std::string fn = std::string("tests/golden/_mine_") + (nm + 4) + ".bin";
            FILE * fp = fopen(fn.c_str(), "wb");
            if (fp) { fwrite(buf.data(), 4, buf.size(), fp); fclose(fp); }
        }
    }
    ggml_tensor * out = s.t_out;   // [1, 512] 池化和 / len
    std::vector<float> raw(ggml_nelements(out));
    ggml_backend_tensor_get(out, raw.data(), 0, raw.size() * 4);
    const float inv = 1.0f / (float) n_valid;
    ge.resize(raw.size());
    for (size_t i = 0; i < raw.size(); i++) ge[i] = raw[i] * inv;
    return true;
}
