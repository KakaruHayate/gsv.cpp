#include "gsv_refcode.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

static const int SSL_DIM = 768, BINS = 1024, K = 2, STRIDE = 2;

// ============================ resampy kaiser_best 表 ============================
namespace {
struct kaiser_tab {
    bool ok = false;
    int  nwin = 0, ntable = 0;
    float rolloff = 0.0f;
    std::vector<float> win, delta;
};
kaiser_tab g_tab;

int gcd_i(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }
}  // namespace

bool gsv_refcode::load_filter_table(const std::string & bin_path) {
    FILE * f = fopen(bin_path.c_str(), "rb");
    if (!f) return false;
    int32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3) { fclose(f); return false; }
    const int nwin = hdr[0], ntable = hdr[1];
    float rolloff = 0.0f;
    if (fread(&rolloff, 4, 1, f) != 1) { fclose(f); return false; }
    if (nwin <= 0 || nwin > (1 << 24)) { fclose(f); return false; }
    g_tab.win.resize((size_t) nwin);
    g_tab.delta.resize((size_t) nwin);
    const bool ok = fread(g_tab.win.data(), 4, (size_t) nwin, f) == (size_t) nwin &&
                    fread(g_tab.delta.data(), 4, (size_t) nwin, f) == (size_t) nwin;
    fclose(f);
    g_tab.nwin = nwin; g_tab.ntable = ntable; g_tab.rolloff = rolloff; g_tab.ok = ok;
    return ok;
}

// repo: resample(audio_tensor, sr0, sr1, device, match_librosa=True) 的等价复刻
// (双边窗口 + 表内线性插值 interp_win + eta*interp_delta, 越界索引 clamp)
void gsv_refcode::resample_librosa(const float * x, int64_t n, int sr_in, int sr_out,
                                   std::vector<float> & out) {
    if (n <= 0 || sr_in <= 0 || sr_out <= 0) { out.clear(); return; }
    if (sr_in == sr_out) { out.assign(x, x + n); return; }
    if (!g_tab.ok) { fprintf(stderr, "[gsv_refcode] filter table 未加载 (load_filter_table)\n"); out.clear(); return; }
    const double ratio = (double) sr_out / (double) sr_in;
    const double scale = std::min(1.0, ratio);
    const int64_t n_out = (int64_t) ((double) n * ratio);
    out.assign((size_t) n_out, 0.0f);
    const int index_step = (int) (scale * (double) g_tab.ntable);
    const int nwin = g_tab.nwin;
    const int max_taps = (nwin + index_step - 1) / index_step;
    const float * w = g_tab.win.data();
    const float * dw = g_tab.delta.data();
    const bool shrink = ratio < 1.0;                       // ratio<1 时表整体乘 ratio

    for (int64_t o = 0; o < n_out; o++) {
        const double t = (double) o / ratio;
        const int64_t nn = (int64_t) std::floor(t);
        const double frac = scale * (t - (double) nn);
        double acc = 0.0;
        for (int side = 0; side < 2; side++) {
            const double fr = (side == 0) ? frac : (scale - frac);
            const double index_frac = fr * (double) g_tab.ntable;
            const int64_t offset = (int64_t) std::floor(index_frac);
            const double eta = index_frac - (double) offset;
            const int64_t limit = (side == 0) ? nn + 1 : (n - nn - 1);
            const int64_t lmax = std::min<int64_t>(limit, (nwin - offset) / index_step);
            for (int64_t ii = 0; ii < lmax && ii < max_taps; ii++) {
                const int64_t li = offset + ii * index_step;
                if (li < 0 || li >= nwin) continue;
                const double ww = (double) w[li] + eta * (double) dw[li];
                const double wv = shrink ? ww * ratio : ww;
                int64_t idx = (side == 0) ? (nn - ii) : (nn + ii + 1);
                idx = std::max<int64_t>(0, std::min<int64_t>(idx, n - 1));
                acc += (double) x[idx] * wv;
            }
        }
        out[(size_t) o] = (float) acc;
    }
}

// ============================ ssl_proj + RVQ 编码 ============================

struct gsv_refcode::impl {
    gsv_refcode_cfg cfg;
    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * w0 = nullptr, * w1 = nullptr, * bw = nullptr, * cb = nullptr;   // k=0/k=1 权重 (转换器已拆)
    std::vector<float> en2;                                                        // ||E||² [bins] (f32)
    int n_threads = 0, T = 0;
    ggml_context * gctx = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_tensor * t_in = nullptr, * t_z = nullptr, * t_d = nullptr;                // 输入 / z [768,T2] / D=E·z [bins,T2]

    ggml_tensor * need(const char * nm) const {
        ggml_tensor * t = ggml_get_tensor(wctx, nm);
        if (!t) { fprintf(stderr, "[gsv_refcode] missing %s\n", nm); abort(); }
        return t;
    }
    bool build(int T_);
};

bool gsv_refcode::impl::build(int T_) {
    // conv k=2 s=2: out[oc, i] = Σ_ic W[oc,ic,0]·x[ic,2i] + W[oc,ic,1]·x[ic,2i+1] + b[oc]
    // x 按 ggml ne{T,768} (raw = torch [768,T] 行主); 偶/奇列 = stride-2 视图
    ggml_init_params ip = { ggml_tensor_overhead() * 512, NULL, true };
    gctx = ggml_init(ip);
    ggml_context * c = gctx;
    const int T2 = T_ / 2;
    t_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, T_, SSL_DIM);   // ne{T,768} = torch [768,T] 行主字节
    ggml_set_input(t_in); ggml_set_name(t_in, "ssl_in");
    // mul_mat 的归约维必须在 ne0 -> 转成 [768, T] (一次 cont, ~0.8MB @T=250)
    ggml_tensor * xt = ggml_cont(c, ggml_permute(c, t_in, 1, 0, 2, 3));
    ggml_tensor * xe = ggml_view_2d(c, xt, SSL_DIM, T2, (size_t) STRIDE * SSL_DIM * 4, 0);
    ggml_tensor * xo = ggml_view_2d(c, xt, SSL_DIM, T2, (size_t) STRIDE * SSL_DIM * 4, (size_t) SSL_DIM * 4);
    ggml_tensor * z = ggml_add(c, ggml_mul_mat(c, w0, xe), ggml_mul_mat(c, w1, xo));
    z = ggml_add(c, z, bw);
    // D = E · z  (E ne{768,1024} -> mul_mat(E, z) = [1024, T2])
    ggml_set_output(z);
    t_z = z;
    t_d = ggml_mul_mat(c, cb, z);
    ggml_set_output(t_d);
    g = ggml_new_graph_custom(c, 64, false);
    ggml_build_forward_expand(g, t_d);
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, g)) { fprintf(stderr, "[gsv_refcode] alloc failed\n"); return false; }
    T = T_;
    return true;
}

gsv_refcode::gsv_refcode() : p(new impl) {}

gsv_refcode * gsv_refcode::load(const std::string & gguf_path, const gsv_refcode_cfg & cfg) {
    gsv_refcode * m = new gsv_refcode();
    impl & s = *m->p;
    s.cfg = cfg;
    s.n_threads = cfg.n_threads;

    ggml_backend_dev_t dev = nullptr;
    {
        const bool want_gpu = cfg.device == "gpu" || cfg.device == "vulkan" || cfg.device == "GPU";
        const int n_dev = (int) ggml_backend_dev_count();
        for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const auto ty = ggml_backend_dev_type(d);
            if (want_gpu) { if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = d; }
            else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (!dev) for (int i = 0; i < n_dev && !dev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
        }
        if (!dev) { fprintf(stderr, "[gsv_refcode] no device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_refcode] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_refcode] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);

    gguf_init_params gip = { true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_refcode] open %s failed\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_refcode] weight buffer failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_refcode] reopen failed\n"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tn = gguf_get_tensor_name(s.gf, ti);
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tn);
            if (!tt) continue;
            const size_t nb = ggml_nbytes(tt);
            std::vector<char> buf(nb);
            if (fseek(fp, (long) (data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nb, fp) != nb) { fprintf(stderr, "[gsv_refcode] read %s failed\n", tn); fclose(fp); delete m; return nullptr; }
            ggml_backend_tensor_set(tt, buf.data(), 0, nb);
        }
        fclose(fp);
    }
    // 权重: ssl_proj.weight ggml ne{k,ic,oc} -> 两个 [ic,oc] 切片视图 (nb1 = nb[2])
    s.w0 = s.need("refcode.ssl_proj.w0");
    s.w1 = s.need("refcode.ssl_proj.w1");
    s.bw = s.need("refcode.ssl_proj.bias");
    s.cb = s.need("refcode.codebook.embed");
    if (s.w0->ne[0] != SSL_DIM || s.w0->ne[1] != SSL_DIM || s.w1->ne[0] != SSL_DIM) {
        fprintf(stderr, "[gsv_refcode] w0/w1 形状不符 (期望 [%d,%d])\n", SSL_DIM, SSL_DIM);
        delete m; return nullptr;
    }
    // ||E||² (f32 累加, 与 repo 的 emb.pow(2).sum(1) 同口径)
    {
        std::vector<float> emb((size_t) BINS * SSL_DIM);
        ggml_backend_tensor_get(s.cb, emb.data(), 0, emb.size() * 4);
        s.en2.assign((size_t) BINS, 0.0f);
        // ggml [768,1024] 的平铺: 元素 (d,b) 在 d + 768*b (不是行主 b*768+d)
        for (int b = 0; b < BINS; b++) {
            float acc = 0.0f;
            for (int d = 0; d < SSL_DIM; d++) {
                const float v = emb[(size_t) d + (size_t) SSL_DIM * b];
                acc += v * v;
            }
            s.en2[(size_t) b] = acc;
        }
    }
    if (cfg.verbose) printf("[gsv_refcode] loaded %s (ssl_dim=%d bins=%d)\n", gguf_path.c_str(), SSL_DIM, BINS);
    return m;
}

gsv_refcode::~gsv_refcode() {
    impl & s = *p;
    if (s.gctx) ggml_free(s.gctx);
    if (s.galloc) ggml_gallocr_free(s.galloc);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.gf) gguf_free(s.gf);
    if (s.wctx) ggml_free(s.wctx);
    delete p;
}

int gsv_refcode::ssl_dim() const { return SSL_DIM; }
int gsv_refcode::bins()    const { return BINS; }

bool gsv_refcode::codes_from_ssl(const float * ssl, int T, std::vector<int32_t> & codes,
                                 std::vector<float> * z_out) {
    impl & s = *p;
    codes.clear();
    if (T < K) { fprintf(stderr, "[gsv_refcode] bad T=%d\n", T); return false; }   // 奇 T 时末帧不参与 (与 stride 2 卷积输出数一致)
    if (T != s.T) {
        if (s.gctx) { ggml_free(s.gctx); s.gctx = nullptr; s.g = nullptr; s.t_in = s.t_z = s.t_d = nullptr; }
        if (!s.build(T)) return false;
    }
    ggml_backend_tensor_set(s.t_in, ssl, 0, (size_t) SSL_DIM * T * 4);
    if (ggml_backend_graph_compute(s.backend, s.g) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_refcode] compute failed\n"); return false;
    }
    const int T2 = T / 2;
    std::vector<float> z((size_t) SSL_DIM * T2);
    ggml_backend_tensor_get(s.t_z, z.data(), 0, z.size() * 4);          // ggml [768,T2]: z[d + 768*i]
    if (z_out) *z_out = z;
    std::vector<float> D((size_t) BINS * T2);
    ggml_backend_tensor_get(s.t_d, D.data(), 0, D.size() * 4);          // ggml [bins,T2]: D[b + BINS*i]
    if (const char * dm = getenv("GSV_REFCODE_DUMP_MAT")) {             // 调试: 落盘 D/en2
        FILE * fp = fopen((std::string(dm) + "/D.bin").c_str(), "wb");
        if (fp) { fwrite(D.data(), 4, D.size(), fp); fclose(fp); }
        fp = fopen((std::string(dm) + "/en2.bin").c_str(), "wb");
        if (fp) { fwrite(s.en2.data(), 4, s.en2.size(), fp); fclose(fp); }
    }
    // 与 repo 完全同式: argmin(x² − 2x·Eᵀ + E²) (三项都算, 避免平局处的 fp 翻转)
    codes.resize((size_t) T2);
    std::vector<float> x2_all((size_t) T2);
    for (int i = 0; i < T2; i++) {
        float x2 = 0.0f;                                                // f32 顺序累加 (与 repo 的 sum(1) 同口径)
        for (int d = 0; d < SSL_DIM; d++) {
            const float v = z[(size_t) d + (size_t) SSL_DIM * i];
            x2 += v * v;
        }
        int best = 0;
        float bestv = x2 - 2.0f * D[(size_t) BINS * i] + s.en2[0];      // 第 i 帧的 b=0 (flat = b + BINS*i)
        for (int b = 1; b < BINS; b++) {
            const float v = x2 - 2.0f * D[(size_t) b + (size_t) BINS * i] + s.en2[(size_t) b];
            if (v < bestv) { bestv = v; best = b; }
        }
        codes[(size_t) i] = best;
        x2_all[(size_t) i] = x2;
    }
    if (const char * dc = getenv("GSV_REFCODE_DUMP_MAT")) {
        FILE * fx = fopen((std::string(dc) + "/x2.bin").c_str(), "wb");
        if (fx) { fwrite(x2_all.data(), 4, x2_all.size(), fx); fclose(fx); }
        FILE * fp = fopen((std::string(dc) + "/codes.bin").c_str(), "wb");
        if (fp) { fwrite(codes.data(), 4, codes.size(), fp); fclose(fp); }
    }
    return true;
}
