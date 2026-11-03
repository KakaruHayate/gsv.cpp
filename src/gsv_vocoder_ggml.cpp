// HiFi-GAN generator (GPT-SoVITS V5 vocoder) in ggml.
//
// Graph (matches module/models.py::Generator for the 48 kHz V5 vocoder):
//   conv_pre [7, 100, 512] -> LeakyReLU(0.1)
//   5 x (ConvTranspose1d k=20/12/4/4/4, stride 10/6/2/2/2, crop (K-s)/2, +bias)
//        -> 3 x ResBlock1 (k=3, dilations 1/3/5) summed -> mean scale 1/3 -> LeakyReLU(0.1)
//   conv_post [7, 16, 1] -> tanh
//
// Weights come from tools/export_vocoder_gguf.py (torch layouts transposed so that
// conv weights are [K, IC, OC] and transposed-conv weights are [K, Cout, Cin]).
#include "gsv_vocoder_ggml.h"

#include "gsv_device.h"
#include "gsv_gguf_slice.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int VOC_MEL = 100;
static const int VOC_UPS = 480;
static const int VOC_SR  = 48000;
static const int N_UPS   = 5;
static const int N_RB    = 3;

struct gsv_vocoder_ggml::impl {
    ggml_backend_t      backend = nullptr;
    ggml_backend_buffer_t wbuf  = nullptr;
    ggml_context *      wctx    = nullptr;
    gguf_context *      gf      = nullptr;
    std::map<std::string, ggml_tensor *> w;
    std::vector<int> ups_strides;   // gsv.voc.ups metadata
    int  n_threads = 0;
    bool verbose   = false;

    ggml_tensor * need(const std::string & n) {
        auto it = w.find(n);
        if (it == w.end()) {
            fprintf(stderr, "[gsv_voc_ggml] missing tensor %s\n", n.c_str());
            return nullptr;
        }
        return it->second;
    }

    // Stride-1 conv1d with the whole epilogue fused: bias + leaky out, optional input
    // scale/leaky (CONV_DIRECT_1D: no im2col buffer, CPU + Vulkan).
    ggml_tensor * conv_same(ggml_context * c, const std::string & pre, ggml_tensor * x,
                            int dil, float leaky, float in_scale = 1.0f, float in_slope = 0.0f) {
        ggml_tensor * ww = need(pre + ".weight");
        if (!ww) return nullptr;
        const int64_t K   = ww->ne[0];
        const int64_t pad = (K / 2) * dil;
        return ggml_conv_direct_1d_fused(c, ww, x, need(pre + ".bias"), nullptr,
                                         (int) pad, dil, leaky, in_scale, in_slope);
    }

    // Sub-pixel upsampling: conv1d on the input grid (M taps) + phase interleave,
    // exactly equivalent to ConvTranspose1d(K, stride) with padding (K-stride)/2.
    ggml_tensor * upsample_subpixel(ggml_context * c, const std::string & pre, ggml_tensor * x,
                                    int stride, float in_scale = 1.0f, float in_slope = 0.0f) {
        ggml_tensor * ww = need(pre + ".weight");    // [M, Cin, Cout*stride] phase-major
        if (!ww) return nullptr;
        const int M   = (int) ww->ne[0];
        const int CS  = (int) ww->ne[2];
        const int cout = CS / stride;
        const int64_t L = x->ne[0];
        ggml_tensor * y = ggml_conv_direct_1d_fused(c, ww, x, need(pre + ".bias"), nullptr,
                                                    M / 2, 1, 0.0f, in_scale, in_slope);
        if (y->ne[0] != L) {
            // symmetric pad produced M-1 extra rows on the right; keep the first L
            y = ggml_cont(c, ggml_view_2d(c, y, L, CS, y->nb[1], 0));
        }

        ggml_tensor * y3 = ggml_reshape_3d(c, y, L, cout, stride);   // [L, Cout, s]
        ggml_tensor * yp = ggml_permute(c, y3, 1, 2, 0, 3);          // [s, L, Cout]
        ggml_tensor * yc = ggml_cont(c, yp);
        return ggml_reshape_2d(c, yc, L * stride, cout);              // [L*s, Cout]
    }

    ggml_tensor * resblock(ggml_context * c, ggml_tensor * x, int idx) {
        const int dil[3] = { 1, 3, 5 };
        ggml_tensor * h = x;
        for (int k = 0; k < 3; k++) {
            char pre[96];
            std::snprintf(pre, sizeof pre, "voc.resblocks.%d.convs1.%d", idx, k);
            ggml_tensor * y = ggml_leaky_relu(c, h, 0.1f, false);
            y = conv_same(c, pre, y, dil[k], 0.1f);
            std::snprintf(pre, sizeof pre, "voc.resblocks.%d.convs2.%d", idx, k);
            y = conv_same(c, pre, y, 1, 0.0f);
            h = ggml_add(c, y, h);
        }
        return h;
    }
};

gsv_vocoder_ggml::gsv_vocoder_ggml() : p(new impl) {}
gsv_vocoder_ggml::~gsv_vocoder_ggml() {
    if (p->wbuf) ggml_backend_buffer_free(p->wbuf);
    if (p->wctx) ggml_free(p->wctx);
    if (p->gf) gguf_free(p->gf);
    if (p->backend) ggml_backend_free(p->backend);
    delete p;
}

int gsv_vocoder_ggml::mel_dim() const { return VOC_MEL; }
int gsv_vocoder_ggml::upsample() const { return VOC_UPS; }
int gsv_vocoder_ggml::sample_rate() const { return VOC_SR; }
const char * gsv_vocoder_ggml::device_name() const {
    return p->backend ? ggml_backend_name(p->backend) : "?";
}

gsv_vocoder_ggml * gsv_vocoder_ggml::load(const std::string & gguf_path, const gsv_vocoder_ggml_cfg & cfg) {
    gsv_vocoder_ggml * m = new gsv_vocoder_ggml();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;

    ggml_backend_dev_t dev = nullptr;
    {
        int gpu_idx = 0; bool gpu_explicit = false;
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
        if (!dev)
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        if (want_gpu && gpu_explicit && (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)) {
            fprintf(stderr, "[device] GPU %d not found (run --list-devices)\n", gpu_idx);
            delete m; return nullptr;
        }
        if (!dev) { fprintf(stderr, "[gsv_voc_ggml] no usable backend device\n"); delete m; return nullptr; }
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_voc_ggml] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_voc_ggml] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }

    ggml_context * slice = nullptr;
    s.wbuf = gsv_alloc_prefix_tensors(s.wctx, s.backend, { "voc." }, &slice);
    if (s.wbuf) { ggml_free(s.wctx); s.wctx = slice; }
    if (!s.wbuf) { fprintf(stderr, "[gsv_voc_ggml] weight buffer alloc failed\n"); delete m; return nullptr; }

    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_voc_ggml] reopen failed\n"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            if (strncmp(tname, "voc.", 4) != 0) continue;
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tname);
            if (!tt) continue;
            const size_t nbytes = ggml_nbytes(tt);
            std::vector<char> buf(nbytes);
            if (fseek(fp, (long) (data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_voc_ggml] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
            s.w[tname] = tt;
        }
        fclose(fp);
    }

    {
        const int64_t k = gguf_find_key(s.gf, "gsv.voc.ups");
        if (k >= 0) {
            const int n = (int) gguf_get_arr_n(s.gf, k);
            const uint32_t * d = (const uint32_t *) gguf_get_arr_data(s.gf, k);
            for (int i = 0; i < n; i++) s.ups_strides.push_back((int) d[i]);
        }
    }

    if (s.verbose)
        printf("[gsv_voc_ggml] loaded %s (%zu tensors, device=%s)\n",
               gguf_path.c_str(), s.w.size(), ggml_backend_name(s.backend));
    return m;
}

bool gsv_vocoder_ggml::forward(const float * mel, int T, std::vector<float> & wav) {
    impl & s = *p;
    wav.clear();
    if (T <= 0 || !mel) return false;

    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_tensor * xin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, VOC_MEL);
    ggml_set_input(xin);

    ggml_tensor * x = s.conv_same(ctx, "voc.conv_pre", xin, 1, 0.1f);
    const float mean_scale = 1.0f / (float) N_RB;
    for (int i = 0; i < N_UPS && x; i++) {
        char pre[64];
        std::snprintf(pre, sizeof pre, "voc.upsub.%d", i);
        const int stride = i < (int) s.ups_strides.size() ? s.ups_strides[i] : 2;
        // entering level i>0 the previous resblock mean still needs /3 and LeakyReLU(0.1);
        // both fold into this convolution's input packing
        x = s.upsample_subpixel(ctx, pre, x, stride,
                                i > 0 ? mean_scale : 1.0f, i > 0 ? 0.1f : 0.0f);
        ggml_tensor * sum = nullptr;
        for (int j = 0; j < N_RB && x; j++) {
            ggml_tensor * rb = s.resblock(ctx, x, i * N_RB + j);
            sum = sum ? ggml_add(ctx, sum, rb) : rb;
        }
        x = sum;
    }
    if (x) {
        x = s.conv_same(ctx, "voc.conv_post", x, 1, 0.0f, mean_scale, 0.01f);
        x = ggml_tanh(ctx, x);
        ggml_set_output(x);
    }
    if (!x) { ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE * 8, false);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "[gsv_voc_ggml] graph alloc failed (T=%d)\n", T);
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(xin, mel, 0, (size_t) VOC_MEL * T * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(s.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[gsv_voc_ggml] compute failed: %d\n", (int) st);
        ggml_gallocr_free(alloc); ggml_free(ctx);
        return false;
    }
    const size_t n = (size_t) ggml_nelements(x);
    wav.resize(n);
    ggml_backend_tensor_get(x, wav.data(), 0, n * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}
