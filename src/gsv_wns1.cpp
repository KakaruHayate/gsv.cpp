#include "gsv_wns1.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int C_HID = 512;
static const int NL = 8;
static const int K = 5;
static const int PAD = 2;

struct gsv_wns1::impl {
    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t s_back = nullptr;
    ggml_context * gctx = nullptr;
    ggml_cgraph * g = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * g_in = nullptr;
    ggml_tensor * g_ge = nullptr;
    ggml_tensor * g_mask = nullptr;
    ggml_tensor * g_out = nullptr;
    int T = 120, LEN = 100;   // T 为"当前图缓存对应的长度"; encode(T != 缓存) 时重建图
    int n_threads = 0;
    bool verbose = false;
    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if(!t){ fprintf(stderr,"[gsv_wns1] missing %s\n",n); abort(); }
        return t;
    }
};

// 1D conv 内核选择 (audio patch 的算子 A/B):
//   0 = ggml_conv_1d                (原样)
//   1 = ggml_conv_1d_fast_1d_im2col (1D 专用 im2col)
//   2 = fast im2col + dst F32       (CPU: 省 F16->F32 转换)
//   3 = ggml_conv_direct_1d         (stride-1 免 im2col, 需 K=1 时 pad=0)
// 输出统一为 3D [OL, OC, 1] (与原 conv_1d 一致), bias 由调用方自行相加。
// 默认 2 (fast im2col + F32 dst): CPU 72ms vs 原 103ms, 且精度 6.5e-4 -> 1.1e-6;
// Vulkan 与之持平且精度略好 (2.9e-3 vs 3.6e-3)。
static int wns1_conv_mode(){
    const char * e = getenv("GSV_WNS1_CONV");
    return e ? atoi(e) : 2;
}
// 3D 视图助手: 量化后的 k=1 权重是 2D ([in, out], 见 quantize_gguf 的压形), 不能 reshape;
// 这里对 2D 原样返回, 由 conv1d_pick 的 2D 分支处理 (直接 mul_mat, 等价 k=1 im2col)。
static ggml_tensor * as3d(ggml_context * c, ggml_tensor * t, int K, int IC, int OC){
    if(t->ne[2] == 1 && t->ne[3] == 1) return t;
    return ggml_reshape_3d(c, t, K, IC, OC);
}
static ggml_tensor * conv1d_pick(ggml_context * c, ggml_tensor * w, ggml_tensor * x, int pad, int mode){
    // 2D 权重 (k=1, 含量化): out[t, oc] = sum_ic x[t, ic]*w[ic, oc]
    // 权重必须作 mul_mat 的 src0 (该 ggml 的量化路径只支持 src0 量化);
    // x [T, IC, 1] → [IC, T]; mul_mat(w [IC, OC], xt) → [OC, T] → 转回 [T, OC, 1]
    if(w->ne[2] == 1 && w->ne[3] == 1){
        GGML_ASSERT(pad == 0);
        ggml_tensor * xt = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));    // [IC, T]
        ggml_tensor * mm = ggml_mul_mat(c, w, xt);                          // [OC, T]
        ggml_tensor * yt = ggml_cont(c, ggml_permute(c, mm, 1, 0, 2, 3));   // [T, OC]
        return ggml_reshape_3d(c, yt, yt->ne[0], yt->ne[1], 1);
    }
    if(mode == 3){
        // conv_direct_1d 需要 2D contiguous x [T, IC] 与 2D 输出
        ggml_tensor * x2 = ggml_view_2d(c, x, x->ne[0], x->ne[1], x->nb[1], 0);
        ggml_tensor * y = ggml_conv_direct_1d(c, w, x2, NULL, pad, 1, 0.0f);   // [OL, OC]
        return ggml_reshape_3d(c, y, y->ne[0], y->ne[1], 1);
    }
    if(mode == 2){
        // 公开的 im2col_fast_1d + 与 conv_1d 内部一致的 mul_mat 接线, 只是 dst 用 F32
        ggml_tensor * im = ggml_im2col_fast_1d(c, w, x, 1, pad, 1, GGML_TYPE_F32, 1);
        ggml_tensor * mm = ggml_mul_mat(c,
                ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]),
                ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]));
        return ggml_reshape_3d(c, mm, im->ne[1], w->ne[2], im->ne[2]);
    }
    if(mode == 4){
        // 混合: 多抽头 (K>1) 走 conv_direct_1d, k=1 走 fast im2col F32
        // (conv_direct_1d 的 Vulkan 实现在 K=1 时会崩, 见 tests/test_conv_direct_vk.cpp)
        if(w->ne[0] > 1){
            ggml_tensor * x2 = ggml_view_2d(c, x, x->ne[0], x->ne[1], x->nb[1], 0);
            ggml_tensor * y = ggml_conv_direct_1d(c, w, x2, NULL, pad, 1, 0.0f);
            return ggml_reshape_3d(c, y, y->ne[0], y->ne[1], 1);
        }
        mode = 2;   // fall through
    }
    if(mode == 1) return ggml_conv_1d_fast_1d_im2col(c, w, x, 1, pad, 1);
    return ggml_conv_1d(c, w, x, 1, pad, 1);
}

static void build_graph(gsv_wns1::impl & s){
    ggml_init_params ip = { ggml_tensor_overhead() * 4096, NULL, true };
    s.gctx = ggml_init(ip);
    ggml_context * c = s.gctx;
    const int T = s.T, C = C_HID;
    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, C);
    ggml_set_input(x); s.g_in = x;
    ggml_tensor * ge = ggml_new_tensor_1d(c, GGML_TYPE_F32, C);
    ggml_set_input(ge); s.g_ge = ge;
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, 1);
    ggml_set_input(mask); s.g_mask = mask;

    // pre: Conv1d 512->512 k=1, then * mask
    {
        ggml_tensor * w = as3d(c, s.need("wns1.pre.weight"), 1, C, C);
        ggml_tensor * y = conv1d_pick(c, w, x, 0, wns1_conv_mode());
        ggml_tensor * b = ggml_reshape_3d(c, s.need("wns1.pre.bias"), 1, C, 1);
        y = ggml_add(c, y, b);
        y = ggml_mul(c, y, mask);
        x = y;
    }

    // cond weight full: ne [512, 8192]; per-layer slice rows [li*2048, li*2048+2048)
    ggml_tensor * cw_full = s.need("wns1.enc.cond_layer.weight_w");
    ggml_tensor * cb_full = s.need("wns1.enc.cond_layer.bias");
    ggml_tensor * ge3 = ggml_reshape_3d(c, ge, 1, C, 1);
    ggml_tensor * sk = NULL;
    char nm[160];
    for(int li = 0; li < NL; li++){
        snprintf(nm,160,"wns1.enc.in_layers.%d.weight_w",li);
        ggml_tensor * iw = ggml_reshape_3d(c, s.need(nm), K, C, 2*C);
        snprintf(nm,160,"wns1.enc.in_layers.%d.bias",li);
        ggml_tensor * ib = ggml_reshape_3d(c, s.need(nm), 1, 2*C, 1);
        ggml_tensor * xin = conv1d_pick(c, iw, x, PAD, wns1_conv_mode());
        xin = ggml_add(c, xin, ib);
        const int off = li * 2 * C;
        const size_t cw_stride = (cw_full->ne[2] == 1 && cw_full->ne[3] == 1) ? cw_full->nb[1] : cw_full->nb[2];
        ggml_tensor * cw = ggml_view_2d(c, cw_full, C, 2*C, cw_stride, (size_t)off * cw_stride);
        ggml_tensor * cw3 = as3d(c, cw, 1, C, 2*C);
        xin = ggml_add(c, xin, conv1d_pick(c, cw3, ge3, 0, wns1_conv_mode()));
        ggml_tensor * cbv = ggml_view_1d(c, cb_full, 2*C, (size_t)off * 4);
        xin = ggml_add(c, xin, ggml_reshape_3d(c, cbv, 1, 2*C, 1));
        // gate: tanh(xin[:C])*sigmoid(xin[C:])  xin [T, 2C, 1] contiguous
        ggml_tensor * a = ggml_view_1d(c, xin, (int64_t)C*T, 0);
        ggml_tensor * b = ggml_view_1d(c, xin, (int64_t)C*T, (size_t)C*T*4);
        ggml_tensor * acts = ggml_mul(c, ggml_tanh(c, a), ggml_sigmoid(c, b));
        acts = ggml_reshape_3d(c, acts, T, C, 1);
        snprintf(nm,160,"wns1.enc.res_skip_layers.%d.weight_w",li);
        int rc = li < NL-1 ? 2*C : C;
        ggml_tensor * rw = as3d(c, s.need(nm), 1, C, rc);
        snprintf(nm,160,"wns1.enc.res_skip_layers.%d.bias",li);
        ggml_tensor * rb = ggml_reshape_3d(c, s.need(nm), 1, rc, 1);
        ggml_tensor * rs = conv1d_pick(c, rw, acts, 0, wns1_conv_mode());
        rs = ggml_add(c, rs, rb);
        if(li < NL-1){
            ggml_tensor * rres = ggml_view_1d(c, rs, (int64_t)C*T, 0);
            ggml_tensor * rsk  = ggml_view_1d(c, rs, (int64_t)C*T, (size_t)C*T*4);
            rres = ggml_reshape_3d(c, rres, T, C, 1);
            x = ggml_mul(c, ggml_add(c, x, rres), mask);
            rsk = ggml_reshape_3d(c, rsk, T, C, 1);
            sk = sk ? ggml_add(c, sk, rsk) : rsk;
        } else {
            sk = sk ? ggml_add(c, sk, rs) : rs;
        }
    }
    // sk [T, C, 1] -> *mask -> proj k=1 -> *mask
    sk = ggml_mul(c, sk, mask);
    {
        ggml_tensor * w = as3d(c, s.need("wns1.proj.weight"), 1, C, C);
        ggml_tensor * y = conv1d_pick(c, w, sk, 0, wns1_conv_mode());
        ggml_tensor * b = ggml_reshape_3d(c, s.need("wns1.proj.bias"), 1, C, 1);
        y = ggml_mul(c, ggml_add(c, y, b), mask);
        s.g_out = y;
    }
    ggml_set_output(s.g_out);
    s.g = ggml_new_graph_custom(c, 4096, false);
    ggml_build_forward_expand(s.g, s.g_out);
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.s_back));
    ggml_gallocr_alloc_graph(s.galloc, s.g);
}

gsv_wns1::gsv_wns1() : p(new impl) {}
gsv_wns1::~gsv_wns1() {
    if(!p) return;
    if(p->galloc) ggml_gallocr_free(p->galloc);
    if(p->gctx) ggml_free(p->gctx);
    if(p->wbuf) ggml_backend_buffer_free(p->wbuf);
    if(p->gf) gguf_free(p->gf);
    if(p->wctx) ggml_free(p->wctx);
    if(p->backend) ggml_backend_free(p->backend);
    delete p;
}

gsv_wns1 * gsv_wns1::load(const std::string & gguf_path, const gsv_wns1_cfg & cfg) {
    gsv_wns1 * m = new gsv_wns1();
    impl & s = *m->p;
    s.n_threads = cfg.n_threads;
    s.verbose = cfg.verbose;
    ggml_backend_dev_t dev = nullptr;
    const bool want_gpu = cfg.device == "gpu" || cfg.device == "vulkan" || cfg.device == "GPU" || cfg.device == "Vulkan";
    const int n_dev = (int) ggml_backend_dev_count();
    for (int i = 0; i < n_dev && !dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(d);
        if (want_gpu) { if (ty == GGML_BACKEND_DEVICE_TYPE_GPU || ty == GGML_BACKEND_DEVICE_TYPE_IGPU) dev = d; }
        else if (ty == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
    }
    if(!dev) for (int i = 0; i < n_dev && !dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
    }
    if(!dev){ fprintf(stderr,"[gsv_wns1] no usable backend device\n"); delete m; return nullptr; }
    if(s.verbose) printf("[gsv_wns1] device: %s\n", ggml_backend_dev_name(dev));
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if(!s.backend){ fprintf(stderr,"[gsv_wns1] backend init failed\n"); delete m; return nullptr; }
    if(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && s.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, s.n_threads);
    s.s_back = s.backend;
    gguf_init_params gip = { true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if(!s.gf){ fprintf(stderr,"[gsv_wns1] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if(!s.wbuf){ fprintf(stderr,"[gsv_wns1] weight buffer alloc failed\n"); delete m; return nullptr; }
    FILE * fp = fopen(gguf_path.c_str(), "rb");
    if(!fp){ fprintf(stderr,"[gsv_wns1] reopen failed\n"); delete m; return nullptr; }
    const size_t doff = gguf_get_data_offset(s.gf);
    const int64_t nt = gguf_get_n_tensors(s.gf);
    for(int64_t ti = 0; ti < nt; ti++){
        const char * tn = gguf_get_tensor_name(s.gf, ti);
        ggml_tensor * tt = ggml_get_tensor(s.wctx, tn);
        if(!tt) continue;
        const size_t nb = ggml_nbytes(tt);
        std::vector<char> buf(nb);
        if(fseek(fp, (long)(doff + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
           fread(buf.data(), 1, nb, fp) != nb){
            fprintf(stderr,"[gsv_wns1] read %s failed\n", tn); fclose(fp); delete m; return nullptr;
        }
        ggml_backend_tensor_set(tt, buf.data(), 0, nb);
    }
    fclose(fp);
    build_graph(s);
    if(s.verbose) printf("[gsv_wns1] loaded %s\n", gguf_path.c_str());
    return m;
}

bool gsv_wns1::encode(const float * fea, const float * ge, int T, int len, std::vector<float> & out) {
    impl & s = *p;
    if(T <= 0){ fprintf(stderr,"[gsv_wns1] bad T=%d\n", T); return false; }
    if(T != s.T){
        // 图按 T 编译 (conv 形状与 mask 长度都绑定 T); T 变化时重建并缓存
        if(s.galloc){ ggml_gallocr_free(s.galloc); s.galloc = nullptr; }
        if(s.gctx){ ggml_free(s.gctx); s.gctx = nullptr; }
        s.g = nullptr; s.g_in = s.g_ge = s.g_mask = s.g_out = nullptr;
        s.T = T;
        build_graph(s);
        if(s.verbose) printf("[gsv_wns1] graph rebuilt for T=%d\n", T);
    }
    std::vector<float> mk((size_t)s.T);
    for(int t = 0; t < s.T; t++) mk[t] = t < len ? 1.0f : 0.0f;
    ggml_backend_tensor_set(s.g_in, fea, 0, (size_t)C_HID * s.T * 4);
    ggml_backend_tensor_set(s.g_ge, ge, 0, (size_t)C_HID * 4);
    ggml_backend_tensor_set(s.g_mask, mk.data(), 0, (size_t)s.T * 4);
    ggml_backend_graph_compute(s.backend, s.g);
    out.assign((size_t)C_HID * s.T, 0.0f);
    ggml_backend_tensor_get(s.g_out, out.data(), 0, out.size() * 4);
    return true;
}
