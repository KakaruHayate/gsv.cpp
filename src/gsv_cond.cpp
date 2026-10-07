#include "gsv_cond.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

struct gsv_cond::impl {
    int DIM = 768, BINS = 1024;

    ggml_tensor * rvq_codebook = nullptr;   // [ne0=DIM, ne1=BINS]

    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;

    // codes [T] -> [DIM, T]; up=true 时 [DIM, 2T]
    void rvq_run(const int32_t * codes, int T, bool up, std::vector<float> & out) {
        ggml_init_params ip = { ggml_tensor_overhead() * 512, NULL, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(ids);
        ggml_tensor * q = ggml_get_rows(ctx, rvq_codebook, ids);          // [DIM, T]
        if (up) q = ggml_interpolate(ctx, q, DIM, 2 * T, 1, 1, GGML_SCALE_MODE_NEAREST);
        ggml_set_output(q);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(graph, q);
        ggml_gallocr_alloc_graph(galloc, graph);
        ggml_backend_tensor_set(ids, codes, 0, (size_t) T * 4);
        ggml_backend_graph_compute(backend, graph);
        out.assign(ggml_nelements(q), 0.0f);
        ggml_backend_tensor_get(q, out.data(), 0, out.size() * 4);
        ggml_free(ctx);
    }
};

gsv_cond::gsv_cond() : p(new impl) {}
gsv_cond::~gsv_cond() {
    impl & s = *p;
    if (s.galloc) ggml_gallocr_free(s.galloc);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.wctx) ggml_free(s.wctx);
    if (s.gf) gguf_free(s.gf);
    delete p;
}

int gsv_cond::rvq_dim()  const { return p->DIM; }
int gsv_cond::rvq_bins() const { return p->BINS; }

void gsv_cond::rvq_decode(const int32_t * codes, int T, bool upsample_x2, std::vector<float> & out) {
    p->rvq_run(codes, T, upsample_x2, out);
}

gsv_cond * gsv_cond::load(const std::string & gguf_path, const gsv_cond_cfg & cfg) {
    gsv_cond * m = new gsv_cond();
    impl & s = *m->p;

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
        if (!dev)
            for (int i = 0; i < n_dev && !dev; i++) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) dev = d;
            }
        if (!dev) { fprintf(stderr, "[gsv_cond] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_cond] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_cond] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_cond] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_cond] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_cond] reopen failed\n"); delete m; return nullptr; }
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
                fprintf(stderr, "[gsv_cond] read tensor %s failed\n", tname);
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
    s.BINS = kv_u32("cond.rvq.bins", 1024);
    s.DIM  = kv_u32("cond.rvq.dim", 768);
    s.rvq_codebook = ggml_get_tensor(s.wctx, "rvq.codebook");
    if (!s.rvq_codebook) { fprintf(stderr, "[gsv_cond] missing rvq.codebook\n"); delete m; return nullptr; }
    if (s.rvq_codebook->ne[0] != s.DIM || s.rvq_codebook->ne[1] != s.BINS) {
        fprintf(stderr, "[gsv_cond] rvq.codebook shape = [%d, %d], 期望 [%d, %d]\n",
                (int) s.rvq_codebook->ne[0], (int) s.rvq_codebook->ne[1], s.DIM, s.BINS);
        delete m; return nullptr;
    }
    if (cfg.verbose)
        printf("[gsv_cond] loaded %s: rvq dim=%d bins=%d (codebook %s)\n",
               gguf_path.c_str(), s.DIM, s.BINS, ggml_type_name(s.rvq_codebook->type));
    return m;
}
