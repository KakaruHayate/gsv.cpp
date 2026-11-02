#include "gsv_bert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>

// 单次 encode 计时 (GSV_BERT_PROFILE=1)
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct bert_layer_w {
    ggml_tensor * qkv_w, * qkv_b, * attn_out_w, * attn_out_b, * attn_ln_w, * attn_ln_b;
    ggml_tensor * ff1_w, * ff1_b, * ff2_w, * ff2_b, * out_ln_w, * out_ln_b;
};

struct gsv_bert::impl {
    int D = 1024, NH = 16, HD = 64, NL = 22, VOCAB = 21128, MAXPOS = 512, FFD = 4096;
    float EPS = 1e-12f;
    bool  is_cpu = false;                 // CPU 后端: 批量路径自动回退逐条 (见 encode_feat_batch)

    ggml_tensor * word_emb = nullptr, * pos_emb = nullptr, * type_emb = nullptr;
    ggml_tensor * emb_ln_w = nullptr, * emb_ln_b = nullptr;
    std::vector<bert_layer_w> ws;

    ggml_context * wctx = nullptr;
    gguf_context * gf = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;

    // 图缓存 (同形状复用 ctx+graph+已规划的 galloc): 短文本下"每次建图/建 ctx"是可见的固定开销
    // (enc_p/ref_enc/wns1/DiT/AR 都已是按形状缓存, BERT 之前逐次重建)。
    // 命中时只做 tensor_set + graph_compute; 形状变化才重建 (同时刻只有一张图, galloc 直接复用)。
    ggml_context * gctx = nullptr;
    ggml_cgraph   * gcache = nullptr;
    int64_t         gkey = -1;
    struct gs_t {
        ggml_tensor * ids = nullptr, * pos = nullptr, * typ = nullptr, * msk = nullptr, * out = nullptr;
    } gt;
    void graph_release() {
        gt = gs_t();
        gcache = nullptr;
        gkey = -1;
        if (gctx) { ggml_free(gctx); gctx = nullptr; }
    }

    bool fuse_ln = true, fuse_act = true;   // 融合算子开关 (GSV_NO_FUSE / GSV_NO_FUSE_LN / GSV_NO_FUSE_ACT)
    // [w;b] 打包张量: 0 = embedding LN, 1 = attn_ln, 2 = out_ln (每层)
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;
    std::vector<ggml_tensor *> ln_pack;   // [1 + 2*NL]

    // LN(x + r + bias)*w + b; wb = [w;b] 打包; r/bias 可为 NULL
    ggml_tensor * ln_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * r, ggml_tensor * bias,
                            ggml_tensor * wb, float eps) {
        if (fuse_ln) return ggml_layernorm_affine(ctx, x, r, bias, wb, eps);
        if (r)    x = ggml_add(ctx, x, r);
        if (bias) x = ggml_add(ctx, x, bias);
        const int n = (int) (wb->ne[0] / 2);
        ggml_tensor * w = ggml_view_1d(ctx, wb, n, 0);
        ggml_tensor * b = ggml_view_1d(ctx, wb, n, (size_t) n * 4);
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
    }

    // 查表 + 转 F32 (F16 表的 get_rows 输出为 F16)
    ggml_tensor * emb_row(ggml_context * ctx, ggml_tensor * tbl, ggml_tensor * idx) {
        ggml_tensor * r = ggml_get_rows(ctx, tbl, idx);
        if (r->type != GGML_TYPE_F32) r = ggml_cast(ctx, r, GGML_TYPE_F32);
        return r;
    }

    // 全序列前向: ids [T] -> x [D, T] (行主序 d + t*D); n_layers < 0 = 全部
    void run(const int32_t * ids, int T, std::vector<float> & out, int n_layers);
    // 通用入口: 显式 pos/typ 与可选 S×S F16 mask (nullptr = 全 0), 供多文本批量路径用
    void run_g(const int32_t * ids, const int32_t * pos, const int32_t * typ, int S,
               const ggml_fp16_t * mask_ss, std::vector<float> & out, int n_layers);

    // encode_feat_cached 的 LRU (键 = T + 完整 ids; 命中直接返回存档特征)
    int  feat_cache_n = 0;
    uint64_t feat_use = 0;
    int  fc_hits = 0, fc_misses = 0;
    struct feat_entry { uint64_t h; uint64_t use; std::vector<int32_t> ids; std::vector<float> feat; };
    std::vector<feat_entry> feat_cache;
};

void gsv_bert::impl::run(const int32_t * ids, int T, std::vector<float> & out, int n_layers) {
    std::vector<int32_t> pos((size_t) T), typ((size_t) T, 0);
    for (int i = 0; i < T; i++) pos[i] = i;
    run_g(ids, pos.data(), typ.data(), T, nullptr, out, n_layers);
}

void gsv_bert::impl::run_g(const int32_t * ids, const int32_t * pos, const int32_t * typ, int S,
                           const ggml_fp16_t * mask_ss, std::vector<float> & out, int n_layers) {
    if (S < 1 || S > 65536) { fprintf(stderr, "[gsv_bert] bad S=%d\n", S); return; }
    if (n_layers < 0 || n_layers > NL) n_layers = NL;

    const int64_t key = ((int64_t) S << 16) | (int64_t) (n_layers + 1);
    if (getenv("GSV_BERT_NO_GRAPHCACHE")) graph_release();   // A/B 用: 强制逐次重建
    if (!gcache || key != gkey) {                      // 形状变化: 重建 (否则直接用缓存)
        graph_release();
        ggml_init_params ip = { ggml_tensor_overhead() * 65536, NULL, true };
        gctx = ggml_init(ip);
        ggml_context * ctx = gctx;

    ggml_tensor * t_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);   ggml_set_input(t_ids);
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);   ggml_set_input(t_pos);
    ggml_tensor * t_typ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);   ggml_set_input(t_typ);
    // mask 由调用方给 (批量时是块对角); 默认全 0 = 无屏蔽 (flash_attn_ext 需要 F16 mask);
    // n_layers=0 时注意力不参与图, 不建 mask
    ggml_tensor * t_msk = nullptr;
    if (n_layers > 0) { t_msk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, S, S, 1, 1); ggml_set_input(t_msk); }

    ggml_tensor * cur = ggml_add(ctx, ggml_add(ctx,
                            emb_row(ctx, word_emb, t_ids),
                            emb_row(ctx, pos_emb, t_pos)),
                            emb_row(ctx, type_emb, t_typ));                    // [D, S]
    cur = ln_affine(ctx, cur, nullptr, nullptr, ln_pack[0], EPS);   // embedding: 无残差/bias

    for (int li = 0; li < n_layers; li++) {
        const bert_layer_w & w = ws[li];
        // QKV 一次 matmul + 三个 (HD,S,NH,1) 视图直送 flash:
        // qkv 的行序是 (hd, nh) 且 S 维步长 = 3D*4, head 维步长 = HD*4 -> 无需 permute/cont
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, w.qkv_w, cur), w.qkv_b);   // [3D, S]
        ggml_tensor * q3 = ggml_reshape_3d(ctx, qkv, D, 3, S);                          // nb=(4, 4D, 12D)
        const size_t nb_s = q3->nb[2], nb_h = 4 * HD;
        ggml_tensor * qh = ggml_view_4d(ctx, q3, HD, S, NH, 1, nb_s, nb_h, nb_h * S, 0);
        ggml_tensor * kh = ggml_view_4d(ctx, q3, HD, S, NH, 1, nb_s, nb_h, nb_h * S, q3->nb[1]);
        ggml_tensor * vh = ggml_view_4d(ctx, q3, HD, S, NH, 1, nb_s, nb_h, nb_h * S, 2 * q3->nb[1]);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, qh, kh, vh, t_msk, 1.0f / std::sqrt((float) HD), 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, ggml_reshape_4d(ctx, attn, D, S, 1, 1), D, S);
        ggml_tensor * ao = ggml_mul_mat(ctx, w.attn_out_w, attn);
        cur = ln_affine(ctx, cur, ao, w.attn_out_b, ln_pack[1 + 2*li + 0], EPS);      // post-LN 1
        ggml_tensor * h = fuse_act
            ? ggml_add_act(ctx, ggml_mul_mat(ctx, w.ff1_w, cur), w.ff1_b, GGML_ACT_GELU_ERF)
            : ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.ff1_w, cur), w.ff1_b));
        ggml_tensor * fo = ggml_mul_mat(ctx, w.ff2_w, h);
        cur = ln_affine(ctx, cur, fo, w.ff2_b, ln_pack[1 + 2*li + 1], EPS);           // post-LN 2
    }
        ggml_set_output(cur);

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, cur);
        if (!ggml_gallocr_alloc_graph(galloc, graph))
            fprintf(stderr, "[gsv_bert] alloc_graph failed (S=%d n_layers=%d)\n", S, n_layers);
        gcache = graph;
        gkey   = key;
        gt.ids = t_ids; gt.pos = t_pos; gt.typ = t_typ; gt.msk = t_msk; gt.out = cur;
    }                                                  // 命中: 直接用缓存 (仅上传输入 + compute)

    ggml_tensor * t_ids = gt.ids, * t_pos = gt.pos, * t_typ = gt.typ;
    ggml_tensor * t_msk = gt.msk;
    ggml_tensor * cur   = gt.out;
    ggml_cgraph * graph = gcache;

    ggml_backend_tensor_set(t_ids, ids, 0, (size_t) S * 4);
    ggml_backend_tensor_set(t_pos, pos, 0, (size_t) S * 4);
    ggml_backend_tensor_set(t_typ, typ, 0, (size_t) S * 4);
    if (t_msk) {
        if (mask_ss) {
            ggml_backend_tensor_set(t_msk, mask_ss, 0, (size_t) S * S * 2);
        } else {
            std::vector<ggml_fp16_t> msk((size_t) S * S, ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(t_msk, msk.data(), 0, msk.size() * 2);
        }
    }

    ggml_backend_graph_compute(backend, graph);

    // ggml 布局: idx = d + t*D; 导出为 [D, S] 行主序 (idx = d*S + t)
    auto to_host_td = [&](ggml_tensor * t, std::vector<float> & dst) {
        std::vector<float> tmp(ggml_nelements(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * 4);
        dst.assign(tmp.size(), 0.0f);
        for (int d = 0; d < D; d++)
            for (int i = 0; i < S; i++) dst[(size_t) d * S + i] = tmp[(size_t) d + (size_t) i * D];
    };
    to_host_td(cur, out);
    // ctx/graph/galloc 保留在缓存中 (graph_release / 析构时释放)
}

gsv_bert::gsv_bert() : p(new impl) {}
gsv_bert::~gsv_bert() {
    impl & s = *p;
    if (s.galloc) ggml_gallocr_free(s.galloc);
    s.graph_release();
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.wctx) ggml_free(s.wctx);
    if (s.gf) gguf_free(s.gf);
    delete p;
}

int gsv_bert::hidden()      const { return p->D; }
int gsv_bert::layers_used() const { return p->NL; }
int gsv_bert::max_pos()     const { return p->MAXPOS; }

void gsv_bert::encode(const int32_t * ids, int T, std::vector<float> & features) {
    p->run(ids, T, features, -1);
}

void gsv_bert::encode_feat(const int32_t * ids, int T, std::vector<float> & feat) {
    // 管线后处理: 去掉 [CLS]/[SEP] -> [D, T-2]
    std::vector<float> full;
    p->run(ids, T, full, -1);
    const int T2 = T - 2;
    if (T2 <= 0) { feat.clear(); return; }
    feat.assign((size_t) p->D * T2, 0.0f);
    for (int d = 0; d < p->D; d++)
        for (int i = 0; i < T2; i++) feat[(size_t) d * T2 + i] = full[(size_t) d * T + i + 1];
}

void gsv_bert::encode_layers(const int32_t * ids, int T, std::vector<float> & features,
                             std::vector<std::vector<float>> & per_layer) {
    // 逐层导出: 每层独立建图 (中间张量不是图根, 常驻 arena 会被复用), 语义 = hidden_states[k]
    features.clear();
    per_layer.assign(p->NL + 1, {});
    for (int k = 0; k <= p->NL; k++) {
        std::vector<float> out;
        p->run(ids, T, out, k);
        if (k == p->NL) features = out;
        per_layer[k] = std::move(out);
    }
}

static uint64_t hash_ids(const int32_t * ids, int T) {
    uint64_t h = 1469598103934665603ull;                 // FNV-1a
    for (int i = 0; i < T; i++) { h ^= (uint32_t) ids[i]; h *= 1099511628211ull; }
    h ^= (uint64_t) (uint32_t) T; h *= 1099511628211ull;
    return h;
}

void gsv_bert::encode_feat_cached(const int32_t * ids, int T, std::vector<float> & feat) {
    impl & s = *p;
    if (s.feat_cache_n <= 0) { encode_feat(ids, T, feat); return; }
    const uint64_t h = hash_ids(ids, T);
    for (impl::feat_entry & e : s.feat_cache) {
        if (e.h == h && (int) e.ids.size() == T && memcmp(e.ids.data(), ids, (size_t) T * 4) == 0) {
            feat = e.feat; e.use = ++s.feat_use; s.fc_hits++;
            return;
        }
    }
    s.fc_misses++;
    encode_feat(ids, T, feat);
    if ((int) s.feat_cache.size() >= s.feat_cache_n) {          // LRU 淘汰
        size_t victim = 0;
        for (size_t i = 1; i < s.feat_cache.size(); i++)
            if (s.feat_cache[i].use < s.feat_cache[victim].use) victim = i;
        s.feat_cache.erase(s.feat_cache.begin() + (long) victim);
    }
    impl::feat_entry e;
    e.h = h; e.use = ++s.feat_use;
    e.ids.assign(ids, ids + T);
    e.feat = feat;
    s.feat_cache.push_back(std::move(e));
}

void gsv_bert::feat_cache_stats(int & hits, int & misses) const {
    hits = p->fc_hits; misses = p->fc_misses;
}

bool gsv_bert::encode_feat_batch(int B, const int32_t * ids_flat, const int * lens, int Tmax,
                                 std::vector<std::vector<float>> & feats) {
    impl & s = *p;
    feats.assign((size_t) (B > 0 ? B : 0), {});
    if (B <= 0 || !ids_flat || !lens || Tmax < 3 || Tmax > s.MAXPOS) {
        fprintf(stderr, "[gsv_bert] encode_feat_batch: bad args (B=%d Tmax=%d max_pos=%d)\n", B, Tmax, s.MAXPOS);
        return false;
    }
    const int S = Tmax * B;
    if (S > 65536) { fprintf(stderr, "[gsv_bert] encode_feat_batch: S=%d too large\n", S); return false; }
    // CPU 后端自动回退逐条: padding 浪费在 CPU 上净亏 (T=25×4 实测 0.7×), GPU 上才是赚的 (2.2×)。
    // GSV_BERT_BATCH_CPU=1 可强制走图路径 (测试覆盖用)。
    if (s.is_cpu && getenv("GSV_BERT_BATCH_CPU") == nullptr) {
        for (int b = 0; b < B; b++) encode_feat(ids_flat + (size_t) b * Tmax, lens[b], feats[b]);
        return true;
    }
    std::vector<int32_t> pos((size_t) S, 0), typ((size_t) S, 0);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < Tmax; i++)
            if (i < lens[b]) pos[(size_t) b * Tmax + i] = i;
    // 块对角 mask: 允许 iff (同序列 && key 是真实 token); F16 0 / -inf (与 AR/DiT 的 pad mask 同款)
    std::vector<ggml_fp16_t> msk((size_t) S * S);
    const ggml_fp16_t f_keep = ggml_fp32_to_fp16(0.0f), f_mask = ggml_fp32_to_fp16(-1e30f);
    for (int q = 0; q < S; q++) {
        const int bq = q / Tmax;
        for (int k = 0; k < S; k++)
            msk[(size_t) k + (size_t) S * q] = (k / Tmax == bq && (k % Tmax) < lens[bq]) ? f_keep : f_mask;
    }
    std::vector<float> out;
    s.run_g(ids_flat, pos.data(), typ.data(), S, msk.data(), out, -1);      // out: [D, S], idx = d*S + t
    if ((int) out.size() != s.D * S) { fprintf(stderr, "[gsv_bert] encode_feat_batch: bad out size\n"); return false; }
    for (int b = 0; b < B; b++) {
        const int T2 = lens[b] - 2;
        if (T2 <= 0) continue;
        feats[b].assign((size_t) s.D * T2, 0.0f);
        for (int d = 0; d < s.D; d++)
            for (int i = 0; i < T2; i++)
                feats[b][(size_t) d * T2 + i] = out[(size_t) d * S + (size_t) b * Tmax + i + 1];
    }
    return true;
}

gsv_bert * gsv_bert::load(const std::string & gguf_path, const gsv_bert_cfg & cfg) {
    gsv_bert * m = new gsv_bert();
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
        if (!dev) { fprintf(stderr, "[gsv_bert] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_bert] device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        // coopmat2 默认开 (2026-10-10 复评): T>=256 时快 21~27%、T=25 持平; 精度 max|Δ|
        // 6.6e-3 -> 3.9e-2 (仍 < 5e-2 阈值, token 稳定余量 ~100x)。见 docs/bert_ggml.md §7。
        // 需要逐位复现旧口径时进程启动前设 GGML_VK_DISABLE_COOPMAT2=1。
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_bert] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.is_cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    s.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.backend));
    s.feat_cache_n = cfg.feat_cache > 0 ? cfg.feat_cache : 0;
    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_bert] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_bert] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_bert] reopen failed\n"); delete m; return nullptr; }
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
                fprintf(stderr, "[gsv_bert] read tensor %s failed\n", tname);
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
    auto kv_f32 = [&](const char * key, float def) {
        const int64_t id = gguf_find_key(s.gf, key);
        return id < 0 ? def : gguf_get_val_f32(s.gf, id);
    };
    s.NL     = kv_u32("bert.layers_used", 22);
    s.D      = kv_u32("bert.hidden", 1024);
    s.NH     = kv_u32("bert.head", 16);
    s.FFD    = kv_u32("bert.inter", 4096);
    s.VOCAB  = kv_u32("bert.vocab", 21128);
    s.MAXPOS = kv_u32("bert.max_pos", 512);
    s.EPS    = kv_f32("bert.ln_eps", 1e-12f);
    s.HD     = s.D / s.NH;

    auto t = [&](const char * n) {
        ggml_tensor * tt = ggml_get_tensor(s.wctx, n);
        if (!tt) fprintf(stderr, "[gsv_bert] missing tensor %s\n", n);
        return tt;
    };
    if (!(s.word_emb  = t("bert.word_emb")))  { delete m; return nullptr; }
    if (!(s.pos_emb   = t("bert.pos_emb")))   { delete m; return nullptr; }
    if (!(s.type_emb  = t("bert.type_emb")))  { delete m; return nullptr; }
    if (!(s.emb_ln_w  = t("bert.emb_ln_w")))  { delete m; return nullptr; }
    if (!(s.emb_ln_b  = t("bert.emb_ln_b")))  { delete m; return nullptr; }
    s.ws.resize(s.NL);
    char buf[128];
    for (int li = 0; li < s.NL; li++) {
        #define GW_L(f, nm) snprintf(buf, sizeof(buf), "bert.l%d." nm, li); if (!(s.ws[li].f = t(buf))) { delete m; return nullptr; }
        GW_L(qkv_w, "qkv_w"); GW_L(qkv_b, "qkv_b");
        GW_L(attn_out_w, "attn_out_w"); GW_L(attn_out_b, "attn_out_b");
        GW_L(attn_ln_w, "attn_ln_w"); GW_L(attn_ln_b, "attn_ln_b");
        GW_L(ff1_w, "ff1_w"); GW_L(ff1_b, "ff1_b"); GW_L(ff2_w, "ff2_w"); GW_L(ff2_b, "ff2_b");
        GW_L(out_ln_w, "out_ln_w"); GW_L(out_ln_b, "out_ln_b");
        #undef GW_L
    }
    // [w;b] 打包 (供 4-src 融合 LN): 0 = embedding LN, 1+2*li+{0,1} = attn_ln / out_ln
    {
        ggml_init_params pp = { ggml_tensor_overhead() * (2 * s.NL + 16), NULL, true };
        s.pctx = ggml_init(pp);
        std::vector<ggml_tensor *> tmp(1 + 2 * s.NL, nullptr);
        for (size_t i = 0; i < tmp.size(); i++)
            tmp[i] = ggml_new_tensor_1d(s.pctx, GGML_TYPE_F32, 2 * s.D);
        s.pbuf = ggml_backend_alloc_ctx_tensors(s.pctx, s.backend);
        if (!s.pbuf) { fprintf(stderr, "[gsv_bert] packed wb buffer alloc failed\n"); delete m; return nullptr; }
        s.ln_pack.resize(1 + 2 * s.NL);
        std::vector<float> buf;
        auto fill = [&](int idx, ggml_tensor * w, ggml_tensor * b) {
            const int n = (int) w->ne[0];
            buf.resize((size_t) 2 * n);
            ggml_backend_tensor_get(w, buf.data(), 0, (size_t) n * 4);
            ggml_backend_tensor_get(b, buf.data() + n, 0, (size_t) n * 4);
            ggml_backend_tensor_set(tmp[idx], buf.data(), 0, (size_t) 2 * n * 4);
            s.ln_pack[idx] = tmp[idx];
        };
        fill(0, s.emb_ln_w, s.emb_ln_b);
        for (int li = 0; li < s.NL; li++) {
            fill(1 + 2*li + 0, s.ws[li].attn_ln_w, s.ws[li].attn_ln_b);
            fill(1 + 2*li + 1, s.ws[li].out_ln_w,  s.ws[li].out_ln_b);
        }
    }
    // 融合算子支持探测 (CPU 恒支持; Vulkan 需 shader 在册)
    {
        ggml_init_params fp = { ggml_tensor_overhead() * 32, NULL, true };
        ggml_context * fctx = ggml_init(fp);
        ggml_tensor * fx = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, 64, 4);
        ggml_tensor * fw = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 64);
        ggml_tensor * fwb = ggml_new_tensor_1d(fctx, GGML_TYPE_F32, 128);   // [w;b] 打包
        ggml_tensor * fln = ggml_layernorm_affine(fctx, fx, nullptr, nullptr, fwb, 1e-5f);
        ggml_tensor * fac = ggml_add_act(fctx, fx, fw, GGML_ACT_GELU_ERF);
        const bool sup = ggml_backend_supports_op(s.backend, fln) && ggml_backend_supports_op(s.backend, fac);
        ggml_free(fctx);
        s.fuse_ln = s.fuse_act = sup;
    }
    if (getenv("GSV_NO_FUSE"))     s.fuse_ln = s.fuse_act = false;
    if (getenv("GSV_NO_FUSE_LN"))  s.fuse_ln = false;
    if (getenv("GSV_NO_FUSE_ACT")) s.fuse_act = false;

    if (cfg.verbose)
        printf("[gsv_bert] fused ops: ln=%s act=%s\n", s.fuse_ln ? "on" : "off", s.fuse_act ? "on" : "off");
    if (cfg.verbose)
        printf("[gsv_bert] loaded %s: D=%d head=%d layers=%d inter=%d vocab=%d max_pos=%d eps=%g\n",
               gguf_path.c_str(), s.D, s.NH, s.NL, s.FFD, s.VOCAB, s.MAXPOS, (double) s.EPS);
    return m;
}
