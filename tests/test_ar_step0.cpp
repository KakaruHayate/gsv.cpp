// GSV AR (Text2SemanticDecoder) ggml graph + golden 对拍
// 结构: post-LN transformer, ReLU FFN, fused QKV, learnable-alpha sine PE
// 对拍目标: tests/golden/ar.step0.* (torch dump_golden_ar.py 导出)
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct ar_hparams {
    int32_t D = 512;        // hidden dim
    int32_t n_head = 16;
    int32_t n_layer = 24;
    int32_t vocab = 1025;
    int32_t phones_vocab = 732;
    int32_t eos = 1024;
    int32_t bert_dim = 1024;
};

struct ar_tensor_map {
    ggml_tensor * text_emb;
    ggml_tensor * audio_emb;
    ggml_tensor * bert_proj;    ggml_tensor * bert_proj_b;
    ggml_tensor * text_pe_alpha;
    ggml_tensor * audio_pe_alpha;
    ggml_tensor * predict;
    std::vector<ggml_tensor *> qkv_w, qkv_b, out_w, out_b;
    std::vector<ggml_tensor *> norm1_w, norm1_b, norm2_w, norm2_b;
    std::vector<ggml_tensor *> ffn1_w, ffn1_b, ffn2_w, ffn2_b;
};

struct sine_pe_table {
    // torch SinePositionalEmbedding: pe[0::2]=sin(pos*div), pe[1::2]=cos(pos*div)
    static void compute(float * dst, int T, int D) {
        for (int pos = 0; pos < T; pos++) {
            for (int i = 0; i < D; i += 2) {
                float div = std::exp(i * -(std::log(10000.0f) / (float)D));
                dst[pos * D + i]     = std::sin(pos * div);
                dst[pos * D + i + 1] = std::cos(pos * div);
            }
        }
    }
};

// pe 行主序 [T, D] -> ggml [D, T] 列主序: 直接按上面顺序写即得到 ggml 视图的
// [ne0=D, ne1=T] 张量（每个位置连续 D 个特征）。上面 compute 已经是
// dst[pos*D + i] 布局，即 ggml ne0=D（特征最快），正确。

static ggml_tensor * build_pe(ggml_context * ctx, int T, int D) {
    ggml_tensor * pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    float * data = (float *) pe->data;
    sine_pe_table::compute(data, T, D);
    return pe;
}

int main(int argc, char ** argv) {
    std::string model_path = argc > 1 ? argv[1] : "models/gsv-ar-f32.gguf";
    std::string golden_dir = argc > 2 ? argv[2] : "tests/golden";

    ar_hparams hp;
    ar_tensor_map m;

    // ---- load gguf ----
    ggml_context * wctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ false, /*ctx*/ &wctx };  // 直接读入数据(CPU 对拍够用)
    gguf_context * gf = gguf_init_from_file(model_path.c_str(), ip);
    if (!gf) { fprintf(stderr, "failed to open %s\n", model_path.c_str()); return 1; }
    auto t = [&](const char * name) {
        ggml_tensor * tt = ggml_get_tensor(wctx, name);
        if (!tt) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
        return tt;
    };
    m.text_emb = t("ar.text_emb");
    m.audio_emb = t("ar.audio_emb");
    m.bert_proj = t("ar.bert_proj");
    m.bert_proj_b = t("ar.bert_proj_b");
    m.text_pe_alpha = t("ar.text_pe_alpha");
    m.audio_pe_alpha = t("ar.audio_pe_alpha");
    m.predict = t("ar.predict");
    char buf[128];
    for (int li = 0; li < hp.n_layer; li++) {
        snprintf(buf, sizeof(buf), "transformer.block%d.qkv_w", li); m.qkv_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.qkv_b", li); m.qkv_b.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.out_w", li); m.out_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.out_b", li); m.out_b.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.norm1_w", li); m.norm1_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.norm1_b", li); m.norm1_b.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.ffn1_w", li); m.ffn1_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.ffn1_b", li); m.ffn1_b.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.ffn2_w", li); m.ffn2_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.ffn2_b", li); m.ffn2_b.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.norm2_w", li); m.norm2_w.push_back(t(buf));
        snprintf(buf, sizeof(buf), "transformer.block%d.norm2_b", li); m.norm2_b.push_back(t(buf));
    }

    // ---- 读取 golden 输入 ----
    auto read_bin = [&](const char * name) -> std::vector<float> {
        std::string path = golden_dir + "/" + name + ".bin";
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "missing golden %s\n", path.c_str()); exit(1); }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<float> v(sz / 4);
        if (fread(v.data(), 4, v.size(), f) != v.size()) exit(1);
        fclose(f);
        return v;
    };

    const int x_len = 32;      // phones
    const int y_len = 24;      // prompt
    const int D = hp.D;
    int recalc_n = 0;
    if (const char * rc = getenv("GSV_AR_RECALC")) recalc_n = atoi(rc);
    const int S = x_len + y_len + (recalc_n > 0 ? recalc_n : 0);

    auto xy_pos_g = read_bin("ar.xy_pos");     // [1, S, D]
    auto attn_g   = read_bin("ar.xy_attn_mask");
    auto dec_g    = read_bin("ar.step0.dec");
    auto logits_g = read_bin("ar.step0.logits");

    // 整段重算模式: GSV_AR_RECALC=<n_new> (1..3) — 新 token embedding+PE 在 C++ 侧生成,
    // 拼接到 xy_pos, mask 扩展, 对拍 ar.recalc{n}.logits
    std::vector<float> xy_ext;
    std::vector<float> mask_ext;
    std::vector<float> logits_ext;
    if (recalc_n > 0) {
        char pb[128];
        // 1) audio embedding 查表 + PE 行 (alpha 缩放), 拼 n_new 个 token
        ggml_tensor * pe_aud = nullptr; // 由 build_pe 生成, 这里直接生成 host 表
        static std::vector<float> pe_tab;    // [T_max, D] 行主序 (ne0=D)
        const int T_max = 4096;
        pe_tab.resize((size_t)T_max * D);
        {
            for (int pos = 0; pos < T_max; pos++)
                for (int i = 0; i < D; i += 2) {
                    float div = std::exp(i * -(std::log(10000.0f) / (float)D));
                    pe_tab[(size_t)pos*D + i]     = std::sin(pos * div);
                    pe_tab[(size_t)pos*D + i + 1] = std::cos(pos * div);
                }
        }
        float alpha = 0;
        { // 读 alpha (F32 标量)
            ggml_tensor * al = t("ar.audio_pe_alpha");
            alpha = *(const float *)al->data;
        }
        auto audio_emb = t("ar.audio_emb"); // [1025, 512] ne0=D? gguf shape [1025 512] → ne0=1025?
        // audio_emb: torch weight [1025,512] row-major; ggml ne0=1025 (行=vocab), ne1=512
        // 查表: token idx → 行 idx, 长度 512, 步长 nb0=4? ne0=1025 是 vocab → 查表按 ne[0]?
        // 我们写 GGUF 时直接写 [1025,512] → ne0=1025(vocab), ne1=512(dim); get_rows 需要 ne0=行内长度。
        // 查表期望 emb[i] = W[tok, i] → W 行主序 [1025,512], ggml 视图 (512, 1025): ne0=512。
        // 简单起见: host 手工查表。
        auto new_tokens = read_bin("ar.dec_tokens");
        xy_ext = xy_pos_g;
        xy_ext.resize((size_t)(56 + recalc_n) * D);
        for (int n = 0; n < recalc_n; n++) {
            int tok = (int) new_tokens[n];
            const float * row = audio_emb->data ? nullptr : nullptr; // no_alloc=false → data 有效
            const float * wmem = (const float *) audio_emb->data;
            // ggml ne=(1025, 512): 元素 (v, d) at mem[v + d*1025]?? ne0=1025 fastest → mem[v*512? no!
            // ggml ne0 fastest: mem[(size_t)v * nb0 + d * nb1] 其中 nb0=4, nb1=1025*4
            // → 行主 [1025,512] 的 mem[v*1025 + d]?? 需确认: numpy [1025,512] row-major
            //   mem = v*512 + d (v 行, d 列)。ggml ne=(1025,512): ne0=1025 (v) fastest → mem[v + d*1025]。
            //   两者不同! numpy [1025,512]: 行 v 连续 512 → mem[v*512+d]。
            //   ggml (ne0=1025, ne1=512) 读 mem[v + d*1025] → 是 numpy 的转置读取!
            // 所以 GGUF 记录 shape [1025 512] 时, C++ 读到的是"转置表"。查表正确方式:
            //   emb[d] = mem[d*1025 + tok] (nb1=1025*4)
            const float * base = (const float *)audio_emb->data;
            // GGUF numpy [1025,512] row-major: emb[d] = base[tok*512 + d]
            for (int d = 0; d < D; d++) {
                float e = base[(size_t)tok * D + d];
                float pe = pe_tab[(size_t)(24 + n)*D + d];
                xy_ext[(size_t)(56 + n)*D + d] = e + alpha * pe;
            }
        }
        // 2) mask 扩展: 拷贝原始 56x56; 新 token 行 causal; 旧 query 对新 token 全可见
        mask_ext.assign((size_t)(56 + recalc_n) * (56 + recalc_n), 0.0f);
        for (int a = 0; a < 56; a++)
            for (int b = 0; b < 56; b++)
                mask_ext[(size_t)a * (56 + recalc_n) + b] = attn_g[(size_t)a * 56 + b];
        for (int n = 0; n < recalc_n; n++) {
            int a = 56 + n;  // 新 token 的 query 行 (绝对位置)
            for (int b = 0; b < 56 + recalc_n; b++)
                mask_ext[(size_t)a * (56 + recalc_n) + b] = (b <= a) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
        // 注意: y 段(含新 token)行对 x 列也可见 — 与官方 decode 一致 (decode 无 mask)。
        // 但上面拷贝的 56x56 原始 mask 中 y 行对 x 列已经是 0(可见) ✓
        for (int a = 0; a < 56; a++)
            for (int n = 0; n < recalc_n; n++)
                // x 段 query 屏蔽 y 列 (torch: x_attn pad True); y 段 query causal 可见
                mask_ext[(size_t)a * (56 + recalc_n) + 56 + n] = (a < 32) ? -std::numeric_limits<float>::infinity() : 0.0f;
    }

    // ---- build graph ----
    const bool no_attn_mode = getenv("GSV_AR_NO_ATTN") != nullptr;
    const int n_node_reserve = 8192;
    ggml_init_params gip = { ggml_tensor_overhead() * n_node_reserve + ggml_graph_overhead_custom(n_node_reserve, false), NULL, true };
    ggml_context * ctx = ggml_init(gip);
    ggml_tensor * xy_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S);
    ggml_tensor * attn   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S);  // ne0=n_kv(列=被看方), ne1=n_batch/q
    ggml_tensor * attn16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
    ggml_tensor * out    = xy_pos;  // placeholder, will build forward

    // post-LN forward: 每层 qkv -> fca -> out -> residual -> LN -> ffn -> residual -> LN
    ggml_tensor * cur = xy_pos;
    const char * dbg = getenv("GSV_AR_DEBUG_LAYERS");
    int n_layers_run = dbg ? atoi(dbg) : hp.n_layer;
    for (int li = 0; li < n_layers_run; li++) {
        // fused qkv: weight [3D, D] -> ggml [D, 3D]; mul_mat(qkv_w, x) = [3D, S]
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, m.qkv_w[li], cur), m.qkv_b[li]);
        // split [3D,S] -> q [D,S] view? qkv ne1 = S; need [S, 3, D] reshape
        ggml_tensor * qkv3 = ggml_reshape_3d(ctx, qkv, D, 3, S);
        ggml_tensor * q = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], 0);
        ggml_tensor * k = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], qkv3->nb[1]);
        ggml_tensor * v = ggml_view_2d(ctx, qkv3, D, S, qkv3->nb[2], 2 * qkv3->nb[1]);
        q = ggml_cont(ctx, q); k = ggml_cont(ctx, k); v = ggml_cont(ctx, v);
        // heads: [D, S] -> [hd, n_head, S]
        const int hd = D / hp.n_head;
        // flash_attn_ext 输入约定: q [hd, S_q, nh], k/v [hd, S_kv, nh] (!! not transposed !!)
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, hd, hp.n_head, S), 0, 2, 1, 3));  // (hd,S,nh)
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, hd, hp.n_head, S), 0, 2, 1, 3));
        ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, hd, hp.n_head, S), 0, 2, 1, 3));
        ggml_tensor * attn_out;
        {
            if (no_attn_mode) {
                attn_out = ggml_scale(ctx, cur, 0.0f);   // 恒 0, 旁路 attention (纯图内路径)
            } else {
                attn_out = ggml_flash_attn_ext(ctx, qh, kh, vh, ggml_cast(ctx, attn, GGML_TYPE_F16), 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
                // res ne = (hd, S_q, nh): fold ne0*ne1 -> D
                attn_out = ggml_reshape_2d(ctx, attn_out, D, S);
            }
        }
        ggml_tensor * o = ggml_add(ctx, ggml_mul_mat(ctx, m.out_w[li], attn_out), m.out_b[li]);
        cur = ggml_add(ctx, cur, o);                       // x = x + attn
        ggml_tensor * n1 = ggml_norm(ctx, cur, 1e-5f);     // LN
        n1 = ggml_mul(ctx, n1, m.norm1_w[li]);
        n1 = ggml_add(ctx, n1, m.norm1_b[li]);
        cur = n1;                                          // post-LN: x = LN(x+attn)
        // ffn
        ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, m.ffn1_w[li], cur), m.ffn1_b[li]);
        h = ggml_relu(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, m.ffn2_w[li], h), m.ffn2_b[li]);
        cur = ggml_add(ctx, cur, h);                       // x = x + ffn
        ggml_tensor * n2 = ggml_norm(ctx, cur, 1e-5f);
        n2 = ggml_mul(ctx, n2, m.norm2_w[li]);
        n2 = ggml_add(ctx, n2, m.norm2_b[li]);
        cur = n2;                                          // x = LN(x+ffn)
    }
    out = cur;
    ggml_tensor * logits = ggml_mul_mat(ctx, m.predict, out);  // [vocab, S] 行对应位置
    ggml_tensor * last_logits;
    {
        // 支持对拍任意 token 的 logits: GSV_AR_DEBUG_TOKEN=<idx>
        const char * dtok = getenv("GSV_AR_DEBUG_TOKEN");
        int tok = dtok ? atoi(dtok) : S - 1;
        last_logits = ggml_view_1d(ctx, logits, hp.vocab, (int64_t)tok * logits->nb[1]);
    }
    ggml_set_input(xy_pos);
    if (!no_attn_mode) {
        ggml_set_input(attn);
        ggml_set_input(attn16);
    }
    ggml_set_output(last_logits);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_node_reserve, false);
    ggml_build_forward_expand(graph, last_logits);

    // ---- backend ----
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, graph);

    // 写输入
    const float * xy_in = recalc_n > 0 ? xy_ext.data() : xy_pos_g.data();
    ggml_backend_tensor_set(xy_pos, xy_in, 0, (size_t)S * D * 4);
    if (!no_attn_mode) {
        const float * mask_in = recalc_n > 0 ? mask_ext.data() : attn_g.data();
        ggml_backend_tensor_set(attn, mask_in, 0, (size_t)S * S * 4);
    }

    // bypass 模式: 清零所有额外的 F32 input (GSV_AR_NO_ATTN 的 attn_out)
    fprintf(stderr, "[dbg] graph nodes=%d\n", ggml_graph_n_nodes(graph));
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        ggml_tensor * nd = ggml_graph_node(graph, i);
        if (i < 3) fprintf(stderr, "[dbg] node%d flags=%x type=%d\n", i, nd->flags, nd->type);
        if ((nd->flags & GGML_TENSOR_FLAG_INPUT) && nd != xy_pos && nd != attn && nd != attn16 && nd->type == GGML_TYPE_F32) {
            std::vector<float> z(ggml_nbytes(nd)/4, 0.0f);
            ggml_backend_tensor_set(nd, z.data(), 0, ggml_nbytes(nd));
            fprintf(stderr, "[dbg] zeroed bypass input %p (%lld bytes)\n", (void*)nd, (long long)ggml_nbytes(nd));
        }
    }

    ggml_backend_graph_compute(backend, graph);

    std::vector<float> got(hp.vocab);
    if (recalc_n > 0) {
        char pb2[160];
        snprintf(pb2, sizeof(pb2), "ar.recalc%d.logits", recalc_n);
        logits_ext = read_bin(pb2);
    }
    std::vector<float> & got_ref = (recalc_n > 0 ? logits_ext : logits_g);
    ggml_backend_tensor_get(last_logits, got.data(), 0, got.size() * 4);

    if (getenv("GSV_AR_DEBUG_DUMP")) {
        FILE * df = fopen(getenv("GSV_AR_DEBUG_DUMP"), "wb");
        if (df) { fwrite(got.data(), 4, got.size(), df); fclose(df); }
    }

    // 对比
    double max_diff = 0;
    for (int i = 0; i < hp.vocab; i++) {
        double d = std::fabs(got[i] - got_ref[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("[ar step0] last-token logits max|diff| = %.6g\n", max_diff);
    printf("%s\n", max_diff < 1e-2 ? "PASS" : "FAIL");

    ggml_free(ctx);
    gguf_free(gf);
    return max_diff < 1e-2 ? 0 : 2;
}
