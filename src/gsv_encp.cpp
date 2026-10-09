#include "gsv_encp.h"

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

// enc_p (V5 TextEncoder + MRTE)
//   ssl_proj (1x1 conv 768→192) → Encoder×3 → text_embedding(732→192) → Encoder×6
//   MRTE (c_pre 192→512 → cross-attn heads=4 dk=128 + cpre 残差 + ge → c_post 512→192)
//   Encoder×3 → proj (1x1 conv 192→384) → split m/logs
// Encoder 层 = rel-pos MHA (heads=2, dk=96, window=4) post-LN + ConvFFN(k=3, ReLU) post-LN
//
// rel-pos (window=4, emb_rel_k/v [1, 2WIN+1, dk]) — torch _rel_to_abs/_abs_to_rel 的 ggml 复刻:
//   rel_logits[tq, w] = q[tq]·ek_pad[w] (1/√dk 缩放), ek_pad = emb_k 两侧 pad (T-WIN-1) 行零
//   scores[tq, ki] += _rel_to_abs(rel_logits);  out[tq, d] += _abs_to_rel(p) @ rel_v_pad
//   R = mul_mat(ek, qh) → [WIN2, T] (elem[w, tq]); concat 零 → x_g [2T-1, T]
//     ggml elem(w, tq) = flat[w + tq*(2T-1)] = torch 行主 x[tq, w] ✓
//   _rel_to_abs: concat 尾 1 列零 → flatten → concat 尾 T-1 零 → reshape [2T-1, T+1]
//     → view_2d(T, T, nb1=(2T-1)*4, off=(T-1)*4) → elem[i0, i1] = flat[T-1+i0+i1*(2T-1)]
//     实测 elem == torch rel[tq=i0, ki=i1] → 直接与 sc (同布局) 相加 ✓
//   _abs_to_rel: p (contig [T,T], flat = torch 行主) → reshape [2T-1, T] (含尾 pad 1 列?)
//     → flatten → concat 头 T 零 → reshape [2T, T] → view(2T-1, T, nb1=2T*4, off=4)
//     → elem[w, tq] = flat[1 + w + tq*2T] = torch rw[tq, w] ✓
//   evp = cont(transpose(emb_v)) 两侧 concat 零 → [2T-1, dk] 稠密 (mul_mat src0 需稠密)
//   out_v = mul_mat(evp, rw) → [dk, T] (elem[d, tq]) 加到 oh ✓
// 零常量张量: ggml_new_tensor + set_input, 每次 encode 前清零 (名字 zero_* , build 后枚举).
// mul_mat 的 src0 必须稠密行主 (llamafile 假设) — 头切片/转置先 cont;
// emb 张量 ne=[dk, WIN2] flat[d + w*dk] 恰是 mul_mat 所需 [K=dk, M] 布局 ✓.
// 层内张量布局: x [C, T] 列主 (mul_mat 友好); FFN 内部临时转 t 主.
// split m/logs: stats [384, T] 的 view 非连续 (nb1=384*4), 必须 cont 后取回.

static const int HID = 192, FLT = 768, NHEAD = 2, DK = 96;
static const int WIN = 4, K_FFN = 3;
static const int N_SSL = 3, N_TXT = 6, N_E2 = 3;

struct gsv_encp::impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_t s_back = nullptr;
    gguf_context * gf = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;

    ggml_context * gctx = nullptr;
    ggml_cgraph  * g = nullptr;
    ggml_gallocr_t galloc = nullptr;
    int T = 0, NT = 0;
    ggml_tensor * t_y = nullptr;     // [768, T]
    ggml_tensor * t_text = nullptr;  // I32 [NT]
    ggml_tensor * t_ge = nullptr;    // [512]
    ggml_tensor * t_m = nullptr;     // [192, T]
    ggml_tensor * t_logs = nullptr;  // [192, T]
    ggml_tensor * t_ym = nullptr;    // [192, T] proj 之前的 encoder2 输出 (喂 bridge)

    int n_threads = 0;
    bool verbose = false;

    ggml_tensor * need(const char * n) const {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) { fprintf(stderr, "[gsv_encp] missing %s\n", n); abort(); }
        return t;
    }

    // 自定义 LayerNorm: x [C, T] 沿通道 (ne0) 归一; gamma/beta [C]
    ggml_tensor * lnc(ggml_context * c, ggml_tensor * x, const char * base) {
        char wn[192], bn[192];
        snprintf(wn, sizeof(wn), "%s.gamma", base);
        snprintf(bn, sizeof(bn), "%s.beta", base);
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), need(wn)), need(bn));
    }

    // rel-pos MHA 自注意力: x [HID, T] → 输出 [HID, T]
    std::string zname(const char * what, const char * seg, int li) {
        static char buf[96];
        snprintf(buf, sizeof(buf), "zero_%s_%s_%d", seg, what, li);
        for (char * q = buf; *q; q++) if (*q == '.') *q = '_';
        return std::string(buf);
    }

    ggml_tensor * rel_mha(ggml_context * c, ggml_tensor * x, int T, const char * seg, int li) {
        char nm[192];
        auto W1 = [&](const char * what) {
            snprintf(nm, sizeof(nm), "%s.attn_layers.%d.%s", seg, li, what);
            return need(nm);
        };
        ggml_tensor * q = ggml_add(c, ggml_mul_mat(c, ggml_reshape_2d(c, W1("conv_q.weight"), HID, HID), x), W1("conv_q.bias"));
        ggml_tensor * k = ggml_add(c, ggml_mul_mat(c, ggml_reshape_2d(c, W1("conv_k.weight"), HID, HID), x), W1("conv_k.bias"));
        ggml_tensor * v = ggml_add(c, ggml_mul_mat(c, ggml_reshape_2d(c, W1("conv_v.weight"), HID, HID), x), W1("conv_v.bias"));
        const float scale = 1.0f / std::sqrt((float) DK);

        ggml_tensor * attn_cat = nullptr;
        const bool dbg = getenv("GSV_ENCP_DEBUG") && strcmp(seg, "enc_p.encoder_ssl") == 0 && li == 0;
        for (int h = 0; h < NHEAD; h++) {
            const size_t off = (size_t) h * DK * 4;
            ggml_tensor * qh = ggml_cont(c, ggml_view_2d(c, q, DK, T, (size_t) HID * 4, off));
            ggml_tensor * kh = ggml_cont(c, ggml_view_2d(c, k, DK, T, (size_t) HID * 4, off));
            ggml_tensor * vh = ggml_cont(c, ggml_view_2d(c, v, DK, T, (size_t) HID * 4, off));
            // 主 scores [T_ki, T_tq]
            ggml_tensor * sc = ggml_scale(c, ggml_mul_mat(c, kh, qh), scale);
            // rel_k: torch _rel_to_abs 复刻 — 见文件头注释
            ggml_tensor * ek = W1("emb_rel_k");   // ne=(DK, 2WIN+1) F32, flat[d + w*DK]
            // R = mul_mat(ek, qh): ek flat[d + w*DK] = [K=dk, M=win2] mul_mat 布局 (a[i=w, l=d])
            ggml_tensor * R = ggml_scale(c, ggml_mul_mat(c, ek, qh), scale);    // [WIN2, T]: R[w, tq]
            // pad 两侧 (T-WIN-1) 行零 → x_g [2T-1, T] (torch 行主 x[tq, w])
            ggml_tensor * zl = ggml_new_tensor_2d(c, GGML_TYPE_F32, T - WIN - 1, T);
            ggml_set_name(zl, zname("zl", seg, li).c_str()); ggml_set_input(zl);
            ggml_tensor * xg = ggml_concat(c, zl, ggml_concat(c, R, zl, 0), 0);  // [2T-1, T]
            // _rel_to_abs: pad 尾 1 → flatten → pad 尾 T-1 → reshape [2T-1, T+1] → [:T, T-1:]
            ggml_tensor * z1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T);
            ggml_set_name(z1, zname("z1", seg, li).c_str()); ggml_set_input(z1);
            ggml_tensor * flat = ggml_reshape_1d(c, ggml_concat(c, xg, z1, 0), (int64_t) T * 2 * T);
            ggml_tensor * z2 = ggml_new_tensor_1d(c, GGML_TYPE_F32, T - 1);
            ggml_set_name(z2, zname("z2", seg, li).c_str()); ggml_set_input(z2);
            ggml_tensor * flat2 = ggml_concat(c, flat, z2, 0);                   // [T*2T + T-1]
            ggml_tensor * rs = ggml_reshape_2d(c, flat2, 2 * T - 1, T + 1);
            ggml_tensor * A = ggml_view_2d(c, rs, T, T, (size_t)(2 * T - 1) * 4, (size_t)(T - 1) * 4);
            // numpy dump [i,j] = ggml elem(i0=j, i1=i) = [tq=i, ki=j]; 实测 A == torch rel([tq,ki]) ✓
            A = ggml_cont(c, A);                                                 // [ki, tq]
            sc = ggml_add(c, sc, A);
            ggml_tensor * p = ggml_soft_max(c, sc);
            if (dbg) {
                ggml_set_output(qh); ggml_set_name(qh, "dbg_qh0");
                ggml_set_output(kh); ggml_set_name(kh, "dbg_kh0");
                ggml_set_output(vh); ggml_set_name(vh, "dbg_vh0");
                ggml_set_output(sc); ggml_set_name(sc, "dbg_sc_h0");
                ggml_set_output(A);  ggml_set_name(A, "dbg_A");
            }
            // rel_v: rw = _abs_to_rel(p), out += rw @ rel_v_pad — 见文件头注释
            // emb_rel_v 可能被量化为 F16/Q8_0, 这里 cast 回 F32 再 concat (拼接要求同类型)
            ggml_tensor * ev = ggml_cast(c, W1("emb_rel_v"), GGML_TYPE_F32);     // [DK, WIN2] F32
            // rel_v_pad: 两侧 pad (T-WIN-1) 行零 → evp ne=[2T-1, DK]:
            // mul_mat(a=evp, b=rw): a[i=d, l=w'] = flat[l + i*(2T-1)] = flat[w' + d*(2T-1)]
            // → evp[e0=w', e1=d]?? a ne=[K=2T-1, M=DK]: flat[l + i*K]: i=d, l=w' → flat[w' + d*(2T-1)]
            // evT [WIN2, DK] flat[w + d*WIN2]: concat 沿 ne0 pad 后 ne=[2T-1, DK] 
            // flat[w' + d*(2T-1)] 其中中心区 = evT flat[w + d*9] → w' 平移后: 
            // concat(z, evT, z, 0) 在 ne0 方向拼接: 每行 d 的 [zeros; ev 行; zeros] → flat[w' + d*(2T-1)] ✓
            ggml_tensor * evT = ggml_cont(c, ggml_transpose(c, ev));             // [WIN2, DK] 真实重排
            ggml_tensor * zp2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, T - WIN - 1, DK);
            ggml_set_name(zp2, zname("zp2", seg, li).c_str()); ggml_set_input(zp2);
            ggml_tensor * evp = ggml_concat(c, zp2, ggml_concat(c, evT, zp2, 0), 0);   // [2T-1, DK] cont
            // p flat[ki + tq*T] = torch p_t[tq, ki] 行主 flat[tq*T + ki] ✓ 直接复刻
            // _abs_to_rel: pad right (T-1) → flatten → pad left T → reshape [2T, T] → [:, 1:]
            ggml_tensor * zp = ggml_new_tensor_2d(c, GGML_TYPE_F32, T - 1, T);
            ggml_set_name(zp, zname("zp", seg, li).c_str()); ggml_set_input(zp);
            ggml_tensor * flatp = ggml_reshape_1d(c, ggml_concat(c, p, zp, 0), (int64_t) T * (2 * T - 1));
            ggml_tensor * z3 = ggml_new_tensor_1d(c, GGML_TYPE_F32, T);
            ggml_set_name(z3, zname("z3", seg, li).c_str()); ggml_set_input(z3);
            ggml_tensor * flatp2 = ggml_concat(c, z3, flatp, 0);                 // [T*2T]
            ggml_tensor * rsp = ggml_reshape_2d(c, flatp2, 2 * T, T);            // [2T, T]
            ggml_tensor * rw = ggml_view_2d(c, rsp, 2 * T - 1, T, (size_t)(2 * T) * 4, 4);
            rw = ggml_cont(c, rw);                                               // [w, tq]
            ggml_tensor * vhT = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));   // [T_ki, dk] 稠密
            ggml_tensor * ov = ggml_mul_mat(c, evp, rw);                         // [DK, T]
            ggml_tensor * oh = ggml_add(c, ggml_mul_mat(c, vhT, p), ov);         // [dk, T]
            if (dbg) { ggml_set_output(oh); ggml_set_name(oh, "dbg_oh_h0"); }
            attn_cat = attn_cat ? ggml_concat(c, attn_cat, oh, 0) : oh;
        }
        // conv_o (循环外: 对 concat 后的 [HID, T])
        return ggml_add(c, ggml_mul_mat(c, ggml_reshape_2d(c, W1("conv_o.weight"), HID, HID), attn_cat), W1("conv_o.bias"));
    }

    // k=3 same-pad ConvFFN: x [C, T] 列主 → 输出 [C, T]
    ggml_tensor * conv_ffn(ggml_context * c, ggml_tensor * x, const char * seg, int li) {
        // im2col_fast_1d 要求 b 的 ne = [T, IC, 1] (时间在 ne0)! 
        // x [C, T] 列主 → 转 t 主 [T, C] (cont, 一次拷贝)
        char n1w[192], n1b[192], n2w[192], n2b[192];
        snprintf(n1w, sizeof(n1w), "%s.ffn_layers.%d.conv_1.weight", seg, li);
        snprintf(n1b, sizeof(n1b), "%s.ffn_layers.%d.conv_1.bias", seg, li);
        snprintf(n2w, sizeof(n2w), "%s.ffn_layers.%d.conv_2.weight", seg, li);
        snprintf(n2b, sizeof(n2b), "%s.ffn_layers.%d.conv_2.bias", seg, li);
        ggml_tensor * xt = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));   // [T, C] (ne0=T)
        ggml_tensor * w1 = need(n1w);   // ne=[K, HID, FLT]
        ggml_tensor * im = ggml_im2col_fast_1d(c, w1, xt, 1, K_FFN / 2, 1, GGML_TYPE_F32, 1);
        ggml_tensor * mm = ggml_mul_mat(c,
                ggml_reshape_2d(c, im, im->ne[0], im->ne[1] * im->ne[2]),
                ggml_reshape_2d(c, w1, w1->ne[0] * w1->ne[1], w1->ne[2]));
        // mm ne = (a.ne1=OL*N, b.ne1=OC)?? — conv_1d 的 result reshape [OL, OC, N] (ne0=OL)
        ggml_tensor * y = ggml_reshape_3d(c, mm, im->ne[1], w1->ne[2], im->ne[2]);   // [OL, OC, N]
        y = ggml_add(c, y, ggml_reshape_3d(c, need(n1b), 1, w1->ne[2], 1));
        y = ggml_relu(c, y);
        ggml_tensor * w2 = need(n2w);   // ne=[K, FLT, HID]
        ggml_tensor * im2 = ggml_im2col_fast_1d(c, w2, y, 1, K_FFN / 2, 1, GGML_TYPE_F32, 1);
        ggml_tensor * mm2 = ggml_mul_mat(c,
                ggml_reshape_2d(c, im2, im2->ne[0], im2->ne[1] * im2->ne[2]),
                ggml_reshape_2d(c, w2, w2->ne[0] * w2->ne[1], w2->ne[2]));
        ggml_tensor * o = ggml_reshape_3d(c, mm2, im2->ne[1], HID, im2->ne[2]);       // [OL, HID, N]
        o = ggml_add(c, o, ggml_reshape_3d(c, need(n2b), 1, HID, 1));
        // 回列主 [HID, T]
        o = ggml_cont(c, ggml_permute(c, o, 1, 0, 2, 3));
        return o;
    }

    // Encoder: n_layers 层
    ggml_tensor * encoder(ggml_context * c, ggml_tensor * x, int T, const char * seg, int n_layers) {
        for (int li = 0; li < n_layers; li++) {
            char base[160];
            ggml_tensor * y = rel_mha(c, x, T, seg, li);
            if (getenv("GSV_ENCP_DEBUG") && strcmp(seg, "enc_p.encoder_ssl") == 0) {
                ggml_set_output(y); ggml_set_name(y, (std::string("dbg_ssl_attn") + std::to_string(li)).c_str());
            }
            snprintf(base, sizeof(base), "%s.norm_layers_1.%d", seg, li);
            x = lnc(c, ggml_add(c, x, y), base);
            y = conv_ffn(c, x, seg, li);
            snprintf(base, sizeof(base), "%s.norm_layers_2.%d", seg, li);
            x = lnc(c, ggml_add(c, x, y), base);
        }
        return x;
    }

    // MRTE: cross-attn(hidden 512, heads 4, dk 128, 无 rel) + ssl 残差 + ge
    ggml_tensor * mrte(ggml_context * c, ggml_tensor * y, ggml_tensor * text_enc, ggml_tensor * ge, int T, int NT) {
        ggml_tensor * cpre = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.c_pre.weight"), 192, 512), y),
                need("enc_p.mrte.c_pre.bias"));                                    // [512, T]
        ggml_tensor * tpre = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.text_pre.weight"), 192, 512), text_enc),
                need("enc_p.mrte.text_pre.bias"));                                 // [512, NT]
        ggml_tensor * attn = nullptr;
        // cross_attention 内部还有 conv_q/k/v (对 cpre/tpre 各做 1x1, torch MHA.forward 里 conv_q(x)/conv_k(c))
        ggml_tensor * qall = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.cross_attention.conv_q.weight"), 512, 512), cpre),
                need("enc_p.mrte.cross_attention.conv_q.bias"));                   // [512, T]
        ggml_tensor * kall = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.cross_attention.conv_k.weight"), 512, 512), tpre),
                need("enc_p.mrte.cross_attention.conv_k.bias"));                   // [512, NT]
        ggml_tensor * vall = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.cross_attention.conv_v.weight"), 512, 512), tpre),
                need("enc_p.mrte.cross_attention.conv_v.bias"));                   // [512, NT]
        for (int h = 0; h < 4; h++) {
            const size_t off = (size_t) h * 128 * 4;
            ggml_tensor * qh = ggml_cont(c, ggml_view_2d(c, qall, 128, T, (size_t) 512 * 4, off));
            ggml_tensor * kh = ggml_cont(c, ggml_view_2d(c, kall, 128, NT, (size_t) 512 * 4, off));
            ggml_tensor * vh = ggml_cont(c, ggml_view_2d(c, vall, 128, NT, (size_t) 512 * 4, off));
            ggml_tensor * sc = ggml_scale(c, ggml_mul_mat(c, kh, qh), 1.0f / std::sqrt((float) 128));
            ggml_tensor * p = ggml_soft_max(c, sc);
            ggml_tensor * vhT = ggml_cont(c, ggml_permute(c, vh, 1, 0, 2, 3));   // [NT, 128]
            ggml_tensor * oh = ggml_mul_mat(c, vhT, p);   // out[128, T]: sum_ki vh[d, ki] p[ki, tq]
            attn = attn ? ggml_concat(c, attn, oh, 0) : oh;
        }
        ggml_tensor * o = ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.cross_attention.conv_o.weight"), 512, 512), attn),
                need("enc_p.mrte.cross_attention.conv_o.bias"));
        o = ggml_add(c, ggml_add(c, o, cpre), ge);                                  // + ssl_enc + ge
        return ggml_add(c,
                ggml_mul_mat(c, ggml_reshape_2d(c, need("enc_p.mrte.c_post.weight"), 512, 192), o),
                need("enc_p.mrte.c_post.bias"));                                    // [192, T]
    }

    void build_graph(int Ty, int NText) {
        ggml_init_params ip = { ggml_tensor_overhead() * 4096, NULL, true };
        gctx = ggml_init(ip);
        ggml_context * c = gctx;
        T = Ty; NT = NText;

        t_y = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, T);
        ggml_set_input(t_y);
        t_text = ggml_new_tensor_1d(c, GGML_TYPE_I32, NT);
        ggml_set_input(t_text);
        t_ge = ggml_new_tensor_1d(c, GGML_TYPE_F32, 512);
        ggml_set_input(t_ge);

        // ssl_proj (1x1 conv): 权重 gguf ne=(768, 192, 1) → mul_mat 需 [in=768, out=192]:
        // reader shape [1, 768, 192] → ggml ne 反转 = [192, 768, 1]: reshape_2d(192, 768)? 
        // viewer: ne=[1,768,192] → ne0=1 ✗. reshape_2d(t, 768, 192): 数据 flat[c_in?..]
        // torch [out=192, in=768] 行主 → ggml ne=[in=768, out=192] 列主 = flat[o*768 + i] ✓
        ggml_tensor * sslw = need("enc_p.ssl_proj.weight");   // ne=[1, 768, 192]
        ggml_tensor * sslw2 = ggml_reshape_2d(c, sslw, 768, 192);
        ggml_tensor * y = ggml_add(c, ggml_mul_mat(c, sslw2, t_y), need("enc_p.ssl_proj.bias"));
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(y); ggml_set_name(y, "dbg_y0"); }

        y = encoder(c, y, T, "enc_p.encoder_ssl", N_SSL);
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(y); ggml_set_name(y, "dbg_stage_ssl"); }

        // text embedding: get_rows → [192, NT]
        ggml_tensor * temb = ggml_get_rows(c, need("enc_p.text_embedding.weight"), t_text);   // [192, NT]
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(temb); ggml_set_name(temb, "dbg_stage_temb"); }
        ggml_tensor * text_enc = encoder(c, temb, NT, "enc_p.encoder_text", N_TXT);
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(text_enc); ggml_set_name(text_enc, "dbg_stage_text"); }

        // MRTE
        ggml_tensor * ym = mrte(c, y, text_enc, t_ge, T, NT);
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(ym); ggml_set_name(ym, "dbg_stage_mrte"); }

        // encoder2
        ym = encoder(c, ym, T, "enc_p.encoder2", N_E2);
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(ym); ggml_set_name(ym, "dbg_stage_enc2"); }

        // proj (1x1 conv 192→384): 权重 ne=[1, 192, 384] → reshape [192, 384]
        ggml_tensor * projw = need("enc_p.proj.weight");
        ggml_tensor * projw2 = ggml_reshape_2d(c, projw, 192, 384);
        ggml_tensor * stats = ggml_add(c, ggml_mul_mat(c, projw2, ym), need("enc_p.proj.bias"));   // [384, T]
        if (getenv("GSV_ENCP_DEBUG")) { ggml_set_output(stats); ggml_set_name(stats, "dbg_stage_stats"); }
        // y (proj 前) 条件链里喂 bridge
        t_ym = ym;
        ggml_set_output(t_ym);
        ggml_set_name(t_ym, "encp_y");
        // split: m = stats[:192], logs = stats[192:]
        // cont: view 是非连续的 (nb1=384*4), 取回时必须稠密
        t_m = ggml_cont(c, ggml_view_2d(c, stats, 192, T, stats->nb[1], 0));
        t_logs = ggml_cont(c, ggml_view_2d(c, stats, 192, T, stats->nb[1], (size_t) 192 * 4));
        ggml_set_output(t_m);
        ggml_set_output(t_logs);

        g = ggml_new_graph_custom(c, 8192, false);
        ggml_build_forward_expand(g, t_m);
        ggml_build_forward_expand(g, t_logs);
        if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s_back));
        if (!ggml_gallocr_alloc_graph(galloc, g)) {
            fprintf(stderr, "[gsv_encp] galloc alloc_graph failed (T=%d NT=%d)\n", T, NT);
            ggml_free(gctx); gctx = nullptr; g = nullptr;
            t_m = t_logs = nullptr;
        }
    }
};

gsv_encp::gsv_encp() : p(new impl) {}
gsv_encp::~gsv_encp() {
    impl & s = *p;
    if (s.galloc) ggml_gallocr_free(s.galloc);
    if (s.gctx) ggml_free(s.gctx);
    if (s.backend) ggml_backend_free(s.backend);
    if (s.wbuf) ggml_backend_buffer_free(s.wbuf);
    if (s.gf) gguf_free(s.gf);
    if (s.wctx) ggml_free(s.wctx);
    delete p;
}



gsv_encp * gsv_encp::load(const std::string & gguf_path, const gsv_encp_cfg & cfg) {
    gsv_encp * m = new gsv_encp();
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
        if (!dev) { fprintf(stderr, "[gsv_encp] no usable backend device\n"); delete m; return nullptr; }
        if (cfg.verbose) printf("[gsv_encp] device: %s\n", ggml_backend_dev_name(dev));
    }
    s.backend = ggml_backend_dev_init(dev, nullptr);
    if (!s.backend) { fprintf(stderr, "[gsv_encp] backend init failed\n"); delete m; return nullptr; }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && cfg.n_threads > 0)
        ggml_backend_cpu_set_n_threads(s.backend, cfg.n_threads);
    s.s_back = s.backend;

    gguf_init_params gip = { /*no_alloc*/ true, &s.wctx };
    s.gf = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!s.gf) { fprintf(stderr, "[gsv_encp] failed to open %s\n", gguf_path.c_str()); delete m; return nullptr; }
    s.wbuf = ggml_backend_alloc_ctx_tensors(s.wctx, s.backend);
    if (!s.wbuf) { fprintf(stderr, "[gsv_encp] weight buffer alloc failed\n"); delete m; return nullptr; }
    {
        FILE * fp = fopen(gguf_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "[gsv_encp] reopen failed\n"); delete m; return nullptr; }
        const size_t data_off = gguf_get_data_offset(s.gf);
        const int64_t n_tensors = gguf_get_n_tensors(s.gf);
        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * tname = gguf_get_tensor_name(s.gf, ti);
            if (strncmp(tname, "enc_p.", 6) != 0) continue;   // 只上传本模块的权重
            ggml_tensor * tt = ggml_get_tensor(s.wctx, tname);
            if (!tt) continue;
            const size_t nbytes = ggml_nbytes(tt);
            std::vector<char> buf(nbytes);
            if (fseek(fp, (long)(data_off + gguf_get_tensor_offset(s.gf, ti)), SEEK_SET) != 0 ||
                fread(buf.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "[gsv_encp] read tensor %s failed\n", tname);
                fclose(fp); delete m; return nullptr;
            }
            ggml_backend_tensor_set(tt, buf.data(), 0, nbytes);
        }
        fclose(fp);
    }
    if (cfg.verbose) printf("[gsv_encp] loaded %s\n", gguf_path.c_str());
    return m;
}

bool gsv_encp::encode(const float * y, int T, const int32_t * text, int n_text,
                      const float * ge, std::vector<float> & out_m, std::vector<float> & out_logs,
                      std::vector<float> * out_y) {
    impl & s = *p;
    if (T <= 0 || n_text <= 0) { fprintf(stderr, "[gsv_encp] bad T/NT\n"); return false; }
    if (T != s.T || n_text != s.NT) {
        if (s.galloc) { ggml_gallocr_free(s.galloc); s.galloc = nullptr; }
        if (s.gctx) { ggml_free(s.gctx); s.gctx = nullptr; }
        s.g = nullptr; s.t_y = s.t_text = s.t_ge = s.t_m = s.t_logs = s.t_ym = nullptr;
        s.build_graph(T, n_text);
        if (!s.g) { fprintf(stderr, "[gsv_encp] build_graph failed\n"); return false; }
        if (s.verbose) printf("[gsv_encp] graph rebuilt for T=%d NT=%d\n", T, n_text);
    }
    // y 外部按 torch [C,T] 行主 (flat[c*T + t]) 传入; ggml t_y ne=[768,T] 行主要求
    // flat[i0 + i1*768] = y[t*768 + c] — 需要 [T,C] 行主, 这里转置后写入.
    {
        std::vector<float> yt((size_t) 768 * T);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 768; c++)
                yt[(size_t) t * 768 + c] = y[(size_t) c * T + t];
        ggml_backend_tensor_set(s.t_y, yt.data(), 0, (size_t) 768 * T * 4);
    }
    ggml_backend_tensor_set(s.t_text, text, 0, (size_t) n_text * 4);
    ggml_backend_tensor_set(s.t_ge, ge, 0, (size_t) 512 * 4);
    // rel-pos 用的零常量张量 (pad) — galloc 分配后内容不保证, 每次清零
    {
        std::vector<float> zbuf;
        for (ggml_tensor * z = ggml_get_first_tensor(s.gctx); z; z = ggml_get_next_tensor(s.gctx, z)) {
            if (strncmp(z->name, "zero_", 5) != 0 || !z->buffer) continue;
            zbuf.assign(ggml_nelements(z), 0.0f);
            ggml_backend_tensor_set(z, zbuf.data(), 0, zbuf.size() * 4);
        }
    }
    ggml_backend_graph_compute(s.backend, s.g);
    if (getenv("GSV_ENCP_DEBUG")) {
        // 阶段转储 (对拍用): ggml [C,T] flat[c + t*C] → numpy reshape(T,C).T
        const char * dbg_names[] = { "dbg_y0", "dbg_stage_ssl", "dbg_stage_temb", "dbg_stage_text",
                                     "dbg_stage_mrte", "dbg_stage_enc2", "dbg_stage_stats",
                                     "dbg_ssl_attn0", "dbg_ssl_attn1", "dbg_ssl_attn2" };
        for (const char * nm : dbg_names) {
            ggml_tensor * t = ggml_get_tensor(s.gctx, nm);
            if (!t) continue;
            std::vector<float> buf(ggml_nelements(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
            std::string fn = std::string("tests/golden/_mine_") + nm + ".bin";
            FILE * fp = fopen(fn.c_str(), "wb");
            if (fp) { fwrite(buf.data(), 4, buf.size(), fp); fclose(fp); }
        }
    }
    out_m.assign((size_t) 192 * T, 0.0f);
    out_logs.assign((size_t) 192 * T, 0.0f);
    // ggml 输出 ne=[192,T] 行主 flat[c + t*192] = torch [T,192]; 转置回 torch [C,T] (c*T + t)
    {
        std::vector<float> tmp((size_t) 192 * T);
        ggml_backend_tensor_get(s.t_m, tmp.data(), 0, tmp.size() * 4);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 192; c++)
                out_m[(size_t) c * T + t] = tmp[(size_t) t * 192 + c];
        ggml_backend_tensor_get(s.t_logs, tmp.data(), 0, tmp.size() * 4);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 192; c++)
                out_logs[(size_t) c * T + t] = tmp[(size_t) t * 192 + c];
        if (out_y) {
            out_y->assign((size_t) 192 * T, 0.0f);
            ggml_backend_tensor_get(s.t_ym, tmp.data(), 0, tmp.size() * 4);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 192; c++)
                    (*out_y)[(size_t) c * T + t] = tmp[(size_t) t * 192 + c];
        }
    }
    return true;
}
