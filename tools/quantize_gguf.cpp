// GGUF 量化器: F32/F16 权重 → 指定类型 (含 K-quants), 供 AR 量化扫描使用
// 用法: quantize_gguf in.gguf out.gguf --spec "attn=q8_0,ffn=q6_k,predict=f16,emb=f32"
// 组: attn(qkv_w/out_w) ffn(ffn1_w/ffn2_w) predict emb; 其余按 --default (默认 f16)
// 依据: ggml_quantize_chunk + gguf 写出 API (不依赖 llama.cpp 的模型加载器)
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static std::map<std::string, ggml_type> TYPE_MAP = {
    { "f32",  GGML_TYPE_F32  }, { "f16",  GGML_TYPE_F16  }, { "bf16", GGML_TYPE_BF16 },
    { "q8_0", GGML_TYPE_Q8_0 }, { "q6_k", GGML_TYPE_Q6_K }, { "q5_k", GGML_TYPE_Q5_K },
    { "q4_k", GGML_TYPE_Q4_K }, { "q5_1", GGML_TYPE_Q5_1 }, { "q5_0", GGML_TYPE_Q5_0 },
    { "q4_1", GGML_TYPE_Q4_1 }, { "q4_0", GGML_TYPE_Q4_0 }, { "iq4_nl", GGML_TYPE_IQ4_NL },
};

static ggml_type parse_type(const std::string & s) {
    auto it = TYPE_MAP.find(s);
    if (it == TYPE_MAP.end()) { fprintf(stderr, "unknown type %s\n", s.c_str()); exit(1); }
    return it->second;
}

// 与 tools/convert_ar.py / tools/convert_bert.py 的 ft() 保持一致的组划分
static std::string group_of(const std::string & name) {
    const bool is_bert = name.rfind("bert.", 0) == 0;   // BERT encoder 张量 (bert.xxx)
    if (name.find("norm") != std::string::npos || name.find("_ln_") != std::string::npos ||
        (name.size() > 2 && name.compare(name.size() - 2, 2, "_b") == 0) ||
        (name.size() > 2 && name.compare(name.size() - 2, 2, ".b") == 0) ||
        name.find(".bias") != std::string::npos ||
        name.find("alpha") != std::string::npos || name.find("bert_proj") != std::string::npos)
        return "fixed";                        // 强制 F32
    // enc_p / ref_enc 前缀分组要在通用规则 (ffn/emb/attn) 之前, 否则被误分类
    // enc_p (TextEncoder+MRTE): 组供 cond 段扫描用 (bias/gamma/beta 已被 fixed 截获)
    if (name.rfind("enc_p.", 0) == 0) {
        if (name.find("emb_rel") != std::string::npos) return "enc_p.rel";
        if (name.find(".attn_layers.") != std::string::npos) return "enc_p.attn";
        if (name.find(".ffn_layers.") != std::string::npos) return "enc_p.ffn";
        if (name.find("mrte.") != std::string::npos) return "enc_p.mrte";
        if (name.find("text_embedding") != std::string::npos) return "enc_p.emb";
        if (name.find("ssl_proj") != std::string::npos) return "enc_p.sslproj";
        if (name.find("proj.") != std::string::npos) return "enc_p.proj";
        return "enc_p.other";
    }
    // ref_enc (MelStyleEncoder)
    if (name.rfind("ref_enc.", 0) == 0) {
        if (name.find("spectral") != std::string::npos) return "ref_enc.spectral";
        if (name.find("temporal") != std::string::npos) return "ref_enc.temporal";
        if (name.find("slf_attn") != std::string::npos) return "ref_enc.attn";
        if (name.find("fc.fc") != std::string::npos) return "ref_enc.fc";
        return "ref_enc.other";
    }
    if (name.find("text_emb") != std::string::npos || name.find("audio_emb") != std::string::npos) return "emb";
    if (name.find("predict") != std::string::npos) return "predict";
    if (is_bert) {
        if (name.find("_emb") != std::string::npos) return "emb";                    // word/pos/type_emb
        if (name.find("ff1_w") != std::string::npos || name.find("ff2_w") != std::string::npos) return "ffn";
        if (name.find("qkv_w") != std::string::npos || name.find("q_w") != std::string::npos ||
            name.find("k_w") != std::string::npos || name.find("v_w") != std::string::npos ||
            name.find("attn_out_w") != std::string::npos) return "attn";
        return "other";
    }
    if (name.find("ffn") != std::string::npos) return "ffn";
    if (name.find("qkv") != std::string::npos || name.find("out_w") != std::string::npos) return "attn";
    // norm_w / enc_norm_w 参与 MUL (F32 语义) → fixed; *_b 同
    if (name.find("norm_w") != std::string::npos || name.find("enc_norm") != std::string::npos)
        return "fixed";
    // hubert.layer.X.ln1_w/ln2_w 被 layernorm_affine 消费 (要求 F32 输入) → fixed
    if (name.find("hubert.layer.") == 0 &&
        (name.find(".ln1_w") != std::string::npos || name.find(".ln2_w") != std::string::npos))
        return "fixed";
    // HuBERT: feat_conv (conv 组) 与 transformer 头切片 q/k/v/out_w (attn 组)
    // pos_conv.w 强制 F32 — host 端 pos_conv/enc_ln 读它按 float*, F16 会越界
    if (name.find("pos_conv") != std::string::npos) return "fixed";
    if (name.find("feat_conv") != std::string::npos) return "conv";
    if (name.rfind("hubert.layer.", 0) == 0) {
        if (name.find(".q_w") != std::string::npos || name.find(".k_w") != std::string::npos ||
            name.find(".v_w") != std::string::npos || name.find(".out_w") != std::string::npos) return "attn";
        if (name.find(".ffn1_w") != std::string::npos || name.find(".ffn2_w") != std::string::npos) return "ffn";
    }
    if (name.rfind("hubert.", 0) == 0) return "other";
    // enc_p (TextEncoder+MRTE): 组供 cond 段扫描用 (bias/gamma/beta 已被 fixed 截获)
    if (name.rfind("enc_p.", 0) == 0) {
        if (name.find("emb_rel") != std::string::npos) return "enc_p.rel";
        if (name.find(".attn_layers.") != std::string::npos) return "enc_p.attn";
        if (name.find(".ffn_layers.") != std::string::npos) return "enc_p.ffn";
        if (name.find("mrte.") != std::string::npos) return "enc_p.mrte";
        if (name.find("text_embedding") != std::string::npos) return "enc_p.emb";
        if (name.find("ssl_proj") != std::string::npos) return "enc_p.sslproj";
        if (name.find("proj.") != std::string::npos) return "enc_p.proj";
        return "enc_p.other";
    }
    // ref_enc (MelStyleEncoder)
    if (name.rfind("ref_enc.", 0) == 0) {
        if (name.find("spectral") != std::string::npos) return "ref_enc.spectral";
        if (name.find("temporal") != std::string::npos) return "ref_enc.temporal";
        if (name.find("slf_attn") != std::string::npos) return "ref_enc.attn";
        if (name.find("fc.fc") != std::string::npos) return "ref_enc.fc";
        return "ref_enc.other";
    }
    if (name.find("rvq.codebook") != std::string::npos) return "rvq";
    // wns1 / cond 段: 权重类 → wns1 组 (bias/codebook 已被 fixed 截获)
    if (name.rfind("wns1.", 0) == 0) return "wns1";
    if (name.rfind("bridge", 0) == 0) return "bridge";
    return "other";
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.gguf out.gguf [--spec attn=q6_k,...] [--default f16]\n", argv[0]); return 1; }
    const std::string in_path = argv[1], out_path = argv[2];
    std::map<std::string, std::string> spec;
    std::string def = "f16";
    for (int i = 3; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--spec") == 0) {
            std::string s = argv[i + 1];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string kv = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                size_t eq = kv.find('=');
                if (eq != std::string::npos) spec[kv.substr(0, eq)] = kv.substr(eq + 1);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (strcmp(argv[i], "--default") == 0) {
            def = argv[i + 1];
        }
    }

    // ---- 读输入 ----
    ggml_context * in_ctx = nullptr;
    gguf_init_params ip = { /*no_alloc*/ false, &in_ctx };
    gguf_context * in = gguf_init_from_file(in_path.c_str(), ip);
    if (!in) { fprintf(stderr, "failed to open %s\n", in_path.c_str()); return 1; }

    // ---- 建输出 + 复制 KV ----
    gguf_context * out = gguf_init_empty();
    const int64_t n_kv = gguf_get_n_kv(in);
    for (int64_t i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(in, i);
        switch (gguf_get_kv_type(in, i)) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (out, key, gguf_get_val_u8  (in, i)); break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (out, key, gguf_get_val_i8  (in, i)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (out, key, gguf_get_val_u16 (in, i)); break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (out, key, gguf_get_val_i16 (in, i)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (out, key, gguf_get_val_u32 (in, i)); break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (out, key, gguf_get_val_i32 (in, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (out, key, gguf_get_val_f32 (in, i)); break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(out, key, gguf_get_val_bool(in, i)); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (out, key, gguf_get_val_u64 (in, i)); break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (out, key, gguf_get_val_i64 (in, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (out, key, gguf_get_val_f64 (in, i)); break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (out, key, gguf_get_val_str (in, i)); break;
            default:
                fprintf(stderr, "warning: skip kv %s (type %d, arrays not copied)\n", key, (int) gguf_get_kv_type(in, i));
                break;
        }
    }

    // ---- 逐张量量化 ----
    ggml_context * meta_ctx = nullptr;
    {
        ggml_init_params mp = { ggml_tensor_overhead() * 512, nullptr, true };
        meta_ctx = ggml_init(mp);
    }
    const int64_t n_tensors = gguf_get_n_tensors(in);
    std::vector<std::vector<uint8_t>> keep;   // 保持数据存活到写出
    keep.reserve(n_tensors * 2);

    for (int64_t ti = 0; ti < n_tensors; ti++) {
        const char * name = gguf_get_tensor_name(in, ti);
        ggml_tensor * t = ggml_get_tensor(in_ctx, name);
        const std::string grp = group_of(name);
        std::string target_name = (grp == "fixed") ? "f32" : (spec.count(grp) ? spec[grp] : def);
        ggml_type tt = parse_type(target_name);
        int64_t ne0 = t->ne[0], ne1 = t->ne[1];
        int64_t nrows = ggml_nrows(t);

        // ne0==1 的 1x1 conv / Conv1d 权重 ([1, in, out]): 纯元数据压成 2D [in, out]
        // (flat 布局不变), 使量化块能沿 in 维排布。加载端 reshape_2d 对两种形状都成立。
        bool squeeze = false;
        if (t->type == GGML_TYPE_F32 && t->ne[0] == 1 && t->ne[3] == 1 && t->ne[2] > 1) {
            squeeze = true;
            ne0 = t->ne[1]; ne1 = t->ne[2]; nrows = t->ne[2] * t->ne[3];
        }

        // 目标类型不可用时的回退 (K-quant 需要 ne0 % 256 == 0; 量化只支持 2D 权重)
        if (tt != GGML_TYPE_F32 && tt != GGML_TYPE_F16) {
            const int blk = ggml_blck_size(tt);
            if (!squeeze && (t->ne[2] != 1 || t->ne[3] != 1)) {
                fprintf(stderr, "  [warn] %s: 3D 无法量化 (ne=[%lld,%lld,%lld]) → 回退 f16\n",
                        name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
                tt = GGML_TYPE_F16;
            } else if ((ne0 % blk) != 0) {
                fprintf(stderr, "  [warn] %s: ne0=%lld 不是 %d 的倍数 → 回退 f16\n", name, (long long) ne0, blk);
                tt = GGML_TYPE_F16;
            }
        }
        ggml_tensor * meta = squeeze && (tt != GGML_TYPE_F32 && tt != GGML_TYPE_F16)
            ? ggml_new_tensor_2d(meta_ctx, tt, ne0, ne1)
            : ggml_new_tensor(meta_ctx, tt, ggml_n_dims(t), t->ne);
        ggml_set_name(meta, name);
        gguf_add_tensor(out, meta);

        const void * src = t->data;
        if (tt == t->type) {   // 同类型直接复制 (squeeze 只改变量化目标的元数据形状)
            keep.emplace_back((const uint8_t *) src, (const uint8_t *) src + ggml_nbytes(t));
            gguf_set_tensor_data(out, name, keep.back().data());
            printf("  %-40s %s (copy)\n", name, ggml_type_name(tt));
            continue;
        }
        if (tt == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf(ggml_nelements(t));
            ggml_fp32_to_fp16_row((const float *) src, buf.data(), (int64_t) ggml_nelements(t));
            keep.emplace_back((const uint8_t *) buf.data(), (const uint8_t *) buf.data() + buf.size() * 2);
            gguf_set_tensor_data(out, name, keep.back().data());
            printf("  %-40s f16 (from %s)\n", name, ggml_type_name(t->type));
            continue;
        }
        // 量化: 逐行块 (squeeze 时按 [ne1, ne2] 行)
        const size_t dst_size = ggml_row_size(tt, ne0) * nrows;
        std::vector<uint8_t> buf(dst_size);
        const size_t written = ggml_quantize_chunk(tt, (const float *) src, buf.data(), 0, nrows, ne0, nullptr);
        if (written != dst_size) { fprintf(stderr, "  [error] %s: quantize size %zu != %zu\n", name, written, dst_size); return 1; }
        keep.push_back(std::move(buf));
        gguf_set_tensor_data(out, name, keep.back().data());
        printf("  %-40s %s (from %s, ne0=%lld)\n", name, ggml_type_name(tt), ggml_type_name(t->type), (long long) ne0);
    }

    if (!gguf_write_to_file(out, out_path.c_str(), false)) { fprintf(stderr, "write failed\n"); return 1; }
    printf("wrote %s\n", out_path.c_str());
    ggml_quantize_free();
    return 0;
}
