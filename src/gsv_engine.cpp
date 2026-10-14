#include "gsv_engine.h"

#include "gsv_ar.h"
#include "gsv_bert.h"
#include "gsv_cond.h"
#include "gsv_dit.h"
#include "gsv_encp.h"
#include "gsv_hubert.h"
#include "gsv_mel.h"
#include "gsv_refcode.h"
#include "gsv_refenc.h"
#include "gsv_sampler.h"
#include "gsv_vocoder.h"
#include "gsv_wns1.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int MEL_DIM = 100;
static const int TXT_DIM = 512;

struct gsv_engine::impl {
    gsv_engine_cfg cfg;

    // 各段模型 (常驻; 与 demo_e2e.cpp 用的是同一批文件)
    gsv_hubert  * hub  = nullptr;
    gsv_refcode * rc   = nullptr;
    gsv_refenc  * re   = nullptr;
    gsv_bert    * bert = nullptr;
    gsv_ar      * ar   = nullptr;
    gsv_cond    * cond = nullptr;
    gsv_encp    * encp = nullptr;
    gsv_wns1    * wns1 = nullptr;
    gsv_dit     * dit  = nullptr;
    gsv_vocoder * voc  = nullptr;

    std::string md;                       // model_dir (带尾部 '/')
    int  ar_vocab = 1025;
    bool ready = false;

    ~impl() {
        delete voc; delete dit; delete wns1; delete encp; delete cond;
        delete ar; delete bert; delete re; delete rc; delete hub;
    }

    std::string path(const char * leaf) const { return md + leaf; }

    // 一次 decode_encp: codes -> RVQ/x2 -> enc_p -> bridge/x2 -> wns1 -> fea [512, 4*Tc]
    bool decode_encp_run(const std::vector<int32_t> & codes,
                         const std::vector<int32_t> & phones,
                         const std::vector<float> & ge,
                         std::vector<float> & fea) {
        const int Tc = (int) codes.size();
        if (Tc <= 0) return false;
        std::vector<float> q;
        cond->rvq_decode(codes.data(), Tc, true, q);            // [768, 2Tc] (idx = d + t*768)
        const int T = 2 * Tc;
        std::vector<float> y_tc((size_t) 768 * T);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 768; c++) y_tc[(size_t) c * T + t] = q[(size_t) c + (size_t) t * 768];
        std::vector<float> m, logs, yenc;
        if (!encp->encode(y_tc.data(), T, phones.data(), (int) phones.size(), ge.data(),
                          m, logs, &yenc))
            return false;
        std::vector<float> xb((size_t) 192 * T), br;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 192; c++) xb[(size_t) c + (size_t) t * 192] = yenc[(size_t) c * T + t];
        cond->bridge_run(xb.data(), T, true, br);               // [2T, 512] (idx = c*2T + t)
        const int T2 = 2 * T;
        fea.clear();
        return wns1->encode(br.data(), ge.data(), T2, T2, fea);
    }
};

gsv_engine::gsv_engine() : p(new impl) {}
gsv_engine::~gsv_engine() { delete p; }

int gsv_engine::mel_dim()  const { return MEL_DIM; }
int gsv_engine::text_dim() const { return TXT_DIM; }
int gsv_engine::ar_vocab() const { return p->ar_vocab; }

gsv_engine * gsv_engine::load(const gsv_engine_cfg & cfg) {
    gsv_engine * m = new gsv_engine();
    impl & s = *m->p;
    s.cfg = cfg;
    s.md = cfg.model_dir.empty() ? std::string("models/") : cfg.model_dir;
    if (s.md.back() != '/' && s.md.back() != '\\') s.md.push_back('/');
    const std::string & dev = cfg.device;
    const bool vb = cfg.verbose;

    // 参考音频三链
    {
        gsv_hubert_cfg c; c.device = dev; c.n_threads = cfg.n_threads; c.verbose = vb;
        s.hub = gsv_hubert::load(s.path("gsv-hubert-f32.gguf"), c);
        gsv_refcode_cfg r; r.device = dev; r.n_threads = cfg.n_threads; r.verbose = vb;
        s.rc = gsv_refcode::load(s.path("gsv-refcode.gguf"), r);
        gsv_refenc_cfg g; g.device = dev; g.n_threads = cfg.n_threads; g.verbose = vb;
        s.re = gsv_refenc::load(s.path("gsv-cond-f32.gguf"), g);
        if (!s.hub || !s.rc || !s.re) {
            fprintf(stderr, "[gsv_engine] 参考音频链加载失败 (hubert/refcode/refenc)\n");
            delete m; return nullptr;
        }
        // resampy kaiser_best 表 (repo resample(match_librosa=True) 的复刻需要)
        if (!gsv_refcode::load_filter_table(s.path("resampy_kaiser_best.bin"))) {
            fprintf(stderr, "[gsv_engine] %s 缺失 (先跑 tools/export_resampy_filter.py)\n",
                    s.path("resampy_kaiser_best.bin").c_str());
            delete m; return nullptr;
        }
    }
    // BERT (LRU 开一点: 同一文本重复合成是服务化常态)
    {
        gsv_bert_cfg b; b.device = dev; b.n_threads = cfg.n_threads; b.verbose = vb;
        b.feat_cache = 8;
        s.bert = gsv_bert::load(s.path("gsv-bert-f16.gguf"), b);
        if (!s.bert) { fprintf(stderr, "[gsv_engine] BERT 加载失败\n"); delete m; return nullptr; }
    }
    // AR
    {
        gsv_ar_cfg a; a.device = dev; a.n_threads = cfg.n_threads; a.verbose = vb;
        s.ar = gsv_ar::load(s.path("gsv-ar-f32.gguf"), a);
        if (!s.ar) { fprintf(stderr, "[gsv_engine] AR 加载失败\n"); delete m; return nullptr; }
        s.ar_vocab = s.ar->vocab_size();
    }
    // 条件段
    {
        gsv_cond_cfg c; c.device = dev; c.n_threads = cfg.n_threads; c.verbose = vb;
        gsv_encp_cfg e; e.device = dev; e.n_threads = cfg.n_threads; e.verbose = vb;
        gsv_wns1_cfg w; w.device = dev; w.n_threads = cfg.n_threads; w.verbose = vb;
        s.cond = gsv_cond::load(s.path("gsv-cond-f32.gguf"), c);
        s.encp = gsv_encp::load(s.path("gsv-cond-encp.gguf"), e);
        s.wns1 = gsv_wns1::load(s.path("gsv-cond-f32.gguf"), w);
        if (!s.cond || !s.encp || !s.wns1) {
            fprintf(stderr, "[gsv_engine] 条件段加载失败 (cond/encp/wns1)\n");
            delete m; return nullptr;
        }
    }
    // DiT (最小可用档 = f16all; adaln 降 F16 的 f16afa 在真实输入上出 NaN)
    {
        gsv_dit_cfg d; d.device = dev; d.verbose = vb;
        s.dit = gsv_dit::load(s.path("gsv-dit-f16all.gguf"), d);
        if (!s.dit) { fprintf(stderr, "[gsv_engine] DiT 加载失败\n"); delete m; return nullptr; }
    }
    // vocoder (ORT; EP 由 gsv_vocoder 自己挑)
    {
        gsv_vocoder_cfg v; v.device = cfg.voc_device; v.verbose = vb;
        s.voc = gsv_vocoder::load(s.path("vocoder_fp32.onnx"), v);
        if (!s.voc) { fprintf(stderr, "[gsv_engine] vocoder 加载失败\n"); delete m; return nullptr; }
    }
    s.ready = true;
    if (vb)
        printf("[gsv_engine] ready: ggml=%s vocoder_EP=%s\n",
               dev.empty() ? "cpu" : dev.c_str(), s.voc->ep_name());
    return m;
}

bool gsv_engine::prep_reference(const std::string & wav_path,
                                std::vector<int32_t> & prompt_codes,
                                std::vector<float> & ge,
                                std::vector<float> & mel2, int & mel2_T) {
    impl & s = *p;
    prompt_codes.clear(); ge.clear(); mel2.clear(); mel2_T = 0;
    if (!s.ready) return false;

    std::vector<float> raw;
    int sr = 0;
    if (!gsv_wav_load(wav_path, raw, sr)) {
        fprintf(stderr, "[gsv_engine] 读不到 %s\n", wav_path.c_str());
        return false;
    }

    // (a) 16k + 0.3s 零尾 -> HuBERT -> ssl_proj -> RVQ -> codes
    std::vector<float> w16;
    gsv_refcode::resample_librosa(raw.data(), (int64_t) raw.size(), sr, 16000, w16);
    const int zero_tail = (int) (32000 * 0.3);          // repo: int(sampling_rate * 0.3)
    std::vector<float> fed(w16.size() + (size_t) zero_tail, 0.0f);
    if (!w16.empty()) std::memcpy(fed.data(), w16.data(), w16.size() * sizeof(float));
    std::vector<float> hout;
    if (!s.hub->encode(fed.data(), (int) fed.size(), hout)) {
        fprintf(stderr, "[gsv_engine] HuBERT 编码失败\n");
        return false;
    }
    const int Th = (int) (hout.size() / 768);
    std::vector<float> ssl((size_t) 768 * Th);          // ggml [768,T] -> torch [768,T]
    for (int d = 0; d < 768; d++)
        for (int t = 0; t < Th; t++) ssl[(size_t) d * Th + t] = hout[(size_t) d + (size_t) 768 * t];
    if (!s.rc->codes_from_ssl(ssl.data(), Th, prompt_codes)) {
        fprintf(stderr, "[gsv_engine] refcode (ssl_proj + RVQ) 失败\n");
        return false;
    }

    // (b) 32k + maxx 归一化 -> spectrogram_torch -> [:, :704] -> ref_enc -> ge
    std::vector<float> spec704;
    int Ts = 0;
    if (!gsv_refspec_wav(raw.data(), (int64_t) raw.size(), sr, spec704, Ts, true)) {
        fprintf(stderr, "[gsv_engine] refer_audio_spec 失败\n");
        return false;
    }
    std::vector<float> spec_gg((size_t) 704 * Ts), fm((size_t) Ts, 1.0f);
    for (int c = 0; c < 704; c++)
        for (int t = 0; t < Ts; t++) spec_gg[(size_t) c + (size_t) t * 704] = spec704[(size_t) c * Ts + t];
    if (!s.re->encode(spec_gg.data(), fm.data(), Ts, ge)) {
        fprintf(stderr, "[gsv_engine] ref_enc 失败\n");
        return false;
    }

    // (c) 32k -> mel_fn_v4 -> norm_spec (torch 布局, DiT 的 prompt mel 直接用)
    std::vector<float> w32;
    gsv_resample(raw.data(), (int64_t) raw.size(), sr, 32000, w32);
    gsv_mel_cfg mc; mc.n_threads = s.cfg.n_threads > 0 ? s.cfg.n_threads : 8;
    gsv_mel mel(mc);
    if (!mel.forward_norm(w32.data(), (int64_t) w32.size(), mel2, mel2_T)) {
        fprintf(stderr, "[gsv_engine] mel_fn_v4 失败\n");
        return false;
    }
    if (s.cfg.verbose)
        printf("[gsv_engine] prep_reference: codes=%zu ge[512] mel2[100,%d]\n",
               prompt_codes.size(), mel2_T);
    return true;
}

bool gsv_engine::bert_feature(const std::vector<int32_t> & ids,
                              const std::vector<int32_t> & word2ph,
                              std::vector<float> & feat1024) {
    impl & s = *p;
    feat1024.clear();
    if (!s.ready || ids.empty() || word2ph.empty()) return false;

    // encode_feat: hidden_states[-3], 已去 CLS/SEP -> [1024, L-2] 行主
    std::vector<float> feat;
    s.bert->encode_feat(ids.data(), (int) ids.size(), feat);
    const int N = (int) word2ph.size();
    if ((int) feat.size() != s.bert->hidden() * N) {
        fprintf(stderr, "[gsv_engine] bert 特征长度 %zu != 1024*%d (ids=%zu)\n",
                feat.size(), N, ids.size());
        return false;
    }
    // repo get_bert_feature: 每字特征按 word2ph[i] 重复 -> [1024, P]
    int64_t P = 0;
    for (int v : word2ph) P += v;
    feat1024.assign((size_t) 1024 * (size_t) P, 0.0f);
    size_t at = 0;
    for (int i = 0; i < N; i++) {
        for (int r = 0; r < word2ph[(size_t) i]; r++) {
            for (int k = 0; k < 1024; k++)
                feat1024[(size_t) k * (size_t) P + at] = feat[(size_t) k * N + (size_t) i];
            at++;
        }
    }
    return true;
}

bool gsv_engine::ar_generate(const std::vector<int32_t> & phones,
                             const std::vector<float> & bert,
                             const std::vector<int32_t> & prompt_codes,
                             int max_steps, int early_stop, uint64_t seed,
                             std::vector<int32_t> & sem) {
    impl & s = *p;
    sem.clear();
    if (!s.ready || phones.empty()) return false;
    if (!bert.empty() && bert.size() != (size_t) 1024 * phones.size()) {
        fprintf(stderr, "[gsv_engine] bert 尺寸 %zu != 1024*%zu\n", bert.size(), phones.size());
        return false;
    }
    gsv_ar_request req;
    req.phones = phones;
    req.bert = bert;                       // 空 = 全零 (非 zh 口径)
    req.prompt = prompt_codes;
    gsv_sampler_cfg sc;                    // 默认 = repo TTS.py 的采样参数
    auto res = s.ar->generate({ req }, sc, seed, early_stop, max_steps);
    if (res.tokens.empty()) { fprintf(stderr, "[gsv_engine] AR 生成返回空\n"); return false; }
    const std::vector<int32_t> & all = res.tokens[0];
    const size_t skip = std::min(prompt_codes.size(), all.size());
    sem.assign(all.begin() + skip, all.end());
    if (s.cfg.verbose)
        printf("[gsv_engine] ar_generate: prompt=%zu -> sem=%zu\n", prompt_codes.size(), sem.size());
    return true;
}

bool gsv_engine::synthesize(const std::vector<int32_t> & sem,
                            const std::vector<int32_t> & phones,
                            const std::vector<int32_t> & prompt_codes,
                            const std::vector<float> & ge,
                            const std::vector<float> & mel2, int mel2_T,
                            int steps, float cfg, uint64_t seed,
                            std::vector<float> & wav) {
    impl & s = *p;
    wav.clear();
    if (!s.ready || sem.empty() || prompt_codes.empty() || mel2_T <= 0) return false;
    if (ge.size() != (size_t) TXT_DIM) {
        fprintf(stderr, "[gsv_engine] ge 尺寸 %zu != %d\n", ge.size(), TXT_DIM);
        return false;
    }
    if (mel2.size() != (size_t) MEL_DIM * (size_t) mel2_T) {
        fprintf(stderr, "[gsv_engine] mel2 尺寸 %zu != %d*%d\n", mel2.size(), MEL_DIM, mel2_T);
        return false;
    }

    std::vector<float> fea_ref, fea_todo;
    if (!s.decode_encp_run(prompt_codes, phones, ge, fea_ref)) {
        fprintf(stderr, "[gsv_engine] decode_encp(参考侧) 失败\n");
        return false;
    }
    if (!s.decode_encp_run(sem, phones, ge, fea_todo)) {
        fprintf(stderr, "[gsv_engine] decode_encp(目标侧) 失败\n");
        return false;
    }
    // wns1 输出 [2T, 512] (idx = c*2T + t) 与 DiT 要的 [512, N] (idx = c*N + t) 同布, 无需转置
    const int R = (int) (fea_ref.size() / TXT_DIM);
    const int N = (int) (fea_todo.size() / TXT_DIM);

    std::vector<float> mel_out;
    if (!s.dit->synthesize(fea_ref.data(), R, fea_todo.data(), N, mel2.data(), mel2_T,
                           steps, cfg, seed, 0, nullptr, mel_out)) {
        fprintf(stderr, "[gsv_engine] DiT synthesize 失败\n");
        return false;
    }
    const int Nmel = (int) (mel_out.size() / MEL_DIM);
    if (Nmel <= 0) { fprintf(stderr, "[gsv_engine] DiT 输出为空\n"); return false; }

    // denorm_spec -> vocoder (输入是 torch 布局 [100, T], 正好是 mel_out 的布局)
    std::vector<float> mel_dn((size_t) MEL_DIM * (size_t) Nmel);
    for (int i = 0; i < MEL_DIM * Nmel; i++) mel_dn[(size_t) i] = gsv_denorm_spec(mel_out[(size_t) i]);
    if (!s.voc->run(mel_dn.data(), Nmel, wav)) {
        fprintf(stderr, "[gsv_engine] vocoder 失败\n");
        return false;
    }
    if (s.cfg.verbose)
        printf("[gsv_engine] synthesize: fea_ref[512,%d] fea_todo[512,%d] -> mel[100,%d] -> wav[%zu]\n",
               R, N, Nmel, wav.size());
    return true;
}
