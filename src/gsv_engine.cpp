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
#include "gsv_vocoder_ggml.h"
#include "gsv_progress.h"
#include "gsv_wns1.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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
    gsv_vocoder_ggml * vgm = nullptr;   // ggml HiFi-GAN vocoder
    std::mutex   voc_mu;   // DML/WebGPU 都不保证并发安全：所有 vocoder 调用串行

    std::string md;                       // model_dir (带尾部 '/')
    int  ar_vocab = 1025;
    bool ready = false;
    double last_cond_ms = 0, last_dit_ms = 0;

    ~impl() {
        delete vgm; delete dit; delete wns1; delete encp; delete cond;
        delete ar; delete bert; delete re; delete rc; delete hub;
    }

    std::string path(const char * leaf) const {
        const std::string f(leaf);
        // 绝对路径原样返回 (voice manifest 会给跨目录的共享模型)
        if (f.size() >= 2 && f[1] == ':') return f;
        if (!f.empty() && (f[0] == '/' || f[0] == '\\')) return f;
        return md + f;
    }
    // 逐模型覆盖: cfg 里给了就用, 否则 model_dir 下的默认文件名
    std::string pick(const std::string & f, const char * def) const {
        return path(f.empty() ? def : f.c_str());
    }

    // 一次 decode_encp: codes -> RVQ/x2 -> enc_p(+speed) -> bridge/x2 -> wns1 -> fea [512, 4*Tc/speed]
    bool decode_encp_run(const std::vector<int32_t> & codes,
                         const std::vector<int32_t> & phones,
                         const std::vector<float> & ge,
                         std::vector<float> & fea,
                         const std::string * diag_dir = nullptr,
                         double speed = 1.0) {
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
                          m, logs, &yenc, (float) speed))
            return false;
        const int Tn = (int) (yenc.size() / 192);               // speed 之后的 enc_p 帧数
        if (diag_dir) {   // 分段对拍: enc_p 的 y (bridge 输入) / bridge 输出 / x2 之后
            auto wr2 = [&](const char * n, const std::vector<float> & v) {
                FILE * f = fopen((*diag_dir + "/" + n).c_str(), "wb");
                if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
            };
            wr2("ours_encp_y.f32", yenc);           // [192, Tn] torch 布局
            wr2("ours_quant.f32", q);               // [768, 2Tc] ggml 布局
        }
        std::vector<float> xb((size_t) 192 * Tn), br;
        for (int t = 0; t < Tn; t++)
            for (int c = 0; c < 192; c++) xb[(size_t) c + (size_t) t * 192] = yenc[(size_t) c * Tn + t];
        cond->bridge_run(xb.data(), Tn, true, br);              // [2Tn, 512] (idx = c*2Tn + t)
        const int T2 = 2 * Tn;
        // repo 的 wns1 y_lengths1 = int(codes*4/speed)+1（speed==1 时 = codes*4）：
        // 它既 mask 输入（该长度之后的帧不进 conv），也把输出对应帧清零；输出长度仍是 2*Tn。
        const int sizee = (speed == 1.0) ? T2 : (int) ((double) Tc * 4.0 / speed) + 1;
        fea.clear();
        if (!wns1->encode(br.data(), ge.data(), T2, sizee > 0 ? sizee : T2, fea)) return false;
        for (int t = (sizee > 0 ? sizee : 0); t < T2; t++)
            for (int c = 0; c < 512; c++) fea[(size_t) c * T2 + t] = 0.0f;
        return true;
    }
};

gsv_engine::gsv_engine() : p(new impl) {}
gsv_engine::~gsv_engine() { delete p; }

int gsv_engine::mel_dim()  const { return MEL_DIM; }

void gsv_engine::set_dbcache(int fn, float threshold, int warmup, int downsample) {
    if (p->dit) p->dit->set_dbcache(fn, threshold, warmup, downsample);
}

void gsv_engine::set_dit_alg(const gsv_dit_alg & a) {
    if (p->dit) p->dit->set_alg(a);
}

gsv_dit_alg gsv_engine::dit_alg() const {
    return p->dit ? p->dit->alg() : gsv_dit_alg();
}
int gsv_engine::dbcache_fn() const { return p->dit ? p->dit->dbcache_fn() : 0; }

void gsv_engine::last_stage_ms(double & cond_ms, double & dit_ms) const {
    cond_ms = p->last_cond_ms;
    dit_ms  = p->last_dit_ms;
}

void gsv_engine::dbcache_stats(int & hits, int & misses) const {
    hits = misses = 0;
    if (!p->dit) return;
    gsv_dit::db_stats st{};
    p->dit->db_get_stats(st);
    for (int b = 0; b < 2; b++) {
        hits += st.hits[b];
        misses += st.misses[b];
    }
}
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
        s.hub = gsv_hubert::load(s.pick(cfg.hubert_file, "gsv-hubert-f16.gguf"), c);
        gsv_refcode_cfg r; r.device = dev; r.n_threads = cfg.n_threads; r.verbose = vb;
        s.rc = gsv_refcode::load(s.pick(cfg.refcode_file, "gsv-refcode.gguf"), r);
        gsv_refenc_cfg g; g.device = dev; g.n_threads = cfg.n_threads; g.verbose = vb;
        s.re = gsv_refenc::load(s.pick(cfg.cond_file, "gsv-cond-f16.gguf"), g);
        if (!s.hub || !s.rc || !s.re) {
            fprintf(stderr, "[gsv_engine] 参考音频链加载失败 (hubert/refcode/refenc)\n");
            delete m; return nullptr;
        }
        // resampy kaiser_best 表 (repo resample(match_librosa=True) 的复刻需要)
        if (!gsv_refcode::load_filter_table(s.pick(cfg.filter_file, "resampy_kaiser_best.bin"))) {
            fprintf(stderr, "[gsv_engine] %s 缺失 (先跑 tools/export_resampy_filter.py)\n",
                    s.pick(cfg.filter_file, "resampy_kaiser_best.bin").c_str());
            delete m; return nullptr;
        }
    }
    // BERT (LRU 开一点: 同一文本重复合成是服务化常态)
    {
        gsv_bert_cfg b; b.device = dev; b.n_threads = cfg.n_threads; b.verbose = vb;
        b.feat_cache = 8;
        s.bert = gsv_bert::load(s.pick(cfg.bert_file, "gsv-bert-f16.gguf"), b);
        if (!s.bert) { fprintf(stderr, "[gsv_engine] BERT 加载失败\n"); delete m; return nullptr; }
    }
    // AR
    {
        gsv_ar_cfg a; a.device = dev; a.n_threads = cfg.n_threads; a.verbose = vb;
        // 锚定档: AR 用 attn+ffn Q8_0 (TV 0.0035 / greedy 3/3); F16 档留作回退
        s.ar = gsv_ar::load(s.pick(cfg.ar_file, "gsv-ar-q8.gguf"), a);
        if (!s.ar) { fprintf(stderr, "[gsv_engine] AR 加载失败\n"); delete m; return nullptr; }
        s.ar_vocab = s.ar->vocab_size();
    }
    // 条件段
    {
        gsv_cond_cfg c; c.device = dev; c.n_threads = cfg.n_threads; c.verbose = vb;
        gsv_encp_cfg e; e.device = dev; e.n_threads = cfg.n_threads; e.verbose = vb;
        gsv_wns1_cfg w; w.device = dev; w.n_threads = cfg.n_threads; w.verbose = vb;
        // 锚定档 (docs/quant_tiers.md): 条件段统一用全 F16 的 cond-f16; 四个模块各自
        // 只分配自己前缀的张量 (gsv_gguf_slice.h), 所以同一文件只占一份权重。
        s.cond = gsv_cond::load(s.pick(cfg.cond_file, "gsv-cond-f16.gguf"), c);
        s.encp = gsv_encp::load(s.pick(cfg.cond_file, "gsv-cond-f16.gguf"), e);
        s.wns1 = gsv_wns1::load(s.pick(cfg.cond_file, "gsv-cond-f16.gguf"), w);
        if (!s.cond || !s.encp || !s.wns1) {
            fprintf(stderr, "[gsv_engine] 条件段加载失败 (cond/encp/wns1)\n");
            delete m; return nullptr;
        }
    }
    // DiT (最小可用档 = f16all; adaln 降 F16 的 f16afa 在真实输入上出 NaN)
    {
        gsv_dit_cfg d; d.device = dev; d.verbose = vb;
        s.dit = gsv_dit::load(s.pick(cfg.dit_file, "gsv-dit-f16all.gguf"), d);
        if (!s.dit) { fprintf(stderr, "[gsv_engine] DiT 加载失败\n"); delete m; return nullptr; }
    }
    // vocoder: ggml HiFi-GAN (the only vocoder backend)
    {
        const std::string vpath = s.pick(cfg.vocoder_file, "gsv-vocoder-f32.gguf");
        gsv_vocoder_ggml_cfg g; g.device = dev; g.n_threads = cfg.n_threads; g.verbose = vb;
        s.vgm = gsv_vocoder_ggml::load(vpath, g);
        if (!s.vgm) {
            fprintf(stderr, "[gsv_engine] vocoder load failed (%s)\n", vpath.c_str());
            delete m; return nullptr;
        }
    }
    s.ready = true;
    if (vb)
        printf("[gsv_engine] ready: ggml=%s vocoder=ggml(%s)\n",
               dev.empty() ? "cpu" : dev.c_str(), s.vgm->device_name());
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
                             const std::vector<int32_t> & prompt_phones,
                             const std::vector<float> & prompt_bert,
                             const std::vector<int32_t> & prompt_codes,
                             int max_steps, int early_stop, uint64_t seed,
                             std::vector<int32_t> & sem,
                             const gsv_sampler_cfg * sampler) {
    impl & s = *p;
    sem.clear();
    if (!s.ready || phones.empty()) return false;
    if (!bert.empty() && bert.size() != (size_t) 1024 * phones.size()) {
        fprintf(stderr, "[gsv_engine] bert 尺寸 %zu != 1024*%zu\n", bert.size(), phones.size());
        return false;
    }
    if (prompt_phones.empty()) {
        fprintf(stderr, "[gsv_engine] prompt_phones 为空: repo 的 AR 输入是 "
                        "参考文本音素 ++ 目标文本音素 (TTS.py:1017)\n");
        return false;
    }
    if (!prompt_bert.empty() && prompt_bert.size() != (size_t) 1024 * prompt_phones.size()) {
        fprintf(stderr, "[gsv_engine] prompt_bert 尺寸 %zu != 1024*%zu\n",
                prompt_bert.size(), prompt_phones.size());
        return false;
    }

    // x = prompt_phones ++ phones (repo: all_phones = prompt_data["phones"] + item["phones"])
    gsv_ar_request req;
    req.phones.reserve(prompt_phones.size() + phones.size());
    req.phones.insert(req.phones.end(), prompt_phones.begin(), prompt_phones.end());
    req.phones.insert(req.phones.end(), phones.begin(), phones.end());

    // bert 沿时间轴拼接 (repo: torch.cat([prompt_bert, target_bert], dim=1))。
    // 非 zh 时两半都是**全零** (repo get_bert_inf 的直接口径) —— 必须真的填一份零缓冲,
    // 不能留空: gsv_ar 的 build_inputs 会无条件读 r.bert[k*T+t]。
    {
        const int Pp = (int) prompt_phones.size();
        const int Pt = (int) phones.size();
        const int P = Pp + Pt;
        req.bert.assign((size_t) 1024 * (size_t) P, 0.0f);
        for (int k = 0; k < 1024; k++) {
            if (!prompt_bert.empty())
                std::memcpy(req.bert.data() + (size_t) k * P, prompt_bert.data() + (size_t) k * Pp,
                            (size_t) Pp * sizeof(float));
            if (!bert.empty())
                std::memcpy(req.bert.data() + (size_t) k * P + Pp, bert.data() + (size_t) k * Pt,
                            (size_t) Pt * sizeof(float));
        }
    }
    req.prompt = prompt_codes;

    gsv_sampler_cfg sc = sampler ? *sampler : gsv_sampler_cfg();   // 默认为 repo TTS.py 的采样参数
    auto res = s.ar->generate({ req }, sc, seed, early_stop, max_steps);
    if (res.tokens.empty()) { fprintf(stderr, "[gsv_engine] AR 生成返回空\n"); return false; }
    const std::vector<int32_t> & all = res.tokens[0];
    const size_t skip = std::min(prompt_codes.size(), all.size());
    sem.assign(all.begin() + skip, all.end());
    if (s.cfg.verbose)
        printf("[gsv_engine] ar_generate: x=%zu(=prompt %zu + target %zu) prompt_codes=%zu -> sem=%zu\n",
               req.phones.size(), prompt_phones.size(), phones.size(), prompt_codes.size(), sem.size());
    return true;
}

bool gsv_engine::ar_generate_batch(const std::vector<ar_batch_item> & items,
                                   const std::vector<int32_t> & prompt_phones,
                                   const std::vector<float> & prompt_bert,
                                   const std::vector<int32_t> & prompt_codes,
                                   int max_steps, int early_stop, uint64_t seed,
                                   std::vector<std::vector<int32_t>> & sems,
                                   const std::vector<uint64_t> * seeds,
                                   const gsv_sampler_cfg * sampler) {
    impl & s = *p;
    sems.assign(items.size(), {});
    if (!s.ready || items.empty()) return false;
    if (prompt_phones.empty()) {
        fprintf(stderr, "[gsv_engine] ar_generate_batch: prompt_phones is empty\n");
        return false;
    }
    if (!prompt_bert.empty() && prompt_bert.size() != (size_t) 1024 * prompt_phones.size()) {
        fprintf(stderr, "[gsv_engine] ar_generate_batch: prompt_bert size mismatch\n");
        return false;
    }

    std::vector<gsv_ar_request> reqs;
    reqs.reserve(items.size());
    for (const auto & it : items) {
        if (it.phones.empty()) return false;
        if (!it.bert.empty() && it.bert.size() != (size_t) 1024 * it.phones.size()) {
            fprintf(stderr, "[gsv_engine] ar_generate_batch: item bert size mismatch\n");
            return false;
        }
        const int Pp = (int) prompt_phones.size();
        const int Pt = (int) it.phones.size();
        const int P = Pp + Pt;
        gsv_ar_request req;
        req.phones.reserve((size_t) P);
        req.phones.insert(req.phones.end(), prompt_phones.begin(), prompt_phones.end());
        req.phones.insert(req.phones.end(), it.phones.begin(), it.phones.end());
        req.bert.assign((size_t) 1024 * (size_t) P, 0.0f);
        for (int k = 0; k < 1024; k++) {
            if (!prompt_bert.empty())
                std::memcpy(req.bert.data() + (size_t) k * P, prompt_bert.data() + (size_t) k * Pp,
                            (size_t) Pp * sizeof(float));
            if (!it.bert.empty())
                std::memcpy(req.bert.data() + (size_t) k * P + Pp, it.bert.data() + (size_t) k * Pt,
                            (size_t) Pt * sizeof(float));
        }
        req.prompt = prompt_codes;
        reqs.push_back(std::move(req));
    }

    gsv_sampler_cfg sc = sampler ? *sampler : gsv_sampler_cfg();
    auto res = s.ar->generate(reqs, sc, seed, early_stop, max_steps, nullptr, seeds);
    if (res.tokens.size() != reqs.size()) {
        fprintf(stderr, "[gsv_engine] ar_generate_batch: got %zu/%zu sequences\n",
                res.tokens.size(), reqs.size());
        return false;
    }
    for (size_t i = 0; i < reqs.size(); i++) {
        const std::vector<int32_t> & all = res.tokens[i];
        const size_t skip = std::min(prompt_codes.size(), all.size());
        sems[i].assign(all.begin() + skip, all.end());
    }
    // 批量 KV 是 B 倍大小；解完立即归还，避免后续 DiT/vocoder 显存紧张
    s.ar->release_cache();
    if (s.cfg.verbose)
        printf("[gsv_engine] ar_generate_batch: %zu seqs, prompt %zu + target (first %zu) -> sem %zu\n",
               reqs.size(), prompt_phones.size(), items[0].phones.size(), sems[0].size());
    return true;
}

bool gsv_engine::synthesize_mel(const std::vector<int32_t> & sem,
                                const std::vector<int32_t> & phones,
                                const std::vector<int32_t> & prompt_codes,
                                const std::vector<int32_t> & prompt_phones,
                                const std::vector<float> & ge,
                                const std::vector<float> & mel2, int mel2_T,
                                int steps, float cfg, uint64_t seed,
                                std::vector<float> & mel, int & mel_T, double speed, int skip_steps) {
    impl & s = *p;
    mel.clear();
    mel_T = 0;
    if (!s.ready || sem.empty() || prompt_codes.empty() || mel2_T <= 0) return false;
    if (ge.size() != (size_t) TXT_DIM) {
        fprintf(stderr, "[gsv_engine] ge 尺寸 %zu != %d\n", ge.size(), TXT_DIM);
        return false;
    }
    if (mel2.size() != (size_t) MEL_DIM * (size_t) mel2_T) {
        fprintf(stderr, "[gsv_engine] mel2 尺寸 %zu != %d*%d\n", mel2.size(), MEL_DIM, mel2_T);
        return false;
    }
    if (prompt_phones.empty()) {
        fprintf(stderr, "[gsv_engine] prompt_phones 为空: 参考侧 fea_ref 必须用**参考文本**的音素\n"
                        "(repo TTS.py::using_vocoder_synthesis 的 decode_encp(prompt_sem, prompt_phones, ...));\n"
                        "传目标文本的音素会让 DiT 的参考条件全错\n");
        return false;
    }

    std::vector<float> fea_ref, fea_todo;
    const std::string diag_dir = getenv("GSV_ENGINE_DUMP") ? getenv("GSV_ENGINE_DUMP") : "";
    const auto t_cond0 = std::chrono::steady_clock::now();
    // 参考侧: prompt_codes + **prompt_phones** (参考文本的音素)
    if (!s.decode_encp_run(prompt_codes, prompt_phones, ge, fea_ref, &diag_dir)) {
        fprintf(stderr, "[gsv_engine] decode_encp(参考侧) 失败\n");
        return false;
    }
    // 目标侧: sem + 目标文本的 phones (+ speed)
    if (!s.decode_encp_run(sem, phones, ge, fea_todo, nullptr, speed)) {
        fprintf(stderr, "[gsv_engine] decode_encp(目标侧) 失败\n");
        return false;
    }
    s.last_cond_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cond0).count();
    const auto t_dit0 = std::chrono::steady_clock::now();
    // wns1 输出 [2T, 512] (idx = c*2T + t) 与 DiT 要的 [512, N] (idx = c*N + t) 同布, 无需转置
    const int R = (int) (fea_ref.size() / TXT_DIM);
    const int N = (int) (fea_todo.size() / TXT_DIM);

    // 诊断: 落盘 DiT 的输入 (GSV_ENGINE_DUMP=<dir>), 供 torch 复算对比
    if (const char * dd = getenv("GSV_ENGINE_DUMP")) {
        auto wr = [&](const char * n, const void * p, size_t bytes) {
            FILE * f = fopen((std::string(dd) + "/" + n).c_str(), "wb");
            if (f) { fwrite(p, 1, bytes, f); fclose(f); }
        };
        wr("fea_ref.f32", fea_ref.data(), fea_ref.size() * 4);
        wr("fea_todo.f32", fea_todo.data(), fea_todo.size() * 4);
        wr("mel2.f32", mel2.data(), mel2.size() * 4);
        wr("sem.i32", sem.data(), sem.size() * 4);
        wr("prompt_codes.i32", prompt_codes.data(), prompt_codes.size() * 4);
        wr("prompt_phones.i32", prompt_phones.data(), prompt_phones.size() * 4);
        wr("phones.i32", phones.data(), phones.size() * 4);
        wr("ge.f32", ge.data(), ge.size() * 4);
        printf("[gsv_engine] dumped DiT inputs to %s (R=%d N=%d melT=%d)\n", dd, R, N, mel2_T);
    }

    std::vector<float> mel_out;
    bool dit_ok = false;
    try {
        dit_ok = s.dit->synthesize(fea_ref.data(), R, fea_todo.data(), N, mel2.data(), mel2_T,
                                   steps, cfg, seed, 0, nullptr, mel_out, skip_steps);
    } catch (const std::exception & e) {
        fprintf(stderr, "[gsv_engine] DiT threw: %s (R=%d N=%d mel2T=%d steps=%d)\n", e.what(), R, N,
                mel2_T, steps);
        return false;
    }
    if (!dit_ok) {
        fprintf(stderr, "[gsv_engine] DiT synthesize 失败\n");
        return false;
    }
    s.last_dit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dit0).count();
    const int Nmel = (int) (mel_out.size() / MEL_DIM);
    if (Nmel <= 0) { fprintf(stderr, "[gsv_engine] DiT 输出为空\n"); return false; }
    if (const char * dd = getenv("GSV_ENGINE_DUMP")) {
        const std::string dir(dd);
        FILE * f = fopen((dir + "/mel_out.f32").c_str(), "wb");
        if (f) { fwrite(mel_out.data(), 4, mel_out.size(), f); fclose(f); }
        // 复刻一份与 C++ 侧完全相同的初始噪声, 供 torch 侧做**同输入**对比:
        // sample() 用 gsv_rng(seed) 逐元素生成 randn*0.875 填满 x [MEL, T]。
        // (与 gsv_dit.cpp::randn01 同式 —— Box-Muller + gsv_rng 的 xorshift)
        const int Tdump = s.dit->cur_T();   // prepare 后的 T = ref_frames + 块长
        gsv_rng rng(seed ? seed : 1);
        std::vector<float> noise((size_t) MEL_DIM * (size_t) Tdump);
        for (size_t i = 0; i < noise.size(); i++) {
            float u1 = rng.uniform01();
            const float u2 = rng.uniform01();
            if (u1 < 1e-30f) u1 = 1e-30f;
            noise[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2) * 0.875f;
        }
        f = fopen((dir + "/noise.f32").c_str(), "wb");
        if (f) { fwrite(noise.data(), 4, noise.size(), f); fclose(f); }

        // 逐层速度场对拍: 用同一份噪声重放 4 步, 每步 dump vel (torch 侧 probe_dit_steps.py 同样拦 4 步)
        std::vector<float> vel((size_t) MEL_DIM * (size_t) Tdump);
        f = fopen((dir + "/vels.f32").c_str(), "wb");
        if (f) {
            const float st = 1.0f / (float) steps;
            const int P = s.dit->cur_prompt_T();
            for (int j = 0; j < steps; j++) {
                const float t = (float) j * st;
                if (!s.dit->velocity(noise.data(), t, false, vel.data())) break;
                // 只对拍生成区 (prompt 区 x 恒 0, 两边都是 0)
                for (int c = 0; c < MEL_DIM; c++)
                    fwrite(vel.data() + (size_t) c * Tdump + P, 4, (size_t) (Tdump - P), f);
                // 逐层探针必须在 **step 0** 之后读 (mark() 记录的是最近一次 velocity 的图);
                // 若放到循环外读到的就是 step 3, 与 torch 的 t=0 对不上。
                if (j == 0) {
                    for (const char * nm : { "dbg_x_lin", "dbg_convpos", "dbg_b0_norm",
                                             "dbg_b0_scale_msa", "dbg_b0_shift_msa",
                                             "dbg_b0_attn", "dbg_b0_ff", "dbg_b0_out",
                                             "dbg_b21_out", "dbg_norm_out" }) {
                        std::vector<float> v;
                        if (s.dit->debug_read(nm, v) && !v.empty()) {
                            FILE * g = fopen((dir + "/" + nm + ".f32").c_str(), "wb");
                            if (g) { fwrite(v.data(), 4, v.size(), g); fclose(g); }
                        }
                    }
                }
            }
            fclose(f);
            printf("[gsv_engine] dumped %d 步 vel + step0 逐层探针 (prompt_T=%d)\n", steps, P);
        }
    }

    // denorm_spec -> vocoder (输入是 torch 布局 [100, T], 正好是 mel_out 的布局)
    mel = std::move(mel_out);
    mel_T = Nmel;
    if (s.cfg.verbose)
        printf("[gsv_engine] synthesize_mel: fea_ref[512,%d] fea_todo[512,%d] -> mel[100,%d]\n", R, N, Nmel);
    return true;
}

bool gsv_engine::vocode_mel(const std::vector<float> & mel, int mel_T, std::vector<float> & wav) {
    impl & s = *p;
    wav.clear();
    // ORT 的 WebGPU/DML EP 都不保证并发安全：所有 vocoder 调用串行
    gsvprog::begin(4, 1);
    std::lock_guard<std::mutex> vl(s.voc_mu);
    if (!s.ready || mel_T <= 0 || mel.size() != (size_t) MEL_DIM * (size_t) mel_T) return false;
    std::vector<float> mel_dn((size_t) MEL_DIM * (size_t) mel_T);
    for (int i = 0; i < MEL_DIM * mel_T; i++) mel_dn[(size_t) i] = gsv_denorm_spec(mel[(size_t) i]);
    if (!s.vgm->forward(mel_dn.data(), mel_T, wav)) {
        fprintf(stderr, "[gsv_engine] vocoder failed\n");
        return false;
    }
    return true;
}

bool gsv_engine::synthesize(const std::vector<int32_t> & sem,
                            const std::vector<int32_t> & phones,
                            const std::vector<int32_t> & prompt_codes,
                            const std::vector<int32_t> & prompt_phones,
                            const std::vector<float> & ge,
                            const std::vector<float> & mel2, int mel2_T,
                            int steps, float cfg, uint64_t seed,
                            std::vector<float> & wav, double speed, int skip_steps) {
    std::vector<float> mel;
    int mel_T = 0;
    if (!synthesize_mel(sem, phones, prompt_codes, prompt_phones, ge, mel2, mel2_T, steps, cfg, seed,
                        mel, mel_T, speed, skip_steps))
        return false;
    return vocode_mel(mel, mel_T, wav);
}
