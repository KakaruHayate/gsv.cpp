// v2 / v2Pro / v2ProPlus S2 分段对拍 + 端到端 benchmark 工具。
//
// 用法:
//   probe_v2sovits <s2.gguf> <fixture_dir> [选项]
//     --models DIR     含 hubert/bert/ar 的模型目录（给了才跑 engine 端到端 + benchmark）
//     --ref-wav PATH   参考音频（engine 端到端时用于 prep_reference；不给则用 fixture 的 ge）
//     --threads N      CPU 线程数
//     --bench N        v2 SoVITS 段重复次数（默认 1）
//     --ar-bench N     AR 段重复次数（默认 1）
//     --speed S        enc_p 的时间缩放（默认 1.0）
//     --ar-max N       AR 最多生成多少 semantic token（默认 2000，~8s 音频约 220）
//     --ar-file PATH   覆盖 AR 的 GGUF（对比 q8 / f16）
//     --seed N         z_p 噪声种子（默认 12345）
//     --noise N        noise_scale（默认 0.5）
//     --dump           把本侧中间量写到 fixture_dir/ours_*.f32（供 python 侧复算/出报告）
//
// fixture 约定（raw little-endian，行主；torch 布局 [C,T] = c*T + t）:
//   ssl.f32        [768, Th]      python: cnhuhbert(16k + 0.3s 零尾) 的 last_hidden_state.T
//   codes_ref.i32  [Tc]           python: vits.extract_latent(ssl)
//   spec704.f32    [704, Ts]      python: spectrogram_torch(32k, maxx norm)[:, :704]
//   ge_ref.f32     [512]          python: vits.ref_enc(spec)
//   phones.i32     [P]            python: 目标文本音素
//   prompt_codes.i32 [Tc] / prompt_phones.i32 / prompt_bert.f32 [1024,Pp]
//   bert.f32       [1024, P]      目标文本 BERT 特征（AR 用）
//   m_ref.f32/logs_ref.f32 [192, T]        python: enc_p 的 m_p / logs_p
//   zp_ref.f32     [192, T]                python: m_p + randn*exp(logs)*noise_scale
//   z_ref.f32      [192, T]                python: flow(z_p, reverse=True)
//   wav_ref.f32    [T*640]                 python: dec(z)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gsv_engine.h"
#include "gsv_refenc.h"
#include "gsv_v2sovits.h"

static bool read_f32(const std::string & p, std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    const size_t rd = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return rd == v.size();
}
static bool read_i32(const std::string & p, std::vector<int32_t> & v) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / 4);
    const size_t rd = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return rd == v.size();
}
static void write_f32(const std::string & p, const std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) return;
    fwrite(v.data(), 4, v.size(), f);
    fclose(f);
}

// 对拍报告: max |a-b|, RMS 差, 参考 RMS, 余弦相似度
static void cmp(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        printf("  %-12s 尺寸不同: ours=%zu ref=%zu\n", name, a.size(), b.size());
        return;
    }
    double mx = 0, se = 0, sr = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        mx = std::max(mx, std::fabs(d));
        se += d * d;
        sr += (double) b[i] * b[i];
        dot += (double) a[i] * b[i];
        na += (double) a[i] * a[i];
        nb += (double) b[i] * b[i];
    }
    const double n = (double) a.size();
    const double cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    printf("  %-12s n=%-8zu max_abs=%.3e rms_diff=%.3e ref_rms=%.3e cos=%.6f\n",
           name, a.size(), mx, std::sqrt(se / n), std::sqrt(sr / n), cos);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: probe_v2sovits <s2.gguf> <fixture_dir> [--models DIR] [--ref-wav PATH]\n"
               "                      [--threads N] [--bench N] [--ar-bench N] [--ar-max N] [--ar-file PATH]\n"
               "                      [--speed S] [--seed N]\n"
               "                      [--noise N] [--dump]\n");
        return 2;
    }
    std::string ec_ar_file;
    const std::string s2_path = argv[1];
    const std::string fix = argv[2];
    std::string models, ref_wav;
    int threads = 0, bench = 1, ar_bench = 1, ar_max = 2000;
    double speed = 1.0, noise = 0.5;
    uint64_t seed = 12345;
    bool dump = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--models") models = next();
        else if (a == "--ref-wav") ref_wav = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--bench") bench = std::max(1, std::atoi(next().c_str()));
        else if (a == "--ar-bench") ar_bench = std::max(1, std::atoi(next().c_str()));
        else if (a == "--ar-max") ar_max = std::max(1, std::atoi(next().c_str()));
        else if (a == "--ar-file") ec_ar_file = next();
        else if (a == "--speed") speed = std::atof(next().c_str());
        else if (a == "--seed") seed = (uint64_t) std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--noise") noise = std::atof(next().c_str());
        else if (a == "--dump") dump = true;
        else { printf("unknown arg %s\n", a.c_str()); return 2; }
    }
    const std::string dir = fix + "/";

    gsv_v2sovits_cfg cfg;
    cfg.n_threads = threads;
    cfg.verbose = true;
    gsv_v2sovits * s2 = gsv_v2sovits::load(s2_path, cfg);
    if (!s2) { printf("load failed\n"); return 1; }
    const int SR = s2->sample_rate();
    printf("\n== v2 SoVITS: version=%s sr=%d upsample=%d codebook=%d frame_rate=%d\n",
           s2->version().c_str(), SR, s2->upsample(), s2->codebook_size(), s2->frame_rate());

    // ---------------- 1) 参考侧 ssl_proj + VQ ----------------
    std::vector<float> ssl, spec704, ge;
    std::vector<int32_t> codes_ref, codes, phones;
    int Th = 0, T = 0;
    if (read_f32(dir + "ssl.f32", ssl) && ssl.size() % 768 == 0) {
        Th = (int) (ssl.size() / 768);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = s2->codes_from_ssl(ssl.data(), Th, codes);
        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        printf("\n[1] ssl_proj+VQ: ssl[768,%d] -> codes[%zu] %s (%.1f ms)\n",
               Th, codes.size(), ok ? "ok" : "FAIL", ms);
        if (read_i32(dir + "codes_ref.i32", codes_ref) && codes_ref.size() == codes.size()) {
            size_t hit = 0;
            for (size_t i = 0; i < codes.size(); i++) hit += (codes[i] == codes_ref[i]);
            printf("  codes        match=%zu/%zu (%.2f%%)\n", hit, codes.size(),
                   100.0 * (double) hit / (double) codes.size());
        }
    } else {
        printf("\n[1] ssl.f32 缺失: 跳过 ssl_proj/VQ 段\n");
        if (read_i32(dir + "prompt_codes.i32", codes)) printf("  用 fixture prompt_codes[%zu]\n", codes.size());
    }

    // ---------------- 2) ref_enc -> ge ----------------
    if (read_f32(dir + "spec704.f32", spec704) && spec704.size() % 704 == 0) {
        const int Ts = (int) (spec704.size() / 704);
        gsv_refenc_cfg rc;
        rc.n_threads = threads;
        gsv_refenc * re = gsv_refenc::load(s2_path, rc);
        if (!re) { printf("ref_enc 加载失败\n"); return 1; }
        std::vector<float> spec_gg(spec704.size()), fm((size_t) Ts, 1.0f);
        for (int c = 0; c < 704; c++)
            for (int t = 0; t < Ts; t++)
                spec_gg[(size_t) c + (size_t) t * 704] = spec704[(size_t) c * Ts + t];
        std::vector<float> my_ge;
        const auto t0 = std::chrono::steady_clock::now();
        if (!re->encode(spec_gg.data(), fm.data(), Ts, my_ge)) { printf("ref_enc FAIL\n"); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        printf("\n[2] ref_enc: spec[704,%d] -> ge[%zu] (%.1f ms)\n", Ts, my_ge.size(), ms);
        std::vector<float> ge_ref;
        if (read_f32(dir + "ge_ref.f32", ge_ref)) cmp("ge", my_ge, ge_ref);
        ge = my_ge;
        delete re;
    } else if (read_f32(dir + "ge_ref.f32", ge)) {
        printf("\n[2] 用 fixture ge_ref[%zu]（无 spec704.f32）\n", ge.size());
    } else {
        printf("\n[2] spec704.f32 / ge_ref.f32 都缺失: 无法验证 ref_enc\n");
    }

    // ---------------- 3) enc_p ----------------
    std::vector<float> m, logs;
    const bool have_phones = read_i32(dir + "phones.i32", phones) && !phones.empty();
    if (have_phones && !codes.empty() && ge.size() == 512) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = s2->encode_p(codes.data(), (int) codes.size(), phones.data(),
                                     (int) phones.size(), ge.data(), m, logs, T, speed);
        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        printf("\n[3] enc_p: codes=%zu phones=%zu -> m/logs[192,%d] %s (%.1f ms)\n",
               codes.size(), phones.size(), T, ok ? "ok" : "FAIL", ms);
        std::vector<float> m_ref, logs_ref;
        if (read_f32(dir + "m_ref.f32", m_ref) && read_f32(dir + "logs_ref.f32", logs_ref)) {
            cmp("m_p", m, m_ref);
            cmp("logs_p", logs, logs_ref);
        }
        if (dump) { write_f32(dir + "ours_m.f32", m); write_f32(dir + "ours_logs.f32", logs); }
    } else {
        printf("\n[3] 缺 phones.i32 / codes / ge: 跳过 enc_p 段\n");
    }

    // ---------------- 4) flow + dec ----------------
    if (T > 0 && ge.size() == 512) {
        std::vector<float> zp, z, z_ref, wav_ref;
        if (read_f32(dir + "zp_ref.f32", zp) && zp.size() == (size_t) 192 * T) {
            printf("\n[4] 用 fixture zp_ref[192,%d]\n", T);
        } else {
            // 自己采样（仅端到端用；逐位对拍请提供 zp_ref.f32）
            zp.assign((size_t) 192 * T, 0.0f);
            for (int c = 0; c < 192; c++)
                for (int t = 0; t < T; t++) {
                    const double s1 = std::sin((double) (c * 7919 + t * 104729 + (int) seed) * 12.9898) * 43758.5453;
                    const double s2v = std::sin((double) (c * 6151 + t * 38699 + 7 + (int) seed) * 78.233) * 12345.6789;
                    const float u1 = (float) (s1 - std::floor(s1));
                    const float u2 = (float) (s2v - std::floor(s2v));
                    const float e = std::sqrt(-2.0f * std::log(u1 + 1e-6f)) * std::cos(6.2831853f * u2);
                    zp[(size_t) c * T + t] = m.empty()
                        ? 0.0f
                        : m[(size_t) c * T + t] + e * std::exp(logs[(size_t) c * T + t]) * (float) noise;
                }
            printf("\n[4] 自采样 z_p（非逐位对拍）\n");
        }
        auto tf = std::chrono::steady_clock::now();
        const bool fok = s2->flow_reverse(zp.data(), T, ge.data(), z);
        const double f_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
        printf("[4] flow: [192,%d] %s (%.1f ms)\n", T, fok ? "ok" : "FAIL", f_ms);
        if (read_f32(dir + "z_ref.f32", z_ref)) cmp("z", z, z_ref);
        if (dump) write_f32(dir + "ours_z.f32", z);

        auto td = std::chrono::steady_clock::now();
        const bool dok = s2->decode_wav(z.data(), T, ge.data(), z_ref);
        const double d_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
        printf("[5] dec: z[192,%d] -> wav[%zu] %s (%.1f ms, %.2fs @ %d Hz)\n", T, z_ref.size(),
               dok ? "ok" : "FAIL", d_ms, (double) z_ref.size() / (double) SR, SR);
        if (read_f32(dir + "wav_ref.f32", wav_ref)) cmp("wav", z_ref, wav_ref);
        double pk = 0;
        for (float v : z_ref) pk = std::max(pk, (double) std::fabs(v));
        printf("  wav peak=%.4f\n", pk);
        if (dump) write_f32(dir + "ours_wav.f32", z_ref);
    }

    // ---------------- 5) engine 端到端 (AR + v2 解码) + benchmark ----------------
    if (!models.empty()) {
        gsv_engine_cfg ec;
        ec.model_dir = models;
        ec.n_threads = threads;
        ec.verbose = true;
        ec.version = "v2";
        ec.s2_file = s2_path;
        ec.ar_file = ec_ar_file;
        ec.v2_noise_scale = (float) noise;
        gsv_engine * eng = gsv_engine::load(ec);
        if (!eng) { printf("engine load failed\n"); delete s2; return 1; }
        printf("\n[6] engine: version=%s sr=%d ar_vocab=%d\n", eng->version().c_str(),
               eng->sample_rate(), eng->ar_vocab());

        std::vector<int32_t> prompt_codes, prompt_phones;
        std::vector<float> prompt_bert, bert, ge_engine, mel2;
        int mel2_T = 0;
        read_i32(dir + "prompt_codes.i32", prompt_codes);
        read_i32(dir + "prompt_phones.i32", prompt_phones);
        read_f32(dir + "prompt_bert.f32", prompt_bert);
        read_f32(dir + "bert.f32", bert);

        if (!ref_wav.empty()) {
            std::vector<int32_t> rc;
            std::vector<float> rge, rmel;
            int rT = 0;
            const auto t0 = std::chrono::steady_clock::now();
            if (!eng->prep_reference(ref_wav, rc, rge, rmel, rT)) {
                printf("prep_reference FAIL\n"); delete eng; delete s2; return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
            printf("  prep_reference: codes=%zu ge[%zu] mel2_T=%d (%.1f ms)\n", rc.size(), rge.size(), rT, ms);
            prompt_codes = rc;
            ge_engine = rge;
            std::vector<int32_t> ref_codes;
            if (read_i32(dir + "prompt_codes.i32", ref_codes) && ref_codes.size() == rc.size()) {
                size_t hit = 0;
                for (size_t i = 0; i < rc.size(); i++) hit += (rc[i] == ref_codes[i]);
                printf("  prompt codes vs python: %zu/%zu\n", hit, rc.size());
            }
        } else {
            ge_engine = ge;
        }

        if (ge_engine.size() == 512 && !prompt_codes.empty() && !prompt_phones.empty() && have_phones) {
            std::vector<int32_t> sem;
            const bool use_bert = bert.size() == (size_t) 1024 * phones.size();
            double ar_ms = 0;
            bool ar_ok = true;
            for (int it = 0; it < ar_bench && ar_ok; it++) {
                const auto t0 = std::chrono::steady_clock::now();
                ar_ok = eng->ar_generate(phones, use_bert ? bert : std::vector<float>(),
                                         prompt_phones, prompt_bert, prompt_codes,
                                         ar_max, ar_max, seed, sem, nullptr);   // (max_steps, early_stop 硬上限)
                const double one = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                ar_ms += one;
                printf("  AR iter %d: %.1f ms (sem=%zu, %.1f ms/token)\n", it + 1, one,
                       sem.size(), sem.empty() ? 0.0 : one / (double) sem.size());
            }
            if (!ar_ok) { printf("  ar_generate FAIL\n"); delete eng; delete s2; return 1; }
            ar_ms /= (double) ar_bench;
            printf("  AR(%s): %d 次, sem=%zu, %.1f ms/次\n", use_bert ? "zh w/ bert" : "no bert",
                   ar_bench, sem.size(), ar_ms);

            double syn_ms = 0;
            std::vector<float> wav;
            for (int it = 0; it < bench; it++) {
                const auto t0 = std::chrono::steady_clock::now();
                if (!eng->synthesize(sem, phones, prompt_codes, prompt_phones, ge_engine, mel2, mel2_T,
                                     0, 0.0f, seed, wav, speed, 0)) {
                    printf("  synthesize FAIL\n"); delete eng; delete s2; return 1;
                }
                syn_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
            }
            syn_ms /= (double) bench;
            double e_ms = 0, f_ms = 0, d_ms = 0;
            s2->last_stage_ms(e_ms, f_ms, d_ms);
            const double dur = (double) wav.size() / (double) eng->sample_rate();
            printf("  v2 synthesize: %d 次, wav=%zu (%.2fs) %.1f ms/次 (enc_p %.1f + flow %.1f + dec %.1f)\n",
                   bench, wav.size(), dur, syn_ms, e_ms, f_ms, d_ms);
            printf("\n[RESULT] duration=%.2fs AR=%.1fms synth=%.1fms total=%.1fms "
                   "RTF(AR+synth)=%.3f RTF(synth)=%.3f\n",
                   dur, ar_ms, syn_ms, ar_ms + syn_ms, (ar_ms + syn_ms) / 1000.0 / dur,
                   syn_ms / 1000.0 / dur);
            write_f32(dir + "ours_engine_wav.f32", wav);
        } else {
            printf("  engine: ge/prompt_codes/prompt_phones/phones 不全，跳过端到端\n");
        }
        delete eng;
    }
    delete s2;
    return 0;
}
