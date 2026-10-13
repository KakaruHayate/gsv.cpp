// 参考音频 -> refer_audio_spec (spectrogram_torch) -> ref_enc -> ge 对拍
//   golden: tools/dump_golden_refspec.py
// 用法: test_refspec [golden_dir] [audio_dir] [cond_gguf]; GSV_REFSPEC_DEVICE=vulkan
#include "../src/gsv_mel.h"
#include "../src/gsv_refenc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

static int n_fail = 0;
static void cmp(const char * what, const std::vector<float> & a, const std::vector<float> & b,
                double tol, double rel_tol = 0.0) {
    if (a.size() != b.size()) { printf("  %-18s SIZE %zu vs %zu  FAIL\n", what, a.size(), b.size()); n_fail++; return; }
    double md = 0, refmax = 1.0;
    for (size_t i = 0; i < a.size(); i++) {
        md = std::max(md, std::fabs((double) a[i] - b[i]));
        refmax = std::max(refmax, (double) std::fabs(b[i]));
    }
    const double thr = std::max(tol, rel_tol * refmax);
    const bool ok = md < thr;
    if (!ok) n_fail++;
    printf("  %-18s max|d|=%.3e refmax=%.3g  %s\n", what, md, refmax, ok ? "PASS" : "FAIL");
}

// torch [C,T] (idx = c*T+t) -> ggml [C,T] (idx = c + t*C)
static std::vector<float> to_ggml(const std::vector<float> & x, int C, int T) {
    std::vector<float> y((size_t) C * T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++) y[(size_t) c + (size_t) t * C] = x[(size_t) c * T + t];
    return y;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const std::string gdir = argc > 1 ? argv[1] : "tests/golden";
    const std::string adir = argc > 2 ? argv[2] : "tests/audio";
    const std::string cgg  = argc > 3 ? argv[3] : "models/gsv-cond-f32.gguf";

    gsv_refenc_cfg rcfg;
    rcfg.verbose = true;
    if (const char * dv = getenv("GSV_REFSPEC_DEVICE")) rcfg.device = dv;
    if (const char * nt = getenv("GSV_REFSPEC_THREADS")) rcfg.n_threads = atoi(nt);
    gsv_refenc * re = gsv_refenc::load(cgg, rcfg);
    if (!re) { fprintf(stderr, "load %s failed\n", cgg.c_str()); return 1; }
    if (re->in_dim() != 704) { fprintf(stderr, "ref_enc in_dim=%d != 704\n", re->in_dim()); return 1; }

    // 1) 合成输入: 全链 (wav32k_norm -> spectrogram_torch -> 704 -> ge)
    //    (wav32k_norm 已是 32k 归一化波形, 隔离重采样误差)
    {
        const std::string p = gdir + "/refspec.syn.";
        const std::vector<float> w32 = read_bin(p + "wav32k_norm.bin");
        const std::vector<float> sref = read_bin(p + "spec.bin");
        const std::vector<float> s704 = read_bin(p + "spec704.bin");
        const std::vector<float> gref = read_bin(p + "ge.bin");
        if (!w32.empty() && !sref.empty() && !gref.empty()) {
            const int T = (int) (sref.size() / 1025);
            printf("[syn] wav32k_norm %zu samples, spec T=%d\n", w32.size(), T);

            std::vector<float> sp704;
            int T2 = 0;
            if (!gsv_refspec_wav(w32.data(), (int64_t) w32.size(), 32000, sp704, T2, false)) {
                printf("  gsv_refspec_wav failed\n"); n_fail++;
            } else {
                // spec (全 1025) golden 需单独算: 内部裁到 704, 故这里直接比对 704 段
                cmp("spec704", sp704, s704, 2e-3);   // FFT 舍入
                if (T2 != T) { printf("  T 不一致: %d vs %d FAIL\n", T2, T); n_fail++; }

                std::vector<float> x = to_ggml(sp704, 704, T2);
                std::vector<float> fm((size_t) T2, 1.0f), ge;
                if (!re->encode(x.data(), fm.data(), T2, ge)) { n_fail++; }
                else {
                    double md = 0, rmax = 1.0;
                    for (size_t i = 0; i < ge.size(); i++) {
                        md = std::max(md, std::fabs((double) ge[i] - gref[i]));
                        rmax = std::max(rmax, (double) std::fabs(gref[i]));
                    }
                    // 与 test_refenc 同口径: 本模块 z 值域 ~±1000, Vulkan F16 底噪 ~1e-3 相对
                    const bool ok = md < 2e-2 * rmax;
                    if (!ok) n_fail++;
                    printf("  %-18s max|d|=%.3e refmax=%.3g  %s\n", "ge", md, rmax, ok ? "PASS" : "FAIL");
                }
            }
        } else {
            printf("[syn] golden 缺失 (先跑 tools/dump_golden_refspec.py --T <n>)\n");
        }
    }

    // 2) 真实素材: 全链 (wav -> resample 32k -> maxx 归一化 -> spec -> 704 -> ge)
    for (auto & spec : { std::pair<const char *, const char *>{ "zh", "ref_zh_32k.wav" },
                         std::pair<const char *, const char *>{ "en", "ref_en_16k.wav" },
                         std::pair<const char *, const char *>{ "zhs", "ref_zh_short_16k.wav" } }) {
        const char * tag = spec.first;
        const std::string wavp = adir + "/" + spec.second;
        const std::string p = gdir + "/refspec." + tag + ".";
        std::vector<float> raw;
        int sr = 0;
        if (!gsv_wav_load(wavp, raw, sr)) { printf("[%s] wav 缺失, skip\n", tag); continue; }
        printf("[%s] wav: %zu samples @ %d Hz\n", tag, raw.size(), sr);

        // 2a) 重采样 + maxx 归一化后的 32k 波形 (隔离重采样误差)
        const std::vector<float> w32g = read_bin(p + "wav32k_norm.bin");
        std::vector<float> r32;
        gsv_resample(raw.data(), (int64_t) raw.size(), sr, 32000, r32);
        if (!w32g.empty()) {
            if (r32.size() == w32g.size()) {
                double maxx = 0;
                for (float v : r32) maxx = std::max(maxx, (double) std::fabs(v));
                if (maxx > 1.0) { const double d = std::min(2.0, maxx); for (float & v : r32) v = (float) ((double) v / d); }
                cmp("resample32k+norm", r32, w32g, 1e-3);
            } else {
                printf("  %-18s SIZE %zu vs %zu  FAIL\n", "resample32k+norm", r32.size(), w32g.size()); n_fail++;
            }
        }

        // 2b) 全链 (含 maxx 归一化) -> spec704 -> ge
        std::vector<float> sp704;
        int T = 0;
        if (!gsv_refspec_wav(raw.data(), (int64_t) raw.size(), sr, sp704, T, true)) {
            printf("  gsv_refspec_wav failed\n"); n_fail++; continue;
        }
        cmp("spec704", sp704, read_bin(p + "spec704.bin"), 2e-3);
        const std::vector<float> gref = read_bin(p + "ge.bin");
        if (gref.empty()) { printf("  [%s] golden ge 缺失\n", tag); continue; }
        std::vector<float> x = to_ggml(sp704, 704, T);
        std::vector<float> fm((size_t) T, 1.0f), ge;
        if (!re->encode(x.data(), fm.data(), T, ge)) { n_fail++; continue; }
        {
            double md = 0, rmax = 1.0;
            for (size_t i = 0; i < ge.size(); i++) {
                md = std::max(md, std::fabs((double) ge[i] - gref[i]));
                rmax = std::max(rmax, (double) std::fabs(gref[i]));
            }
            const bool ok = md < 2e-2 * rmax;
            if (!ok) n_fail++;
            printf("  %-18s max|d|=%.3e refmax=%.3g (T=%d)  %s\n", "ge", md, rmax, T, ok ? "PASS" : "FAIL");
        }

        // 2c) 参考帧数口径 (synthesize_v5_mel::reference_frames = min(mel2.T, fea_ref.T)):
        //     mel2 = mel_fn_v4(ref_audio) 的帧数 (hop 320 @32k)
        //     fea_ref = 4·prompt_codes  (25Hz -> x2 -> enc_p 50Hz -> bridge x2 -> 100Hz)
        //   prompt codes 来自参考音频 + 0.6s 零尾 -> fea_ref 通常比 mel2 多 ~60 帧,
        //   即 min() 恒取 mel2 (参考 mel 决定 prompt 长度)。这里把这个约定钉死。
        {
            const std::vector<float> melg = read_bin(gdir + "/ref." + tag + ".mel.bin");
            std::vector<float> cg;
            {   // codes 以 float32 存 (与 refcode golden 同格式)
                FILE * f = fopen((gdir + "/ref." + tag + ".codes.bin").c_str(), "rb");
                if (f) {
                    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
                    cg.resize((size_t) sz / 4);
                    if (fread(cg.data(), 4, cg.size(), f) != cg.size()) cg.clear();
                    fclose(f);
                }
            }
            if (!melg.empty() && !cg.empty()) {
                const int T_mel = (int) (melg.size() / 100);
                const int T_fea = 4 * (int) cg.size();
                const bool mel_wins = T_mel <= T_fea;
                // 若 mel2 更长, synthesize 会裁 fea_ref -> prompt 仍是 500 帧上限, 但
                // prompt 内容来自 mel 的那一侧; 真实素材上必须 T_mel <= T_fea (零尾所致)
                if (!mel_wins) n_fail++;
                printf("  %-18s T_mel=%d T_fea=4*%zu=%d -> min 取 %s  %s\n",
                       "ref_frames", T_mel, cg.size(), T_fea,
                       mel_wins ? "mel2" : "fea_ref", mel_wins ? "PASS" : "FAIL");
            } else if (melg.empty()) {
                printf("  %-18s skip (缺 ref.%s.mel.bin)\n", "ref_frames", tag);
            } else {
                // dump_golden_ref.py 的 tag 是 zh/en, dump_golden_refcode.py 的是 zhs/en/zh32
                printf("  %-18s skip (缺 ref.%s.codes.bin, 见 refcode golden 的 tag 命名)\n",
                       "ref_frames", tag);
            }
        }
    }
    delete re;
    printf("\n%s\n", n_fail == 0 ? "REFSPEC ALL PASSED" : "REFSPEC FAILED");
    return n_fail == 0 ? 0 : 1;
}
