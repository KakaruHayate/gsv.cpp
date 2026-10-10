// 参考音频预处理对拍: wav 读取 / 重采样 / mel_fn_v4 / norm_spec
//   golden: tools/dump_golden_ref.py (torchaudio + repo 的 mel_spectrogram_torch)
// 用法: test_mel [golden_dir] [audio_dir]; GSV_MEL_THREADS=N
#include "../src/gsv_mel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static void cmp(const char * what, const std::vector<float> & a, const std::vector<float> & b, double tol) {
    if (a.size() != b.size()) { printf("  %-16s SIZE %zu vs %zu  FAIL\n", what, a.size(), b.size()); return; }
    double md = 0, mean = 0, refmax = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = std::fabs((double) a[i] - b[i]);
        md = std::max(md, d);
        mean += d;
        refmax = std::max(refmax, (double) std::fabs(b[i]));
    }
    mean /= (double) a.size();
    printf("  %-16s max|d|=%.3e mean=%.3e refmax=%.3g  %s\n", what, md, mean, refmax,
           md < tol ? "PASS" : "FAIL");
}

int main(int argc, char ** argv) {
    const std::string gdir = argc > 1 ? argv[1] : "tests/golden";
    const std::string adir = argc > 2 ? argv[2] : "tests/audio";

    gsv_mel_cfg cfg = {};   // 默认 = mel_fn_v4
    if (const char * nt = getenv("GSV_MEL_THREADS")) cfg.n_threads = atoi(nt);
    gsv_mel mel(cfg);

    int n_fail = 0;
    for (auto & spec : { std::pair<const char *, const char *>{ "zh", "ref_zh_32k.wav" },
                         std::pair<const char *, const char *>{ "en", "ref_en_16k.wav" } }) {
        const char * tag = spec.first;
        const std::string wavp = adir + "/" + spec.second;
        std::vector<float> raw;
        int sr = 0;
        if (!gsv_wav_load(wavp, raw, sr)) { printf("[%s] wav load failed: %s\n", tag, wavp.c_str()); n_fail++; continue; }
        printf("[%s] wav: %zu samples @ %d Hz (%.2fs)\n", tag, raw.size(), sr, (double) raw.size() / sr);

        const std::string p = gdir + "/ref." + tag + ".";
        // 32k / 16k 重采样对照 (源 sr -> 目标)
        std::vector<float> r32, r16;
        gsv_resample(raw.data(), (int64_t) raw.size(), sr, 32000, r32);
        gsv_resample(raw.data(), (int64_t) raw.size(), sr, 16000, r16);
        cmp("resample->32k", r32, read_bin(p + "wav32k.bin"), 1e-3);
        cmp("resample->16k", r16, read_bin(p + "wav16k.bin"), 1e-3);

        // mel (用 golden 的 32k 样本, 隔离重采样误差) + norm_spec
        const std::vector<float> w32 = read_bin(p + "wav32k.bin");
        std::vector<float> m, mn;
        int T = 0, T2 = 0;
        if (!mel.forward(w32.data(), (int64_t) w32.size(), m, T)) { printf("  mel forward failed\n"); n_fail++; continue; }
        mel.forward_norm(w32.data(), (int64_t) w32.size(), mn, T2);
        printf("  mel frames: C++ T=%d (expect %zu)\n", T, read_bin(p + "mel.bin").size() / 100);
        cmp("mel", m, read_bin(p + "mel.bin"), 3e-3);   // log 域; FFT 舍入 + log 放大
        cmp("mel_norm", mn, read_bin(p + "mel_norm.bin"), 5e-4);
    }
    printf("\n%s\n", n_fail == 0 ? "MEL ALL PASSED" : "MEL FAILED");
    return n_fail == 0 ? 0 : 1;
}
