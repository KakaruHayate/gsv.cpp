// probe: distilled HuBERT (gsv-dhubert GGUF) 数值对拍 + CPU 计时
//
//   dhubert_probe <dhubert.gguf> <in.wav> [out.f32] [--ref r.f32] [--runs N]
//                 [--threads N] [--device cpu|vulkan|1] [--dump prefix]
//
//   in.wav  任意采样率单声道/多声道 wav (用 gsv_wav_load + gsv_resample 变成 16k mono)
//   out.f32 [hidden, T] f32 行主序 (idx = d + t*hidden), 与 tools/dump_ref_dhubert.py 的
//           输出逐字节可比
//   --ref   参考特征文件, 计算 max abs / mean abs / 相关系数 并打印对照表
//   --dump  额外写出中间张量 (prefix.front.f32 / .tcn.f32 / .attnN.f32), 供分段定位数值偏差
#include "gsv_dhubert.h"
#include "gsv_mel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string & p, size_t expect = 0) {
    std::vector<float> v;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return v; }
    v.resize((size_t) n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) v.clear();
    fclose(f);
    if (expect && v.size() != expect) {
        fprintf(stderr, "[probe] %s: %zu floats, expected %zu\n", p.c_str(), v.size(), expect);
        v.clear();
    }
    return v;
}

static bool write_f32(const std::string & p, const std::vector<float> & v) {
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "[probe] cannot write %s\n", p.c_str()); return false; }
    const bool ok = fwrite(v.data(), 4, v.size(), f) == v.size();
    fclose(f);
    return ok;
}

struct diffstat {
    double max_abs = 0, mean_abs = 0, corr = 0, rel = 0;
    int    at_d = 0, at_t = 0;
    double ref_max = 0, out_max = 0;
};

// a = ours, b = reference; both [D, T] row-major
static diffstat compare(const std::vector<float> & a, const std::vector<float> & b, int D, int T) {
    diffstat s;
    const size_t n = a.size();
    if (n != b.size() || n == 0) return s;
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, ad = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = a[i], y = b[i], d = std::fabs(x - y);
        ad += d;
        if (d > s.max_abs) { s.max_abs = d; s.at_d = (int) (i % (size_t) D); s.at_t = (int) (i / (size_t) D); }
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
        s.ref_max = std::max(s.ref_max, (double) std::fabs(y));
        s.out_max = std::max(s.out_max, (double) std::fabs(x));
    }
    s.mean_abs = ad / (double) n;
    const double cov = sab / (double) n - (sa / (double) n) * (sb / (double) n);
    const double va  = saa / (double) n - (sa / (double) n) * (sa / (double) n);
    const double vb  = sbb / (double) n - (sb / (double) n) * (sb / (double) n);
    s.corr = (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0.0;
    s.rel  = s.max_abs / (s.ref_max > 0 ? s.ref_max : 1.0);
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <dhubert.gguf> <in.wav> [out.f32] [--ref r.f32] [--runs N]\n"
               "          [--threads N] [--device cpu|vulkan|1] [--dump prefix] [--check-rebuild]\n"
               "  --ref           参考 f32 (tools/dump_ref_dhubert.py), 打印 diff 对照表\n"
               "  --dump          写中间张量 (prefix.front{i}/tcn/attn{i}.f32) 分段对拍\n"
               "  --check-rebuild 换长度跑一遍再换回来, 校验图重建后输出逐位一致\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1], wav = argv[2];
    std::string out_path, ref_path, dev, dump;
    int runs = 3, threads = 0;
    bool check_rebuild = false;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--ref" && i + 1 < argc) ref_path = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--device" && i + 1 < argc) dev = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--check-rebuild") check_rebuild = true;
        else if (!a.empty() && a[0] != '-') out_path = a;
    }
    if (!dump.empty() && !getenv("GSV_DHUBERT_DUMP")) setenv("GSV_DHUBERT_DUMP", dump.c_str(), 1);

    std::vector<float> audio;
    int sr = 0;
    if (!gsv_wav_load(wav, audio, sr)) { fprintf(stderr, "[probe] cannot read %s\n", wav.c_str()); return 1; }
    if (sr != 16000) {
        std::vector<float> rs;
        gsv_resample(audio.data(), (int64_t) audio.size(), sr, 16000, rs);
        printf("resample: %d Hz -> 16000 Hz (%zu -> %zu samples)\n", sr, audio.size(), rs.size());
        audio.swap(rs);
    }
    printf("wav: %s  %zu samples (%.3f s @16k)\n", wav.c_str(), audio.size(), (double) audio.size() / 16000.0);

    gsv_dhubert_cfg cfg;
    cfg.n_threads = threads;
    cfg.device = dev;
    cfg.verbose = true;
    const auto t_load0 = std::chrono::steady_clock::now();
    gsv_dhubert * m = gsv_dhubert::load(gguf, cfg);
    const double t_load = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_load0).count();
    if (!m) { fprintf(stderr, "[probe] load failed\n"); return 1; }
    printf("load: %.1f ms\n", t_load);

    std::vector<float> feat;
    int rc_rebuild = 0;
    double t_first = 0;
    std::vector<double> times;
    const char * act = getenv("GSV_DHUBERT_ACT") ? getenv("GSV_DHUBERT_ACT") : "relu";
    for (int r = 0; r < std::max(1, runs); r++) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!m->encode(audio.data(), (int) audio.size(), feat)) { fprintf(stderr, "[probe] encode failed\n"); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        if (r == 0) t_first = ms; else times.push_back(ms);
        printf("run%d: %.2f ms (T=%d, dim=%d, act=%s)\n", r + 1, ms, m->n_frames(), m->hidden(), act);
    }
    if (!times.empty()) {
        std::vector<double> s = times;
        std::sort(s.begin(), s.end());
        const double med = s[s.size() / 2];
        const double audio_s = (double) audio.size() / 16000.0;
        printf("steady-state: median %.2f ms  (min %.2f / max %.2f), %.3f s audio -> %.1fx realtime\n",
               med, s.front(), s.back(), audio_s, audio_s * 1000.0 / med);
        printf("first call (incl. graph build): %.2f ms\n", t_first);
    }

    if (!out_path.empty()) {
        if (!write_f32(out_path, feat)) return 1;
        printf("wrote %s (%zu floats, [%d, %d])\n", out_path.c_str(), feat.size(), m->hidden(), m->n_frames());
    }

    // 图重建回归: 先跑一个不同长度的片段 (触发 build 重建), 再跑原始长度, 结果必须逐位相同
    if (check_rebuild) {
        const std::vector<float> base = feat;
        std::vector<float> mid, again;
        const int n_mid = std::max((int) audio.size() / 2, 400);
        if (!m->encode(audio.data(), n_mid, mid)) { fprintf(stderr, "[probe] mid encode failed\n"); return 1; }
        printf("rebuild: %d samples -> T=%d\n", n_mid, m->n_frames());
        if (!m->encode(audio.data(), (int) audio.size(), again)) { fprintf(stderr, "[probe] re-encode failed\n"); return 1; }
        double dmax = 0;
        for (size_t i = 0; i < base.size() && i < again.size(); i++)
            dmax = std::max(dmax, (double) std::fabs(base[i] - again[i]));
        const bool same = base.size() == again.size() && dmax == 0.0;
        printf("rebuild: re-encode of the original length is %s (max diff %.3e)\n",
               same ? "bit-identical" : "DIFFERENT", dmax);
        feat = base;
        if (!same) rc_rebuild = 3;
    }

    int rc = rc_rebuild;

    if (!ref_path.empty()) {
        const std::vector<float> ref = read_f32(ref_path, feat.size());
        if (ref.empty()) { fprintf(stderr, "[probe] reference %s unusable\n", ref_path.c_str()); return 1; }
        const diffstat d = compare(feat, ref, m->hidden(), m->n_frames());
        printf("\n=== vs %s (%d x %d) ===\n", ref_path.c_str(), m->hidden(), m->n_frames());
        printf("  max abs diff  : %.6e   (at d=%d t=%d)\n", d.max_abs, d.at_d, d.at_t);
        printf("  mean abs diff : %.6e\n", d.mean_abs);
        printf("  corr          : %.9f\n", d.corr);
        printf("  rel max       : %.3e  (|ref|max=%.4f, |out|max=%.4f)\n", d.rel, d.ref_max, d.out_max);
        const bool pass = d.max_abs < 1e-2 && d.corr > 0.999;
        printf("  verdict       : %s (need max_abs < 1e-2 && corr > 0.999)\n", pass ? "PASS" : "FAIL");
        if (!pass) rc = 3;
    }
    delete m;
    return rc;
}
