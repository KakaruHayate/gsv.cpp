// vocoder (ORT ONNX) 对拍 + 基准
//   golden: tools/dump_golden_vocoder.py (torch Generator 前向, fp32)
// 用法: test_vocoder [onnx_path] [golden_dir]
//   GSV_VOC_DEVICE=cpu|gpu|webgpu|dml, GSV_VOC_ORT=<onnxruntime.dll 或其目录>,
//   GSV_VOC_BENCH=N, GSV_VOC_VERBOSE=1
#include "../src/gsv_vocoder.h"

#include <algorithm>
#include <chrono>
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

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const std::string onnx = argc > 1 ? argv[1] : "models/vocoder_fp32.onnx";
    const std::string gdir = argc > 2 ? argv[2] : "tests/golden";

    gsv_vocoder_cfg cfg;
    if (const char * dv = getenv("GSV_VOC_DEVICE")) cfg.device = dv;
    if (const char * od = getenv("GSV_VOC_ORT"))    cfg.ort_dll = od;
    if (const char * nt = getenv("GSV_VOC_THREADS")) cfg.n_threads = atoi(nt);
    cfg.verbose = getenv("GSV_VOC_VERBOSE") != nullptr;

    gsv_vocoder * v = gsv_vocoder::load(onnx, cfg);
    if (!v) { fprintf(stderr, "load %s failed\n", onnx.c_str()); return 1; }
    printf("vocoder: mel_dim=%d upsample=%d EP=%s\n", v->mel_dim(), v->upsample(), v->ep_name());

    int n_fail = 0;
    for (int T : { 1, 100, 200, 257, 300, 500, 1000 }) {
        const std::vector<float> mel = read_bin(gdir + "/voc." + std::to_string(T) + ".mel.bin");
        const std::vector<float> ref = read_bin(gdir + "/voc." + std::to_string(T) + ".wav.bin");
        if (mel.empty() || ref.empty()) { printf("[T=%d] golden 缺失, skip\n", T); continue; }

        std::vector<float> wav;
        if (!v->run(mel.data(), T, wav)) { printf("[T=%d] run failed\n", T); n_fail++; continue; }
        if (wav.size() != ref.size()) {
            printf("[T=%d] SIZE %zu vs %zu  FAIL\n", T, wav.size(), ref.size()); n_fail++; continue;
        }
        double md = 0, mean = 0, rmax = 1.0;
        int n_nan = 0;
        for (size_t i = 0; i < wav.size(); i++) {
            if (!std::isfinite((double) wav[i])) { n_nan++; continue; }
            const double d = std::fabs((double) wav[i] - ref[i]);
            md = std::max(md, d);
            mean += d;
            rmax = std::max(rmax, (double) std::fabs(ref[i]));
        }
        mean /= (double) wav.size();
        // 判据: 导出时 torch-vs-ONNX 自身就是 1e-4 级; ORT EP 与 CPU 的差异 ~1e-6。
        // 用绝对 1e-3 + 相对 1e-3 双门 (波形 |wav|<=1, tanh 饱和区误差天然大一点)。
        const double thr = std::max(1e-3, 1e-3 * rmax);
        const bool ok = md < thr && n_nan == 0;
        if (!ok) n_fail++;
        printf("[T=%-5d] max|d|=%.3e mean=%.3e refmax=%.4f%s  %s\n",
               T, md, mean, rmax, n_nan ? " (NaN!!)" : "", ok ? "PASS" : "FAIL");
    }

    const char * be = getenv("GSV_VOC_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        for (int T : { 200, 500, 1000 }) {
            const std::vector<float> mel = read_bin(gdir + "/voc." + std::to_string(T) + ".mel.bin");
            if (mel.empty()) continue;
            std::vector<float> wav;
            for (int i = 0; i < 3; i++) v->run(mel.data(), T, wav);      // warmup
            std::vector<double> ts;
            for (int i = 0; i < bn; i++) {
                const auto t0 = std::chrono::steady_clock::now();
                v->run(mel.data(), T, wav);
                const auto t1 = std::chrono::steady_clock::now();
                ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ts.begin(), ts.end());
            double sum = 0; for (double x : ts) sum += x;
            printf("bench T=%-5d (%.2fs audio): avg %7.2f ms  min %7.2f ms  n=%d  [%s]\n",
                   T, T / 100.0, sum / ts.size(), ts.front(), bn, v->ep_name());
        }
    }
    delete v;
    printf("\n%s\n", n_fail == 0 ? "VOCODER ALL PASSED" : "VOCODER FAILED");
    return n_fail == 0 ? 0 : 1;
}
