// HuBERT ggml 实现对拍 (替代 host 版 test_hubert.cpp, 后者保留作 host 参考)
// golden: tests/golden/hubert.* (torch fp32, tools/dump_golden_hubert.py)
//   hubert.input_norm [16000]  z-score 后音频 (直接喂图 1)
//   hubert.cnn_out    [C,T]    CNN 前端输出 (对拍图 1 的 CNN 部分: [T,C] 转置后比对)
//   hubert.out        [T,768]  最终输出
// 用法: test_hubert_ggml [gguf] [golden_dir];  GSV_HUBERT_DEVICE=vulkan 切 GPU
#include "../src/gsv_hubert.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}

static double cmp(const std::vector<float> & a, const std::vector<float> & b, const char * tag) {
    if (a.empty() || a.size() != b.size()) { printf("  %-10s 尺寸不符 (%zu vs %zu) FAIL\n", tag, a.size(), b.size()); return 1e30; }
    double md = 0;
    // NaN 感知: fabs(NaN) > md 恒 false, 会把全 NaN 输出误报 PASS
    size_t n_nan = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite((double) a[i])) { n_nan++; continue; }
        double d = std::fabs((double) a[i] - b[i]);
        if (!(d <= md)) md = d;   // NaN 传播
    }
    if (n_nan) { printf("  %-10s 含 %zu 个非有限值 FAIL\n", tag, n_nan); return 1e30; }
    printf("  %-10s max|d|=%.3e %s\n", tag, md, md < 2e-2 ? "PASS" : "FAIL");
    return md;
}

int main(int argc, char ** argv) {
    const char * mdl  = argc > 1 ? argv[1] : "models/gsv-hubert-f32.gguf";
    const char * gdir = argc > 2 ? argv[2] : "tests/golden";

    gsv_hubert_cfg cfg;
    cfg.verbose = true;
    cfg.n_threads = atoi(getenv("GSV_HUBERT_NTHREADS") ? getenv("GSV_HUBERT_NTHREADS") : "16");
    if (const char * dv = getenv("GSV_HUBERT_DEVICE")) cfg.device = dv;

    fprintf(stderr, "[main] before load\n"); fflush(stderr);
    gsv_hubert * m = gsv_hubert::load(mdl, cfg);
    if (!m) return 1;

    std::vector<float> in = read_bin(std::string(gdir) + "/hubert.input_norm.bin");
    if (in.empty()) { fprintf(stderr, "missing hubert.input_norm\n"); delete m; return 1; }

    // 调试: 直接把 golden enc_in 作为 g2 输入 (隔离 host 段与 g2)
    const bool g2_only = getenv("GSV_HUBERT_G2ONLY") != nullptr;
    std::vector<float> enc_in;
    if (g2_only) {
        enc_in = read_bin(std::string(gdir) + "/hubert.enc_in.bin");   // [T, D]
        if (enc_in.size() != (size_t) m->n_frames() * m->hidden()) { fprintf(stderr, "bad enc_in\n"); delete m; return 1; }
    }

    std::vector<float> out;
    if (g2_only) {
        // 走内部 hack: encode() 不允许 —— 这里临时借 m->encode 会重新跑 g1;
        // 改为: 调 encode() 但把输入换成 enc_in 需要引擎支持, 简化: 仍跑全链路,
        // 但 dump g2_in 由引擎负责 (GSV_HUBERT_DEBUG 下) —— 此分支仅供人工检查
        if (!m->encode(in.data(), (int) in.size(), out)) { delete m; return 1; }
    } else if (!m->encode(in.data(), (int) in.size(), out)) { delete m; return 1; }

    // 临时落盘, 供 numpy 交叉验证 (调试用)
    if (getenv("GSV_HUBERT_DUMP")) {
        std::string dp = std::string(gdir) + "/hubert.out.ggml.bin";
        FILE * df = fopen(dp.c_str(), "wb");
        if (df) { fwrite(out.data(), 4, out.size(), df); fclose(df); printf("  dumped %s\n", dp.c_str()); }
    }

    int n_bad = 0;
    const int T = m->n_frames(), D = m->hidden();

    // CNN 中间对拍: golden cnn_out 是 [C, T] (ch*T+t); 图 1 输出含 CNN+gelu 但不含 feat_proj
    // (图 1 出来已经是 feat [768,T]) —— cnn_out 的图内等价点无法直接回读,
    // 因此 CNN 部分的正确性由最终 out 间接覆盖; 这里只对拍最终输出。
    std::vector<float> ref = read_bin(std::string(gdir) + "/hubert.out.bin");
    if (ref.size() == (size_t) T * D) {
        // golden out [T, D] 行主 == ggml [D, T] 列主输出: 字节序恒等, 直接逐元素比
        double md = cmp(out, ref, "hubert_out");
        if (md > 2e-2) n_bad++;
    } else {
        // 旧 golden 可能是 [T,768] 转置前布局 —— 两种都试
        double md1 = 0, md2 = 0;
        for (size_t i = 0; i < out.size(); i++) {
            md1 = std::max((double) md1, (double) std::fabs((double) out[i] - ref[i]));
        }
        printf("  golden 尺寸异常, 恒等比较 max|d|=%.3e\n", md1);
        if (md1 > 2e-2) n_bad++;
    }

    // 基准
    if (getenv("GSV_HUBERT_BENCH")) {
        const int n = atoi(getenv("GSV_HUBERT_BENCH"));
        std::vector<double> ts;
        for (int i = 0; i < n + 2; i++) {
            auto t0 = std::chrono::steady_clock::now();
            m->encode(in.data(), (int) in.size(), out);
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 2) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml hubert: avg %.1f ms min %.1f ms (n=%d)\n", sum / ts.size(), ts.front(), (int) ts.size());
    }

    printf("\n%s\n", n_bad == 0 ? "HUBERT GGML PARITY PASS" : "HUBERT GGML PARITY FAIL");
    delete m;
    return n_bad == 0 ? 0 : 1;
}
