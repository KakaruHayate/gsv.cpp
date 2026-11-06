// Regression: the reference-audio preprocessing must be repeatable.
//
// 背景 (2026-11-06 用户报的 "第一次渲染正常, 第二次崩坏"):
//   gsv_hubert 的 g2 注意力 mask 是**建图时写一次**的常量, 但当时它住在 galloc 的 arena 里。
//   gallocr 按"最后一次使用"回收区域, 于是同一次 compute 的尾声 (最后一层的 FFN) 就把 mask
//   覆盖成了别的张量的位模式; 第二次 encode() 起读到的 mask 全是垃圾 -> 注意力被随机加权 ->
//   g2 输出沿时间轴几乎恒定 -> ssl 每帧相同 -> refcode 的 argmin 退化成同一个码 ->
//   prompt_codes 变成一串常数 -> AR 失去参考条件 -> 第二次渲染开始语无伦次/时长暴涨。
//
// 这个测试只跑参考链路 (resample -> HuBERT -> ssl_proj/RVQ), 断言:
//   1) 同一段音频连续预处理 N 次, codes 必须逐位相同;
//   2) codes 不能退化 (不同码的数量要够多);
//   3) 中间插入一段不同长度的参考 (迫使两张图重建) 之后, 结果仍要可重复。
//
//   test_prep_repeat <models_dir> <ref_audio> [other_ref_audio]
//   models_dir 里需要有 gsv-hubert-f16.gguf / gsv-refcode.gguf / resampy_kaiser_best.bin
#include "gsv_hubert.h"
#include "gsv_mel.h"
#include "gsv_refcode.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int D_SSL = 768;

// gsv_engine::prep_reference 的 (a) 段: 16k + 0.3s 零尾 -> HuBERT -> ssl_proj + RVQ
static bool prep(gsv_hubert * hub, gsv_refcode * rc, const std::string & wav_path,
                 std::vector<int32_t> & codes) {
    codes.clear();
    std::vector<float> raw;
    int sr = 0;
    if (!gsv_wav_load(wav_path, raw, sr)) { printf("  wav load failed: %s\n", wav_path.c_str()); return false; }
    std::vector<float> w16;
    gsv_refcode::resample_librosa(raw.data(), (int64_t) raw.size(), sr, 16000, w16);
    const int zero_tail = (int) (32000 * 0.3);
    std::vector<float> fed(w16.size() + (size_t) zero_tail, 0.0f);
    if (!w16.empty()) std::memcpy(fed.data(), w16.data(), w16.size() * sizeof(float));
    std::vector<float> hout;
    if (!hub->encode(fed.data(), (int) fed.size(), hout)) { printf("  hubert failed\n"); return false; }
    const int Th = (int) (hout.size() / D_SSL);
    std::vector<float> ssl((size_t) D_SSL * Th);
    for (int d = 0; d < D_SSL; d++)
        for (int t = 0; t < Th; t++) ssl[(size_t) d * Th + t] = hout[(size_t) d + (size_t) D_SSL * t];
    return rc->codes_from_ssl(ssl.data(), Th, codes);
}

static int uniq(const std::vector<int32_t> & v) {
    std::vector<int32_t> s = v;
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    return (int) s.size();
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <models_dir> <ref_audio> [other_ref_audio]\n", argv[0]);
        return 2;
    }
    const std::string md = std::string(argv[1]) + "/";
    const std::string ref = argv[2];
    const char * other = argc > 3 ? argv[3] : nullptr;

    const char * dev = getenv("GSV_TEST_DEVICE");   // "" = CPU; "vulkan" 也复现过同一个 bug
    gsv_hubert_cfg hc;  hc.device = dev ? dev : "";
    gsv_refcode_cfg rcfg; rcfg.device = dev ? dev : "";
    gsv_hubert * hub = gsv_hubert::load(md + "gsv-hubert-f16.gguf", hc);
    gsv_refcode * rc = gsv_refcode::load(md + "gsv-refcode.gguf", rcfg);
    if (!hub || !rc) { printf("model load failed\n"); return 1; }
    if (!gsv_refcode::load_filter_table(md + "resampy_kaiser_best.bin")) {
        printf("resampy filter table missing\n"); return 1;
    }

    int fails = 0;
    std::vector<int32_t> first;
    for (int i = 0; i < 4; i++) {
        std::vector<int32_t> codes;
        if (!prep(hub, rc, ref, codes)) { printf("FAIL prep #%d\n", i + 1); return 1; }
        if (i == 0) first = codes;
        const bool same = (codes == first);
        const int u = uniq(codes);
        printf("prep #%d: codes=%zu uniq=%d %s\n", i + 1, codes.size(), u, same ? "== #1" : "!= #1 <-- BUG");
        if (!same) fails++;
        if (u < 8) { printf("  degenerate codes (uniq=%d) <-- BUG\n", u); fails++; }
        // 中间插入一段不同长度的参考: 迫使 g1/g2 按新长度重建, 检验重建路径也可重复
        if (other && i == 1) {
            std::vector<int32_t> dummy;
            if (!prep(hub, rc, other, dummy)) { printf("FAIL prep(other)\n"); return 1; }
            printf("prep(other len): codes=%zu uniq=%d\n", dummy.size(), uniq(dummy));
        }
    }
    delete hub;
    delete rc;
    printf(fails == 0 ? "PREP_REPEAT_OK\n" : "PREP_REPEAT_FAILED (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
