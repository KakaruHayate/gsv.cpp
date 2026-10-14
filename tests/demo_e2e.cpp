// 端到端 (半程) 合成演示: phones+bert -> AR -> codes -> refspec/hubert 参考侧 ->
//   cond (RVQ->enc_p->bridge->wns1) -> DiT synthesize_v5_mel -> vocoder -> wav
//
// 覆盖: AR 引擎 / 参考音频三链 (16k codes + 32k spec->ge + 32k mel) / 条件段 / DiT 分块 / vocoder
// 未覆盖: 文本前端 (字符 -> phones/bert, 需要 G2PW + tokenizer + BERT 前端接线)
//        —— 这里 phones/bert 由 golden 提供 (tests/golden/ar.phones.bin)
//
// 用法: demo_e2e [--ref <wav>] [--out <wav>] [--steps 4] [--device cpu|vulkan]
//   GSV_E2E_VOC_DEVICE / GSV_E2E_DEVICE 控制 vocoder 与 ggml 侧后端
// 输出: 16-bit PCM wav (32kHz)
#include "../src/gsv_ar.h"
#include "../src/gsv_bert.h"
#include "../src/gsv_cond.h"
#include "../src/gsv_dit.h"
#include "../src/gsv_encp.h"
#include "../src/gsv_hubert.h"
#include "../src/gsv_mel.h"
#include "../src/gsv_refcode.h"
#include "../src/gsv_refenc.h"
#include "../src/gsv_vocoder.h"
#include "../src/gsv_wns1.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------- wav 写出 (16-bit PCM) ----------------
static bool wav_write(const std::string & path, const std::vector<float> & x, int sr) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const int32_t n = (int32_t) x.size();
    const int32_t byte_rate = sr * 2;
    fwrite("RIFF", 1, 4, f);
    int32_t chunk = 36 + n * 2;      fwrite(&chunk, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int32_t fmt_sz = 16;             fwrite(&fmt_sz, 4, 1, f);
    int16_t af = 1;                  fwrite(&af, 2, 1, f);
    int16_t ch = 1;                  fwrite(&ch, 2, 1, f);
    int32_t sr32 = sr;               fwrite(&sr32, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    int16_t ba = 2;                  fwrite(&ba, 2, 1, f);
    int16_t bits = 16;               fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    int32_t data_sz = n * 2;         fwrite(&data_sz, 4, 1, f);
    std::vector<int16_t> pcm((size_t) n);
    for (int32_t i = 0; i < n; i++) {
        float v = x[(size_t) i];
        v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        pcm[(size_t) i] = (int16_t) (v * (v >= 0 ? 32767.0f : 32768.0f));
    }
    const bool ok = fwrite(pcm.data(), 2, pcm.size(), f) == pcm.size();
    fclose(f);
    return ok;
}

static std::vector<float> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fclose(f); return {}; }
    fclose(f);
    return v;
}
static double ms_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::string ref_wav = "tests/audio/ref_zh_32k.wav";
    std::string out_wav = "tests/e2e_out.wav";
    int  steps = 4;
    const char * gdir = "tests/golden";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref_wav = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_wav = argv[++i];
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--golden") && i + 1 < argc) gdir = argv[++i];
    }
    const char * dev = getenv("GSV_E2E_DEVICE");
    const std::string device = dev ? dev : "";

    // ================================================================
    // A) 参考音频 -> 三份产出
    // ================================================================
    printf("=== A) 参考音频: %s ===\n", ref_wav.c_str());
    const auto tA = std::chrono::steady_clock::now();

    // A1) 16k -> HuBERT -> ssl_proj -> RVQ -> prompt codes (AR 的语义 prompt)
    std::vector<float> raw;
    int sr = 0;
    if (!gsv_wav_load(ref_wav, raw, sr)) { fprintf(stderr, "wav 读取失败\n"); return 1; }
    printf("  wav %zu samples @ %d Hz (%.2fs)\n", raw.size(), sr, (double) raw.size() / sr);

    if (!gsv_refcode::load_filter_table("models/resampy_kaiser_best.bin")) {
        fprintf(stderr, "resampy 表缺失\n"); return 1;
    }
    std::vector<float> w16;
    gsv_refcode::resample_librosa(raw.data(), (int64_t) raw.size(), sr, 16000, w16);
    // repo: 拼 0.3s 零尾 (int(32000*0.3) = 9600)
    const int zero_tail = (int) (32000 * 0.3);
    std::vector<float> fed(w16.size() + (size_t) zero_tail, 0.0f);
    memcpy(fed.data(), w16.data(), w16.size() * 4);
    printf("  16k (%zu) + 零尾 %d = %zu\n", w16.size(), zero_tail, fed.size());

    gsv_hubert_cfg hcfg; hcfg.device = device;
    gsv_hubert * hub = gsv_hubert::load("models/gsv-hubert-f32.gguf", hcfg);
    if (!hub) { fprintf(stderr, "hubert 加载失败\n"); return 1; }
    std::vector<float> hout;
    if (!hub->encode(fed.data(), (int) fed.size(), hout)) { fprintf(stderr, "hubert 编码失败\n"); return 1; }
    const int Th = (int) (hout.size() / 768);
    std::vector<float> ssl((size_t) 768 * Th);
    for (int d = 0; d < 768; d++)
        for (int t = 0; t < Th; t++) ssl[(size_t) d * Th + t] = hout[(size_t) d + (size_t) 768 * t];

    gsv_refcode_cfg rcfg; rcfg.device = device;
    gsv_refcode * rc = gsv_refcode::load("models/gsv-refcode.gguf", rcfg);
    if (!rc) { fprintf(stderr, "refcode 加载失败\n"); return 1; }
    std::vector<int32_t> prompt_codes;
    if (!rc->codes_from_ssl(ssl.data(), Th, prompt_codes)) { fprintf(stderr, "refcode 失败\n"); return 1; }
    printf("  prompt codes: %zu (%.2fs @25Hz)\n", prompt_codes.size(), prompt_codes.size() / 25.0);

    // A2) 32k -> maxx 归一化 -> spectrogram_torch(2048/640) -> [:, :704] -> ref_enc -> ge[512]
    std::vector<float> spec704;
    int Ts = 0;
    if (!gsv_refspec_wav(raw.data(), (int64_t) raw.size(), sr, spec704, Ts, true)) {
        fprintf(stderr, "refspec 失败\n"); return 1;
    }
    gsv_refenc_cfg gcfg; gcfg.device = device;
    gsv_refenc * re = gsv_refenc::load("models/gsv-cond-f32.gguf", gcfg);
    if (!re) { fprintf(stderr, "ref_enc 加载失败\n"); return 1; }
    std::vector<float> spec_gg((size_t) 704 * Ts), fm((size_t) Ts, 1.0f), ge;
    for (int c = 0; c < 704; c++)
        for (int t = 0; t < Ts; t++) spec_gg[(size_t) c + (size_t) t * 704] = spec704[(size_t) c * Ts + t];
    if (!re->encode(spec_gg.data(), fm.data(), Ts, ge)) { fprintf(stderr, "ref_enc 失败\n"); return 1; }
    printf("  refer_audio_spec [704, %d] -> ge[512]\n", Ts);

    // A3) 32k -> mel_fn_v4 -> norm_spec -> mel2 [100, Tm] (DiT 的 prompt mel)
    std::vector<float> w32;
    gsv_resample(raw.data(), (int64_t) raw.size(), sr, 32000, w32);
    gsv_mel_cfg mcfg; mcfg.n_threads = 8;
    gsv_mel mel(mcfg);
    std::vector<float> mel2;
    int Tm = 0;
    if (!mel.forward_norm(w32.data(), (int64_t) w32.size(), mel2, Tm)) { fprintf(stderr, "mel 失败\n"); return 1; }
    printf("  mel2 (norm_spec) [100, %d] (%.2fs)\n", Tm, Tm / 100.0);
    printf("  [A] %.1f ms\n", ms_since(tA));

    // ================================================================
    // B) 文本 -> AR -> 语义 tokens (phones/bert 由 golden 提供, 文本前端未接线)
    // ================================================================
    printf("\n=== B) AR 生成 (phones/bert 来自 golden) ===\n");
    const auto tB = std::chrono::steady_clock::now();
    std::vector<float> phf = read_bin(std::string(gdir) + "/ar.phones.bin");
    std::vector<float> bef = read_bin(std::string(gdir) + "/ar.bert.bin");
    if (phf.empty() || bef.empty()) { fprintf(stderr, "golden phones/bert 缺失\n"); return 1; }
    const int Tph = (int) phf.size();

    gsv_ar_cfg acfg; acfg.n_threads = 8; acfg.device = device;
    gsv_ar * ar = gsv_ar::load("models/gsv-ar-f32.gguf", acfg);
    if (!ar) { fprintf(stderr, "AR 加载失败\n"); return 1; }
    if ((int) bef.size() != 1024 * Tph) {
        fprintf(stderr, "bert 尺寸不符: %zu vs %d\n", bef.size(), 1024 * Tph);
        return 1;
    }
    gsv_ar_request req;
    req.phones.resize(Tph);
    for (int i = 0; i < Tph; i++) req.phones[i] = (int32_t) phf[i];
    req.bert = bef;                              // [1024, T] 行主
    req.prompt = prompt_codes;
    gsv_sampler_cfg sc;                          // 默认采样参数 (与 repo TTS.py 一致)
    auto res = ar->generate({ req }, sc, 20261013, 20, 1000);
    if (res.tokens.empty() || res.lens.empty()) { fprintf(stderr, "AR 生成失败\n"); return 1; }
    const std::vector<int32_t> & gen = res.tokens[0];
    // tokens = prompt + 生成; 取生成部分
    std::vector<int32_t> sem(gen.begin() + std::min(prompt_codes.size(), gen.size()), gen.end());
    printf("  AR: prompt %zu + 生成 %d -> sem %zu (%.2fs @25Hz)\n",
           prompt_codes.size(), res.lens[0], sem.size(), sem.size() / 25.0);
    printf("  [B] %.1f ms\n", ms_since(tB));

    // ================================================================
    // C) 条件段: decode_encp (参考侧 + 目标侧)
    // ================================================================
    printf("\n=== C) 条件段 (decode_encp x2) ===\n");
    const auto tC = std::chrono::steady_clock::now();
    gsv_cond_cfg ccfg; ccfg.device = device;
    gsv_encp_cfg ecfg; ecfg.device = device;
    gsv_wns1_cfg wcfg; wcfg.device = device;
    gsv_cond * cond = gsv_cond::load("models/gsv-cond-f32.gguf", ccfg);
    gsv_encp * encp = gsv_encp::load("models/gsv-cond-encp.gguf", ecfg);
    gsv_wns1 * wns1 = gsv_wns1::load("models/gsv-cond-f32.gguf", wcfg);
    if (!cond || !encp || !wns1) { fprintf(stderr, "条件段加载失败\n"); return 1; }

    // 一次 decode_encp: codes -> RVQ.decode -> x2 -> enc_p -> bridge -> x2 -> wns1 -> fea[512, 4*Tc]
    auto decode_encp = [&](const std::vector<int32_t> & codes, std::vector<float> & fea) -> bool {
        const int Tc = (int) codes.size();
        if (Tc <= 0) return false;
        std::vector<float> q;
        cond->rvq_decode(codes.data(), Tc, true, q);              // [768, 2Tc] (idx = d + t*768)
        const int T = 2 * Tc;
        std::vector<float> y_tc((size_t) 768 * T);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 768; c++) y_tc[(size_t) c * T + t] = q[(size_t) c + (size_t) t * 768];
        std::vector<float> m, logs, yenc;
        if (!encp->encode(y_tc.data(), T, req.phones.data(), Tph, ge.data(), m, logs, &yenc)) return false;
        std::vector<float> xb((size_t) 192 * T), br;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 192; c++) xb[(size_t) c + (size_t) t * 192] = yenc[(size_t) c * T + t];
        cond->bridge_run(xb.data(), T, true, br);                  // [2T, 512] (idx = c*2T + t)
        const int T2 = 2 * T;
        fea.clear();
        return wns1->encode(br.data(), ge.data(), T2, T2, fea);    // [2T, 512] (idx = c*2T + t)
    };

    std::vector<float> fea_ref, fea_todo;
    if (!decode_encp(prompt_codes, fea_ref)) { fprintf(stderr, "decode_encp(ref) 失败\n"); return 1; }
    if (!decode_encp(sem, fea_todo)) { fprintf(stderr, "decode_encp(todo) 失败\n"); return 1; }
    // wns1 输出 [2T,512] (idx = c*2T + t) -> DiT 的 synthesize 要 [512, N] (idx = c*N + t)
    auto to_dit_layout = [](std::vector<float> & f, int T) {
        const int C = 512;
        std::vector<float> g((size_t) C * T);
        for (int c = 0; c < C; c++)
            for (int t = 0; t < T; t++) g[(size_t) c * T + t] = f[(size_t) c * T + t];
        return g;
    };
    const int Rref = (int) (fea_ref.size() / 512);
    const int Ntodo = (int) (fea_todo.size() / 512);
    std::vector<float> fr = to_dit_layout(fea_ref, Rref);
    std::vector<float> ft = to_dit_layout(fea_todo, Ntodo);
    printf("  fea_ref [512, %d] (%.2fs)  fea_todo [512, %d] (%.2fs)\n",
           Rref, Rref / 100.0, Ntodo, Ntodo / 100.0);
    printf("  [C] %.1f ms\n", ms_since(tC));

    // ================================================================
    // D) DiT: synthesize_v5_mel (分块 + rolling prompt)
    // ================================================================
    printf("\n=== D) DiT synthesize_v5_mel (%d steps) ===\n", steps);
    const auto tD = std::chrono::steady_clock::now();
    gsv_dit_cfg dcfg; dcfg.device = device;
    // DiT 最小可用档 = f16all (docs/dit_ggml.md §6); adaln 降 F16 的 f16afa 在真实输入上出 NaN
    // (adaLN 的调制量级大, F16 溢出) —— 对拍用的也是 f16all, 这里保持一致
    gsv_dit * dit = gsv_dit::load("models/gsv-dit-f16all.gguf", dcfg);
    if (!dit) { fprintf(stderr, "DiT 加载失败\n"); return 1; }
    // cfg: v5turbo 用 0.0 (v5dev 才 1.30)
    std::vector<float> mel_out;
    if (!dit->synthesize(fr.data(), Rref, ft.data(), Ntodo, mel2.data(), Tm, steps, 0.0f,
                         20261013, 0, nullptr, mel_out)) {
        fprintf(stderr, "DiT synthesize 失败\n"); return 1;
    }
    const int Nmel = (int) (mel_out.size() / 100);
    printf("  mel_out [100, %d] (%.2fs)\n", Nmel, Nmel / 100.0);
    if (getenv("GSV_E2E_DUMP")) {
        auto dmp = [](const char * n, const std::vector<float> & v) {
            double mx = 0, sum = 0;
            for (float x : v) { mx = std::max(mx, (double) std::fabs(x)); sum += x; }
            printf("    [dbg] %-12s n=%-8zu max|.|=%9.4f mean=%11.5f\n",
                   n, v.size(), mx, v.empty() ? 0 : sum / v.size());
            FILE * f = fopen((std::string("tests/_e2e_") + n + ".bin").c_str(), "wb");
            if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
        };
        dmp("fea_ref", fr); dmp("fea_todo", ft); dmp("mel2", mel2); dmp("mel_out", mel_out);
    }
    printf("  [D] %.1f ms\n", ms_since(tD));

    // ================================================================
    // E) vocoder: denorm_spec -> ORT ONNX -> wav [32k]
    // ================================================================
    printf("\n=== E) vocoder (ORT ONNX) ===\n");
    const auto tE = std::chrono::steady_clock::now();
    std::vector<float> mel_dn((size_t) 100 * Nmel);
    for (int i = 0; i < 100 * Nmel; i++) mel_dn[(size_t) i] = gsv_denorm_spec(mel_out[(size_t) i]);

    gsv_vocoder_cfg vcfg;
    if (const char * vd = getenv("GSV_E2E_VOC_DEVICE")) vcfg.device = vd;
    else vcfg.device = "";                       // 自动: WebGPU > DML > CPU
    gsv_vocoder * voc = gsv_vocoder::load("models/vocoder_fp32.onnx", vcfg);
    if (!voc) { fprintf(stderr, "vocoder 加载失败\n"); return 1; }
    std::vector<float> wav;
    if (!voc->run(mel_dn.data(), Nmel, wav)) { fprintf(stderr, "vocoder 失败\n"); return 1; }
    printf("  EP=%s wav [%zu] (%.2fs @32k)\n", voc->ep_name(), wav.size(), (double) wav.size() / 32000.0);
    printf("  [E] %.1f ms\n", ms_since(tE));

    // 峰值归一化到 -3 dBFS (0.708): AR 提前停/参考音频短时输出幅度会很低,
    // 写盘前统一增益, 与常见 TTS 输出电平一致 (可用 GSV_E2E_NONORM=1 关掉)
    if (getenv("GSV_E2E_NONORM") == nullptr) {
        float pk = 0.0f;
        for (float v : wav) pk = std::max(pk, std::fabs(v));
        if (pk > 1e-9f) {
            const float g = 0.708f / pk;
            for (float & v : wav) v *= g;
            double rms = 0; for (float v : wav) rms += (double) v * v;
            rms = std::sqrt(rms / (double) wav.size());
            printf("  peak-norm: gain %.1fx -> rms %.4f (%.1f dBFS)\n",
                   g, rms, 20.0 * std::log10(std::max(rms, 1e-12)));
        }
    }

    if (!wav_write(out_wav, wav, 32000)) { fprintf(stderr, "wav 写出失败: %s\n", out_wav.c_str()); return 1; }
    printf("\n=== 完成 -> %s (%.2fs 音频, 全链 %.0f ms) ===\n",
           out_wav.c_str(), (double) wav.size() / 32000.0,
           ms_since(tA));

    delete voc; delete dit; delete cond; delete encp; delete wns1;
    delete ar; delete rc; delete re; delete hub;
    return 0;
}
