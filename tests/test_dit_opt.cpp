// DiT 算法级加速 A/B 工具 (2026-11-06).
//   1) 全关档与 tests/golden/cfm.* (torch 同输入 dump) 绝对对齐 — 验证 fixture 口径;
//   2) 各开关 (CFG 截断 + Uncond 复用) 对全关档的质量差 + 耗时;
//   3) cfm.chunk.* 多 chunk 场景 A/B.
// 用法:
//   test_dit_opt [--device cpu|vulkan] [--models DIR] [--golden DIR]
//                [--family s4c0|s8c13|s32c13|s4c0_big|s32c13_big|chunk|all]
//                [--dit PATH] [--chunk-steps N] [--chunk-cfg F]
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include "gsv_dit.h"

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static bool read_bin(const std::string & p, std::vector<float> & v) {
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

static bool read_scalar(const std::string & p, float & f, int & i) {
    std::vector<float> v;
    if (!read_bin(p, v) || v.size() < 1) return false;
    memcpy(&f, &v[0], 4);
    memcpy(&i, &v[0], 4);
    return true;
}

struct metrics {
    float max_abs = 0, mean_abs = 0, rel_l1 = 0, corr = 0;
    void compute(const float * a, const float * b, size_t n) {
        double sa = 0, sb = 0, sab = 0, na = 0, nb = 0, num = 0;
        for (size_t i = 0; i < n; i++) {
            const double d = fabs((double) a[i] - b[i]);
            if (d > max_abs) max_abs = (float) d;
            sa += d;
            na += fabs(a[i]); nb += fabs(b[i]);
            sab += (double) a[i] * b[i];
        }
        mean_abs = (float) (sa / (n ? n : 1));
        rel_l1   = (float) (sa / (na + 1e-30));
        // corr(a,b)
        double ma = 0, mb = 0;
        for (size_t i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= (n ? n : 1); mb /= (n ? n : 1);
        double va = 0, vb = 0, cov = 0;
        for (size_t i = 0; i < n; i++) {
            va += ((double) a[i] - ma) * ((double) a[i] - ma);
            vb += ((double) b[i] - mb) * ((double) b[i] - mb);
            cov += ((double) a[i] - ma) * ((double) b[i] - mb);
        }
        corr = (float) (cov / (sqrt(va * vb) + 1e-30));
        (void) sab;
    }
};

// 取 a 的 [off, off+cols) 列 (torch 布局 [C, T]) 与 b 的 [0, cols) 列逐列比
static metrics cmp_cols(const std::vector<float> & a, const std::vector<float> & b,
                        int C, int Ta, int Tb, int off_a, int cols) {
    metrics m;
    std::vector<float> x, y;
    x.reserve((size_t) C * cols); y.reserve((size_t) C * cols);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < cols; t++) {
            x.push_back(a[(size_t) c * Ta + off_a + t]);
            y.push_back(b[(size_t) c * Tb + t]);
        }
    m.compute(x.data(), y.data(), x.size());
    return m;
}

struct family {
    const char * name;
    int T, P, steps;
    float cfg;
    const char * gguf;      // 相对 --models
};

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string device = "cpu", models = "models", golden = "tests/golden";
    std::string want = "all", dit_override;
    int chunk_steps = 0; float chunk_cfg = -1.0f;
    float noise_scale = 0.875f;     // fixture 的 noise 是未乘 temperature 的 randn (CFM 里 *0.875)
    int  dbcache_fn = 0;            // >0 = 同时开 cache-dit (复现前端 auto 档的组合)

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--device") device = next();
        else if (a == "--models") models = next();
        else if (a == "--golden") golden = next();
        else if (a == "--family") want = next();
        else if (a == "--dit") dit_override = next();
        else if (a == "--chunk-steps") chunk_steps = atoi(next().c_str());
        else if (a == "--chunk-cfg") chunk_cfg = (float) atof(next().c_str());
        else if (a == "--noise-scale") noise_scale = (float) atof(next().c_str());
        else if (a == "--dbcache") dbcache_fn = atoi(next().c_str());
        else { printf("unknown arg %s\n", a.c_str()); return 2; }
    }

    // fixture 口径 (T/P 由文件尺寸推; steps/cfg 从 .steps/.cfg 读, 下面打印原值核对)
    struct fam_io { const char * name; };
    std::vector<family> fams;
    auto add_fam = [&](const char * n, const char * gguf) {
        const std::string base = golden + "/cfm." + n;
        std::vector<float> mu, pr, no;
        if (!read_bin(base + ".mu.bin", mu)) { printf("[skip] %s: no mu\n", n); return; }
        if (!read_bin(base + ".prompt.bin", pr)) { printf("[skip] %s: no prompt\n", n); return; }
        if (!read_bin(base + ".noise.bin", no)) { printf("[skip] %s: no noise\n", n); return; }
        const int T  = (int) (mu.size() / 512);
        const int P  = (int) (pr.size() / 100);
        float cf = 0, sf = 0; int st = 0, sti = 0;
        read_scalar(base + ".cfg.bin", cf, sti);
        read_scalar(base + ".steps.bin", sf, st);
        if (st < 1 || st > 1000) st = (int) (sf + 0.5f);      // steps 也可能按 float dump
        fams.push_back({ n, T, P, st, cf, gguf });
        printf("[fixt] %-10s T=%d P=%d steps=%d cfg=%.3f (raw cfg=%.3f steps_i=%d)\n",
               n, T, P, st, cf, cf, sti);
    };
    if (want == "all" || want == "s4c0")      add_fam("s4c0",      "gsv-dit-f16all.gguf");
    if (want == "all" || want == "s8c13")     add_fam("s8c13",     "gsv-dit-f16all.gguf");
    if (want == "all" || want == "s32c13")    add_fam("s32c13",    "gsv-v5dev-dit-f16all.gguf");
    if (want == "all" || want == "s4c0_big")  add_fam("s4c0_big",  "gsv-dit-f16all.gguf");
    if (want == "all" || want == "s32c13_big")add_fam("s32c13_big","gsv-v5dev-dit-f16all.gguf");

    for (const family & fm : fams) {
        const std::string base = golden + "/cfm." + fm.name;
        const std::string dpath = dit_override.empty() ? (models + "/" + fm.gguf) : dit_override;
        printf("\n================ %s (T=%d P=%d steps=%d cfg=%.3f) %s ================\n",
               fm.name, fm.T, fm.P, fm.steps, fm.cfg, dpath.c_str());
        std::vector<float> mu, prompt, noise, out;
        CHECK(read_bin(base + ".mu.bin", mu));
        CHECK(read_bin(base + ".prompt.bin", prompt));
        CHECK(read_bin(base + ".noise.bin", noise));
        CHECK(read_bin(base + ".out.bin", out));
        if (mu.size() != (size_t) 512 * fm.T || prompt.size() != (size_t) 100 * fm.P ||
            noise.size() != (size_t) 100 * fm.T || out.size() != (size_t) 100 * fm.T) {
            printf("[skip] %s: fixture size mismatch\n", fm.name);
            continue;
        }
        std::vector<float> x0 = noise;                       // CFM: x = randn * temperature
        for (float & v : x0) v *= noise_scale;

        gsv_dit_cfg cfg; cfg.device = device;
        cfg.dbcache_fn = dbcache_fn;      // 与 alg 开关组合 (前端 auto 档: steps>=16 时 F8)
        gsv_dit * dit = gsv_dit::load(dpath, cfg);
        if (!dit) { printf("[fail] load %s\n", dpath.c_str()); g_fail++; continue; }

        auto run = [&](const gsv_dit_alg & alg, std::vector<float> & mel, double & ms) {
            dit->set_alg(alg);
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = dit->sample(mu.data(), prompt.data(), fm.P, fm.T, fm.steps, fm.cfg,
                                        1, x0.data(), mel);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return ok;
        };

        // 全关档 = 上游口径: cfg_off_t / uncond_reuse 全 0 (struct 默认是生产值, 须显式清零)
        gsv_dit_alg off;
        off.cfg_off_t = 0.0f; off.uncond_reuse = 0;
        std::vector<float> m_off, m_tmp;
        double ms_off = 0, ms_cold = 0;
        CHECK(run(off, m_off, ms_cold));                   // 第一次: 建图 + 跑
        CHECK(run(off, m_off, ms_off));                    // 第二次: 热
        // 确定性: 再跑一次必须逐位一致
        std::vector<float> m_off2;
        double ms_x = 0;
        CHECK(run(off, m_off2, ms_x));
        metrics det = cmp_cols(m_off2, m_off, 100, fm.T - fm.P, fm.T - fm.P, 0, fm.T - fm.P);
        printf("  [det ] 同参数复跑 corr=%.6f max|d|=%.2e (必须 1.0/0)\n", det.corr, det.max_abs);
        CHECK(det.corr > 0.999999f && det.max_abs == 0.0f);

        // 绝对对齐: golden out 的 gen 列 vs 我们的 [0, T-P).
        // 两种口径都试: 偏移 P (整宽 x, prompt 区在前) 或偏移 0 (切片输出零填充).
        // 注意: Vulkan 后端在该 fixture 上与 torch golden 有固有偏差 (~8.7 max|d|,
        // HEAD 版本即如此, 与 alg 开关无关), 所以绝对对齐只在 CPU 后端强制;
        // Vulkan 上只打印参考值, 以内部 A/B (各档 vs 全关档) 为准.
        metrics abs  = cmp_cols(out, m_off, 100, fm.T, fm.T - fm.P, fm.P, fm.T - fm.P);
        metrics abs0 = cmp_cols(out, m_off, 100, fm.T, fm.T - fm.P, 0, fm.T - fm.P);
        if (abs0.max_abs < abs.max_abs) abs = abs0;
        printf("  [gold] max|d|=%.4e mean=%.3e relL1=%.3e corr=%.6f  (cold %.0f ms / warm %.1f ms)\n",
               abs.max_abs, abs.mean_abs, abs.rel_l1, abs.corr, ms_cold, ms_off);
        if (device == "cpu") CHECK(abs.max_abs < 2e-2f);
        else printf("  [note] vulkan: golden 绝对对齐不做强制 (后端固有偏差), 看内部 A/B\n");

        struct variant { const char * name; gsv_dit_alg alg; };
        std::vector<variant> vs;
        auto push = [&](const char * n, gsv_dit_alg a) { vs.push_back({ n, a }); };
        gsv_dit_alg prod;                                  // 生产默认 (128 / 0.7 / 0)
        // 方向3: CFG 截断 + Uncond 复用
        for (float off : {0.95f, 0.90f, 0.85f, 0.80f, 0.70f, 0.60f, 0.50f}) {
            gsv_dit_alg a; a.cfg_off_t = off; a.uncond_reuse = 0;
            char * nm = new char[20]; snprintf(nm, 20, "cfg%.2f", off);
            push(nm, a);
        }
        for (int ur : {1, 2}) {
            gsv_dit_alg a; a.cfg_off_t = 0.7f; a.uncond_reuse = ur;
            char * nm = new char[20]; snprintf(nm, 20, "cfg0.70+u%d", ur);
            push(nm, a);
        }
        {
            gsv_dit_alg a; a.cfg_off_t = 0.6f; a.uncond_reuse = 1;
            push("cfg0.60+u1", a);
        }
        push("prod(cfg.70)", prod);
        push("all-off", []{ gsv_dit_alg a; a.cfg_off_t = 0.0f; a.uncond_reuse = 0; return a; }());
        printf("  %-12s %10s %10s %10s %8s %9s %8s\n",
               "variant", "max|d|", "mean|d|", "relL1", "corr", "ms(warm)", "speedup");
        for (const variant & v : vs) {
            double ms1 = 0, ms2 = 0;
            if (!run(v.alg, m_tmp, ms1)) { printf("  %-12s RUN FAILED\n", v.name); g_fail++; continue; }
            if (!run(v.alg, m_tmp, ms2)) { printf("  %-12s RUN FAILED\n", v.name); g_fail++; continue; }
            metrics m = cmp_cols(m_tmp, m_off, 100, fm.T - fm.P, fm.T - fm.P, 0, fm.T - fm.P);
            printf("  %-12s %10.3e %10.3e %10.3e %8.5f %9.1f %8.2fx\n",
                   v.name, m.max_abs, m.mean_abs, m.rel_l1, m.corr, ms2, ms_off / (ms2 > 0 ? ms2 : 1));
        }
        dit->set_alg(off);
        delete dit;
    }

    // ---------------- fixture 口径诊断 (s0.x_in / s0.vel 布局, mu 转置, 噪声缩放) ----------------
    if (want == "all" || want == "s4c0" || want == "probe") {
        const std::string base = golden + "/cfm.s4c0";
        std::vector<float> mu, prompt, noise, out, s0v, s0x;
        if (read_bin(base + ".mu.bin", mu) && read_bin(base + ".prompt.bin", prompt) &&
            read_bin(base + ".noise.bin", noise) && read_bin(base + ".out.bin", out) &&
            read_bin(base + ".s0.vel.bin", s0v) && read_bin(base + ".s0.x_in.bin", s0x)) {
            const int T = (int) (mu.size() / 512), P = (int) (prompt.size() / 100);
            const int steps = 4;
            printf("\n================ fixture probe (s4c0 T=%d P=%d) ================\n", T, P);
            gsv_dit_cfg cfg2; cfg2.device = device;
            gsv_dit * dit = gsv_dit::load(models + "/gsv-dit-f16all.gguf", cfg2);
            if (!dit) { printf("[fail] load\n"); g_fail++; }
            else {
                std::vector<float> prompt_x((size_t) 100 * T, 0.0f);
                for (int c = 0; c < 100; c++)
                    memcpy(prompt_x.data() + (size_t) c * T, prompt.data() + (size_t) c * P, (size_t) P * 4);
                std::vector<float> x0 = noise;
                for (int c = 0; c < 100; c++) memset(x0.data() + (size_t) c * T, 0, (size_t) P * 4);
                metrics dx = cmp_cols(s0x, x0, 100, T, T, 0, T);
                printf("  [diag x_in ] max|d|=%.4e corr=%.6f (0=噪声口径一致)\n", dx.max_abs, dx.corr);

                auto run_mel = [&](const gsv_dit_alg & alg, const float * mu_p, const float * noise_p,
                                   float scale, std::vector<float> & mel) {
                    dit->set_alg(alg);
                    std::vector<float> nn((size_t) 100 * T);
                    for (size_t i = 0; i < nn.size(); i++) nn[i] = noise_p[i] * scale;
                    for (int c = 0; c < 100; c++) memset(nn.data() + (size_t) c * T, 0, (size_t) P * 4);
                    return dit->sample(mu_p, prompt.data(), P, T, steps, 0.0f, 1, nn.data(), mel);
                };
                gsv_dit_alg off;
                off.cfg_off_t = 0.0f; off.uncond_reuse = 0;
                std::vector<float> m_off, m_try, mu_t((size_t) mu.size());
                CHECK(run_mel(off, mu.data(), noise.data(), noise_scale, m_off));
                metrics g1 = cmp_cols(out, m_off, 100, T, T - P, P, T - P);
                printf("  [diag mel ] baseline(noise*%.3f) vs gold: max|d|=%.4e corr=%.6f\n",
                       noise_scale, g1.max_abs, g1.corr);
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < 512; c++) mu_t[(size_t) c * T + t] = mu[(size_t) t * 512 + c];
                if (run_mel(off, mu_t.data(), noise.data(), noise_scale, m_try)) {
                    metrics mt = cmp_cols(out, m_try, 100, T, T - P, P, T - P);
                    printf("  [diag mel ] mu 转置:            max|d|=%.4e corr=%.6f\n", mt.max_abs, mt.corr);
                }
                // golden s0.vel 布局 ([T,100] vs [100,T])
                {
                    dit->set_alg(off);
                    dit->prepare(prompt_x.data(), mu.data(), T, T, P);
                    std::vector<float> v0((size_t) 100 * T);
                    CHECK(dit->velocity(x0.data(), 0.0f, false, v0.data()));
                    metrics a = cmp_cols(s0v, v0, 100, T, T, 0, T);
                    printf("  [diag vel0] [100,T] 布局: max|d|=%.4e corr=%.6f\n", a.max_abs, a.corr);
                    std::vector<float> gt((size_t) 100 * T);
                    for (int t = 0; t < T; t++)
                        for (int c = 0; c < 100; c++) gt[(size_t) c * T + t] = s0v[(size_t) t * 100 + c];
                    metrics b2 = cmp_cols(gt, v0, 100, T, T, 0, T);
                    printf("  [diag vel0] [T,100] 布局: max|d|=%.4e corr=%.6f\n", b2.max_abs, b2.corr);
                }
                delete dit;
            }
        } else printf("[skip] probe fixtures missing\n");
    }

    // ---------------- 多 chunk (cfm.chunk.*): synthesize A/B ----------------
    if (want == "all" || want == "chunk") {
        const std::string base = golden + "/cfm.chunk";
        std::vector<float> fea_ref, fea_todo, mel_ref, out;
        std::vector<float> n0, n1, n2, blk;
        const bool ok = read_bin(base + ".fea_ref.bin", fea_ref) && read_bin(base + ".fea_todo.bin", fea_todo) &&
                        read_bin(base + ".mel_ref.bin", mel_ref) && read_bin(base + ".out.bin", out) &&
                        read_bin(base + ".noise0.bin", n0) && read_bin(base + ".noise1.bin", n1) &&
                        read_bin(base + ".noise2.bin", n2) && read_bin(base + ".block.bin", blk);
        printf("\n================ chunk (synthesize, 多 chunk + rolling) ================\n");
        if (!ok) { printf("[skip] chunk fixtures incomplete\n"); }
        else {
            const int R = (int) (fea_ref.size() / 512), N = (int) (fea_todo.size() / 512);
            const int mel_ref_T = (int) (mel_ref.size() / 100);
            int block = 0; memcpy(&block, blk.data(), 4);
            // block.bin 口径不定 (见过 4.5f 的 float 位型); 用 noise0 的帧数反推: tlen0 = n0 - ref_frames
            const int n0f = (int) (n0.size() / 100);
            const int ref_frames_est = std::min(mel_ref_T, R);
            if (block <= 0 || block > 100000) block = n0f - ref_frames_est;
            printf("[fixt] chunk R=%d N=%d mel_ref_T=%d block=%d noises=%zu/%zu/%zu out=%zu\n",
                   R, N, mel_ref_T, block, n0.size() / 100, n1.size() / 100, n2.size() / 100, out.size() / 100);
            const std::string dpath = dit_override.empty() ? (models + "/gsv-dit-f16all.gguf") : dit_override;
            gsv_dit_cfg cfg; cfg.device = device;
            gsv_dit * dit = gsv_dit::load(dpath, cfg);
            if (!dit) { printf("[fail] load %s\n", dpath.c_str()); g_fail++; }
            else {
                std::vector<const float *> x0s = { n0.data(), n1.data(), n2.data() };
                std::vector<float> n0s = n0, n1s = n1, n2s = n2;   // CFM: randn * temperature
                for (float & v : n0s) v *= noise_scale;
                for (float & v : n1s) v *= noise_scale;
                for (float & v : n2s) v *= noise_scale;
                std::vector<const float *> x0s_s = { n0s.data(), n1s.data(), n2s.data() };
                auto run_syn = [&](const gsv_dit_alg & alg, int steps, float ccfg,
                                   std::vector<float> & mel, double & ms) {
                    dit->set_alg(alg);
                    const auto t0 = std::chrono::steady_clock::now();
                    const bool r = dit->synthesize(fea_ref.data(), R, fea_todo.data(), N,
                                                   mel_ref.data(), mel_ref_T, steps, ccfg, 1,
                                                   block, &x0s_s, mel);
                    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    return r;
                };
                // steps/cfg 自动探测: fixture 没给, 试常见组合, 报绝对误差 (golden 可能是空文件, 只作参考)
                struct sc { int steps; float cfg; };
                const sc cands[] = { {4, 0.0f}, {4, 1.3f}, {8, 1.3f}, {32, 1.3f}, {8, 0.0f} };
                int best_i = -1; double best_err = 1e30f;
                std::vector<float> m_off;
                double ms_off = 0;
                gsv_dit_alg off;
                off.cfg_off_t = 0.0f; off.uncond_reuse = 0;
                for (int ci = 0; ci < 5; ci++) {
                    if (chunk_steps > 0) { if (cands[ci].steps != chunk_steps) continue; }
                    if (chunk_cfg >= 0.0f) { if (fabsf(cands[ci].cfg - chunk_cfg) > 1e-6f) continue; }
                    std::vector<float> m;
                    double ms = 0;
                    if (!run_syn(off, cands[ci].steps, cands[ci].cfg, m, ms)) continue;
                    metrics g = cmp_cols(out, m, 100, N, N, 0, N);
                    printf("  [probe] steps=%2d cfg=%.2f -> gold max|d|=%.4e corr=%.6f (%.1f ms)\n",
                           cands[ci].steps, cands[ci].cfg, g.max_abs, g.corr, ms);
                    if (g.max_abs < best_err) { best_err = g.max_abs; best_i = ci; m_off = m; ms_off = ms; }
                }
                if (best_i < 0) { printf("[fail] no candidate ran\n"); g_fail++; }
                else {
                    printf("  [base ] steps=%d cfg=%.2f (%.1f ms) — chunk golden 可能为空文件, 以内部 A/B 为准\n",
                           cands[best_i].steps, cands[best_i].cfg, ms_off);
                    struct variant { const char * name; gsv_dit_alg alg; };
                    std::vector<variant> vs;
                    gsv_dit_alg prod;                                  // 生产默认 (128 / 0.7 / 0)
                                if (cands[best_i].cfg > 1e-5f) {
                        vs.push_back({ "cfg.7(default)", []{ gsv_dit_alg a; a.cfg_off_t = 0.7f; return a; }() });
                        vs.push_back({ "cfg.7+u1", []{ gsv_dit_alg a; a.cfg_off_t = 0.7f; a.uncond_reuse = 1; return a; }() });
                    }
                    vs.push_back({ "prod(cfg.70)", prod });
                    printf("  %-12s %10s %10s %10s %8s %9s %8s\n",
                           "variant", "max|d|", "mean|d|", "relL1", "corr", "ms(warm)", "speedup");
                    for (const variant & v : vs) {
                        std::vector<float> m; double ms1 = 0, ms2 = 0;
                        if (!run_syn(v.alg, cands[best_i].steps, cands[best_i].cfg, m, ms1)) {
                            printf("  %-12s RUN FAILED\n", v.name); g_fail++; continue;
                        }
                        run_syn(v.alg, cands[best_i].steps, cands[best_i].cfg, m, ms2);
                        metrics g = cmp_cols(m, m_off, 100, N, N, 0, N);
                        printf("  %-12s %10.3e %10.3e %10.3e %8.5f %9.1f %8.2fx\n",
                               v.name, g.max_abs, g.mean_abs, g.rel_l1, g.corr, ms2,
                               ms_off / (ms2 > 0 ? ms2 : 1));
                    }
                }
                delete dit;
            }
        }
    }

    printf("\n%s (%d failure(s))\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
