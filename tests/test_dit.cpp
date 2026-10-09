// DiT/CFM 对拍 + 基准
// golden: tests/golden/dit.t96|pad|t0.* (单步前向) + cfm.s4c0|s8c13|chunk.* (CFM 采样)
// 用法: test_dit [gguf] [golden_dir]; GSV_DIT_DEVICE=vulkan, GSV_DIT_BENCH=N
// 布局: golden 的 [1,T,C] 张量 = 我们图的 Domain B (ggml ne{C,T}) 字节序, 直接比;
//       [1,C,T] 张量 (x/prompt/mu/vel) 需按 torch [C,T] 布局读入 (我们的 API 同布局)
#include "../src/gsv_dit.h"

#include <algorithm>
#include <chrono>
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

static double maxdiff_nan(const std::vector<float> & a, const std::vector<float> & b, size_t & n_bad) {
    if (a.size() != b.size()) return 1e30;
    double md = 0; n_bad = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite((double) a[i]) || !std::isfinite((double) b[i])) { n_bad++; continue; }
        double d = std::fabs((double) a[i] - b[i]);
        if (!(d <= md)) md = d;
    }
    return md;
}
static double amax(const std::vector<float> & v) {
    double m = 0;
    for (float x : v) if (std::isfinite((double) x)) m = std::max(m, std::fabs((double) x));
    return m;
}
// 失败时打印误差最大的位置 (layout [C,T]: idx = c + C*t) — 区分"值错"还是"布局/通道错位"
struct TopE { double d; size_t i; };
static int g_x_lens = 0;   // run_case 设置: >0 时把误差分有效区/pad 区统计
static void report_top(const char * what, const std::vector<float> & mine, const std::vector<float> & ref,
                       int C, int T, int topn = 6) {
    std::vector<TopE> es;
    for (size_t i = 0; i < mine.size() && i < ref.size(); i++) {
        if (!std::isfinite((double) mine[i]) || !std::isfinite((double) ref[i])) continue;
        es.push_back({ std::fabs((double) mine[i] - ref[i]), i });
    }
    size_t k = std::min<size_t>((size_t) topn, es.size());
    std::partial_sort(es.begin(), es.begin() + k, es.end(),
                      [](const TopE & a, const TopE & b) { return a.d > b.d; });
    // 误差元素占比 (>1% * refmax), 并分有效区/pad 区
    double rmax = amax(ref);
    size_t n_big = 0, n_val = 0, n_pad = 0;
    for (const auto & e : es) {
        if (e.d < 0.01 * rmax) continue;
        n_big++;
        if (g_x_lens > 0 && g_x_lens < T) { if ((int)(e.i / C) < g_x_lens) n_val++; else n_pad++; }
    }
    printf("      %-16s top:", what);
    for (size_t j = 0; j < k; j++) {
        const size_t i = es[j].i;
        printf(" (c=%lld,t=%lld|%.2g/%.2g)", (long long)(i % C), (long long)(i / C),
               (double) mine[i], (double) ref[i]);
    }
    if (g_x_lens > 0 && g_x_lens < T)
        printf("  [>1%%: %zu/%zu valid=%zu pad=%zu]", n_big, es.size(), n_val, n_pad);
    else
        printf("  [>1%%: %zu/%zu]", n_big, es.size());
    printf("\n");
}

// torch [1,T,C] (flat t*C+c) -> torch [1,C,T] (flat c*T+t); 尺寸不符 (golden 缺失) 返回空
static std::vector<float> t2ct(const std::vector<float> & v, int T, int C) {
    if (v.size() != (size_t) T * C) return {};
    std::vector<float> o(v.size());
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) o[(size_t) c * T + t] = v[(size_t) t * C + c];
    return o;
}
// torch [1,T,C] 的前 k 个通道 -> ggml [k, T] 布局 (flat c + k*t)
static std::vector<float> head_slice(const std::vector<float> & v, int T, int C, int k) {
    std::vector<float> o((size_t) k * T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < k; c++) o[(size_t) c + (size_t) k * t] = v[(size_t) t * C + c];
    return o;
}

static int n_fail = 0;
static bool   g_quiet    = false;   // 仅 FAIL 打印 (32 步 / 手工循环)
static bool   g_sect_on  = false;   // 段内统计
static size_t g_checks = 0, g_fails = 0;
static double g_sect_max = 0.0;
static double rel_tol = 2e-2;      // 判定: d < rel_tol * max|ref| (尺度自适应)

static void check(const char * what, const std::vector<float> & mine, const std::vector<float> & ref,
                  double abs_floor = 0.0, int C = 0, int T = 0) {
    size_t nb = 0;
    const double d = maxdiff_nan(mine, ref, nb);
    const double thr = std::max(rel_tol * amax(ref), abs_floor);
    const bool ok = (d >= 0 && d < thr && nb == 0);
    if (!ok) n_fail++;
    g_checks++; if (!ok) g_fails++;
    if (g_sect_on && d < 1e29) g_sect_max = std::max(g_sect_max, d);
    if (ok && g_quiet) return;
    printf("    %-22s max|d|=%.3e  refmax=%.3e  thr=%.2e %s%s\n", what, d, amax(ref), thr,
           ok ? "PASS" : "FAIL", nb ? " (nan/inf!)" : "");
    if (!ok && C > 0 && T > 0 && mine.size() == ref.size() && (int) mine.size() == C * T)
        report_top(what, mine, ref, C, T);
}
// 读中间张量并与 golden 比 (name=nullptr 表示跳过)
static void check_dbg(gsv_dit * m, const char * dbg_name, const char * what,
                      const std::vector<float> & ref, const std::vector<float> & ref_all = {},
                      double abs_floor = 0.0, int C = 0, int T = 0) {
    std::vector<float> mine;
    if (!m->debug_read(dbg_name, mine)) { printf("    %-22s MISSING (%s)\n", what, dbg_name); n_fail++; return; }
    const std::vector<float> & r = ref.empty() ? ref_all : ref;
    check(what, mine, r, abs_floor, C, T);
}

// 单步前向用例
static void run_case(gsv_dit * m, const std::string & gdir, const char * tag, int T, int x_lens) {
    const std::string p = gdir + "/dit." + tag + ".";
    auto x0 = read_bin(p + "x0.bin"), prompt = read_bin(p + "prompt_x.bin"), mu = read_bin(p + "mu.bin");
    auto tref = read_bin(p + "t.bin"), vref = read_bin(p + "vel.bin");
    if (x0.empty() || vref.empty() || prompt.empty() || mu.empty()) {
        printf("  [%s] golden missing -> skip\n", tag); return;
    }
    printf("  [%s] T=%d x_lens=%d t=%.3f\n", tag, T, x_lens, tref.empty() ? -1.0 : tref[0]);
    g_x_lens = x_lens;
    if (!m->prepare(prompt.data(), mu.data(), T, x_lens)) { n_fail++; return; }
    std::vector<float> vel((size_t) 100 * T);
    if (!m->velocity(x0.data(), tref[0], false, vel.data())) { n_fail++; return; }
    check("vel", vel, t2ct(vref, T, 100), 0.0, 100, T);

    // 中间量 (GSV_DIT_DEBUG=1 时由 mark() 标出; 均为 Domain B [C,T] = torch [1,T,C] 字节序)
    auto raw = [&](const char * n) { return read_bin(p + n); };
    {   // cache 图输出 (condition 常驻 c_out; text_embed 为 cache 图内标记)
        auto st = raw("static.bin");
        if (!st.empty()) check_dbg(m, "condition", "condition", st, {}, 0, 1024, T);
        check_dbg(m, "dbg_te_pos", "te_pos", raw("te_pos.bin"), {}, 0, 512, T);
        check_dbg(m, "dbg_te_cn0", "te_cn0", raw("te_cn0.bin"), {}, 0, 512, T);
        check_dbg(m, "dbg_te_cn1", "te_cn1", raw("te_cn1.bin"), {}, 0, 512, T);
        check_dbg(m, "dbg_te_cn2", "te_cn2", raw("te_cn2.bin"), {}, 0, 512, T);
        auto te = raw("text_embed.bin");
        if (!te.empty()) check_dbg(m, "dbg_text_embed", "text_embed", te, {}, 0, 512, T);
    }
    check_dbg(m, "dbg_x_lin",    "x_lin",    raw("x_lin.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_convpos",  "convpos",  raw("convpos_out.bin"), {}, 0, 1024, T);
    {   // trunk 输入 = x_lin + convpos_out (torch 逐元素相加)
        auto cp = raw("convpos_out.bin"), xl = raw("x_lin.bin");
        if (!cp.empty() && cp.size() == xl.size()) {
            std::vector<float> ti(cp.size());
            for (size_t i = 0; i < ti.size(); i++) ti[i] = cp[i] + xl[i];
            check_dbg(m, "dbg_trunk_in", "trunk_in", ti, {}, 0, 1024, T);
        }
    }
    check_dbg(m, "dbg_b0_norm",  "b0_norm",  raw("b0_norm.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_gate_msa",  "b0_gate_msa",  raw("b0_gate_msa.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_scale_mlp", "b0_scale_mlp", raw("b0_scale_mlp.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_shift_mlp", "b0_shift_mlp", raw("b0_shift_mlp.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_gate_mlp",  "b0_gate_mlp",  raw("b0_gate_mlp.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_attn",  "b0_attn",  raw("b0_attn.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_ff",    "b0_ff",    raw("b0_ff.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b0_out",   "b0_out",   raw("b0_out.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_b21_out",  "b21_out",  raw("b21_out.bin"), {}, 0, 1024, T);
    check_dbg(m, "dbg_norm_out", "norm_out", raw("norm_out.bin"), {}, 0, 1024, T);
    {   // head0 rope 后的 q (golden 是整个 1024 维的 q, 取前 64 通道)
        auto q0 = raw("q0_roped.bin");
        if (!q0.empty()) check_dbg(m, "dbg_q0_roped", "q0_roped", head_slice(q0, T, 1024, 64), {}, 0, 64, T);
        auto qr = raw("q0_raw.bin");   // rope 前 (attn 线性输出)
        if (!qr.empty()) check_dbg(m, "dbg_q0_raw", "q0_raw", head_slice(qr, T, 1024, 64), {}, 0, 64, T);
    }
    {   // rope 角度表自检: golden 存的是角度 = n*inv_freq[i], 我们存 cos/sin
        auto rope = raw("rope.bin");   // [1,T,64]
        std::vector<float> cs, sn;
        m->debug_tables(cs, sn);
        if (!rope.empty() && cs.size() == (size_t) 32 * T) {
            std::vector<float> c_exp((size_t) 32 * T), s_exp((size_t) 32 * T);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < 32; i++) {
                    const float a = rope[(size_t) t * 64 + 2 * i];
                    c_exp[(size_t) i + 32 * (size_t) t] = std::cos(a);
                    s_exp[(size_t) i + 32 * (size_t) t] = std::sin(a);
                }
            check("rope_cos", cs, c_exp, 1e-6);
            check("rope_sin", sn, s_exp, 1e-6);
        }
    }
}

// ---- cache-dit 消融辅助 ----

// 参数串: "fn=8,thr=0.08,warmup=8,interval=1,maxcached=-1,maxcont=-1,maxaccum=0"
static void parse_dbcache_str(gsv_dit_cfg & cfg, const char * s) {
    if (!s || !*s) return;
    std::string str(s);
    auto num = [&](const char * key, double def) -> double {
        std::string k = std::string(key) + "=";
        size_t p = str.find(k);
        if (p == std::string::npos) return def;
        return atof(str.c_str() + p + k.size());
    };
    cfg.dbcache_fn              = (int)   num("fn",        cfg.dbcache_fn);
    cfg.dbcache_threshold       = (float) num("thr",       cfg.dbcache_threshold);
    cfg.dbcache_warmup          = (int)   num("warmup",    cfg.dbcache_warmup);
    cfg.dbcache_warmup_interval = (int)   num("interval",  cfg.dbcache_warmup_interval);
    cfg.dbcache_max_cached      = (int)   num("maxcached", cfg.dbcache_max_cached);
    cfg.dbcache_max_cont        = (int)   num("maxcont",   cfg.dbcache_max_cont);
    cfg.dbcache_max_accum       = (float) num("maxaccum",  cfg.dbcache_max_accum);
    cfg.dbcache_downsample      = (int)   num("ds",        cfg.dbcache_downsample);
}
static void parse_dbcache_env(gsv_dit_cfg & cfg) { parse_dbcache_str(cfg, getenv("GSV_DIT_DBCACHE")); }

static double mean_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double s = 0;
    for (size_t i = 0; i < a.size(); i++) s += std::fabs((double) a[i] - b[i]);
    return s / (double) a.size();
}
static double max_abs_diff2(const std::vector<float> & a, const std::vector<float> & b) {
    size_t nb = 0;
    return maxdiff_nan(a, b, nb);
}
// golden out [1,100,T] (c*T+t) 裁出尾部 [P:] -> [100, T-P] (engine sample() 输出布局)
static std::vector<float> crop_mel_tail(const std::vector<float> & out_full, int T, int P) {
    std::vector<float> crop((size_t) 100 * (T - P));
    if (out_full.size() != (size_t) 100 * T) return crop;
    for (int c = 0; c < 100; c++)
        memcpy(crop.data() + (size_t) c * (T - P), out_full.data() + (size_t) c * T + P, (size_t) (T - P) * 4);
    return crop;
}

// 消融一例: 独立加载实例 -> sample() 端到端 -> 与 torch golden / 无缓存参考对比
struct ab_case { const char * tag; int T, P, steps; float cfg_rate; const char * conf; };
static void ab_run(const char * mdl, const char * gdir, const char * name, const ab_case & cs,
                   std::vector<float> & ref_mel) {
    const std::string p = std::string(gdir) + "/cfm." + cs.tag + ".";
    auto mu = read_bin(p + "mu.bin"), pr = read_bin(p + "prompt.bin"), nz = read_bin(p + "noise.bin");
    auto out_ref = read_bin(p + "out.bin");
    if (mu.empty() || nz.empty()) { printf("  [%s] golden missing -> skip\n", name); return; }
    gsv_dit_cfg cfg;
    if (const char * dv = getenv("GSV_DIT_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_DIT_THREADS")) cfg.n_threads = atoi(nt);
    parse_dbcache_str(cfg, cs.conf);                       // "" = 参考 (fn=0, 不建缓存图)
    gsv_dit * mm = gsv_dit::load(mdl, cfg);
    if (!mm) { printf("  [%s] load failed\n", name); n_fail++; return; }
    std::vector<float> x0s(nz.size());
    for (size_t i = 0; i < x0s.size(); i++) x0s[i] = nz[i] * 0.875f;
    std::vector<float> mel;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = mm->sample(mu.data(), pr.data(), cs.P, cs.T, cs.steps, cs.cfg_rate, 0, x0s.data(), mel);
    const auto t1 = std::chrono::steady_clock::now();
    if (!ok) { printf("  [%s] sample failed\n", name); n_fail++; delete mm; return; }
    gsv_dit::db_stats st{};
    mm->db_get_stats(st);
    delete mm;
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const std::vector<float> crop = crop_mel_tail(out_ref, cs.T, cs.P);
    printf("  %-20s %s\n", name, cs.conf[0] ? cs.conf : "(no-cache, 参考)");
    printf("      vs torch:  max|d|=%.3e  mean|d|=%.4f\n", max_abs_diff2(mel, crop), mean_abs_diff(mel, crop));
    if (!ref_mel.empty() && ref_mel.size() == mel.size())
        printf("      vs 无缓存: max|d|=%.3e  mean|d|=%.4f\n", max_abs_diff2(mel, ref_mel), mean_abs_diff(mel, ref_mel));
    printf("      hits=%d/%d misses=%d/%d | %.0f ms\n",
           st.hits[0], st.hits[1], st.misses[0], st.misses[1], ms);
    if (!cs.conf[0]) ref_mel = mel;                        // 记录无缓存参考 (供后续档位比较)
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // 崩溃时也能看到进度
    setvbuf(stderr, NULL, _IONBF, 0);
    const char * mdl  = argc > 1 ? argv[1] : "models/gsv-dit-f32.gguf";
    const char * gdir = argc > 2 ? argv[2] : "tests/golden";

    if (const char * dbg = getenv("GSV_DIT_TEST_DEBUG")) rel_tol = atof(dbg);
#ifdef _WIN32
    _putenv_s("GSV_DIT_DEBUG", "1");       // 打开中间量标记
#else
    setenv("GSV_DIT_DEBUG", "1", 1);
#endif

    gsv_dit_cfg cfg;
    cfg.verbose = true;
    if (const char * dv = getenv("GSV_DIT_DEVICE")) cfg.device = dv;
    if (const char * nt = getenv("GSV_DIT_THREADS")) cfg.n_threads = atoi(nt);

    gsv_dit * m = gsv_dit::load(mdl, cfg);
    if (!m) return 1;

    printf("== 1) 单步前向 ==\n");
    run_case(m, gdir, "t96", 96, 96);
    run_case(m, gdir, "pad", 96, 80);
    run_case(m, gdir, "t0", 64, 64);

    printf("== 2) CFM 采样 ==\n");
    // ---- 4 步 cfg=0 (v5turbo 生产形态) ----
    {
        const std::string p = std::string(gdir) + "/cfm.s4c0.";
        auto mu = read_bin(p + "mu.bin"), pr = read_bin(p + "prompt.bin"), nz = read_bin(p + "noise.bin");
        auto out_ref = read_bin(p + "out.bin");
        const int T = 96, P = 32, steps = 4;
        if (mu.empty() || nz.empty()) { printf("  [s4c0] golden missing -> skip\n"); }
        else {
            printf("  [s4c0] steps=%d cfg=0 T=%d P=%d\n", steps, T, P);
            // 逐步对拍 (复刻 CFM 循环: prepare 一次, 每步 velocity)
            std::vector<float> prompt_x((size_t) 100 * T, 0.0f);
            for (int c = 0; c < 100; c++) memcpy(prompt_x.data() + (size_t) c * T, pr.data() + (size_t) c * P, P * 4);
            if (!m->prepare(prompt_x.data(), mu.data(), T, T)) n_fail++;
            std::vector<float> x(nz.size()), vel((size_t) 100 * T);
            for (size_t i = 0; i < x.size(); i++) x[i] = nz[i] * 0.875f;   // CFMV5.noise_temperature
            for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            const float step = 1.0f / steps;
            for (int j = 0; j < steps; j++) {
                char nm[64];
                snprintf(nm, sizeof(nm), "s%d.x_in", j);
                check(nm, x, read_bin(p + nm + ".bin"), 1e-6);
                snprintf(nm, sizeof(nm), "s%d.vel", j);
                m->velocity(x.data(), (float) j * step, false, vel.data());
                check(nm, vel, t2ct(read_bin(p + nm + ".bin"), T, 100));
                for (size_t i = 0; i < vel.size(); i++) x[i] += step * vel[i];
                for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            }
            // 端到端 sample() (注入 golden 噪声; 引擎返回 result[..., P:] = [100, T-P], golden 是完整 [1,T,100])
            std::vector<float> x0s(nz.size());
            for (size_t i = 0; i < x0s.size(); i++) x0s[i] = nz[i] * 0.875f;
            std::vector<float> mel;
            m->sample(mu.data(), pr.data(), P, T, steps, 0.0f, 0, x0s.data(), mel);
            {
                // out_ref 是 CFM.inference 返回的完整 x [1, 100, T] (torch 布局 c*T+t) — 不是 [1,T,C], 直接按 c 切
                std::vector<float> crop((size_t) 100 * (T - P));
                for (int c = 0; c < 100; c++)
                    memcpy(crop.data() + (size_t) c * (T - P), out_ref.data() + (size_t) c * T + P, (size_t) (T - P) * 4);
                check("sample_out", mel, crop);
            }
        }
    }
    // ---- 8 步 cfg=1.30 (每步 pos+neg 两个前向) ----
    {
        const std::string p = std::string(gdir) + "/cfm.s8c13.";
        auto mu = read_bin(p + "mu.bin"), pr = read_bin(p + "prompt.bin"), nz = read_bin(p + "noise.bin");
        auto out_ref = read_bin(p + "out.bin");
        const int T = 96, P = 32, steps = 8;
        const float cfg_rate = 1.30f;
        if (mu.empty() || nz.empty()) { printf("  [s8c13] golden missing -> skip\n"); }
        else {
            printf("  [s8c13] steps=%d cfg=%.2f\n", steps, cfg_rate);
            std::vector<float> prompt_x((size_t) 100 * T, 0.0f);
            for (int c = 0; c < 100; c++) memcpy(prompt_x.data() + (size_t) c * T, pr.data() + (size_t) c * P, P * 4);
            if (!m->prepare(prompt_x.data(), mu.data(), T, T)) n_fail++;
            std::vector<float> x(nz.size()), vel((size_t) 100 * T), neg((size_t) 100 * T);
            for (size_t i = 0; i < x.size(); i++) x[i] = nz[i] * 0.875f;   // CFMV5.noise_temperature
            for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            const float step = 1.0f / steps;
            for (int j = 0; j < steps; j++) {
                char nm[64];
                snprintf(nm, sizeof(nm), "s%d.x_in", 2 * j);
                check(nm, x, read_bin(p + nm + ".bin"), 1e-6);
                m->velocity(x.data(), (float) j * step, false, vel.data());
                snprintf(nm, sizeof(nm), "s%d.vel", 2 * j);                  // 拦截序号: pos=2j, neg=2j+1
                check(nm, vel, t2ct(read_bin(p + nm + ".bin"), T, 100));
                m->velocity(x.data(), (float) j * step, true, neg.data());
                snprintf(nm, sizeof(nm), "s%d.vel", 2 * j + 1);
                check(nm, neg, t2ct(read_bin(p + nm + ".bin"), T, 100));
                for (size_t i = 0; i < vel.size(); i++) vel[i] = vel[i] + (vel[i] - neg[i]) * cfg_rate;
                for (size_t i = 0; i < vel.size(); i++) x[i] += step * vel[i];
                for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            }
            std::vector<float> x0s(nz.size());
            for (size_t i = 0; i < x0s.size(); i++) x0s[i] = nz[i] * 0.875f;
            std::vector<float> mel;
            m->sample(mu.data(), pr.data(), P, T, steps, cfg_rate, 0, x0s.data(), mel);
            {
                // out_ref 是 CFM.inference 返回的完整 x [1, 100, T] (torch 布局 c*T+t) — 不是 [1,T,C], 直接按 c 切
                std::vector<float> crop((size_t) 100 * (T - P));
                for (int c = 0; c < 100; c++)
                    memcpy(crop.data() + (size_t) c * (T - P), out_ref.data() + (size_t) c * T + P, (size_t) (T - P) * 4);
                check("sample_out", mel, crop);
            }
        }
    }
    // ---- 32 步 cfg=1.30 (v5dev 生产形态, uncached 逐步对拍; 静默汇总) ----
    {
        const std::string p = std::string(gdir) + "/cfm.s32c13.";
        auto mu = read_bin(p + "mu.bin"), pr = read_bin(p + "prompt.bin"), nz = read_bin(p + "noise.bin");
        auto out_ref = read_bin(p + "out.bin");
        const int T = 96, P = 32, steps = 32;
        const float cfg_rate = 1.30f;
        if (mu.empty() || nz.empty()) { printf("  [s32c13] golden missing -> skip\n"); }
        else {
            printf("  [s32c13] steps=%d cfg=%.2f (uncached)\n", steps, cfg_rate);
            std::vector<float> prompt_x((size_t) 100 * T, 0.0f);
            for (int c = 0; c < 100; c++) memcpy(prompt_x.data() + (size_t) c * T, pr.data() + (size_t) c * P, P * 4);
            if (!m->prepare(prompt_x.data(), mu.data(), T, T)) n_fail++;
            std::vector<float> x(nz.size()), vel((size_t) 100 * T), neg((size_t) 100 * T);
            for (size_t i = 0; i < x.size(); i++) x[i] = nz[i] * 0.875f;
            for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            const float step = 1.0f / steps;
            g_quiet = true; g_sect_on = true; g_sect_max = 0.0;
            const size_t c0 = g_checks, f0 = g_fails;
            for (int j = 0; j < steps; j++) {
                char nm[64];
                snprintf(nm, sizeof(nm), "s%d.x_in", 2 * j);
                check(nm, x, read_bin(p + nm + ".bin"), 1e-6);
                m->velocity(x.data(), (float) j * step, false, vel.data());
                snprintf(nm, sizeof(nm), "s%d.vel", 2 * j);
                check(nm, vel, t2ct(read_bin(p + nm + ".bin"), T, 100));
                m->velocity(x.data(), (float) j * step, true, neg.data());
                snprintf(nm, sizeof(nm), "s%d.vel", 2 * j + 1);
                check(nm, neg, t2ct(read_bin(p + nm + ".bin"), T, 100));
                for (size_t i = 0; i < vel.size(); i++) vel[i] = vel[i] + (vel[i] - neg[i]) * cfg_rate;
                for (size_t i = 0; i < vel.size(); i++) x[i] += step * vel[i];
                for (int c = 0; c < 100; c++) memset(x.data() + (size_t) c * T, 0, P * 4);
            }
            g_quiet = false; g_sect_on = false;
            printf("    (32 步逐步: %zu 检查, %zu 失败, 段内 max|d|=%.3e) %s\n",
                   g_checks - c0, g_fails - f0, g_sect_max, (g_fails - f0) ? "FAIL" : "PASS");
            std::vector<float> x0s(nz.size());
            for (size_t i = 0; i < x0s.size(); i++) x0s[i] = nz[i] * 0.875f;
            std::vector<float> mel;
            m->sample(mu.data(), pr.data(), P, T, steps, cfg_rate, 0, x0s.data(), mel);
            check("sample_out", mel, crop_mel_tail(out_ref, T, P));
        }
    }
    // ---- 分块调度 (rolling prompt, block=5 -> 3 块) ----
    {
        const std::string p = std::string(gdir) + "/cfm.chunk.";
        auto fr = read_bin(p + "fea_ref.bin"), ft = read_bin(p + "fea_todo.bin");
        auto mr = read_bin(p + "mel_ref.bin"), out_ref = read_bin(p + "out.bin");
        auto blk = read_bin(p + "block.bin");
        if (fr.empty() || out_ref.empty()) { printf("  [chunk] golden missing -> skip\n"); }
        else {
            const int R = 4, N = 12, block = blk.empty() ? 5 : (int) blk[0];
            printf("  [chunk] R=%d N=%d block=%d (rolling prompt)\n", R, N, block);
            std::vector<std::vector<float>> noises;
            for (int i = 0; i < 8; i++) {
                auto nz = read_bin(p + "noise" + std::to_string(i) + ".bin");
                if (nz.empty()) break;
                for (auto & v : nz) v *= 0.875f;   // CFMV5.noise_temperature (golden 存的是原始 randn)
                noises.push_back(nz);
            }
            std::vector<const float *> nzptrs;
            for (auto & v : noises) nzptrs.push_back(v.data());
            std::vector<float> mel;
            if (!m->synthesize(fr.data(), R, ft.data(), N, mr.data(), 4, 0.0f, 12345, block,
                               &nzptrs, mel)) n_fail++;
            else check("chunk_out", mel, out_ref, 0.0, 100, N);
            printf("    (chunks=%d)\n", (int) noises.size());
        }
    }

    // ---- 3) cache-dit 消融 (T=96; sample() 端到端, 每档独立实例) ----
    if (!getenv("GSV_DIT_FAST")) {
        printf("== 3) cache-dit 消融 (T=96) ==\n");
        std::vector<float> ref4, ref32;
        const ab_case cases[] = {
            { "s4c0",   96, 32, 4,  0.0f, "" },                            // turbo 4 步 (DMD 蒸馏)
            { "s4c0",   96, 32, 4,  0.0f, "fn=8,thr=0.10,warmup=1" },
            { "s4c0",   96, 32, 4,  0.0f, "fn=12,thr=0.15,warmup=1" },
            { "s4c0",   96, 32, 4,  0.0f, "fn=8,thr=0.25,warmup=0" },      // 激进档 (game.cpp 默认阈值)
            { "s32c13", 96, 32, 32, 1.3f, "" },                            // v5dev 32 步
            { "s32c13", 96, 32, 32, 1.3f, "fn=8,thr=0.08,warmup=8" },
            { "s32c13", 96, 32, 32, 1.3f, "fn=8,thr=0.12,warmup=8" },
            { "s32c13", 96, 32, 32, 1.3f, "fn=12,thr=0.12,warmup=8" },
            { "s32c13", 96, 32, 32, 1.3f, "fn=16,thr=0.20,warmup=8" },
        };
        const char * names[] = { "s4c0 turbo", "s4c0 F8/.10", "s4c0 F12/.15", "s4c0 F8/.25-a",
                                 "s32c13 参考", "s32c13 F8/.08", "s32c13 F8/.12",
                                 "s32c13 F12/.12", "s32c13 F16/.20" };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            std::vector<float> & rf = (std::string(cases[i].tag) == "s4c0") ? ref4 : ref32;
            ab_run(mdl, gdir, names[i], cases[i], rf);
        }
    }
    // ---- 4) 生产尺寸 (T=1000) 端到端 (GSV_DIT_BIG=1) ----
    if (getenv("GSV_DIT_BIG")) {
        printf("== 4) T=1000 端到端 (s4c0_big / s32c13_big) ==\n");
        std::vector<float> ref4b, ref32b;
        const ab_case bigs[] = {
            { "s4c0_big",   1000, 500, 4,  0.0f, "" },
            { "s4c0_big",   1000, 500, 4,  0.0f, "fn=8,thr=0.10,warmup=1" },
            { "s32c13_big", 1000, 500, 32, 1.3f, "" },
            { "s32c13_big", 1000, 500, 32, 1.3f, "fn=8,thr=0.08,warmup=8" },
            { "s32c13_big", 1000, 500, 32, 1.3f, "fn=8,thr=0.12,warmup=8" },
        };
        const char * bnames[] = { "big s4c0 参考", "big s4c0 F8/.10", "big s32c13 参考",
                                  "big s32c13 F8/.08", "big s32c13 F8/.12" };
        for (size_t i = 0; i < sizeof(bigs) / sizeof(bigs[0]); i++) {
            std::vector<float> & rf = (std::string(bigs[i].tag) == "s4c0_big") ? ref4b : ref32b;
            ab_run(mdl, gdir, bnames[i], bigs[i], rf);
        }
    }

    const char * be = getenv("GSV_DIT_BENCH");
    const int bn = be ? atoi(be) : 0;
    if (bn > 0) {
        // 生产尺寸基准: T=1000 (prompt 500 + target 500), 固定 t=0.25 单步前向
        // GSV_DIT_DBCACHE="fn=.." 时改用独立实例走缓存路径
        gsv_dit_cfg bcfg;
        bcfg.n_threads = cfg.n_threads; bcfg.verbose = false; bcfg.device = cfg.device;
        parse_dbcache_env(bcfg);
        gsv_dit * bm = m;
        if (bcfg.dbcache_fn > 0) {
            bm = gsv_dit::load(mdl, bcfg);
            if (!bm) { printf("bench: dbcache 实例加载失败\n"); return 1; }
        }
        const int T = 1000, P = 500;
        std::vector<float> prompt((size_t) 100 * T, 0.0f);
        for (size_t i = 0; i < prompt.size(); i++) prompt[i] = 0.1f * (float) ((i % 17) - 8);
        std::vector<float> mu((size_t) 512 * T);
        for (size_t i = 0; i < mu.size(); i++) mu[i] = 0.1f * (float) ((i % 13) - 6);
        bm->prepare(prompt.data(), mu.data(), T, T);
        std::vector<float> x((size_t) 100 * T, 0.5f), vel((size_t) 100 * T);
        std::vector<double> ts;
        for (int i = 0; i < bn + 2; i++) {
            bm->db_set_step(i);            // dbcache: warmup 后按固定 t 反复推 (fd 支配命中)
            auto t0 = std::chrono::steady_clock::now();
            bm->velocity(x.data(), 0.25f, false, vel.data());
            auto t1 = std::chrono::steady_clock::now();
            if (i >= 2) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double sum = 0; for (double v : ts) sum += v;
        printf("ggml DiT step (%s, T=%d%s): avg %.1f ms min %.1f ms n=%d\n",
               cfg.device.empty() ? "cpu" : "vulkan", T,
               bcfg.dbcache_fn > 0 ? ", dbcache" : "", sum / ts.size(), ts.front(), (int) ts.size());
        if (bcfg.dbcache_fn > 0) {
            gsv_dit::db_stats st{};
            bm->db_get_stats(st);
            printf("  dbcache hits=%d misses=%d (last fd=%.5f)\n", st.hits[0], st.misses[0], st.last_fd[0]);
        }
        if (bm != m) delete bm;
    }

    printf("\n%s\n", n_fail == 0 ? "DIT PARITY PASS" : "DIT PARITY FAIL");
    delete m;
    return n_fail == 0 ? 0 : 1;
}
