// wns1 ggml impl parity + bench. golden: tests/golden/wns1.*
// usage: test_wns1_ggml [gguf] [golden_dir]; GSV_WNS1_DEVICE=vulkan, GSV_WNS1_BENCH=N
#include "../src/gsv_wns1.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

static std::vector<float> read_bin(const std::string & p){
    FILE * f = fopen(p.c_str(), "rb");
    if(!f) return {};
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t)sz / 4);
    if(fread(v.data(), 4, v.size(), f) != v.size()){ fclose(f); return {}; }
    fclose(f); return v;
}
static double maxdiff(const std::vector<float> & a, const std::vector<float> & b){
    double md = 0; for(size_t i = 0; i < a.size(); i++){ double d = std::fabs((double)a[i]-b[i]); if(d>md) md=d; } return md;
}
int main(int argc, char ** argv){
    const char * mdl = argc>1?argv[1]:"models/gsv-cond-f32.gguf";
    const char * gdir = argc>2?argv[2]:"tests/golden";
    const int T = 120, C = 512, LEN = 100;
    gsv_wns1_cfg cfg;
    cfg.verbose = true;
    if(const char * dv = getenv("GSV_WNS1_DEVICE")) cfg.device = dv;
    if(const char * nt = getenv("GSV_WNS1_THREADS")) cfg.n_threads = atoi(nt);
    gsv_wns1 * m = gsv_wns1::load(mdl, cfg);
    if(!m){ fprintf(stderr,"load failed %s\n", mdl); return 1; }
    std::vector<float> x = read_bin(std::string(gdir) + "/wns1.input.bin");
    std::vector<float> ge = read_bin(std::string(gdir) + "/wns1.ge.bin");
    std::vector<float> ref = read_bin(std::string(gdir) + "/wns1.out.bin");
    if(x.empty() || ge.empty() || ref.empty()){ fprintf(stderr,"missing golden\n"); delete m; return 1; }
    std::vector<float> out;
    if(!m->encode(x.data(), ge.data(), T, LEN, out)){ delete m; return 1; }
    double md = maxdiff(out, ref);
    printf("  wns1_out   max|d|=%.3e %s\n", md, md < 2e-2 ? "PASS" : "FAIL");
    // T 变化: 图应按新 T 重建 (集成侧长度可变)
    {
        const int T2 = 60, LEN2 = 50;
        std::vector<float> small((size_t)C * T2), o2;
        for(int c = 0; c < C; c++)
            for(int t = 0; t < T2; t++) small[(size_t)c*T2+t] = x[(size_t)c*T + t];
        bool ok2 = m->encode(small.data(), ge.data(), T2, LEN2, o2);
        size_t nan2 = 0; for(float v : o2) if(!std::isfinite(v)) nan2++;
        printf("  T=%d 重建路径: %s (out %zu, 非有限 %zu)\n",
               T2, (ok2 && nan2 == 0 && o2.size() == (size_t)C*T2) ? "PASS" : "FAIL", o2.size(), nan2);
        if(!ok2 || nan2 || o2.size() != (size_t)C*T2) return 3;
    }

    const char * be = getenv("GSV_WNS1_BENCH");
    int bn = be ? atoi(be) : 0;
    if(bn > 0){
        std::vector<double> ts;
        for(int i = 0; i < bn + 3; i++){
            auto t0 = std::chrono::steady_clock::now();
            m->encode(x.data(), ge.data(), T, LEN, out);
            auto t1 = std::chrono::steady_clock::now();
            if(i >= 3) ts.push_back(std::chrono::duration<double, std::milli>(t1-t0).count());
        }
        std::sort(ts.begin(), ts.end());
        double ssum = 0; for(double v : ts) ssum += v;
        printf("ggml %s: avg %.3f ms min %.3f max %.3f n=%d\n", cfg.device.empty()?"cpu":"vulkan", ssum/ts.size(), ts.front(), ts.back(), (int)ts.size());
    }
    delete m;
    return md < 2e-2 ? 0 : 2;
}
