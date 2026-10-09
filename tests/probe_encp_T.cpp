// 探针: enc_p 在给定 T 下的最小调用 (定位 F16 权重 + T=80 的 concat 断言)
#include "../src/gsv_encp.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    const char * mdl = argc > 1 ? argv[1] : "models/gsv-cond-a16.gguf";
    const int T = argc > 2 ? atoi(argv[2]) : 80;
    const int NT = 50;
    gsv_encp_cfg cfg; cfg.verbose = true;
    gsv_encp * m = gsv_encp::load(mdl, cfg);
    if (!m) return 1;
    std::vector<float> y((size_t) 768 * T, 0.1f);
    std::vector<int32_t> text(NT, 3);
    std::vector<float> ge(512, 0.05f);
    std::vector<float> om, ologs, oy;
    printf("encode T=%d ...\n", T);
    bool ok = m->encode(y.data(), T, text.data(), NT, ge.data(), om, ologs, &oy);
    printf("ok=%d m[0]=%f y[0]=%f\n", (int) ok, om.empty() ? 0.f : om[0], oy.empty() ? 0.f : oy[0]);
    delete m;
    return ok ? 0 : 1;
}
