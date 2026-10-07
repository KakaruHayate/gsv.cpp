// wns1 (VITS WN Encoder: 8-layer WaveNet k=5 d=1 pad=2, gin=512) parity test.
// Host-side reference impl. Goldens from tools/dump_golden_wns1.py, weights models/gsv-cond-f32.gguf.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <crtdbg.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

static std::vector<float> load_bin(const std::string & p, size_t n){
    FILE * f = fopen(p.c_str(), "rb");
    if(!f){ fprintf(stderr, "open %s fail\n", p.c_str()); exit(3); }
    std::vector<float> v(n);
    if(fread(v.data(), 4, n, f) != n){ fprintf(stderr, "read %s fail\n", p.c_str()); exit(3); }
    fclose(f); return v;
}
static struct ggml_tensor * need(struct ggml_context * c, const char * n){
    struct ggml_tensor * t = ggml_get_tensor(c, n);
    if(!t){ fprintf(stderr, "missing weight %s\n", n); exit(3); }
    return t;
}
static float maxdiff(const std::vector<float> & a, const std::vector<float> & b){
    float md = 0;
    for(size_t i = 0; i < a.size(); i++){ float d = fabsf(a[i]-b[i]); if(d>md) md=d; }
    return md;
}

// conv1d channel-major: x[Cin][T] -> y[Cout][T], zero pad, w torch [Cout][Cin][K] contiguous
static void conv1d_cm(const std::vector<float> & x, int T, int Cin,
        const float * w, const float * b, int Cout, int K, int pad, std::vector<float> & y){
    y.assign((size_t)Cout * T, 0.0f);
    for(int oc = 0; oc < Cout; oc++){
        const float * wo = w + (size_t)oc * Cin * K;
        for(int t = 0; t < T; t++){
            float acc = b[oc];
            for(int ic = 0; ic < Cin; ic++){
                const float * wi = wo + (size_t)ic * K;
                const float * xr = &x[(size_t)ic * T];
                for(int kk = 0; kk < K; kk++){
                    int ti = t + kk - pad;
                    if(ti < 0 || ti >= T) continue;
                    acc += wi[kk] * xr[ti];
                }
            }
            y[(size_t)oc * T + t] = acc;
        }
    }
}

int main(int argc, char ** argv){
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    setvbuf(stdout, NULL, _IONBF, 0);
    const char * model = argc>1?argv[1]:"models/gsv-cond-f32.gguf";
    const char * gold  = argc>2?argv[2]:"tests/golden";
    const int T = 120, C = 512, NL = 8, K = 5, PAD = 2, LEN = 100;
    struct ggml_context * wctx = NULL;
    struct gguf_init_params gp = { false, &wctx };
    struct gguf_context * gctx = gguf_init_from_file(model, gp);
    if(!gctx){ fprintf(stderr,"gguf open fail %s\n",model); return 3; }
    char pb[512]; char nm[128];
    snprintf(pb,512,"%s/wns1.input.bin",gold); std::vector<float> xin = load_bin(pb, (size_t)C*T);
    snprintf(pb,512,"%s/wns1.ge.bin",gold); std::vector<float> ge = load_bin(pb, (size_t)C);
    snprintf(pb,512,"%s/wns1.pre_out.bin",gold); std::vector<float> rpre = load_bin(pb, (size_t)C*T);
    snprintf(pb,512,"%s/wns1.enc_out.bin",gold); std::vector<float> renc = load_bin(pb, (size_t)C*T);
    snprintf(pb,512,"%s/wns1.out.bin",gold); std::vector<float> rout = load_bin(pb, (size_t)C*T);
    std::vector<float> mask((size_t)T);
    for(int t = 0; t < T; t++) mask[t] = t < LEN ? 1.0f : 0.0f;
    float worst = 0;
    // pre: Conv1d(512->512, k=1) * mask
    std::vector<float> x;
    conv1d_cm(xin, T, C, (const float *)need(wctx,"wns1.pre.weight")->data, (const float *)need(wctx,"wns1.pre.bias")->data, C, 1, 0, x);
    for(int c = 0; c < C; c++) for(int t = 0; t < T; t++) x[(size_t)c*T+t] *= mask[t];
    { float md = maxdiff(x, rpre); printf("  %-10s max|d|=%.3e\n","pre_out",md); if(md>worst)worst=md; }
    // cond: gfull[t] = cond_w @ ge + cond_b, 8192 ch, sliced per layer
    const float * cw = (const float *)need(wctx,"wns1.enc.cond_layer.weight_w")->data;
    const float * cb = (const float *)need(wctx,"wns1.enc.cond_layer.bias")->data;
    std::vector<float> gfull((size_t)8192*T, 0.0f);
    for(int o = 0; o < 8192; o++){
        const float * wr = cw + (size_t)o * C;
        float gb = cb[o];
        for(int t = 0; t < T; t++){
            float acc = gb;
            for(int c = 0; c < C; c++) acc += wr[c] * ge[c];
            gfull[(size_t)o*T+t] = acc;
        }
    }
    // WN 8 layers: in_conv(k=5,pad=2) + cond slice -> tanh*sigmoid -> res_skip
    std::vector<float> sk((size_t)C*T, 0.0f);
    for(int li = 0; li < NL; li++){
        snprintf(nm,128,"wns1.enc.in_layers.%d.weight_w",li);
        const float * iw = (const float *)need(wctx, nm)->data;
        snprintf(nm,128,"wns1.enc.in_layers.%d.bias",li);
        const float * ib = (const float *)need(wctx, nm)->data;
        std::vector<float> xin2;
        conv1d_cm(x, T, C, iw, ib, 2*C, K, PAD, xin2);
        const int off = li * 2 * C;
        for(int o = 0; o < 2*C; o++) for(int t = 0; t < T; t++)
            xin2[(size_t)o*T+t] += gfull[(size_t)(off+o)*T+t];
        std::vector<float> acts((size_t)C*T);
        for(int t = 0; t < T; t++){
            for(int c = 0; c < C; c++){
                float a = xin2[(size_t)c*T+t];
                float b = xin2[(size_t)(C+c)*T+t];
                acts[(size_t)c*T+t] = tanhf(a) / (1.0f + expf(-b));
            }
        }
        snprintf(nm,128,"wns1.enc.res_skip_layers.%d.weight_w",li);
        const float * rw = (const float *)need(wctx, nm)->data;
        snprintf(nm,128,"wns1.enc.res_skip_layers.%d.bias",li);
        const float * rb = (const float *)need(wctx, nm)->data;
        int rc = li < NL-1 ? 2*C : C;
        std::vector<float> rs;
        conv1d_cm(acts, T, C, rw, rb, rc, 1, 0, rs);
        if(li < NL-1){
            for(int c = 0; c < C; c++) for(int t = 0; t < T; t++){
                x[(size_t)c*T+t] = (x[(size_t)c*T+t] + rs[(size_t)c*T+t]) * mask[t];
                sk[(size_t)c*T+t] += rs[(size_t)(C+c)*T+t];
            }
        } else {
            for(int c = 0; c < C; c++) for(int t = 0; t < T; t++)
                sk[(size_t)c*T+t] += rs[(size_t)c*T+t];
        }
    }
    // enc out = skip-sum * mask; proj: Conv1d(512->512, k=1) * mask
    for(int c = 0; c < C; c++) for(int t = 0; t < T; t++) sk[(size_t)c*T+t] *= mask[t];
    { float md = maxdiff(sk, renc); printf("  %-10s max|d|=%.3e\n","enc_out",md); if(md>worst)worst=md; }
    std::vector<float> yout;
    conv1d_cm(sk, T, C, (const float *)need(wctx,"wns1.proj.weight")->data, (const float *)need(wctx,"wns1.proj.bias")->data, C, 1, 0, yout);
    for(int c = 0; c < C; c++) for(int t = 0; t < T; t++) yout[(size_t)c*T+t] *= mask[t];
    { float md = maxdiff(yout, rout); printf("  %-10s max|d|=%.3e\n","wns1_out",md); if(md>worst)worst=md; }
    gguf_free(gctx); ggml_free(wctx);
    printf(worst < 2e-2f ? "WNS1 PARITY PASS\n" : "WNS1 PARITY FAIL\n");
    return worst < 2e-2f ? 0 : 2;
}
