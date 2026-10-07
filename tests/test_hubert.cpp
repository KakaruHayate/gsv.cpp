// HuBERT ggml parity test. Host CNN frontend + host pos_conv (rvc.cpp pattern);
// ggml covers feat_proj + enc LN + 12-layer transformer. Goldens from dump_golden_hubert.py.
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
static float gelu_erf(float x){ return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f))); }

// conv1d no-pad: x[L][Cin] -> y[OL][Cout], w stored [K][Cin][Cout]
static void conv1d_host(const std::vector<float> & x, int L, int Cin,
        const float * w, int Cout, int K, int s, std::vector<float> & y, int OL){
    y.assign((size_t)OL * Cout, 0.0f);
    for(int o = 0; o < OL; o++){
        const float * xo = &x[(size_t)(o * s) * Cin];
        for(int k = 0; k < K; k++){
            const float * xk = xo + (size_t)k * Cin;
            const float * wk = w + (size_t)k * Cin * Cout;
            for(int c = 0; c < Cin; c++){
                float xv = xk[c];
                const float * wc = wk + (size_t)c * Cout;
                float * yo = &y[(size_t)o * Cout];
                for(int oc = 0; oc < Cout; oc++) yo[oc] += xv * wc[oc];
            }
        }
    }
}

// GroupNorm(groups=C): per-channel normalize over time
static void groupnorm_host(std::vector<float> & x, int T, int C, const float * w, const float * b){
    for(int c = 0; c < C; c++){
        double m = 0; for(int t = 0; t < T; t++) m += x[(size_t)t * C + c]; m /= T;
        double v = 0; for(int t = 0; t < T; t++){ double d = x[(size_t)t * C + c] - m; v += d * d; }
        float inv = 1.0f / sqrtf((float)(v / T) + 1e-5f);
        for(int t = 0; t < T; t++){ float & r = x[(size_t)t * C + c]; r = (r - (float)m) * inv * w[c] + b[c]; }
    }
}

// pos conv: grouped k=128 groups=16, torch F.pad(64,63), w stored [K][48][768]
static void pos_conv_host(const std::vector<float> & x, int T,
        const float * w, const float * bias, std::vector<float> & y){
    const int K = 128, G = 16, CG = 48, C = 768;
    y.assign((size_t)T * C, 0.0f);
    for(int t = 0; t < T; t++){
        for(int g = 0; g < G; g++){
            for(int k = 0; k < K; k++){
                int ti = t + k - 64;
                if(ti < 0 || ti >= T) continue;
                for(int ic = 0; ic < CG; ic++){
                    float xv = x[(size_t)ti * C + g * CG + ic];
                    const float * wr = w + ((size_t)k * CG + ic) * C + g * CG;
                    float * yo = &y[(size_t)t * C + g * CG];
                    for(int oc = 0; oc < CG; oc++) yo[oc] += xv * wr[oc];
                }
            }
        }
    }
    for(int t = 0; t < T; t++) for(int c = 0; c < C; c++){ y[(size_t)t * C + c] = gelu_erf(y[(size_t)t * C + c] + bias[c]); }
}

static void layernorm_host(std::vector<float> & x, int T, int C, const float * w, const float * b){
    for(int t = 0; t < T; t++){
        float * r = &x[(size_t)t * C];
        double m = 0; for(int c = 0; c < C; c++) m += r[c]; m /= C;
        double v = 0; for(int c = 0; c < C; c++){ double d = r[c] - m; v += d * d; }
        float inv = 1.0f / sqrtf((float)(v / C) + 1e-5f);
        for(int c = 0; c < C; c++) r[c] = (r[c] - (float)m) * inv * w[c] + b[c];
    }
}
static void linear_host(const std::vector<float> & x, int T, int Cin, const float * w, const float * b, int Cout, std::vector<float> & y){
    y.assign((size_t)T * Cout, 0.0f);
    for(int t = 0; t < T; t++) for(int oc = 0; oc < Cout; oc++){
        const float * wr = w + (size_t)oc * Cin;
        const float * xr = &x[(size_t)t * Cin];
        double acc = b[oc];
        for(int c = 0; c < Cin; c++) acc += (double)xr[c] * wr[c];
        y[(size_t)t * Cout + oc] = (float)acc;
   
 }
}

int main(int argc, char ** argv){
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    setvbuf(stdout, NULL, _IONBF, 0);
    const char * model = argc>1?argv[1]:"models/gsv-hubert-f32.gguf";
    const char * gold  = argc>2?argv[2]:"tests/golden";
    const int TRAW=16000, T=49, D=768, NH=12, HD=64, NL=12;
    const int CK[7]={10,3,3,3,3,2,2}, CS[7]={5,2,2,2,2,2,2};
    const int CLEN[7]={3199,1599,799,399,199,99,49};

    struct ggml_context * wctx = NULL;
    struct gguf_init_params gp = { false, &wctx };
    struct gguf_context * gctx = gguf_init_from_file(model, gp);
    if(!gctx){ fprintf(stderr,"gguf open fail %s\n",model); return 3; }

    char pb[512]; snprintf(pb,512,"%s/hubert.input.bin",gold);
    std::vector<float> raw = load_bin(pb, TRAW);
    double m=0; for(int i=0;i<TRAW;i++) m+=raw[i]; m/=TRAW;
    double sq=0; for(int i=0;i<TRAW;i++){double d=raw[i]-m; sq+=d*d;}
    float sd=(float)sqrt(sq/TRAW);
    std::vector<float> x((size_t)TRAW);
    for(int i=0;i<TRAW;i++) x[i]=(float)((raw[i]-m)/(sd+1e-7));

    // host CNN frontend
    int L = TRAW, C = 1;
    char nm[128];
    for(int i=0;i<7;i++){
        snprintf(nm,128,"hubert.feat_conv.%d.w",i);
        const float * w = (const float *)need(wctx, nm)->data;
        std::vector<float> y;
        conv1d_host(x, L, C, w, 512, CK[i], CS[i], y, CLEN[i]);
        if(i==0){
            const float * gw = (const float *)need(wctx, "hubert.feat_conv.0.norm_w")->data;
            const float * gb = (const float *)need(wctx, "hubert.feat_conv.0.norm_b")->data;
            groupnorm_host(y, CLEN[i], 512, gw, gb);
        }
        for(auto & v : y) v = gelu_erf(v);
        x = std::move(y); L = CLEN[i]; C = 512;
    }
    float worst = 0;
    snprintf(pb,512,"%s/hubert.cnn_out.bin",gold);
    std::vector<float> rc = load_bin(pb, (size_t)512*T);
    { float md=0; for(int ch=0;ch<512;ch++)for(int t=0;t<T;t++){ float d=fabsf(x[(size_t)t*512+ch]-rc[(size_t)ch*T+t]); if(d>md)md=d; }
      printf("  %-10s max|d|=%.3e\n","cnn_out",md); if(md>worst)worst=md; }

    // host feat_proj: LN(512) + Linear(512->768)
    layernorm_host(x, T, 512, (const float *)need(wctx,"hubert.feat_proj.norm_w")->data, (const float *)need(wctx,"hubert.feat_proj.norm_b")->data);
    std::vector<float> fp;
    linear_host(x, T, 512, (const float *)need(wctx,"hubert.feat_proj.w")->data, (const float *)need(wctx,"hubert.feat_proj.b")->data, 768, fp);
    snprintf(pb,512,"%s/hubert.feat_proj.bin",gold);
    std::vector<float> rfp = load_bin(pb, (size_t)768*T);
    { float md=0; for(size_t i=0;i<fp.size();i++){ float d=fabsf(fp[i]-rfp[i]); if(d>md)md=d; }
      printf("  %-10s max|d|=%.3e\n","feat_proj",md); if(md>worst)worst=md; }

    // host pos_conv + residual add
    std::vector<float> pos;
    pos_conv_host(fp, T, (const float *)need(wctx,"hubert.pos_conv.w")->data, (const float *)need(wctx,"hubert.pos_conv.b")->data, pos);
    for(size_t i=0;i<fp.size();i++) fp[i] += pos[i];
    // enc_in = LN(feat_proj + pos)
    layernorm_host(fp, T, 768, (const float *)need(wctx,"hubert.enc_norm_w")->data, (const float *)need(wctx,"hubert.enc_norm_b")->data);
    snprintf(pb,512,"%s/hubert.enc_in.bin",gold);
    std::vector<float> rei = load_bin(pb, (size_t)768*T);
    { float md=0; for(size_t i=0;i<fp.size();i++){ float d=fabsf(fp[i]-rei[i]); if(d>md)md=d; }
      printf("  %-10s max|d|=%.3e\n","enc_in",md); if(md>worst)worst=md; }


    // host 12-layer post-norm transformer (fp32, T=49)
    for(int li = 0; li < NL; li++){
        char base[128]; snprintf(base, 128, "hubert.layer.%d", li);
        std::vector<float> q, k, v;
        snprintf(nm,128,"%s.q_w",base); linear_host(fp, T, D, (const float *)need(wctx, nm)->data, (const float *)need(wctx, (std::string(base)+".q_b").c_str())->data, D, q);
        snprintf(nm,128,"%s.k_w",base); linear_host(fp, T, D, (const float *)need(wctx, nm)->data, (const float *)need(wctx, (std::string(base)+".k_b").c_str())->data, D, k);
        snprintf(nm,128,"%s.v_w",base); linear_host(fp, T, D, (const float *)need(wctx, nm)->data, (const float *)need(wctx, (std::string(base)+".v_b").c_str())->data, D, v);
        // attention: q/k/v [T, NH*HD], heads split
        std::vector<float> attn((size_t)T * D, 0.0f);
        const float scale = 1.0f / sqrtf((float)HD);
        for(int h = 0; h < NH; h++){
            std::vector<float> sc((size_t)T * T, 0.0f);
            for(int tq = 0; tq < T; tq++){
                const float * qr = &q[(size_t)tq * D + h * HD];
                for(int tk = 0; tk < T; tk++){
                    const float * kr = &k[(size_t)tk * D + h * HD];
                    double acc = 0; for(int d = 0; d < HD; d++) acc += (double)qr[d] * kr[d];
                    sc[(size_t)tq * T + tk] = (float)(acc * scale);
                }
                // softmax over row
                float mx = sc[(size_t)tq * T];
                for(int tk = 1; tk < T; tk++) if(sc[(size_t)tq * T + tk] > mx) mx = sc[(size_t)tq * T + tk];
                double sm = 0; for(int tk = 0; tk < T; tk++){ float e = expf(sc[(size_t)tq * T + tk] - mx); sc[(size_t)tq * T + tk] = e; sm += e; }
                for(int tk = 0; tk < T; tk++) sc[(size_t)tq * T + tk] = (float)(sc[(size_t)tq * T + tk] / sm);
            }
            for(int tq = 0; tq < T; tq++){
                for(int d = 0; d < HD; d++){
                    double acc = 0;
                    for(int tk = 0; tk < T; tk++) acc += (double)sc[(size_t)tq * T + tk] * v[(size_t)tk * D + h * HD + d];
                    attn[(size_t)tq * D + h * HD + d] = (float)acc;
                }
            }
        }
        std::vector<float> ao;
        linear_host(attn, T, D, (const float *)need(wctx, (std::string(base)+".out_w").c_str())->data, (const float *)need(wctx, (std::string(base)+".out_b").c_str())->data, D, ao);
        for(size_t i = 0; i < fp.size(); i++) fp[i] += ao[i];
        layernorm_host(fp, T, D, (const float *)need(wctx, (std::string(base)+".ln1_w").c_str())->data, (const float *)need(wctx, (std::string(base)+".ln1_b").c_str())->data);
        std::vector<float> h;
        linear_host(fp, T, D, (const float *)need(wctx, (std::string(base)+".ffn1_w").c_str())->data, (const float *)need(wctx, (std::string(base)+".ffn1_b").c_str())->data, 4*D, h);
        for(auto & x : h) x = gelu_erf(x);
        std::vector<float> h2;
        linear_host(h, T, 4*D, (const float *)need(wctx, (std::string(base)+".ffn2_w").c_str())->data, (const float *)need(wctx, (std::string(base)+".ffn2_b").c_str())->data, D, h2);
        for(size_t i = 0; i < fp.size(); i++) fp[i] += h2[i];
        layernorm_host(fp, T, D, (const float *)need(wctx, (std::string(base)+".ln2_w").c_str())->data, (const float *)need(wctx, (std::string(base)+".ln2_b").c_str())->data);
    }

    snprintf(pb,512,"%s/hubert.out.bin",gold);
    std::vector<float> ro = load_bin(pb, (size_t)D*T);
    { float md=0; double sa=0,sb=0,sab=0; for(int d=0;d<D;d++)for(int t=0;t<T;t++){ float a=fp[(size_t)t*D+d], b=ro[(size_t)t*D+d]; float e=fabsf(a-b); if(e>md)md=e; sa+=(double)a*a; sb+=(double)b*b; sab+=(double)a*b; }
      printf("  %-10s max|d|=%.3e cos=%.6f\n","hub_out",md,sab/(sqrt(sa)*sqrt(sb)+1e-30)); if(md>worst)worst=md; }

    gguf_free(gctx); ggml_free(wctx);
    printf(worst < 2e-2f ? "HUBERT PARITY PASS\n" : "HUBERT PARITY FAIL\n");
    return worst < 2e-2f ? 0 : 2;
}
