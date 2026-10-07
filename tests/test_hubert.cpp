// HuBERT (chinese-hubert-base) ggml parity test vs torch golden.
// Order per modeling_hubert.py: pos_conv -> +x -> LayerNorm (NOT LN then +pos).
// GGUF per convert_hubert.py: linear [out,in] row-major == ggml [ne0=in,ne1=out];
//   conv1d stored [K,IC,OC]; pos_conv folded weight-norm, stored [128,48,768] (16 groups).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <crtdbg.h>

static std::vector<float> load_bin(const std::string & p, size_t n){
    FILE * f = fopen(p.c_str(), "rb");
    if(!f){ fprintf(stderr, "open %s fail\n", p.c_str()); exit(3); }
    std::vector<float> v(n);
    if(fread(v.data(), 4, n, f) != n){ fprintf(stderr, "read %s fail\n", p.c_str()); exit(3); }

    fclose(f); return v;
}
// LayerNorm over ne0 (feature dim) then affine with w,b [D]
static struct ggml_tensor * LN(struct ggml_context * c, struct ggml_tensor * x,
        struct ggml_tensor * w, struct ggml_tensor * b){
    struct ggml_tensor * w2 = ggml_reshape_2d(c, w, w->ne[0], 1);
    struct ggml_tensor * b2 = ggml_reshape_2d(c, b, b->ne[0], 1);
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w2), b2);
}
static struct ggml_tensor * lin(struct ggml_context * c, struct ggml_tensor * x,
        struct ggml_tensor * w, struct ggml_tensor * b){
    x = ggml_mul_mat(c, w, x); if(b) x = ggml_add(c, x, ggml_reshape_2d(c, b, b->ne[0], 1)); return x;
}

int main(int argc, char ** argv){
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
    const char * model = argc>1?argv[1]:"models/gsv-hubert-f32.gguf";
    const char * gold  = argc>2?argv[2]:"tests/golden";
    const int TRAW=16000, T=49, D=768, NH=12, HD=64, NL=12;
    const int CK[7]={10,3,3,3,3,2,2}, CS[7]={5,2,2,2,2,2,2};

    struct ggml_context * wctx = NULL;
    struct gguf_init_params gp = { false, &wctx };
    struct gguf_context * gctx = gguf_init_from_file(model, gp);
    if(!gctx){ fprintf(stderr,"gguf open fail %s\n",model); return 3; }

    struct ggml_init_params gip = { (size_t)64*1024*1024, NULL, true };
    struct ggml_context * c = ggml_init(gip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(c, 16384, false);

    char pb[512]; snprintf(pb,512,"%s/hubert.input.bin",gold);
    std::vector<float> raw = load_bin(pb, TRAW);
    double m=0; for(int i=0;i<TRAW;i++) m+=raw[i]; m/=TRAW;
    double sq=0; for(int i=0;i<TRAW;i++){double d=raw[i]-m; sq+=d*d;}
    float sd=(float)sqrt(sq/TRAW);
    std::vector<float> xin(TRAW);
    for(int i=0;i<TRAW;i++) xin[i]=(float)((raw[i]-m)/(sd+1e-7));
    fprintf(stderr,"[mk] audio_in\n");
    struct ggml_tensor * attn_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, T);
    ggml_set_name(attn_mask,"attn_mask"); ggml_set_input(attn_mask);
    struct ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, TRAW, 1);
    ggml_set_name(x,"audio_in"); ggml_set_input(x);

    int nconv = 7; { const char * e2 = getenv("GSV_HUBERT_NCONV"); if(e2) nconv = atoi(e2); }
    int skipgn = 0; { const char * e3 = getenv("GSV_HUBERT_SKIPGN"); if(e3) skipgn = atoi(e3); }
    // CNN frontend: data [len, C] with channel on ne1
    for(int i=0;i<nconv;i++){
        char nm[128]; snprintf(nm,128,"hubert.feat_conv.%d.w",i);
        struct ggml_tensor * wc = ggml_new_tensor_3d(c, GGML_TYPE_F32, CK[i], i==0?1:512, 512);
        ggml_set_name(wc, nm); ggml_set_input(wc);
        x = ggml_conv_1d(c, wc, x, CS[i], 0, 1);
        if(i==0 && !skipgn){
            // GroupNorm(512 groups=512): reshape [len,512] -> [len,1,512], channel on ne2
            x = ggml_reshape_3d(c, x, x->ne[0], 1, 512);
            x = ggml_group_norm(c, x, 512, 1e-5f);
            struct ggml_tensor * gw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 1, 512);
            ggml_set_name(gw,"hubert.feat_conv.0.norm_w"); ggml_set_input(gw);
            struct ggml_tensor * gb = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 1, 512);
            ggml_set_name(gb,"hubert.feat_conv.0.norm_b"); ggml_set_input(gb);
            x = ggml_add(c,
 ggml_mul(c, x, gw), gb);
            x = ggml_reshape_2d(c, x, x->ne[0], 512);
        }
        x = ggml_gelu(c, x);
    }
    fprintf(stderr,"[mk] cnn done\n");
    struct ggml_tensor * cnn_out = ggml_cont(c, x);
    ggml_set_name(cnn_out,"cnn_out"); ggml_set_output(cnn_out);

    // feature projection: LN(512) -> Linear(512->768), work in [C, T]
    x = ggml_cont(c, ggml_permute(c, cnn_out, 1, 0, 2, 3));   // [512, T]
    struct ggml_tensor * fp_nw = ggml_new_tensor_1d(c, GGML_TYPE_F32, 512);
    ggml_set_name(fp_nw,"hubert.feat_proj.norm_w"); ggml_set_input(fp_nw);
    struct ggml_tensor * fp_nb = ggml_new_tensor_1d(c, GGML_TYPE_F32, 512);
    ggml_set_name(fp_nb,"hubert.feat_proj.norm_b"); ggml_set_input(fp_nb);
    x = LN(c, x, fp_nw, fp_nb);
    struct ggml_tensor * fp_w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 512, 768);
    ggml_set_name(fp_w,"hubert.feat_proj.w"); ggml_set_input(fp_w);
    struct ggml_tensor * fp_b = ggml_new_tensor_1d(c, GGML_TYPE_F32, 768);
    ggml_set_name(fp_b,"hubert.feat_proj.b"); ggml_set_input(fp_b);
    x = lin(c, x, fp_w, fp_b);                                  // [768, T]
    fprintf(stderr,"[mk] feat_proj done\n");
    struct ggml_tensor * feat_proj = ggml_cont(c, x);
    ggml_set_name(feat_proj,"feat_proj"); ggml_set_output(feat_proj);

    // pos_conv_embed: grouped conv k=128 groups=16, same-pad, crop last frame
    struct ggml_tensor * xc = ggml_cont(c, ggml_permute(c, feat_proj, 1, 0, 2, 3)); // [T,768]
    struct ggml_tensor * posw = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 48, 768);
    ggml_set_name(posw,"hubert.pos_conv.w"); ggml_set_input(posw);
    struct ggml_tensor * posb = ggml_new_tensor_1d(c, GGML_TYPE_F32, 768);
    ggml_set_name(posb,"hubert.pos_conv.b"); ggml_set_input(posb);
    struct ggml_tensor * posacc = NULL;
    for(int g=0; g<16; g++){
        struct ggml_tensor * xin_g = ggml_cont(c, ggml_view_2d(c, xc, xc->ne[0], 48, xc->nb[1], (size_t)g*48*xc->nb[1]));
        struct ggml_tensor * w_g = ggml_cont(c, ggml_view_3d(c, posw, 128, 48, 48, posw->nb[1], posw->nb[2], (size_t)g*48*posw->nb[2]));
        struct ggml_tensor * yg = ggml_conv_1d(c, w_g, xin_g, 1, 64, 1);
        yg = ggml_cont(c, ggml_view_2d(c, yg, T, 48, yg->nb[1], 0));
        posacc = (g==0) ? yg : ggml_concat(c, posacc, yg, 1);
    }
    posacc = ggml_add(c, posacc, ggml_reshape_2d(c, posb, 1, 768));
    struct ggml_tensor * pos_t = ggml_cont(c, ggml_permute(c, posacc, 1, 0, 2, 3)); // [768, T]
    x = ggml_add(c, feat_proj, pos_t);
    struct ggml_tensor * en_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, 768);
    ggml_set_name(en_w,"hubert.enc_norm_w"); ggml_set_input(en_w);
    struct ggml_tensor * en_b = ggml_new_tensor_1d(c, GGML_TYPE_F32, 768);
    ggml_set_name(en_b,"hubert.enc_norm_b"); ggml_set_input(en_b);
    x = LN(c, x, en_w, en_b);
    fprintf(stderr,"[mk] enc_in done\n");
    struct ggml_tensor * enc_in = ggml_cont(c, x);
    ggml_set_name(enc_in,"enc_in"); ggml_set_output(enc_in);

    // 12x post-norm transformer layers, x as [768, T]
    for(int li=0; li<NL; li++){
        fprintf(stderr,"[mk] layer %d\n", li); char base[128]; snprintf(base,128,"hubert.layer.%d",li);
        char nm[160];
        snprintf(nm,160,"%s.q_w",base); struct ggml_tensor * qw=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D); ggml_set_name(qw,nm); ggml_set_input(qw);
        snprintf(nm,160,"%s.q_b",base); struct ggml_tensor * qb=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(qb,nm); ggml_set_input(qb);
        snprintf(nm,160,"%s.k_w",base); struct ggml_tensor * kw=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D); ggml_set_name(kw,nm); ggml_set_input(kw);
        snprintf(nm,160,"%s.k_b",base); struct ggml_tensor * kb=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(kb,nm); ggml_set_input(kb);
        snprintf(nm,160,"%s.v_w",base); struct ggml_tensor * vw=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D); ggml_set_name(vw,nm); ggml_set_input(vw);
        snprintf(nm,160,"%s.v_b",base); struct ggml_tensor * vb=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(vb,nm); ggml_set_input(vb);
        struct ggml_tensor * q = lin(c, x, qw, qb);
        struct ggml_tensor * k = lin(c, x, kw, kb);
        struct ggml_tensor * v = lin(c, x, vw, vb);
        struct ggml_tensor * qh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, q, HD, NH, T), 0, 2, 1, 3));
        struct ggml_tensor * kh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, k, HD, NH, T), 0, 2, 1, 3));
        struct ggml_tensor * vh = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, HD, NH, T), 0, 2, 1, 3));
        struct ggml_tensor * ao = ggml_flash_attn_ext(c, qh, kh, vh, attn_mask, 1.0f/sqrtf((float)HD), 0.0f, 0.0f);
        ao = ggml_cont(c, ggml_permute(c, ao, 0, 2, 1, 3));
        ao = ggml_reshape_2d(c, ao, D, T);
        snprintf(nm,160,"%s.out_w",base); struct ggml_tensor * ow=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,D); ggml_set_name(ow,nm); ggml_set_input(ow);
        snprintf(nm,160,"%s.out_b",base); struct ggml_tensor * ob=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(ob,nm); ggml_set_input(ob);
        ao = lin(c, ao, ow, ob);
        x = ggml_add(c, x, ao);
        snprintf(nm,160,"%s.ln1_w",base); struct ggml_tensor * l1w=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(l1w,nm); ggml_set_input(l1w);
        snprintf(nm,160,"%s.ln1_b",base); struct ggml_tensor * l1b=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(l1b,nm); ggml_set_input(l1b);
        x = LN(c, x, l1w, l1b);
        snprintf(nm,160,"%s.ffn1_w",base); struct ggml_tensor * f1w=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,4*D); ggml_set_name(f1w,nm); ggml_set_input(f1w);
        snprintf(nm,160,"%s.ffn1_b",base); struct ggml_tensor * f1b=ggml_new_tensor_1d(c,GGML_TYPE_F32,4*D); ggml_set_name(f1b,nm); ggml_set_input(f1b);
        snprintf(nm,160,"%s.ffn2_w",base); struct ggml_tensor * f2w=ggml_new_tensor_2d(c,GGML_TYPE_F32,4*D,D); ggml_set_name(f2w,nm); ggml_set_input(f2w);
        snprintf(nm,160,"%s.ffn2_b",base); struct ggml_tensor * f2b=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(f2b,nm); ggml_set_input(f2b);
        struct ggml_tensor * h = ggml_gelu(c, lin(c, x, f1w, f1b));
        h = lin(c, h, f2w, f2b);
        x = ggml_add(c, x, h);
        snprintf(nm,160,"%s.ln2_w",base); struct ggml_tensor * l2w=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(l2w,nm); ggml_set_input(l2w);
        snprintf(nm,160,"%s.ln2_b",base); struct ggml_tensor * l2b=ggml_new_tensor_1d(c,GGML_TYPE_F32,D); ggml_set_name(l2b,nm); ggml_set_input(l2b);
        x = LN(c, x, l2w, l2b);
    }
    fprintf(stderr,"[mk] out done\n");
    struct ggml_tensor * out = ggml_cont(c, x);
    ggml_set_name(out,"hub_out"); ggml_set_output(out);

    // build forward from all outputs, alloc graph, fill inputs, compute
    int stage = 4;
    { const char * es = getenv("GSV_HUBERT_STAGE");
      if(es) stage = atoi(es); }
    struct ggml_tensor * fout = out;
    if(stage==1) fout = cnn_out;
    if(stage==2) fout = feat_proj;
    if(stage==3) fout = enc_in;
    ggml_build_forward_expand(gf, fout);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 1);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(galloc, gf);
    for(int i=0;i<ggml_graph_n_nodes(gf);i++){
        struct ggml_tensor * t = ggml_graph_node(gf, i);
        if(!(t->flags & GGML_TENSOR_FLAG_INPUT)) continue;
        if(strcmp(t->name,"audio_in")==0){ ggml_backend_tensor_set(t, xin.data(), 0, (size_t)TRAW*4); continue; }
        if(strcmp(t->name,"attn_mask")==0){ std::vector<ggml_fp16_t> zv((size_t)T*T, ggml_fp32_to_fp16(0.0f)); ggml_backend_tensor_set(t, zv.data(), 0, (size_t)T*T*2); continue; }
        struct ggml_tensor * w = ggml_get_tensor(wctx, t->name);
        if(!w){ fprintf(stderr,"no weight %s\n", t->name); return 3; }
        if(ggml_nbytes(w) != ggml_nbytes(t)){ fprintf(stderr,"size mismatch %s\n", t->name); return 3; }
        ggml_backend_tensor_set(t, w->data, 0, ggml_nbytes(w));
    }
    fprintf(stderr,"[mk] compute start\n"); ggml_backend_graph_compute(be, gf); fprintf(stderr,"[mk] compute end\n");

    // stage-wise parity vs torch golden (transpose-compare: ours [C,T], golden [1,T,C] or [1,C,T])
    float worst = 0;
    snprintf(pb,512,"%s/hubert.cnn_out.bin",gold); std::vector<float> rc = load_bin(pb, (size_t)512*T);
    std::vector<float> gc((size_t)512*T); ggml_backend_tensor_get(cnn_out, gc.data(), 0, (size_t)512*T*4);
    { float md=0; for(int ch=0;ch<512;ch++)for(int t=0;t<T;t++){ float d=fabsf(gc[t*512+ch]-rc[ch*T+t]); if(d>md)md=d; }
      printf("  %-10s max|d|=%.3e\n","cnn_out",md); if(md>worst)worst=md; }
    snprintf(pb,512,"%s/hubert.feat_proj.bin",gold); std::vector<float> rfp = load_bin(pb, (size_t)768*T);
    std::vector<float> gfp((size_t)768*T); ggml_backend_tensor_get(feat_proj, gfp.data(), 0, (size_t)768*T*4);
    { float md=0; for(int d=0;d<768;d++)for(int t=0;t<T;t++){ float e=fabsf(gfp[d*T+t]-rfp[t*768+d]); if(e>md)md=e; }
      printf("  %-10s max|d|=%.3e\n","feat_proj",md); if(md>worst)worst=md; }
    snprintf(pb,512,"%s/hubert.enc_in.bin",gold); std::vector<float> rei = load_bin(pb, (size_t)768*T);
    std::vector<float> gei((size_t)768*T); ggml_backend_tensor_get(enc_in, gei.data(), 0, (size_t)768*T*4);
    { float md=0; for(int d=0;d<768;d++)for(int t=0;t<T;t++){ float e=fabsf(gei[d*T+t]-rei[t*768+d]); if(e>md)md=e; }
      printf("  %-10s max|d|=%.3e\n","enc_in",md); if(md>worst)worst=md; }
    snprintf(pb,512,"%s/hubert.out.bin",gold); std::vector<float> ro = load_bin(pb, (size_t)768*T);
    std::vector<float> go((size_t)768*T); ggml_backend_tensor_get(out, go.data(), 0, (size_t)768*T*4);
    { float md=0; double sa=0,sb=0,sab=0; for(int d=0;d<768;d++)for(int t=0;t<T;t++){ float a=go[d*T+t], b=ro[t*768+d]; float e=fabsf(a-b); if(e>md)md=e; sa+=(double)a*a; sb+=(double)b*b; sab+=(double)a*b; }
      double cs=sab/(sqrt(sa)*sqrt(sb)+1e-30); printf("  %-10s max|d|=%.3e cos=%.6f\n","hub_out",md,cs); if(md>worst)worst=md; }

    ggml_gallocr_free(galloc); ggml_backend_free(be); gguf_free(gctx); ggml_free(wctx); ggml_free(c);
    printf(worst < 2e-2f ? "HUBERT PARITY PASS\n" : "HUBERT PARITY FAIL\n");
    return worst < 2e-2f ? 0 : 2;
}
