// wns1 (VITS WN Encoder: 8-layer WaveNet k=5 d=1 pad=2, gin=512) ggml impl.
// fea[T,512](t-major) + ge[512] + len -> out[T,512]; weights models/gsv-cond-f32.gguf.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
struct gsv_wns1_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;   // "" = CPU; "vulkan"/"gpu" = first GPU device
};
class gsv_wns1 {
public:
    static gsv_wns1 * load(const std::string & gguf_path, const gsv_wns1_cfg & cfg);
    ~gsv_wns1();
    // fea: [T,512] t-major (idx = c*T+t); ge: [512]; out: [T,512] t-major
    bool encode(const float * fea, const float * ge, int T, int len, std::vector<float> & out);
    int hidden() const { return 512; }
private:
    gsv_wns1();
    struct impl;
    impl * p;
    friend void build_graph(impl & s);
};

