#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct gsv_encp_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;
};

class gsv_encp {
public:
    static gsv_encp * load(const std::string & gguf_path, const gsv_encp_cfg & cfg);
    ~gsv_encp();

    // y: [768, T] RVQ 输出 (torch 行主); text: [n_text] token id; ge: [512]
    // out_m/out_logs: [192, T] torch 行主 (flat[c*T + t], 与 golden 的 .m/.logs 同构)
    // out_y (可选): [192, T] torch 行主 — proj 之前的 encoder2 输出, 条件链里喂 bridge
    bool encode(const float * y, int T, const int32_t * text, int n_text,
                const float * ge, std::vector<float> & out_m, std::vector<float> & out_logs,
                std::vector<float> * out_y = nullptr);
    int hidden() const { return 192; }

private:
    gsv_encp();
    struct impl;
    impl * p;
};
