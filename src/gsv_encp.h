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
    // out_m/out_logs: [192, T'] torch 行主 (flat[c*T' + t])；speed!=1 时 T'=int(T/speed)+1
    // （与 repo Encoder.forward 末尾的 F.interpolate(mode="linear") 等价：proj 是逐帧线性的，
    //   所以对 m/logs 做时间插值 == 先插值 y 再 proj）
    // out_y (可选): [192, T'] torch 行主 — proj 之前（已按 speed 插值）的 encoder2 输出, 喂 bridge
    bool encode(const float * y, int T, const int32_t * text, int n_text,
                const float * ge, std::vector<float> & out_m, std::vector<float> & out_logs,
                std::vector<float> * out_y = nullptr, float speed = 1.0f);
    int hidden() const { return 192; }

private:
    gsv_encp();
    struct impl;
    impl * p;
};
