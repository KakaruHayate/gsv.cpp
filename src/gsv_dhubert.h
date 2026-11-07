// 蒸馏 HuBERT (student_v0.pt, 2.97M 参数) 的 ggml 实现 —— CN-HuBERT 在特征层面的 drop-in
//
// 结构对齐 GPT_SoVITS/feature_extractor/student_hubert.py::StudentSSL:
//   1) front: 7 层 [conv1d(k, stride, 无 padding) -> GroupNorm(8) -> SiLU], 总步长 320 -> 50Hz
//   2) tcn  : 6 个残差块  h = SiLU(GN(conv1d(x, k=3, dilation=d, padding=d)));
//                         h = GN(conv1d(h, k=3, dilation=d, padding=d)); out = x + h
//   3) attn : 2 层 pre-norm TransformerEncoderLayer (D=192, NH=4, FFN=768, 无位置编码)
//   4) head : conv1d 1x1 192 -> 768 (输出教师的隐藏维)
//
// 布局: 卷积段用 ggml_conv_1d 原生 [T, C, B]; 进 attention 前转 [C, T] (mul_mat 友好);
//       输出 [768, T] 行主序 (idx = d + t*768) == torch head 输出的转置。
// 权重来自 tools/export_gguf/convert_distilled_hubert.py (arch=gsv-dhubert);
// 卷积权重在 load() 时一次性从 [K][IC][OC] 转置成 ggml im2col 需要的 k 最内布局。
//
// 与教师模型的关系: 输入同样是 16 kHz 原始波形 (不做归一化/不做 z-score),
// 输出同为 50 Hz 的 768 维特征, 因此可以直接替换 CNHubert 的 ssl 特征。
// 数值对拍: tools/dump_ref_dhubert.py (torch fp32) 与本类的 encode() 输出逐元素比较,
//           tools/dhubert_compare.py 打印 max/mean abs diff 与相关系数。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_dhubert_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;        // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备
    // FFN 激活。torch 的 nn.TransformerEncoderLayer 默认 activation=relu
    // (student_hubert.py 没有显式传参), 所以默认 relu; "gelu" 只用于对照实验。
    // 环境变量 GSV_DHUBERT_ACT=relu|gelu 可覆盖。
    std::string ffn_act = "relu";
};

class gsv_dhubert {
public:
    static gsv_dhubert * load(const std::string & gguf_path, const gsv_dhubert_cfg & cfg);
    ~gsv_dhubert();

    // audio: 16 kHz 单声道波形 (变长, >= 400 + 320 个采样点; 帧数 T = (n - 400) / 320 + 1)。
    //        喂的是**原始波形** ([-1,1] 量纲), 与 torch 参考一致; 不要额外归一化。
    // out   : [768, T] 行主序 (idx = d + t*768)。帧数变化时自动重建图。
    bool encode(const float * audio, int n_samples, std::vector<float> & out);

    int  hidden()   const;     // 768 (out_dim)
    int  n_frames() const;     // 最近一次 encode 的帧数
    int  frame_rate() const { return 50; }

private:
    gsv_dhubert();
    struct impl;
    impl * p;
};
