// GSV AR (Text2SemanticDecoder) 引擎
// 功能: GGUF 加载 + 文本/bert/参考语义 token 前端 + batched 首步/增量解码 + 采样生成
// 语义严格对齐 GPT-SoVITS AR/models/t2s_model.py::infer_panel_batch_infer
// (左 padding + padding/causal mask, EOS 与 early-stop 规则, PE 位置 = prompt_len + step)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gsv_sampler.h"

struct gsv_ar_request {
    std::vector<int32_t> phones;   // 文本音素 id (长度可变, 引擎内部左 padding)
    std::vector<float>   bert;     // [bert_dim, phones_len] 行主序 (bert_dim=1024): bert[k*T + t]
    std::vector<int32_t> prompt;   // 参考语义 token (批内所有请求必须一致, 与 TTS.py 一致)
};

struct gsv_ar_cfg {
    int  n_threads  = 0;      // CPU 后端线程数 (0 = ggml 默认)
    int  max_batch  = 8;      // 预留 (KV cache 上限按实际请求数动态分配)
    bool verbose    = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 选第一个 GPU 设备
    bool disable_coopmat2 = true;  // Vulkan: coopmat2 在 F32 matmul 上掉精度 (实测 logits 0.0035->0.0201),
                                   // 且性能几乎无差 -> 默认禁用
};

struct gsv_ar_result {
    std::vector<std::vector<int32_t>> tokens;   // 每序列: prompt + 生成 (不含触发停止的最后一个 token)
    std::vector<int> lens;                      // 每序列生成 token 数 (与 torch 的 idx_list 同义)
};

class gsv_ar {
public:
    static gsv_ar * load(const std::string & gguf_path, const gsv_ar_cfg & cfg);
    ~gsv_ar();

    int hidden_dim()          const;
    int vocab_size()          const;
    int eos()                 const;
    int bert_dim()            const;
    int prompt_len_expected() const;   // 0 = 未初始化

    // 生成。sampler 用 gsv_sampler_cfg (默认即 TTS.py 的推断默认: top_k=15/top_p=1.0/temp=1.0/rep=1.35)
    // early_stop_num: -1 表示不启用 (与 torch 一致: 生成 token 数 > early_stop_num 即停止)
    // max_steps: 迭代上限 (torch 为 1500)
    gsv_ar_result generate(const std::vector<gsv_ar_request> & reqs,
                           const gsv_sampler_cfg & sampler,
                           uint64_t seed_base,
                           int early_stop_num = -1,
                           int max_steps = 1500);

    // 测试/诊断: 首步 last-token logits (b 行 × vocab)
    void first_logits(const std::vector<gsv_ar_request> & reqs, std::vector<float> & logits_out);

private:
    gsv_ar();
    struct impl;
    impl * p;
};
