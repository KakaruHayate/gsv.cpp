// 引擎门面: 把 demo_e2e.cpp 里写死的链提炼成可复用接口, 供 gsv_ggml_cli (server) 调用。
// 所有推理都复用 src/gsv_*.cpp 里已对拍通过的模块, 这里只做编排与布局转换。
//
// 阶段划分与 repo 的 TTS.py 一致:
//   set_ref_audio  -> prep_reference (参考音频三链, 可缓存)
//   get_phones_and_bert -> Python 前端 (phones / word2ph / bert_ids)
//   T2S (AR)       -> ar_generate   (语义 token)
//   decode_encp + synthesize_v5_mel + vocoder -> synthesize (wav)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_engine_cfg {
    std::string device;        // ggml 后端: "" = CPU, "vulkan"/"gpu" = 第一个 GPU
    std::string voc_device;    // ORT EP: "" = 自动 (WebGPU > DML > CPU), 或 cpu/dml/webgpu
    std::string model_dir;     // 模型目录 (models/)
    int  n_threads = 0;        // CPU 后端线程数 (0 = 默认)
    bool verbose    = false;
};

class gsv_engine {
public:
    static gsv_engine * load(const gsv_engine_cfg & cfg);
    ~gsv_engine();

    // ---- 参考音频三链 (repo TTS.py::set_ref_audio) ----
    //   (a) 16k + 0.3s 零尾 -> HuBERT -> ssl_proj -> RVQ -> prompt_codes [T]
    //   (b) 32k + maxx 归一化 -> spectrogram_torch(2048/640) -> [:, :704] -> ref_enc -> ge [512]
    //   (c) 32k -> mel_fn_v4 -> norm_spec -> mel2 [100, Tm] (torch 布局 idx = m*Tm + t)
    bool prep_reference(const std::string & wav_path,
                        std::vector<int32_t> & prompt_codes,
                        std::vector<float> & ge,
                        std::vector<float> & mel2, int & mel2_T);

    // ---- BERT 特征 (repo TextPreprocessor.get_bert_feature) ----
    //   ids -> gsv_bert::encode_feat (hidden_states[-3], 去 CLS/SEP) -> [1024, L-2]
    //   -> 按 word2ph 逐字重复 -> [1024, P] 行主 (P = Σword2ph = phones.size())
    bool bert_feature(const std::vector<int32_t> & ids,
                      const std::vector<int32_t> & word2ph,
                      std::vector<float> & feat1024);

    // ---- AR (repo TTS.py 的 T2S 段) ----
    //   repo 的 AR 输入 x = **参考文本音素 ++ 目标文本音素** (TTS.py:1017 `all_phones =
    //   prompt_data["phones"] + item["phones"]`), bert 同理沿 dim=1 拼接; 语义 prompt 用参考码。
    //   只喂目标那一半会让条件缺一半 -> 模型早早判 EOS。
    //   prompt_bert 为空表示两半都用全零 (非 zh 口径)。
    bool ar_generate(const std::vector<int32_t> & phones,
                     const std::vector<float> & bert,           // [1024, Pt] 行主
                     const std::vector<int32_t> & prompt_phones,
                     const std::vector<float> & prompt_bert,    // [1024, Pp] 行主 (可空)
                     const std::vector<int32_t> & prompt_codes,
                     int max_steps, int early_stop, uint64_t seed,
                     std::vector<int32_t> & sem);

    // ---- 主链 (repo decode_encp + synthesize_v5_mel + vocoder) ----
    //   prompt_codes + **prompt_phones** -> 参考侧 fea_ref   (repo: decode_encp(prompt_sem,
    //       prompt_phones, ...) —— 参考侧用的是**参考音频对应文本**的音素, 不是目标文本的!)
    //   sem + phones                       -> 目标侧 fea_todo
    //   -> DiT synthesize_v5_mel (分块 rolling) -> denorm_spec -> vocoder -> wav [32k]
    bool synthesize(const std::vector<int32_t> & sem,
                    const std::vector<int32_t> & phones,
                    const std::vector<int32_t> & prompt_codes,
                    const std::vector<int32_t> & prompt_phones,
                    const std::vector<float> & ge,
                    const std::vector<float> & mel2, int mel2_T,
                    int steps, float cfg, uint64_t seed,
                    std::vector<float> & wav);

    int  mel_dim()  const;      // 100
    int  text_dim() const;      // 512
    int  ar_vocab() const;

private:
    gsv_engine();
    struct impl;
    impl * p;
};
