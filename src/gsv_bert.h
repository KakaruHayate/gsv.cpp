// chinese-roberta-wwm-ext-large (BERT-large) 的 ggml 实现
// 只跑 0..n_layer-3 层 (hidden_states[-3] 即第 21 层输出), 输出 [hidden, T] 特征
// 语义对齐 HF BertLayer (post-LN): x = LN(x + attn); x = LN(x + FFN), FFN 激活 = gelu(erf)
// 张量布局: 权重 torch [out,in] 直写 = ggml [ne0=in, ne1=out]
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_bert_cfg {
    int  n_threads = 0;       // CPU 后端线程数
    bool verbose   = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备
    int  feat_cache = 0;      // encode_feat_cached 的 LRU 条目数 (0 = 关闭; 服务化场景省 TTFT)
};

class gsv_bert {
public:
    static gsv_bert * load(const std::string & gguf_path, const gsv_bert_cfg & cfg);
    ~gsv_bert();

    int hidden()      const;
    int layers_used() const;
    int max_pos()     const;

    // ids: token id 序列 (含 [CLS]/[SEP], 无 padding, T <= max_pos)
    // features: [hidden, T] 行主序 —— 即 hidden_states[-3]（未做去 CLS/SEP 与 word2ph 重复）
    void encode(const int32_t * ids, int T, std::vector<float> & features);

    // 管线用特征: 去 [CLS]/[SEP] -> [hidden, T-2] 行主序（送 AR 的 bert 输入）
    void encode_feat(const int32_t * ids, int T, std::vector<float> & feat);

    // 带 LRU 特征缓存 (需 cfg.feat_cache > 0; 否则等价 encode_feat)。
    // 键 = (T, 完整 ids): 服务化场景里同一文本重复合成时省掉整段 BERT。
    void encode_feat_cached(const int32_t * ids, int T, std::vector<float> & feat);
    void feat_cache_stats(int & hits, int & misses) const;

    // 多文本批量 (摊薄小 GEMM): B 条序列, 各自长度 lens[b] (含 [CLS]/[SEP], <= max_pos),
    // ids_flat 按序列顺序拼接、每条占 Tmax 个槽 (不足处随意填, 由块对角 mask 屏蔽)。
    // 输出 feats[b] = [hidden, lens[b]-2] 行主序 (与逐条 encode_feat 同语义; 非逐位一致,
    // 见测试容差)。注意 S = Tmax*B 较大时 mask 为 S^2 (F16): B=4/Tmax=64 ~ 32KB。
    bool encode_feat_batch(int B, const int32_t * ids_flat, const int * lens, int Tmax,
                           std::vector<std::vector<float>> & feats);

    // 逐层导出 (调试用): per_layer[0] = embedding+LN, per_layer[k] = 第 k-1 层输出, 共 layers_used+1 项
    void encode_layers(const int32_t * ids, int T, std::vector<float> & features,
                       std::vector<std::vector<float>> & per_layer);

private:
    gsv_bert();
    struct impl;
    impl * p;
};
