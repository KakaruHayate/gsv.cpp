// G2PW (grapheme-to-phoneme, 多音字消歧) —— ONNX 推理, onnxruntime C API, DML/WebGPU/CPU EP
// 语义对照 repo/GPT_SoVITS/text/g2pw/{onnx_api.py,dataset.py,utils.py,g2pw.py}
//
// 模型 = 12 层 BERT (bert-base-chinese, hidden 768, heads 12) + 条件描述符头:
//   seq        = bert(input_ids)                                     [B, L, 768]
//   h          = seq[position_ids]                                   [N, 768]  取多音字所在 token
//   pred_pos   = argmax(pos_classifier(h))                           [N]       11 类词性
//   affect     = descriptor_bias[0] + char_descriptor[char_ids]
//                + second_order_descriptor[char_ids*11 + pred_pos]   [N, 1305]
//   w          = phoneme_mask * sigmoid(affect)                      [N, 1305]
//   logits     = classifier(h)                                       [N, 1305]
//   probs      = clip( softmax(logits) * w / Σ(softmax*w), 1-eps, eps )
//   pred       = argmax(probs) -> labels[pred] (注音符号 ㄅㄆㄇ...)
//
// **本模块只做到 probs/argmax 为止** (即 ONNX 的输出), 注音->拼音/音素的映射与文本规整
// (数字/英文/日文处理) 属前端另一层, 由调用方按 repo 的 text/ 处理。
// 中文 G2P 的另一半 (pypinyin 的 simple_seg / char_bopomofo_dict 兜底) 也留给调用方。
//
// 关键口径 (dataset.py::prepare_onnx_input):
//   - text **先 .lower()**; [CLS] 前缀 + [SEP] 后缀; token_type 全 0; attention_mask 全 1
//   - 变长序列在 batch 内 **右侧 pad 0**; phoneme_mask 是 [N,1305] 的 0/1 (非 padding, 不 pad)
//   - position_id = text2token[query_id] + 1  ([CLS] 占了位置 0)
//   - max_len = 512; 超长文本 repo 会按 query 附近截断 (_truncate)
//   - eps = 1e-6 (probs 的 clip 上下界)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_g2pw_cfg {
    std::string device;        // "" = 自动 (GPU EP 优先); "cpu"; "gpu"/"dml"/"webgpu"
    std::string ort_dll;       // onnxruntime.dll 路径/目录 ("" = 默认搜索顺序)
    std::string model_dir;     // G2PWModel 目录 (含 g2pW.onnx + labels.tsv + char2id.tsv)
    std::string vocab_txt;     // BERT vocab.txt (与 roberta/bert 前端共用)
    int  n_threads = 0;
    int  max_len   = 512;
    bool verbose   = false;
};

class gsv_g2pw {
public:
    static gsv_g2pw * load(const gsv_g2pw_cfg & cfg);
    ~gsv_g2pw();

    int  n_labels() const;     // 1305
    const char * ep_name() const;

    // 单条查询: text (UTF-8 中文串) 中第 query_id 个字符是多音字
    //   -> 返回 labels 下标 (0..n_labels-1); confidence 非空时给出 probs 最大值
    // 内部: 自己完成 tokenize (wordize_and_map + BERT WP 切分) -> 组 batch=1 -> argmax
    bool predict_one(const std::string & text, int query_id, int & label_idx, float * confidence = nullptr);

    // 批量: 每条 (text, query_id) 独立成 batch 内一行 (右侧 pad), 一次 Run 完成
    //   texts 已按 repo 的 _prepare_data 裁到多音字附近上下文, query_id 是裁后下标
    bool predict_batch(const std::vector<std::string> & texts,
                       const std::vector<int> & query_ids,
                       std::vector<int> & label_idx,
                       std::vector<float> * confidence = nullptr);

    // 供 golden 对拍/调试: 直接喂已构造好的 batch (torch 布局)
    bool run_raw(const std::vector<std::vector<int64_t>> & input_ids,
                 const std::vector<std::vector<int64_t>> & token_type_ids,
                 const std::vector<std::vector<int64_t>> & attention_masks,
                 const std::vector<std::vector<float>> & phoneme_masks,
                 const std::vector<int64_t> & char_ids,
                 const std::vector<int64_t> & position_ids,
                 std::vector<int> & label_idx,
                 std::vector<std::vector<float>> * probs_out = nullptr);

    // labels[idx] (注音符号, UTF-8); idx 越界返回 nullptr
    const char * label(int idx) const;

private:
    gsv_g2pw();
    // impl 定义在 .cpp 的匿名 namespace 里 (实现细节不外泄);
    // 这里只存 void* 由 .cpp 里的函数强转, 避免 private 嵌套类型被自由函数引用。
    void * p;
};
