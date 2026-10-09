// DiT (F5 式 CFM estimator, v5: use_step_embedding=False) ggml 实现.
// 语义对照 repo/GPT_SoVITS/f5_tts/model/backbones/dit.py + modules.py (DiTBlock/AdaLayerNormZero/
// ConvNeXtV2Block/GRN/ConvPositionEmbedding/SinusPositionEmbedding) 与
// x_transformers 的 RotaryEmbedding/apply_rotary_pos_emb; golden 见 tests/golden/dit.*
//
// 关键语义 (2026-10-09 探针逐条确认, 与 torch 对拍时不可改动):
//   1. RoPE 只作用于 q/k 的前 64 维 (= attention 的第 0 号头): x_transformers 的
//      apply_rotary_pos_emb 用 rot_dim = freqs.shape[-1] = dim_head = 64 截断, 其余 15 头不旋转.
//      角度表 = n * inv_freq[i], 同对相邻两维共享; rotate_half 是交错 (GPT-NeoX) 配对.
//   2. inv_freq 取 checkpoint 里的 fp16 存储值 (与 fp32 公式差 ~1.6e-4, 用公式重算会破坏大 n 的相位).
//   3. time_embed 正弦: cat(sin, cos)(1000 * t * exp(-i*log(10000)/127)), i<128; MLP 256->1024->1024.
//   4. FFN 的 GELU 用 tanh 近似 (approximate="tanh"), ConvNeXtV2 的 GELU 是精确 erf 版.
//   5. static cache = prepare_static_cache: condition/negative_condition [1024,T] 一次算好,
//      每步只算 F.linear(x, proj[:, :100]) + static + conv_pos + 22 块.
//
// 布局: 图内主干 Domain B (ggml ne{C,T} = torch [1,T,C] 字节), conv 段用 Domain A (ne{T,C} = torch [1,C,T] 字节);
// 对外 API 一律 torch 布局 (x/prompt/mu/vel 都是 [1,C,T] 字节 = ggml ne{T,C}).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_dit_cfg {
    int  n_threads = 0;
    bool verbose   = false;
    std::string device;       // "" = CPU; "vulkan"/"gpu" = 第一个 GPU 设备

    // cache-dit (DBCache 三段式, vipshop/cache-dit 语义; dbcache_fn = 0 关闭)
    int   dbcache_fn              = 0;      // Fn_compute_blocks: 前 n 块每步必算 (判据 + 复用基)
    float dbcache_threshold       = 0.08f;  // residual_diff_threshold (Fn 残差相对 L1 门)
    int   dbcache_warmup          = 8;      // max_warmup_steps: 前 n 步强制全算
    int   dbcache_warmup_interval = 1;      // warmup_interval
    int   dbcache_max_cached      = -1;     // max_cached_steps (per 分支; -1 不限)
    int   dbcache_max_cont        = -1;     // max_continuous_cached_steps (-1 不限)
    float dbcache_max_accum       = 0.0f;   // max_accumulated_residual_diff_threshold (0 关)
    int   dbcache_downsample      = 4;      // Fn 残差缓冲时间下采样 (上游 downsample_factor; 1=关)
};

class gsv_dit {
public:
    static gsv_dit * load(const std::string & gguf_path, const gsv_dit_cfg & cfg);
    ~gsv_dit();

    int mel_dim() const;      // 100
    int dim()     const;      // 1024
    int depth()   const;      // 22

    // prepare_static_cache (torch 布局):
    //   prompt_mel: [mel_dim, T]   (idx = c*T + t) = DiT 的 cond0 = prompt_x
    //   mu        : [text_dim,T]   = enc_p/wns1 的 fea (condition 的 text 路)
    //   x_lens    : 有效帧数 (<= T; v5 生产路径恒为 T)
    bool prepare(const float * prompt_mel, const float * mu, int T, int x_lens);
    int  cur_T() const;          // 当前图缓存对应的序列长 (prepare 后)
    int  cur_prompt_T() const;   // 最近一次 sample/synthesize 的 prompt 帧数

    // 单步速度 (estimator 前向): x_mel [mel_dim, T] torch 布局 -> vel [mel_dim, T]
    //   negative=true 时用 negative_condition (cfg 负分支)
    bool velocity(const float * x_mel, float t, bool negative, float * vel);

    // CFM.inference 等价 (torch 布局): mu [text_dim, T], prompt [mel_dim, P] -> mel [mel_dim, T-P]
    //   x0 非空时用其作为初始噪声 (torch CFM 的 randn*temp, 未置零; 对拍用), 否则用 seed 生成
    bool sample(const float * mu, const float * prompt, int P, int T, int steps, float cfg,
                uint64_t seed, const float * x0, std::vector<float> & mel_out);

    // synthesize_v5_mel 等价 (分块 + rolling prompt).
    //   fea_ref [512,R] / fea_todo [512,N] / mel_ref [100,R] (都 torch 布局)
    //   block_override > 0 时覆盖 chunk_frames (测试用小值), 否则用生产值 640
    //   x0_chunks 非空时: 第 i 块用 (*x0_chunks)[i] 作初始噪声 (对拍用, 长度 = 块数)
    bool synthesize(const float * fea_ref, int R, const float * fea_todo, int N,
                    const float * mel_ref, int steps, float cfg, uint64_t seed,
                    int block_override, const std::vector<const float *> * x0_chunks,
                    std::vector<float> & mel_out);

    // SinusPositionEmbedding(256) 的 host 端实现 (fp32, 与 torch 逐位一致)
    static void time_sin(float t, float * out256);

    // ---- cache-dit 状态接口 (sample()/synthesize() 内部已自动管理; 手工消融循环用) ----
    struct db_stats {
        int   steps;              // 已设置的最大 step (含 warmup)
        int   hits[2];            // [pos, neg]
        int   misses[2];
        float last_fd[2];
        float acc[2];             // 累计残差 diff
    };
    void db_set_step(int j);      // 手工循环里在 velocity() 前调用 (warmup/gates 依据)
    void db_get_stats(db_stats & st) const;
    // rope 角度表 (每次 prepare 后有效): cos/sin 各 [32*T], idx = i + 32*t
    void debug_tables(std::vector<float> & cos_tab, std::vector<float> & sin_tab) const;
    // 读回中间张量 (GSV_DIT_DEBUG=1 + 最近一次 velocity 之后可用; 名字见 gsv_dit.cpp 的 mark())
    bool debug_read(const char * name, std::vector<float> & out);

private:
    gsv_dit();
    struct impl;
    impl * p;
    int cur_T_ = 0;
    int cur_prompt_T_ = 0;
};
