// GSV AR 采样链 — 严格复刻 GPT-SoVITS AR/models/utils.py 的 logits_to_probs / sample
//
// torch 原实现顺序（不可调换）:
//   1. repetition penalty: score=gather(logits, prev); score = where(score<0, score*rep, score/rep);
//      scatter 回原位置（重复 token 幂等）
//   2. top_p: 降序排序 -> softmax -> cumsum -> 去掉 cum > top_p 的（rank 0 保留）-> masked_fill(-inf)
//   3. temperature: logits /= max(temperature, 1e-5)
//   4. top_k: pivot = 第 k 大值; logits < pivot -> -inf（相等保留，故可能多于 k 个）
//   5. softmax
//   采样: idx = argmax(probs / q), q ~ Exp(1)（torch multinomial_sample_one_no_sync 的等价形式）
#pragma once

#include <cstdint>
#include <vector>

struct gsv_sampler_cfg {
    int   top_k = 15;                 // <=0 表示不启用
    float top_p = 1.0f;               // >=1 表示不启用
    float temperature = 1.0f;
    float repetition_penalty = 1.35f;
};

// 确定性部分: 原地消费 logits（与 torch 一致，会修改传入数组），产出 probs
// logits: [vocab] 长度 V; prev: 已生成 token; probs: 输出 [V]
void gsv_logits_to_probs(const gsv_sampler_cfg & cfg,
                         float * logits, int V,
                         const int32_t * prev, int n_prev,
                         float * probs);

// 采样规则: argmax(probs / q)；q 为空则用 rng 生成
struct gsv_rng {
    uint64_t s;
    explicit gsv_rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
    uint64_t next() {                       // xorshift64*
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    float uniform01() { return (float)((next() >> 40) * (1.0 / 16777216.0)); }  // [0,1)
    float exponential() { return -logf(1.0f - uniform01()); }                   // 与 torch 的 -log(1-U) 同式
};

int gsv_argmax_probs_over_q(const float * probs, const float * q, int V);
int gsv_sample(const float * probs, int V, gsv_rng & rng, float * q_scratch /*[V] 可为 nullptr*/);
