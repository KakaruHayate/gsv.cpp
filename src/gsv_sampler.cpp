#include "gsv_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

void gsv_logits_to_probs(const gsv_sampler_cfg & cfg,
                         float * logits, int V,
                         const int32_t * prev, int n_prev,
                         float * probs) {
    // 1) repetition penalty
    if (prev != nullptr && n_prev > 0 && cfg.repetition_penalty != 1.0f) {
        for (int i = 0; i < n_prev; i++) {
            const int t = prev[i];
            if (t < 0 || t >= V) continue;
            const float s = logits[t];
            logits[t] = (s < 0.0f) ? s * cfg.repetition_penalty : s / cfg.repetition_penalty;
        }
    }

    // 2) top_p (nucleus)
    if (cfg.top_p < 1.0f) {
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

        // softmax(sorted) —— 排序后最大值在首位
        const float mx = logits[idx[0]];
        float sum = 0.0f;
        for (int k = 0; k < V; k++) {
            sum += std::exp(logits[idx[k]] - mx);
        }
        const float inv_sum = 1.0f / sum;
        float cum = 0.0f;
        for (int k = 0; k < V; k++) {
            cum += std::exp(logits[idx[k]] - mx) * inv_sum;
            if (k > 0 && cum > cfg.top_p) {   // rank 0 永远保留（与 torch 的 [:,0]=False 一致）
                logits[idx[k]] = -INFINITY;
            }
        }
    }

    // 3) temperature
    const float t = std::max(cfg.temperature, 1e-5f);
    for (int i = 0; i < V; i++) logits[i] /= t;

    // 4) top_k (pivot = 第 k 大, 用 < 比较 -> 等值保留)
    if (cfg.top_k > 0 && cfg.top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (cfg.top_k - 1), tmp.end(), std::greater<float>());
        const float pivot = tmp[cfg.top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < pivot) logits[i] = -INFINITY;
        }
    }

    // 5) softmax (float32 累加, 与 torch CPU 核一致)
    float mx = -INFINITY;
    for (int i = 0; i < V; i++) mx = std::max(mx, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = (logits[i] == -INFINITY) ? 0.0f : std::exp(logits[i] - mx);
        sum += probs[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < V; i++) probs[i] *= inv;
}

int gsv_argmax_probs_over_q(const float * probs, const float * q, int V) {
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < V; i++) {
        const float v = (probs[i] > 0.0f) ? probs[i] / q[i] : 0.0f;   // probs=0 的位置不参与
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}

int gsv_sample(const float * probs, int V, gsv_rng & rng, float * q_scratch) {
    std::vector<float> local;
    float * q = q_scratch;
    if (q == nullptr) {
        local.resize(V);
        q = local.data();
    }
    for (int i = 0; i < V; i++) q[i] = rng.exponential();
    return gsv_argmax_probs_over_q(probs, q, V);
}
