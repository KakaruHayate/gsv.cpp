// 共享 GGUF 的按前缀加载。
//
// 条件段把 enc_p / ref_enc / wns1 / bridge / rvq 装在**同一个** GGUF 里（298 张量），
// 而 gsv_refenc / gsv_cond / gsv_wns1 / gsv_encp 是四个独立对象、各自加载一次。
// 如果每处都 `ggml_backend_alloc_ctx_tensors(wctx)`，就会按整个文件大小重复分配显存
// （实测 3 份 cond-f32 + 1 份 cond-encp = 652.8 MiB，真正需要的只有 177.3 MiB）。
//
// 这里按名字前缀重建一个只含本模块张量的 ggml_context，再在该 context 上分配后端缓冲；
// 数据上传仍由调用方按 (name, offset) 完成（未命中的名字 `ggml_get_tensor` 返回 NULL，
// 现有上传循环本来就有 `if (!tt) continue;`）。
#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>

#include "ggml.h"
#include "ggml-backend.h"

static inline bool gsv_gguf_prefix_match(const char * name, std::initializer_list<const char *> prefixes) {
    for (const char * p : prefixes) {
        const size_t n = std::strlen(p);
        if (std::strncmp(name, p, n) == 0) return true;
    }
    return false;
}

// 成功时返回后端缓冲并把 *out_ctx 设为新建的过滤上下文（调用方负责 ggml_free）；
// 失败返回 NULL。原 ctx 不会被释放，调用方在切换成功后自行 ggml_free。
static inline ggml_backend_buffer_t gsv_alloc_prefix_tensors(
        ggml_context * src, ggml_backend_t backend,
        std::initializer_list<const char *> prefixes, ggml_context ** out_ctx) {
    if (src == nullptr || backend == nullptr || out_ctx == nullptr) return nullptr;

    int n = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(src); t != nullptr; t = ggml_get_next_tensor(src, t))
        if (gsv_gguf_prefix_match(t->name, prefixes)) n++;
    if (n == 0) return nullptr;

    ggml_init_params ip = { (size_t) n * ggml_tensor_overhead() + 16 * 1024, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) return nullptr;

    for (ggml_tensor * t = ggml_get_first_tensor(src); t != nullptr; t = ggml_get_next_tensor(src, t)) {
        if (!gsv_gguf_prefix_match(t->name, prefixes)) continue;
        ggml_tensor * nt = ggml_dup_tensor(ctx, t);
        if (nt == nullptr) { ggml_free(ctx); return nullptr; }
        ggml_set_name(nt, t->name);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) { ggml_free(ctx); return nullptr; }
    *out_ctx = ctx;
    return buf;
}
