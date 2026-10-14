#pragma once

/*
 * XXH64 摘要 + 协议版本号。
 *
 * 设计来源: KakaruHayate/openutau-vst-bridge (MPL-2.0) 的 src/hash.h ——
 *   - hash 一律以**十进制字符串**上线: 64 位值超出 JSON number 的 2^53 安全整数范围;
 *   - 解析严格 (只允许数字, 无符号/空白/越界), 因为畸形 hash 会让数据面失同步。
 * 本文件把 OpenUtau 专用的常量换成本项目的, 其余语义保持一致。
 */

#include <cstddef>
#include <cstdint>
#include <string>

namespace gsvipc {

/// 协议版本, 由 discovery file 承载并在 ping 响应里回显 (PROTOCOL.md §4 的同款做法)。
/// 主版本不匹配 -> 拒绝连接; 次版本偏移 -> 新侧自我限制到旧侧认识的字段。
inline constexpr const char *kApiVersion = "1.0";

/// 数据面单帧字节上限 (PROTOCOL.md §5.2 的同款保护): 长度由对端控制,
/// 超限必须按协议错误拒绝而不是分配, 否则对端一个假长度就能耗尽我们内存。
/// 256 MiB ≈ 11 分钟的 32kHz 单声道 float32 (本项目的 wav 就是这个规格)。
inline constexpr size_t kMaxFrameBytes = 256u * 1024u * 1024u;

/// XXH64 (seed 0) 摘要 —— tensor 帧的名字, Python 侧据此去重。
uint64_t Xxh64(const void *data, size_t size);

/// hash 以十进制字符串上线 (见文件头)。
std::string FormatHash(uint64_t hash);

/// 严格十进制解析: 只允许数字, 无符号、无空白、不越界 (与 bridge 侧同口径)。
bool TryParseHash(const std::string &text, uint64_t *hash);

}  // namespace gsvipc
