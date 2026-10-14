#pragma once

/*
 * Line and frame syntax for both planes (docs/e2e_server.md §3). Ported from
 * KakaruHayate/openutau-vst-bridge (MPL-2.0) src/frame.h, with the OpenUtau-specific
 * frame kind (`audio`) changed to ours (`tensor`) and the tolerance rules kept as-is:
 * a malformed length desynchronizes the stream past recovery, so both sides parse
 * strictly or not at all.
 */

#include <cstdint>
#include <string>

namespace gsvipc {

/// A received line starting with this introduces binary, not control (data plane).
inline constexpr const char *kFramePrefix = "tensor ";

enum class ControlKind {
    /// Malformed or a header this version does not know. Logged and survived, never fatal.
    Unknown,
    /// Bare `close`, no payload.
    Close,
    Request,
    Response,
    Notification,
};

struct ControlLine {
    ControlKind kind = ControlKind::Unknown;
    /// Set for Request and Response.
    std::string uuid;
    /// Request kind or notification kind.
    std::string name;
    /// Raw JSON text after the first space. Empty when the line carried none.
    std::string payload;
};

/// Splits one control line (newline already stripped) into header parts and raw payload.
ControlLine ParseControlLine(const std::string &line);

bool IsFrameHeader(const std::string &line);

/// Parses `tensor <name> <hash> <length>`. Strict on every field — no sign, no whitespace,
/// nothing past int32 for the length — because a malformed length would desynchronize the
/// stream past recovery.
bool TryParseFrameHeader(const std::string &line, std::string *name, std::string *hash,
                         int32_t *length);

/// Includes the trailing newline: unlike a control line, the newline is part of the binary
/// frame and the payload follows it verbatim.
std::string BuildFrameHeader(const std::string &name, const std::string &hash, size_t length);

/// These return the line without its newline; the writer appends one.
std::string BuildRequestLine(const std::string &uuid, const std::string &kind, const std::string &json);
std::string BuildResponseLine(const std::string &uuid, const std::string &json);
std::string BuildNotificationLine(const std::string &kind, const std::string &json);

/// A v4-shaped random identifier for outbound requests. Only has to be unique per
/// connection and free of ':' and ' '.
std::string NewUuid();

}  // namespace gsvipc
