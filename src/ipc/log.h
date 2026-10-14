#pragma once

/*
 * Minimal logging for the IPC + server layer. The bridge this transport was ported from had a
 * full logger with levels and sinks; we only need a timestamped stderr line, so the surface is
 * three functions. Everything is header-inline so no extra translation unit is needed.
 */

#include <cstdarg>
#include <cstdio>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

namespace gsvipc {

enum class LogLevel { Info, Warn, Error };

inline const char *LevelTag(LogLevel level) {
    switch (level) {
        case LogLevel::Info:  return "I";
        case LogLevel::Warn:  return "W";
        case LogLevel::Error: return "E";
    }
    return "?";
}

/// One line to stderr, prefixed with the level. Newline included.
inline void LogLine(LogLevel level, const char *text) {
    std::fprintf(stderr, "[gsv:%s] %s\n", LevelTag(level), text ? text : "");
}

inline void LogInfo(const char *text)  { LogLine(LogLevel::Info, text); }
inline void LogWarn(const char *text)  { LogLine(LogLevel::Warn, text); }
inline void LogError(const char *text) { LogLine(LogLevel::Error, text); }

/// printf-style variants (what the ported socket.cpp calls).
inline void LogLinef(LogLevel level, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    LogLine(level, buf);
}

#define GSV_LOG_INFO(...)  ::gsvipc::LogLinef(::gsvipc::LogLevel::Info, __VA_ARGS__)
#define GSV_LOG_WARN(...)  ::gsvipc::LogLinef(::gsvipc::LogLevel::Warn, __VA_ARGS__)
#define GSV_LOG_ERROR(...) ::gsvipc::LogLinef(::gsvipc::LogLevel::Error, __VA_ARGS__)

}  // namespace gsvipc
