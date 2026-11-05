#pragma once
// Path helpers: JSON bodies carry UTF-8, but on Windows a narrow std::filesystem::path is
// interpreted with the ANSI code page, so any non-ASCII path (Chinese/Japanese file names)
// fails with "No mapping for the Unicode character exists in the target multi-byte code page".
// Convert explicitly and use the wide APIs instead.
#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
// windows.h defines min/max macros unless told otherwise; pocketfft (and other headers) break
// on them, so keep the same guard the server uses.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace gsvpath {

inline std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                      s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return std::wstring(s.begin(), s.end());   // not valid UTF-8: pass bytes through
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), &w[0], n);
    return w;
}

inline std::filesystem::path from_utf8(const std::string & s) {
    return std::filesystem::path(utf8_to_wide(s));
}

inline FILE * open_read(const std::string & s) {
    return _wfopen(utf8_to_wide(s).c_str(), L"rb");
}

}  // namespace gsvpath
#else

namespace gsvpath {

inline std::filesystem::path from_utf8(const std::string & s) { return std::filesystem::path(s); }
inline FILE * open_read(const std::string & s) { return std::fopen(s.c_str(), "rb"); }

}  // namespace gsvpath
#endif
