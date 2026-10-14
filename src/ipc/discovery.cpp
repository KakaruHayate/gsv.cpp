#include "discovery.h"

#include "hash.h"
#include "log.h"
#include "socket.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

namespace gsvipc {
namespace {

/// Matches `Path.GetTempPath()` on Windows (TMP, then TEMP, then the profile), else `$TMPDIR`.
/// Wide throughout on Windows: a temp path under a non-ASCII user name would not survive the
/// active code page.
std::string TempDirectory() {
#if defined(_WIN32)
    DWORD needed = ::GetTempPathW(0, nullptr);
    if (needed == 0) return ".";
    std::wstring buffer(needed, L'\0');
    DWORD length = ::GetTempPathW(needed, buffer.data());
    if (length == 0 || length >= needed) return ".";
    buffer.resize(length);
    // UTF-8 so a non-ASCII temp path survives (we only ever pass it back to narrow APIs as text).
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, buffer.data(), (int) buffer.size(),
                                        nullptr, 0, nullptr, nullptr);
    std::string out((size_t) std::max(n, 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, buffer.data(), (int) buffer.size(), out.data(), n,
                          nullptr, nullptr);
    if (n > 0) out.resize((size_t) n);
    return out;
#else
    const char *configured = std::getenv("TMPDIR");
    if (configured != nullptr && configured[0] != '\0') return configured;
    return "/tmp";
#endif
}

bool JoinPath(std::string *out, const std::string &dir, const std::string &leaf) {
    if (dir.empty()) return false;
    *out = dir;
    const char last = dir[dir.size() - 1];
    if (last != '/' && last != '\\') *out += "/";
    *out += leaf;
    return true;
}

/// Escapes a JSON string body (quotes, backslash, control chars). Names are ASCII in practice,
/// but the display name is free text.
std::string JsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }
    return out;
}

/// Pulls `"key": "value"` out of a small JSON object without a full parser. The discovery
/// schema is three known string fields plus one int, which is all we ever write.
bool JsonFindString(const std::string &body, const std::string &key, std::string *out) {
    const std::string needle = "\"" + key + "\"";
    size_t at = body.find(needle);
    if (at == std::string::npos) return false;
    at = body.find(':', at + needle.size());
    if (at == std::string::npos) return false;
    at = body.find('"', at + 1);
    if (at == std::string::npos) return false;
    std::string value;
    for (size_t i = at + 1; i < body.size(); i++) {
        const char c = body[i];
        if (c == '\\' && i + 1 < body.size()) { value.push_back(body[++i]); continue; }
        if (c == '"') { *out = value; return true; }
        value.push_back(c);
    }
    return false;
}

bool JsonFindInt(const std::string &body, const std::string &key, int *out) {
    const std::string needle = "\"" + key + "\"";
    size_t at = body.find(needle);
    if (at == std::string::npos) return false;
    at = body.find(':', at + needle.size());
    if (at == std::string::npos) return false;
    while (at < body.size() && (body[at] == ':' || body[at] == ' ')) at++;
    std::string digits;
    while (at < body.size() && body[at] >= '0' && body[at] <= '9') digits.push_back(body[at++]);
    if (digits.empty()) return false;
    *out = std::atoi(digits.c_str());
    return true;
}

bool ReadWholeFile(const std::string &path, std::string *out) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
}

}  // namespace

std::string DiscoveryDir() {
    std::string dir;
    JoinPath(&dir, TempDirectory(), "gsv-ggml");
    std::string servers;
    JoinPath(&servers, dir, "servers");
    return servers;
}

std::string WriteDiscoveryFile(const std::string &key, int port, const std::string &name) {
    if (port <= 0 || port > 65535) {
        GSV_LOG_ERROR("Refusing to advertise port %d, which is not a usable port.", port);
        return {};
    }
    const std::string dir = DiscoveryDir();
#if defined(_WIN32)
    // One mkdir level at a time: SHCreateDirectoryEx is the wide-correct way, but the temp
    // path is already wide and this directory name is ASCII, so the narrow form is safe here.
    std::string partial;
    for (size_t i = 0; i <= dir.size();) {
        const size_t next = dir.find_first_of("/\\", i);
        const bool last = (next == std::string::npos);
        partial = last ? dir : dir.substr(0, next);
        if (!partial.empty()) ::CreateDirectoryA(partial.c_str(), nullptr);
        if (last) break;
        i = next + 1;
    }
#else
    std::string partial;
    for (size_t i = 0; i <= dir.size();) {
        const size_t next = dir.find('/', i);
        const bool last = (next == std::string::npos);
        partial = last ? dir : dir.substr(0, next);
        if (!partial.empty()) ::mkdir(partial.c_str(), 0755);
        if (last) break;
        i = next + 1;
    }
#endif

    std::string target;
    if (!JoinPath(&target, dir, key + ".json")) return {};
    const std::string display = name.empty() ? ("gsv-ggml " + std::to_string(port)) : name;
    std::ostringstream body;
    body << "{\n"
         << "  \"port\": " << port << ",\n"
         << "  \"name\": \"" << JsonEscape(display) << "\",\n"
         << "  \"apiVersion\": \"" << kApiVersion << "\"\n"
         << "}\n";

    // Staging + rename: a client scanning at any moment must never see a half-written file.
    const std::string staging = target + ".tmp";
    {
        std::ofstream out(staging.c_str(), std::ios::binary | std::ios::trunc);
        if (!out) {
            GSV_LOG_ERROR("Could not write %s.", staging.c_str());
            return {};
        }
        out.write(body.str().data(), (std::streamsize) body.str().size());
        if (!out) {
            GSV_LOG_ERROR("Could not write %s.", staging.c_str());
            std::remove(staging.c_str());
            return {};
        }
    }
#if defined(_WIN32)
    if (!::MoveFileExA(staging.c_str(), target.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        GSV_LOG_ERROR("Could not publish %s (error %lu).", target.c_str(), ::GetLastError());
        std::remove(staging.c_str());
        return {};
    }
#else
    if (std::rename(staging.c_str(), target.c_str()) != 0) {
        GSV_LOG_ERROR("Could not publish %s.", target.c_str());
        std::remove(staging.c_str());
        return {};
    }
#endif
    GSV_LOG_INFO("Advertised port %d as %s.", port, target.c_str());
    return target;
}

void RemoveDiscoveryFile(const std::string &path) {
    if (path.empty()) return;
    if (std::remove(path.c_str()) != 0) {
        // Not fatal: the client probes the port and prunes what no longer answers.
        GSV_LOG_WARN("Could not delete %s.", path.c_str());
    }
}

std::vector<DiscoveryEntry> ScanDiscoveryDir() {
    std::vector<DiscoveryEntry> out;
    const std::string dir = DiscoveryDir();
#if defined(_WIN32)
    std::string pattern;
    JoinPath(&pattern, dir, "*.json");
    WIN32_FIND_DATAA fd;
    HANDLE find = ::FindFirstFileA(pattern.c_str(), &fd);
    if (find == INVALID_HANDLE_VALUE) return out;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        std::string path;
        if (!JoinPath(&path, dir, fd.cFileName)) continue;
        DiscoveryEntry e;
        e.path = path;
        std::string body;
        if (!ReadWholeFile(path, &body)) continue;
        if (!JsonFindInt(body, "port", &e.port)) continue;
        JsonFindString(body, "name", &e.name);
        JsonFindString(body, "apiVersion", &e.apiVersion);
        out.push_back(e);
    } while (::FindNextFileA(find, &fd));
    ::FindClose(find);
#else
    DIR *d = ::opendir(dir.c_str());
    if (!d) return out;
    while (dirent *de = ::readdir(d)) {
        const std::string name = de->d_name;
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".json") != 0) continue;
        std::string path;
        if (!JoinPath(&path, dir, name)) continue;
        DiscoveryEntry e;
        e.path = path;
        std::string body;
        if (!ReadWholeFile(path, &body)) continue;
        if (!JsonFindInt(body, "port", &e.port)) continue;
        JsonFindString(body, "name", &e.name);
        JsonFindString(body, "apiVersion", &e.apiVersion);
        out.push_back(e);
    }
    ::closedir(d);
#endif
    std::sort(out.begin(), out.end(),
              [](const DiscoveryEntry &a, const DiscoveryEntry &b) { return a.path < b.path; });
    return out;
}

bool PortIsFree(int port) {
    if (port <= 0 || port > 65535) return false;
    std::unique_ptr<SocketStream> probe = ConnectLoopback(port);
    if (probe) {
        // Something answered: the server is alive.
        probe->Close();
        return false;
    }
    return true;
}

void PruneStaleDiscoveryFiles() {
    for (const DiscoveryEntry &e : ScanDiscoveryDir()) {
        if (PortIsFree(e.port)) {
            GSV_LOG_INFO("Pruning stale discovery file %s (nothing on port %d).",
                         e.path.c_str(), e.port);
            RemoveDiscoveryFile(e.path);
        }
    }
}

}  // namespace gsvipc
