#pragma once

/*
 * Service discovery: the server writes one JSON file per instance into a per-user temp
 * directory, and the Python frontend scans that directory instead of being told a port.
 * Ported from KakaruHayate/openutau-vst-bridge (MPL-2.0) src/discovery.h, with the OpenUtau
 * schema replaced by ours.
 *
 * Why a file rather than a fixed port: two servers (e.g. one on CPU, one on Vulkan) can run
 * side by side, and a crashed server leaves a stale file the client can detect and clean.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace gsvipc {

struct DiscoveryEntry {
    std::string path;      ///< The JSON file this was read from.
    int port = 0;
    std::string name;
    std::string apiVersion;
};

/// `%TEMP%/gsv-ggml/servers` on Windows, `$TMPDIR/gsv-ggml/servers` elsewhere.
/// Created on demand by WriteDiscoveryFile.
std::string DiscoveryDir();

/// Writes `<dir>/<key>.json` and returns its path, or empty on failure.
/// Overwrites any previous file for the same key, so a restarted server refreshes its port.
std::string WriteDiscoveryFile(const std::string &key, int port, const std::string &name);

void RemoveDiscoveryFile(const std::string &path);

/// Every parseable `*.json` in the discovery directory, sorted by path.
std::vector<DiscoveryEntry> ScanDiscoveryDir();

/// True when nothing is listening on 127.0.0.1:<port> (a successful bind means the server is
/// gone). Used to prune stale files left by a crash.
bool PortIsFree(int port);

/// Removes discovery files whose server is no longer reachable.
void PruneStaleDiscoveryFiles();

}  // namespace gsvipc
