#pragma once

// Device-string parsing shared by all model loaders.
//
//   ""            CPU
//   "cpu"         CPU
//   "vulkan"      first GPU (same as "gpu", "cuda", case variants)
//   "vulkan:1"    GPU #1 (0-based, as printed by --list-devices)
//   "gpu:1"       same
//   "1"           bare number = GPU #1
//
// *idx is the GPU ordinal; *explicit_idx is true when the string named a specific GPU
// (so callers can fail instead of silently falling back to CPU).

#include <cstdlib>
#include <cstring>
#include <string>

inline bool gsv_device_parse(const std::string & d, int * idx = nullptr, bool * explicit_idx = nullptr) {
    if (idx) *idx = 0;
    if (explicit_idx) *explicit_idx = false;
    if (d.empty()) return false;
    if (d == "cpu" || d == "CPU" || d == "CPUExecutionProvider") return false;

    const size_t colon = d.rfind(':');
    if (colon != std::string::npos) {
        if (idx) *idx = std::atoi(d.c_str() + colon + 1);
        if (explicit_idx) *explicit_idx = true;
        return true;
    }
    if (d.find_first_not_of("0123456789") == std::string::npos) {
        if (idx) *idx = std::atoi(d.c_str());
        if (explicit_idx) *explicit_idx = true;
        return true;
    }
    return d == "gpu" || d == "vulkan" || d == "cuda" || d == "GPU" || d == "Vulkan" || d == "CUDA";
}
