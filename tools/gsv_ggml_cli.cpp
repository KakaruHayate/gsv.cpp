// gsv_ggml_cli —— GPT-SoVITS V5 推理 server (ggml 侧), 供 Python 前端通过 IPC 驱动。
//
// 协议见 docs/e2e_server.md (两平面 framing, 源自 KakaruHayate/openutau-vst-bridge 的设计):
//   控制面  request:<uuid>:<kind> <json>\n / response:<uuid> <json>\n / notification:... / close
//   数据面  tensor <name> <hash> <len>\n<len bytes raw>   (float32 大块走这里, 不进 JSON)
//
// 用法:
//   gsv_ggml_cli [--models models] [--device cpu|vulkan] [--voc-device auto|cpu|dml|webgpu]
//                [--threads N] [--name <显示名>] [--verbose] [--once]
//   --once: 接受一个连接后退出 (调试用)
//
// 模型在启动时全部载入, 之后每个请求只跑推理。常驻期间可通过 SIGINT/Ctrl-C 退出
// (监听器轮询 + discovery 文件在析构里删除)。

#if defined(_WIN32)
// windows.h's min/max macros break nlohmann/json and the std library; NOMINMAX keeps them out.
// It must come before any standard header that uses std::min/std::max.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

#include "gsv_engine.h"
#include "ipc/discovery.h"
#include "ipc/frame.h"
#include "ipc/hash.h"
#include "ipc/log.h"
#include "ipc/reader.h"
#include "ipc/session.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

std::atomic<bool> g_stop{false};

#if defined(_WIN32)
BOOL WINAPI OnConsoleCtrl(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        g_stop.store(true);
        return TRUE;
    }
    return FALSE;
}
#else
void OnSignal(int) { g_stop.store(true); }
#endif

/// Reinterprets a parked tensor's bytes as float32. Rejects a size that is not a whole number
/// of elements so a malformed frame cannot silently read past the end.
bool AsF32(const gsvipc::ParkedTensor &t, std::vector<float> *out, size_t *count = nullptr) {
    if (t.bytes.size() % sizeof(float) != 0) return false;
    const size_t n = t.bytes.size() / sizeof(float);
    out->resize(n);
    if (n > 0) std::memcpy(out->data(), t.bytes.data(), t.bytes.size());
    if (count) *count = n;
    return true;
}

bool AsI32(const gsvipc::ParkedTensor &t, std::vector<int32_t> *out, size_t *count = nullptr) {
    if (t.bytes.size() % sizeof(int32_t) != 0) return false;
    const size_t n = t.bytes.size() / sizeof(int32_t);
    out->resize(n);
    if (n > 0) std::memcpy(out->data(), t.bytes.data(), t.bytes.size());
    if (count) *count = n;
    return true;
}

std::vector<uint8_t> AsBytes(const void *data, size_t n) {
    const auto *p = static_cast<const uint8_t *>(data);
    return std::vector<uint8_t>(p, p + n);
}

/// The engine behind every request. One instance, shared by all connections: the ggml modules
/// rebuild their graphs per sequence length and keep no mutable cross-request state, and
/// gsv_bert's feature LRU is only a cache (a hit returns the same values a miss would compute).
class EngineRequests : public gsvipc::RequestHandler {
public:
    explicit EngineRequests(gsv_engine *e) : engine_(e) {}

    bool Handle(const std::string &kind, const std::string &params,
                const std::map<std::string, gsvipc::ParkedTensor> &tensors,
                std::string &responseJson,
                std::map<std::string, std::vector<uint8_t>> &outTensors,
                std::string &error) override;

private:
    gsv_engine *engine_;
};

bool EngineRequests::Handle(const std::string &kind, const std::string &params,
                            const std::map<std::string, gsvipc::ParkedTensor> &tensors,
                            std::string &responseJson,
                            std::map<std::string, std::vector<uint8_t>> &outTensors,
                            std::string &error) {
    json req = json::object();
    if (!params.empty()) {
        try {
            req = json::parse(params);
        } catch (const std::exception &e) {
            error = std::string("bad JSON: ") + e.what();
            return false;
        }
    }
    auto need = [&](const char *key) -> const gsvipc::ParkedTensor * {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            error = std::string("missing tensor '") + key + "'";
            return nullptr;
        }
        return &it->second;
    };

    if (kind == "ping") {
        json out;
        out["apiVersion"] = gsvipc::kApiVersion;
        out["name"] = "gsv-ggml";
        out["melDim"] = engine_->mel_dim();
        out["textDim"] = engine_->text_dim();
        out["arVocab"] = engine_->ar_vocab();
        responseJson = out.dump();
        return true;
    }

    if (kind == "prepRef") {
        if (!req.contains("wavPath") || !req["wavPath"].is_string()) {
            error = "prepRef needs a string wavPath";
            return false;
        }
        std::vector<int32_t> codes;
        std::vector<float> ge, mel2;
        int melT = 0;
        if (!engine_->prep_reference(req["wavPath"].get<std::string>(), codes, ge, mel2, melT)) {
            error = "prep_reference failed";
            return false;
        }
        json out;
        out["codes"] = codes.size();
        out["melT"] = melT;
        responseJson = out.dump();
        // 大块走数据面: codes / ge / mel2
        if (!codes.empty())
            outTensors["codes"] = AsBytes(codes.data(), codes.size() * sizeof(int32_t));
        outTensors["ge"] = AsBytes(ge.data(), ge.size() * sizeof(float));
        outTensors["mel2"] = AsBytes(mel2.data(), mel2.size() * sizeof(float));
        return true;
    }

    if (kind == "bert") {
        const gsvipc::ParkedTensor *ids = need("ids");
        const gsvipc::ParkedTensor *w2p = need("word2ph");
        if (!ids || !w2p) return false;
        std::vector<int32_t> idsV, w2pV;
        if (!AsI32(*ids, &idsV) || !AsI32(*w2p, &w2pV)) {
            error = "ids/word2ph must be int32 arrays";
            return false;
        }
        std::vector<float> feat;
        if (!engine_->bert_feature(idsV, w2pV, feat)) {
            error = "bert_feature failed";
            return false;
        }
        json out;
        out["phones"] = feat.empty() ? 0 : (int64_t) feat.size() / 1024;
        responseJson = out.dump();
        outTensors["feat"] = AsBytes(feat.data(), feat.size() * sizeof(float));
        return true;
    }

    if (kind == "ar") {
        const gsvipc::ParkedTensor *ph = need("phones");
        const gsvipc::ParkedTensor *pr = need("prompt");
        if (!ph || !pr) return false;
        std::vector<int32_t> phones, prompt;
        if (!AsI32(*ph, &phones) || !AsI32(*pr, &prompt)) {
            error = "phones/prompt must be int32 arrays";
            return false;
        }
        // bert 可省 (非 zh 口径 = 全零)
        std::vector<float> bert;
        auto bit = tensors.find("bert");
        if (bit != tensors.end() && !AsF32(bit->second, &bert)) {
            error = "bert must be a float32 array";
            return false;
        }
        const int maxSteps = req.value("maxSteps", 1500);
        const int earlyStop = req.value("earlyStop", -1);
        const uint64_t seed = req.value("seed", (uint64_t) 0);
        std::vector<int32_t> sem;
        if (!engine_->ar_generate(phones, bert, prompt, maxSteps, earlyStop, seed, sem)) {
            error = "ar_generate failed";
            return false;
        }
        json out;
        out["sem"] = sem.size();
        responseJson = out.dump();
        if (!sem.empty())
            outTensors["sem"] = AsBytes(sem.data(), sem.size() * sizeof(int32_t));
        return true;
    }

    if (kind == "synth") {
        const gsvipc::ParkedTensor *sm = need("sem");
        const gsvipc::ParkedTensor *ph = need("phones");
        const gsvipc::ParkedTensor *pr = need("prompt");
        const gsvipc::ParkedTensor *ge = need("ge");
        const gsvipc::ParkedTensor *m2 = need("mel2");
        if (!sm || !ph || !pr || !ge || !m2) return false;
        std::vector<int32_t> sem, phones, prompt;
        std::vector<float> geV, mel2;
        if (!AsI32(*sm, &sem) || !AsI32(*ph, &phones) || !AsI32(*pr, &prompt) ||
            !AsF32(*ge, &geV) || !AsF32(*m2, &mel2)) {
            error = "synth tensors have the wrong element size";
            return false;
        }
        if (!req.contains("melT") || !req["melT"].is_number_integer()) {
            error = "synth needs an integer melT";
            return false;
        }
        const int melT = req["melT"].get<int>();
        const int steps = req.value("steps", 4);
        const float cfgRate = req.value("cfg", 0.0f);
        const uint64_t seed = req.value("seed", (uint64_t) 0);
        std::vector<float> wav;
        if (!engine_->synthesize(sem, phones, prompt, geV, mel2, melT, steps, cfgRate, seed, wav)) {
            error = "synthesize failed";
            return false;
        }
        json out;
        out["wav"] = wav.size();
        out["sampleRate"] = 32000;
        responseJson = out.dump();
        if (!wav.empty()) outTensors["wav"] = AsBytes(wav.data(), wav.size() * sizeof(float));
        return true;
    }

    error = "unknown request kind '" + kind + "'";
    return false;
}

void Usage() {
    std::printf(
        "usage: gsv_ggml_cli [--models DIR] [--device cpu|vulkan] [--voc-device auto|cpu|dml|webgpu]\n"
        "                     [--threads N] [--name STR] [--verbose] [--once]\n");
}

}  // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    gsv_engine_cfg cfg;
    cfg.model_dir = "models";
    std::string name = "gsv-ggml";
    bool once = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--models") cfg.model_dir = next("--models");
        else if (a == "--device") cfg.device = next("--device");
        else if (a == "--voc-device") cfg.voc_device = next("--voc-device");
        else if (a == "--threads") cfg.n_threads = std::atoi(next("--threads").c_str());
        else if (a == "--name") name = next("--name");
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "--once") once = true;
        else if (a == "-h" || a == "--help") { Usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); Usage(); return 2; }
    }

#if defined(_WIN32)
    ::SetConsoleCtrlHandler(OnConsoleCtrl, TRUE);
#else
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);
#endif

    std::printf("[gsv_ggml_cli] loading models from %s (ggml=%s, vocoder=%s)...\n",
                cfg.model_dir.c_str(), cfg.device.empty() ? "cpu" : cfg.device.c_str(),
                cfg.voc_device.empty() ? "auto" : cfg.voc_device.c_str());
    gsv_engine *engine = gsv_engine::load(cfg);
    if (!engine) { std::fprintf(stderr, "[gsv_ggml_cli] engine load failed\n"); return 1; }

    gsvipc::Listener listener;
    if (!listener.Start()) { std::fprintf(stderr, "[gsv_ggml_cli] could not listen\n"); return 1; }
    // Stale files from a crashed run would make the client dial a dead port.
    gsvipc::PruneStaleDiscoveryFiles();
    const std::string discoveryPath =
        gsvipc::WriteDiscoveryFile("gsv-ggml-" + std::to_string(listener.Port()),
                                   listener.Port(), name);
    std::printf("[gsv_ggml_cli] listening on 127.0.0.1:%d (api %s)%s\n",
                listener.Port(), gsvipc::kApiVersion,
                discoveryPath.empty() ? "" : ("  discovery: " + discoveryPath).c_str());

    EngineRequests handler(engine);
    while (!g_stop.load()) {
        std::unique_ptr<gsvipc::SocketStream> conn = listener.Accept(500);
        if (!conn) continue;
        std::printf("[gsv_ggml_cli] client connected\n");
        // One thread per connection; the engine is shared and read-only across requests.
        gsvipc::SocketStream *raw = conn.release();
        std::thread([raw, &handler, once]() {
            gsvipc::Session session(raw, &handler, true);
            session.RunWithReadyNotification();
            delete raw;
            if (once) g_stop.store(true);
        }).detach();
    }

    std::printf("[gsv_ggml_cli] shutting down\n");
    listener.Stop();
    gsvipc::RemoveDiscoveryFile(discoveryPath);
    delete engine;
    return 0;
}
