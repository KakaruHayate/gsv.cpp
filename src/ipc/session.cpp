#include "session.h"

#include "hash.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>

namespace gsvipc {
namespace {

constexpr int kReadTimeoutMs = 120000;   // generous: a big tensor may take a while to arrive

}  // namespace

Session::Session(SocketStream *stream, RequestHandler *handler, bool verbose)
    : stream_(stream), handler_(handler), reader_(stream), verbose_(verbose) {
    peer_ = "peer";
}

Session::~Session() = default;

bool Session::SendLine(const std::string &line) {
    std::string withNewline = line;
    withNewline.push_back('\n');
    return stream_->Write(withNewline.data(), withNewline.size());
}

bool Session::SendTensor(const std::string &name, const std::vector<uint8_t> &bytes) {
    const uint64_t hash = Xxh64(bytes.data(), bytes.size());
    const std::string header = BuildFrameHeader(name, FormatHash(hash), bytes.size());
    if (!stream_->Write(header.data(), header.size())) return false;
    if (!bytes.empty() && !stream_->Write(bytes.data(), bytes.size())) return false;
    return true;
}

bool Session::SendResponse(const std::string &uuid, bool success, const std::string &json,
                           const std::string &error,
                           const std::map<std::string, std::vector<uint8_t>> &tensors) {
    nlohmann::json envelope;
    envelope["success"] = success;
    if (success) {
        if (!json.empty()) {
            try {
                envelope["data"] = nlohmann::json::parse(json);
            } catch (const std::exception &e) {
                GSV_LOG_ERROR("Handler produced invalid JSON for %s: %s", uuid.c_str(), e.what());
                envelope["success"] = false;
                envelope["data"] = nlohmann::json::object();
                envelope["error"] = std::string("handler produced invalid JSON: ") + e.what();
                return SendLine(BuildResponseLine(uuid, envelope.dump()));
            }
        } else {
            envelope["data"] = nlohmann::json::object();
        }
        envelope["error"] = nullptr;
    } else {
        envelope["data"] = nlohmann::json::object();
        envelope["error"] = error.empty() ? "unspecified failure" : error;
    }
    // Declare the tensor frames that follow, in order. Without this the client cannot tell a
    // response with no tensors from one whose frames have not arrived yet, and would block.
    nlohmann::json names = nlohmann::json::array();
    for (const auto &kv : tensors) names.push_back(kv.first);
    envelope["tensors"] = names;
    if (!SendLine(BuildResponseLine(uuid, envelope.dump()))) return false;

    // Tensors follow the envelope. A failure here desynchronizes the stream, which is a
    // protocol error, so the caller drops the connection.
    for (const auto &kv : tensors) {
        if (!SendTensor(kv.first, kv.second)) return false;
    }
    return true;
}

void Session::Run() {
    // Wire order for one exchange: the client sends its `tensor` frames first, then the
    // `request` line that consumes them. So this loop reads lines and branches on the plane:
    // a frame header means "park the payload", anything else is a control line.
    while (true) {
        std::string line;
        const ReadStatus st = reader_.ReadLine(&line, kReadTimeoutMs);
        if (st == ReadStatus::Timeout) continue;
        if (st == ReadStatus::EndOfStream) {
            if (verbose_) GSV_LOG_INFO("connection closed by peer.");
            return;
        }
        if (st == ReadStatus::ProtocolError) {
            GSV_LOG_WARN("framing violation: %s", reader_.Error().c_str());
            return;
        }

        if (IsFrameHeader(line)) {
            std::string name, hash;
            int32_t length = 0;
            if (!TryParseFrameHeader(line, &name, &hash, &length)) {
                GSV_LOG_WARN("malformed tensor header.");
                return;
            }
            if (length < 0 || (size_t) length > kMaxFrameBytes) {
                // Peer-controlled length: refuse rather than allocate.
                GSV_LOG_WARN("out-of-range tensor length %d.", length);
                return;
            }
            std::vector<uint8_t> payload;
            if (reader_.ReadExactly((size_t) length, &payload) != ReadStatus::Ok) {
                GSV_LOG_WARN("tensor payload short or failed.");
                return;
            }
            ParkedTensor t;
            t.bytes = std::move(payload);
            t.hash = hash;
            parked_[name] = std::move(t);
            continue;
        }

        const ControlLine control = ParseControlLine(line);
        if (control.kind == ControlKind::Close) {
            if (verbose_) GSV_LOG_INFO("peer asked to close.");
            return;
        }
        if (control.kind == ControlKind::Unknown) {
            // Survive: a line we do not understand is never fatal.
            GSV_LOG_WARN("unparseable control line.");
            continue;
        }
        if (control.kind == ControlKind::Notification) {
            if (verbose_) GSV_LOG_INFO("notification %s ignored.", control.name.c_str());
            continue;
        }
        if (control.kind != ControlKind::Request) continue;

        std::string responseJson, error;
        std::map<std::string, std::vector<uint8_t>> outTensors;
        bool ok = false;
        try {
            ok = handler_->Handle(control.name, control.payload, parked_, responseJson,
                                  outTensors, error);
        } catch (const std::exception &e) {
            ok = false;
            error = std::string("handler threw: ") + e.what();
        }
        if (!SendResponse(control.uuid, ok, ok ? responseJson : std::string(), error, outTensors)) {
            GSV_LOG_WARN("write failed; dropping connection.");
            return;
        }
        if (verbose_) {
            GSV_LOG_INFO("%s -> %s (%zu tensor frames)", control.name.c_str(),
                         ok ? "ok" : "failed", outTensors.size());
        }
        // Tensors are consumed by the request that named them.
        parked_.clear();
    }
}

void Session::RunWithReadyNotification() {
    nlohmann::json ready;
    ready["apiVersion"] = kApiVersion;
    ready["name"] = "gsv-ggml";
    SendLine(BuildNotificationLine("ready", ready.dump()));
    Run();
}

}  // namespace gsvipc
