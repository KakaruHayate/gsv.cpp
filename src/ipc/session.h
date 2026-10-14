#pragma once

/*
 * One connection's request/response loop.
 *
 * Ported in spirit from KakaruHayate/openutau-vst-bridge (MPL-2.0) src/session.{h,cpp}: the
 * framing lives in frame/reader, this layer owns the message vocabulary and the rule that a
 * malformed request produces a failed envelope rather than a dropped connection.
 *
 * Large float32 payloads (wav, mel, bert features) do not travel inside JSON. The requester
 * sends `tensor <name> <hash> <len>\n<bytes>` frames before the request that consumes them;
 * the session parks them by name so the handler can pick them up. Responses work the same
 * way: the JSON envelope carries names, shapes and hashes, and the tensors follow.
 */

#include "frame.h"
#include "reader.h"
#include "socket.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gsvipc {

/// A raw byte payload parked by tensor name, plus the hash the sender claimed for it.
struct ParkedTensor {
    std::vector<uint8_t> bytes;
    std::string hash;
};

/// The engine-facing surface a session drives. Implemented by the CLI's engine wrapper.
class RequestHandler {
public:
    virtual ~RequestHandler() = default;

    /// `kind` is the request kind from the control line; `params` is the parsed JSON body.
    /// `tensors` holds the data-plane frames that preceded this request, keyed by name.
    /// On success the handler fills `responseJson` and, for large outputs, `outTensors`
    /// (name -> raw bytes); the session then emits the tensor frames after the envelope.
    /// Return false to produce a failed envelope with `error` set.
    virtual bool Handle(const std::string &kind, const std::string &params,
                        const std::map<std::string, ParkedTensor> &tensors,
                        std::string &responseJson,
                        std::map<std::string, std::vector<uint8_t>> &outTensors,
                        std::string &error) = 0;
};

/// Drives one accepted connection until `close`, an error, or the peer going away.
class Session {
public:
    Session(SocketStream *stream, RequestHandler *handler, bool verbose);
    ~Session();

    /// Blocks. Returns when the connection ends; the reason is logged.
    void Run();

    /// Reads requests forever but also emits `notification:ready` on connect, so a client can
    /// wait for the model load to finish before sending work.
    void RunWithReadyNotification();

    const std::string &PeerLabel() const { return peer_; }

private:
    /// Sends the JSON envelope, then any tensor frames, for one request.
    bool SendResponse(const std::string &uuid, bool success, const std::string &json,
                      const std::string &error,
                      const std::map<std::string, std::vector<uint8_t>> &tensors);

    /// Sends one data-plane frame: header line, then exactly `bytes.size()` bytes.
    bool SendTensor(const std::string &name, const std::vector<uint8_t> &bytes);

    bool SendLine(const std::string &line);

    SocketStream *stream_;
    RequestHandler *handler_;
    FrameReader reader_;
    std::string peer_;
    bool verbose_;
    std::map<std::string, ParkedTensor> parked_;
};

}  // namespace gsvipc
