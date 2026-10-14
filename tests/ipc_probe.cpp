// 编译期 + 运行期冒烟: framing 的两个经典坑 (跨读的数据帧 / 头已预读负载未到)
#include "ipc/frame.h"
#include "ipc/reader.h"
#include "ipc/stream.h"
#include "ipc/hash.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
using namespace gsvipc;
// 脚本化字节流: 按 chunk 边界喂, 制造"跨两次读"与"头先到负载后到"
class ScriptStream : public ByteStream {
public:
    explicit ScriptStream(std::string data, size_t chunk) : d_(std::move(data)), c_(chunk) {}
    int Read(void *b, size_t n, int) override {
        if (p_ >= d_.size()) return kStreamEnded;
        size_t k = std::min(n, std::min(c_, d_.size() - p_));
        memcpy(b, d_.data() + p_, k); p_ += k; return (int) k;
    }
    bool Write(const void *, size_t) override { return true; }
private:
    std::string d_; size_t c_, p_ = 0;
};
int main() {
    // 1) 解析 round-trip
    std::string req = BuildRequestLine("abc-123", "ping", "{\"x\":1}");
    ControlLine cl = ParseControlLine(req);
    assert(cl.kind == ControlKind::Request && cl.uuid == "abc-123" && cl.name == "ping");
    assert(cl.payload == "{\"x\":1}");
    // 2) tensor 帧头三段式。BuildFrameHeader 带尾换行 (线上就是 LF 结尾);
    //    读取侧 ReadLine 会剥掉换行再交给解析器, 所以这里也剥。
    const std::string fhWire = BuildFrameHeader("wav", "13507256038857166760", 12);
    std::string fh = fhWire;
    if (!fh.empty() && fh.back() == '\n') fh.pop_back();
    std::string nm, h; int32_t len = 0;
    assert(IsFrameHeader(fh) && TryParseFrameHeader(fh, &nm, &h, &len));
    assert(nm == "wav" && len == 12);
    // 3) 畸形长度必须拒绝
    assert(!TryParseFrameHeader("tensor wav 1 -5", &nm, &h, &len));
    assert(!TryParseFrameHeader("tensor wav 1 5 6", &nm, &h, &len));
    // 4) hash 十进制 round-trip
    uint64_t v = 0;
    assert(TryParseHash("18446744073709551615", &v) && v == 18446744073709551615ULL);
    uint64_t big=0; assert(!TryParseHash("18446744073709551616", &big));
    // 5) 跨读的数据帧: 一次喂 1 字节, 头与负载都会被拼完整
    std::string wire = fhWire + std::string(12, 'Z') + "close\n";
    for (size_t chunk : {1u, 3u, 7u, 1000u}) {
        ScriptStream s(wire, chunk);
        FrameReader r(&s);
        std::string line;
        assert(r.ReadLine(&line, 1000) == ReadStatus::Ok && IsFrameHeader(line));
        std::vector<uint8_t> pay;
        assert(r.ReadExactly(12, &pay) == ReadStatus::Ok && pay.size() == 12);
        assert(std::string(pay.begin(), pay.end()) == std::string(12, 'Z'));
        std::string close;
        assert(r.ReadLine(&close, 1000) == ReadStatus::Ok && close == "close");
    }
    printf("IPC FRAMING SMOKE PASS\n");
    return 0;
}
