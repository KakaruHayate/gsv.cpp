// Unit tests for tools/svc_core.h (no engine, no GPU, no models).
// Built by scripts/build-tests.bat, also run by CI.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "svc_core.h"

using namespace svc;

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            g_failures++;                                                  \
        }                                                                  \
    } while (0)

int main() {
    // utf8_len
    CHECK(utf8_len("") == 0);
    CHECK(utf8_len("abc") == 3);
    CHECK(utf8_len("你好") == 2);

    // split_text: cut0 = single segment
    const auto s0 = split_text("你好。世界！", "cut0");
    CHECK(s0.size() == 1);
    CHECK(s0[0] == "你好。世界！");

    // cut5: short text stays one segment, long text splits
    CHECK(split_text("你好。世界！", "cut5").size() == 1);
    const auto s5 = split_text(
        "这是一个很长的句子，用来测试切分逻辑是否正常工作。第二个句子也有不少字符！", "cut5");
    CHECK(s5.size() >= 2);
    CHECK(s5[0] != "");

    // wav_bytes: canonical 44-byte header + 16-bit mono 48 kHz
    std::vector<float> x(1000, 0.1f);
    const std::string w = wav_bytes(x);
    CHECK(w.size() == 44 + 1000 * 2);
    CHECK(w.compare(0, 4, "RIFF") == 0);
    CHECK(w.compare(8, 4, "WAVE") == 0);
    CHECK(w.compare(36, 4, "data") == 0);

    // normalize: DC removed, peak = 0.708
    std::vector<float> y = {-0.5f, 0.5f};
    normalize(y);
    CHECK(std::fabs(y[0] + 0.708f) < 1e-3f);
    CHECK(std::fabs(y[1] - 0.708f) < 1e-3f);

    // Lru: recency + eviction
    Lru<int> lru(2);
    lru.put("a", 1);
    lru.put("b", 2);
    int v = 0;
    CHECK(lru.get("a", v) && v == 1);
    lru.put("c", 3);   // evicts b
    CHECK(!lru.get("b", v));
    CHECK(lru.get("a", v) && v == 1);
    CHECK(lru.get("c", v) && v == 3);
    lru.clear();
    CHECK(!lru.get("a", v));

    // path_allowed: prefix must be a path boundary
    g_allow.clear();
    g_allow.push_back("C:/data/models");
    CHECK(path_allowed(""));
    CHECK(path_allowed("C:/data/models/ref.wav"));
    CHECK(path_allowed("C:/data/models/sub/ref.wav"));
    CHECK(!path_allowed("C:/data/other/ref.wav"));
    CHECK(!path_allowed("C:/data/modelsX/ref.wav"));

    // authorized
    httplib::Request req;
    g_token.clear();
    CHECK(authorized(req));
    g_token = "t0k";
    CHECK(!authorized(req));
    req.set_header("Authorization", "Bearer t0k");
    CHECK(authorized(req));

    if (g_failures == 0) std::printf("ALL TESTS PASSED\n");
    else std::printf("%d TEST(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
