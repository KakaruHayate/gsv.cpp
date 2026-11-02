// gsv 服务层核心：voice 注册、LRU 缓存、合成流水线、请求日志。
// 供 tools/gsv_server.cpp 使用（HTTP 路由留在那边）。
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "httplib.h"
#include "json.hpp"

#include "gsv_engine.h"

namespace svc {

using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr int kSampleRate = 48000;

// ---------------------------------------------------------------- sidecar
struct SidecarCfg {
    std::string host = "127.0.0.1";
    int port = 9882;
};

struct BertSeg {
    std::vector<int32_t> ids;
    std::vector<int32_t> w2p;
    int start = 0;    // 该段在 phones 轴上的起始位置
};

struct G2POut {
    std::vector<int32_t> phones;
    std::vector<int32_t> word2ph;
    std::vector<int32_t> bert_ids;
    std::vector<BertSeg> bert_segs;   // auto 混合语言：逐段 BERT
};

inline bool parse_host_port(const std::string & url, SidecarCfg & out) {
    std::string s = url;
    if (s.rfind("http://", 0) == 0) s = s.substr(7);
    if (s.rfind("https://", 0) == 0) s = s.substr(8);
    const size_t slash = s.find('/');
    if (slash != std::string::npos) s = s.substr(0, slash);
    const size_t colon = s.rfind(':');
    if (colon == std::string::npos) { out.host = s; return !s.empty(); }
    out.host = s.substr(0, colon);
    out.port = std::atoi(s.substr(colon + 1).c_str());
    return !out.host.empty() && out.port > 0;
}

// 契约测试用：无需 sidecar 也能返回确定性的 G2P
inline bool g_mock_sidecar = false;

inline bool sidecar_g2p(const SidecarCfg & sc, const std::string & text, const std::string & lang,
                        G2POut & out, std::string & err) {
    if (g_mock_sidecar) {
        out.phones = {1, 2, 3};
        out.word2ph = {1, 1, 1};
        out.bert_ids.clear();
        out.bert_segs.clear();
        return true;
    }
    httplib::Client cli(sc.host, sc.port);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(180, 0);
    json req{{"text", text}, {"lang", lang}};
    auto res = cli.Post("/g2p", req.dump(), "application/json");
    if (!res) {
        err = "text sidecar 不可达 (" + sc.host + ":" + std::to_string(sc.port) + "): " +
              httplib::to_string(res.error());
        return false;
    }
    if (res->status != 200) {
        err = "text sidecar 返回 " + std::to_string(res->status) + ": " + res->body;
        return false;
    }
    try {
        const json j = json::parse(res->body);
        out.phones = j.at("phones").get<std::vector<int32_t>>();
        out.word2ph = j.at("word2ph").get<std::vector<int32_t>>();
        out.bert_ids = j.value("bert_ids", std::vector<int32_t>{});
        if (j.contains("bert_segs") && j["bert_segs"].is_array()) {
            for (const auto & s : j["bert_segs"]) {
                BertSeg seg;
                seg.ids = s.value("ids", std::vector<int32_t>{});
                seg.w2p = s.value("w2p", std::vector<int32_t>{});
                seg.start = s.value("start", 0);
                if (!seg.ids.empty() && !seg.w2p.empty()) out.bert_segs.push_back(seg);
            }
        }
    } catch (const std::exception & e) {
        err = std::string("text sidecar 响应解析失败: ") + e.what();
        return false;
    }
    return true;
}

inline bool sidecar_alive(const SidecarCfg & sc) {
    httplib::Client cli(sc.host, sc.port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    auto res = cli.Get("/health");
    return res && res->status == 200;
}

// ---------------------------------------------------------------- utils
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void put_u32(std::string & b, uint32_t v) {
    for (int i = 0; i < 4; i++) b.push_back((char) ((v >> (8 * i)) & 0xFF));
}
inline void put_u16(std::string & b, uint16_t v) {
    b.push_back((char) (v & 0xFF));
    b.push_back((char) ((v >> 8) & 0xFF));
}

// dc-remove + peak normalize (target 0.708)，与 tools/frontend/tts.py 同口径
inline void normalize(std::vector<float> & x) {
    if (x.empty()) return;
    double mean = 0.0;
    for (float v : x) mean += v;
    mean /= (double) x.size();
    double peak = 0.0;
    for (float & v : x) { v = (float) (v - mean); peak = std::max(peak, (double) std::fabs(v)); }
    if (peak > 1e-9) {
        const double k = 0.708 / peak;
        for (float & v : x) v = (float) (v * k);
    }
}

inline std::string wav_bytes(const std::vector<float> & x) {
    std::string out;
    const uint32_t n = (uint32_t) x.size();
    const uint32_t data_bytes = n * 2;
    out.reserve(44 + data_bytes);
    out += "RIFF";
    put_u32(out, 36 + data_bytes);
    out += "WAVEfmt ";
    put_u32(out, 16);
    put_u16(out, 1);
    put_u16(out, 1);
    put_u32(out, (uint32_t) kSampleRate);
    put_u32(out, (uint32_t) kSampleRate * 2);
    put_u16(out, 2);
    put_u16(out, 16);
    out += "data";
    put_u32(out, data_bytes);
    for (float v : x) {
        const float c = std::max(-1.0f, std::min(1.0f, v));
        const int16_t s = (int16_t) (c * 32767.0f);   // 向零截断，与 numpy astype 对齐
        put_u16(out, (uint16_t) s);
    }
    return out;
}

// ---------------------------------------------------------------- splitting
inline bool is_punct(const std::string & ch) {
    static const char * p[] = {"。", "！", "？", "；", "：", "，", "、", "…", "—",
                               ".", "!", "?", ";", ":", ",", "~", "\n"};
    for (const char * s : p) if (ch == s) return true;
    return false;
}

inline int utf8_len(const std::string & s) {
    int n = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        i += (c < 0x80) ? 1 : ((c >> 5) == 6) ? 2 : ((c >> 4) == 14) ? 3 : 4;
        n++;
    }
    return n;
}

inline std::vector<std::string> split_text(const std::string & text, const std::string & method) {
    std::vector<std::string> out;
    if (method.empty() || method == "cut0" || text.empty()) { out.push_back(text); return out; }

    std::vector<std::string> raw;
    std::string cur;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char) text[i];
        const size_t len = (c < 0x80) ? 1 : ((c >> 5) == 6) ? 2 : ((c >> 4) == 14) ? 3 : 4;
        std::string ch = text.substr(i, len);
        i += len;
        cur += ch;
        if (is_punct(ch)) { raw.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) raw.push_back(cur);

    std::string acc;
    for (const auto & s : raw) {
        acc += s;
        if (utf8_len(acc) >= 8) { out.push_back(acc); acc.clear(); }
    }
    if (!acc.empty()) {
        if (!out.empty()) out.back() += acc;
        else out.push_back(acc);
    }
    std::vector<std::string> bounded;
    for (const auto & s : out) {
        if (utf8_len(s) <= 300) { bounded.push_back(s); continue; }
        std::string piece;
        for (size_t i = 0; i < s.size();) {
            const unsigned char c = (unsigned char) s[i];
            const size_t len = (c < 0x80) ? 1 : ((c >> 5) == 6) ? 2 : ((c >> 4) == 14) ? 3 : 4;
            piece += s.substr(i, len);
            i += len;
            if (utf8_len(piece) >= 200) { bounded.push_back(piece); piece.clear(); }
        }
        if (!piece.empty()) bounded.push_back(piece);
    }
    return bounded;
}

// ---------------------------------------------------------------- security
inline std::string g_token;                 // 空 = 不校验（仅限 loopback）
inline std::vector<std::string> g_allow;    // 允许访问的目录（绝对路径）

inline bool is_loopback(const std::string & h) {
    return h == "127.0.0.1" || h == "localhost" || h == "::1" || h == "[::1]";
}

inline std::string canon_path(const std::string & p) {
    std::error_code ec;
    fs::path fp = fs::weakly_canonical(fs::absolute(p), ec);
    if (ec) fp = fs::absolute(p);
    std::string s = fp.string();
    for (char & c : s) {
        if (c == '\\') c = '/';
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

inline bool path_allowed(const std::string & p) {
    if (p.empty()) return true;
    const std::string c = canon_path(p);
    for (const auto & a : g_allow) {
        const std::string ca = canon_path(a);
        if (c == ca) return true;
        if (c.size() > ca.size() && c.compare(0, ca.size(), ca) == 0 &&
            (ca.back() == '/' || c[ca.size()] == '/'))
            return true;
    }
    return false;
}

inline bool authorized(const httplib::Request & req) {
    if (g_token.empty()) return true;
    if (req.get_header_value("Authorization") == "Bearer " + g_token) return true;
    if (req.has_param("token") && req.get_param_value("token") == g_token) return true;
    return false;
}

// ---------------------------------------------------------------- params
struct SynthParams {
    std::string text, text_lang = "zh";
    std::string ref_path, prompt_text, prompt_lang = "zh";
    std::string voice;
    int steps = 4;
    float cfg = 0.0f;
    int seed = -1;
    int max_steps = 1500, early_stop = 400;
    double speed = 1.0;
    int skip_steps = 0;   // 邪修：跳过 DiT 前 N 步模型调用（v=0）
    // cache-dit：-1 = auto（V5dev 32 步档自动 F8/.12；turbo 4 步自动关）
    int dbcache_fn = -1;
    float dbcache_thr = 0.02f;
    int dbcache_warmup = 8;
    int dbcache_ds = 4;
    gsv_sampler_cfg sampler;
    std::string split_method = "cut5";
    float fragment_interval = 0.3f;
};

struct SynthTiming {
    double g2p_ms = 0, prep_ms = 0, bert_ms = 0, ar_ms = 0, synth_ms = 0;
    int segments = 0;
    int phones = 0;
    int cache_hits = 0, cache_misses = 0;   // cache-dit (pos+neg)
    double cond_ms = 0, dit_ms = 0, voc_ms = 0;
};

// ---------------------------------------------------------------- voices
struct Voice {
    std::string name;
    std::string dir;
    std::string version;  // v5turbo / v5dev（决定 DiT 默认步数与 cfg）
    gsv_engine_cfg cfg;   // device/threads 由 State::base 覆盖
};

class VoiceRegistry {
public:
    void init(const std::string & models_dir) {
        std::lock_guard<std::mutex> lk(mu_);
        map_.clear();
        Voice def;
        def.name = "default";
        def.dir = models_dir;
        def.cfg.model_dir = models_dir;
        map_["default"] = def;

        const fs::path root = fs::path(models_dir) / "voices";
        std::error_code ec;
        if (!fs::is_directory(root, ec)) return;
        for (const auto & e : fs::directory_iterator(root, ec)) {
            if (!e.is_directory()) continue;
            const fs::path mf = e.path() / "manifest.json";
            if (!fs::exists(mf)) continue;
            try {
                std::ifstream f(mf);
                json j;
                f >> j;
                Voice v;
                v.name = j.value("name", e.path().filename().string());
                v.dir = e.path().string();
                v.cfg.model_dir = v.dir;
                auto p = [&](const char * key, std::string & dst) {
                    if (!j.contains(key) || !j[key].is_string()) return;
                    const std::string val = j[key].get<std::string>();
                    const fs::path fp(val);
                    dst = fp.is_absolute() ? val : (e.path() / val).string();
                };
                p("ar", v.cfg.ar_file);
                p("cond", v.cfg.cond_file);
                p("dit", v.cfg.dit_file);
                p("bert", v.cfg.bert_file);
                p("hubert", v.cfg.hubert_file);
                p("refcode", v.cfg.refcode_file);
                p("vocoder", v.cfg.vocoder_file);
                p("filter", v.cfg.filter_file);
                if (j.contains("version") && j["version"].is_string())
                    v.version = j["version"].get<std::string>();
                if (v.version.empty()) {
                    const std::string & d = v.cfg.dit_file;
                    v.version = d.find("v5dev") != std::string::npos ? "v5dev" : "v5turbo";
                }
                map_[v.name] = v;
            } catch (const std::exception & ex) {
                std::fprintf(stderr, "[voices] bad manifest %s: %s\n", mf.string().c_str(), ex.what());
            }
        }
    }

    std::vector<Voice> list() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<Voice> out;
        for (const auto & kv : map_) out.push_back(kv.second);
        return out;
    }

    bool get(const std::string & name, Voice & out) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(name);
        if (it == map_.end()) return false;
        out = it->second;
        return true;
    }

private:
    mutable std::mutex mu_;
    std::map<std::string, Voice> map_;
};

// ---------------------------------------------------------------- LRU
template <typename V>
class Lru {
public:
    explicit Lru(size_t cap) : cap_(cap) {}

    bool get(const std::string & k, V & out) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(k);
        if (it == map_.end()) return false;
        order_.splice(order_.begin(), order_, it->second.it);
        out = it->second.value;
        return true;
    }

    void put(const std::string & k, const V & v) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(k);
        if (it != map_.end()) {
            it->second.value = v;
            order_.splice(order_.begin(), order_, it->second.it);
            return;
        }
        order_.push_front(k);
        map_[k] = {v, order_.begin()};
        while (map_.size() > cap_) {
            const std::string & last = order_.back();
            map_.erase(last);
            order_.pop_back();
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        map_.clear();
        order_.clear();
    }

private:
    size_t cap_;
    std::mutex mu_;
    std::list<std::string> order_;
    struct Entry { V value; std::list<std::string>::iterator it; };
    std::unordered_map<std::string, Entry> map_;
};

// ---------------------------------------------------------------- state
struct RefPrep {
    std::vector<int32_t> codes;
    std::vector<float> ge, mel2;
    int mel2_T = 0;
};

struct PromptPrep {
    std::vector<int32_t> phones;
    std::vector<float> bert;
};

struct ArBatcher;      // AR 级聚合（保留，供单请求直连路径）
struct SynthScheduler; // 请求级聚合（/tts 与 /batch 都走这里）

struct State {
    gsv_engine_cfg base;                 // device / threads / default model_dir
    std::shared_ptr<gsv_engine> eng;     // 当前引擎
    std::mutex eng_mu;                   // 保护 eng 指针
    std::timed_mutex compute_mu;         // 合成串行（单 GPU）
    std::mutex reload_mu;                // 同一时刻只允许一次换模型
    std::string voice = "default";
    std::atomic<bool> reloading{false};
    VoiceRegistry voices;
    Lru<std::shared_ptr<const RefPrep>> ref_cache{8};
    Lru<std::shared_ptr<const PromptPrep>> prompt_cache{16};
    std::atomic<uint64_t> n_tts{0}, n_errors{0}, n_jobs{0}, n_timeouts{0}, n_busy{0}, n_reloads{0};
    std::atomic<int> waiters{0};
    std::atomic<double> last_total_ms{0};
    std::string version = "v5turbo";   // 当前 voice 的版本（决定 DiT 默认步数/cfg）
    ArBatcher * ar_batcher = nullptr;
    SynthScheduler * sched = nullptr;
    bool mock_engine = false;   // 契约测试：跳过真实引擎
};

inline std::shared_ptr<gsv_engine> snapshot(State & st) {
    std::lock_guard<std::mutex> lk(st.eng_mu);
    return st.eng;
}

// 同步换 voice（阻塞；期间旧引擎继续服务）。不预热：加载成功后原子换指针。
inline bool switch_voice(State & st, const std::string & name, std::string & err, double & load_ms) {
    if (name.empty()) return true;
    {
        std::lock_guard<std::mutex> lk(st.eng_mu);
        if (st.voice == name && st.eng) return true;
    }
    Voice v;
    if (!st.voices.get(name, v)) { err = "unknown voice: " + name; return false; }

    std::lock_guard<std::mutex> rl(st.reload_mu);
    {
        std::lock_guard<std::mutex> lk(st.eng_mu);
        if (st.voice == name && st.eng) return true;
    }
    st.reloading = true;
    gsv_engine_cfg cfg = v.cfg;
    cfg.device = st.base.device;
    cfg.voc_device = st.base.voc_device;
    cfg.n_threads = st.base.n_threads;
    cfg.verbose = st.base.verbose;
    if (cfg.model_dir.empty()) cfg.model_dir = st.base.model_dir;
    if (cfg.filter_file.empty() && !st.base.model_dir.empty()) {
        // 共享资源：resampy 表在顶层 model_dir，不在 voice 目录里
        cfg.filter_file = fs::absolute(fs::path(st.base.model_dir) / "resampy_kaiser_best.bin").string();
    }

    const double t0 = now_s();
    gsv_engine * e = gsv_engine::load(cfg);
    load_ms = (now_s() - t0) * 1000;
    st.reloading = false;
    if (!e) { err = "engine load failed for voice: " + name; return false; }

    {
        std::lock_guard<std::mutex> lk(st.eng_mu);
        st.eng.reset(e);
        st.voice = name;
        st.version = v.version;
    }
    st.ref_cache.clear();
    st.prompt_cache.clear();
    st.n_reloads++;
    std::printf("[voices] switched to '%s' (%.0f ms)\n", name.c_str(), load_ms);
    return true;
}

// 未显式给 sample_steps/cfg_rate 时按当前 voice 版本取 python 默认（v5dev: 32 / 1.30）
inline void resolve_auto_params(const State & st, SynthParams & p) {
    const bool dev = st.version.find("v5dev") != std::string::npos;
    if (p.steps <= 0) p.steps = dev ? 32 : 4;
    if (p.cfg < 0)    p.cfg = dev ? 1.30f : 0.0f;
}

// BERT 特征：单段走 bert_ids/word2ph；auto 混合语言按 bert_segs 逐段前向后 scatter，// 与 repo「zh 段真特征 + 非 zh 全零」的逐段拼接一致。
inline bool compute_bert(gsv_engine & eng, const G2POut & g, std::vector<float> & out) {
    out.clear();
    const int P = (int) g.phones.size();
    if (!g.bert_segs.empty()) {
        out.assign((size_t) 1024 * (size_t) P, 0.0f);
        for (const auto & s : g.bert_segs) {
            std::vector<float> block;
            if (!eng.bert_feature(s.ids, s.w2p, block)) return false;
            const int n = (int) (block.size() / 1024);
            for (int c = 0; c < 1024; c++) {
                for (int t = 0; t < n; t++) {
                    const int dst = s.start + t;
                    if (dst >= 0 && dst < P && (size_t) (c * n + t) < block.size())
                        out[(size_t) c * P + dst] = block[(size_t) c * n + t];
                }
            }
        }
        return true;
    }
    if (g.bert_ids.empty()) return true;
    return eng.bert_feature(g.bert_ids, g.word2ph, out);
}

// ---------------------------------------------------------------- synth
inline bool ensure_ref(State & st, const std::string & voice, gsv_engine & eng, const SidecarCfg & side,
                       const SynthParams & p, std::shared_ptr<const RefPrep> & ref,
                       std::shared_ptr<const PromptPrep> & prm, SynthTiming & tm, std::string & err) {
    const std::string ref_key = voice + "|" + p.ref_path + "|" + p.prompt_text + "|" + p.prompt_lang;
    if (!st.ref_cache.get(ref_key, ref)) {
        auto rp = std::make_shared<RefPrep>();
        const double t0 = now_s();
        if (!eng.prep_reference(p.ref_path, rp->codes, rp->ge, rp->mel2, rp->mel2_T)) {
            err = "prep_reference failed: " + p.ref_path;
            return false;
        }
        tm.prep_ms += (now_s() - t0) * 1000;
        ref = rp;
        st.ref_cache.put(ref_key, ref);
    }

    const std::string pkey = voice + "|" + p.prompt_text + "|" + p.prompt_lang;
    if (!st.prompt_cache.get(pkey, prm)) {
        auto pp = std::make_shared<PromptPrep>();
        G2POut g;
        const double t0 = now_s();
        if (!sidecar_g2p(side, p.prompt_text, p.prompt_lang, g, err)) return false;
        pp->phones = g.phones;
        if (!compute_bert(eng, g, pp->bert)) {
            err = "bert_feature(prompt) failed";
            return false;
        }
        tm.g2p_ms += (now_s() - t0) * 1000;
        prm = pp;
        st.prompt_cache.put(pkey, prm);
    }
    return true;
}

inline bool synth_segment(State & st, gsv_engine & eng, const SidecarCfg & side, const SynthParams & p,
                          const std::shared_ptr<const RefPrep> & ref,
                          const std::shared_ptr<const PromptPrep> & prm,
                          std::vector<float> & wav, SynthTiming & tm, std::string & err) {
    int stage = 0;
    try {
    stage = 1;
    const double t0 = now_s();
    G2POut tgt;
    if (!sidecar_g2p(side, p.text, p.text_lang, tgt, err)) return false;
    tm.g2p_ms += (now_s() - t0) * 1000;
    tm.phones += (int) tgt.phones.size();
    if (tgt.phones.empty()) {
        err = "no phonemes produced for this text (unsupported characters or empty input)";
        return false;
    }

    std::vector<float> bert;
    if (!tgt.bert_segs.empty() || !tgt.bert_ids.empty()) {
        stage = 2;
        const double tb = now_s();
        if (!compute_bert(eng, tgt, bert)) {
            err = "bert_feature(target) failed";
            return false;
        }
        tm.bert_ms += (now_s() - tb) * 1000;
    }

    int seed = p.seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int) (rd() & 0x7fffffff);
    }

    stage = 3;
    const double t_ar = now_s();
    std::vector<int32_t> sem;
    if (!eng.ar_generate(tgt.phones, bert, prm->phones, prm->bert, ref->codes, p.max_steps,
                         p.early_stop, (uint64_t) seed, sem, &p.sampler)) {
        err = "ar_generate failed";
        return false;
    }
    tm.ar_ms += (now_s() - t_ar) * 1000;

    stage = 4;
    const double t_sy = now_s();
    if (!eng.synthesize(sem, tgt.phones, ref->codes, prm->phones, ref->ge, ref->mel2, ref->mel2_T,
                        p.steps, p.cfg, (uint64_t) seed, wav, p.speed)) {
        err = "synthesize failed";
        return false;
    }
    tm.synth_ms += (now_s() - t_sy) * 1000;
    return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[synth_segment] stage=%d text='%s': %s\n", stage, p.text.c_str(), e.what());
        throw;
    }
}

inline bool synth_text(State & st, gsv_engine & eng, const SidecarCfg & side, const SynthParams & p,
                       std::vector<float> & out, SynthTiming & tm, std::string & err) {
    int stage = 0;
    try {
    stage = 1;
    const auto segs = split_text(p.text, p.split_method);
    tm.segments = (int) segs.size();
    out.clear();

    stage = 2;
    std::shared_ptr<const RefPrep> ref;
    std::shared_ptr<const PromptPrep> prm;
    if (!ensure_ref(st, st.voice, eng, side, p, ref, prm, tm, err)) return false;

    stage = 3;
    for (size_t i = 0; i < segs.size(); i++) {
        SynthParams sp = p;
        sp.text = segs[i];
        std::vector<float> wav;
        if (const char * dbg = std::getenv("GSV_DEBUG_SYNTH")) {
            std::fprintf(stderr, "[synth_text] seg %zu call (text='%.24s' steps=%d max=%d)\n", i,
                         sp.text.c_str(), sp.steps, sp.max_steps);
        }
        // 每个 segment 只做目标文本 G2P + AR + synth；prompt 已在上面缓存
        if (!synth_segment(st, eng, side, sp, ref, prm, wav, tm, err)) return false;
        if (const char * dbg = std::getenv("GSV_DEBUG_SYNTH")) {
            std::fprintf(stderr, "[synth_text] seg %zu wav=%zu out=%zu gap=%.3f\n", i, wav.size(),
                         out.size(), (double) p.fragment_interval);
        }
        out.insert(out.end(), wav.begin(), wav.end());
        if (i + 1 < segs.size() && p.fragment_interval > 0.0f) {
            out.insert(out.end(), (size_t) (p.fragment_interval * kSampleRate), 0.0f);
        }
    }
    return true;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[synth_text] stage=%d len=%zu text='%.40s' voice=%s: %s\n", stage,
                     p.text.size(), p.text.c_str(), st.voice.c_str(), e.what());
        throw;
    }
}

// 批量 AR：同一参考、多条目标文本，先分组跑 ar_generate_batch 再逐条 synthesize。
// 目前只用于 /batch 且 text_split_method=cut0（每条一段）的场景，语义同 repo 的
// parallel_infer（pad + mask，长度可不同）。
inline bool synth_batch_segments(State & st, const SidecarCfg & side, const SynthParams & base,
                                 const std::vector<std::string> & texts,
                                 std::vector<std::vector<float>> & wavs,
                                 std::vector<SynthTiming> & tms,
                                 std::vector<std::string> & errs,
                                 std::string & err, std::string & ec) {
    const size_t n = texts.size();
    wavs.assign(n, {});
    tms.assign(n, SynthTiming{});
    errs.assign(n, std::string{});

    if (!base.voice.empty() && base.voice != st.voice) {
        double ms = 0;
        if (!switch_voice(st, base.voice, err, ms)) { ec = "engine"; return false; }
    }
    auto eng = snapshot(st);
    if (!eng) { err = "engine not ready"; ec = "engine"; return false; }

    if (st.waiters.fetch_add(1) >= 32) {
        st.waiters.fetch_sub(1);
        st.n_busy++;
        err = "server busy (too many waiting requests)";
        ec = "busy";
        return false;
    }
    std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
    const bool got = lk.try_lock_for(std::chrono::seconds(300));
    st.waiters.fetch_sub(1);
    if (!got) { st.n_timeouts++; err = "queue timeout"; ec = "timeout"; return false; }

    SynthParams p = base;
    std::shared_ptr<const RefPrep> ref;
    std::shared_ptr<const PromptPrep> prm;
    SynthTiming pre;
    if (!ensure_ref(st, st.voice, *eng, side, p, ref, prm, pre, err)) { ec = "sidecar"; return false; }

    std::vector<G2POut> g2p(n);
    std::vector<std::vector<float>> berts(n);
    std::vector<size_t> valid;
    for (size_t i = 0; i < n; i++) {
        if (!sidecar_g2p(side, texts[i], p.text_lang, g2p[i], errs[i])) { ec = "sidecar"; continue; }
        if (g2p[i].phones.empty()) {
            errs[i] = "no phonemes produced for this text";
            continue;
        }
        if (!compute_bert(*eng, g2p[i], berts[i])) { errs[i] = "bert_feature failed"; continue; }
        tms[i].g2p_ms = pre.g2p_ms;
        tms[i].prep_ms = pre.prep_ms;
        valid.push_back(i);
    }

    const size_t kChunk = 8;
    for (size_t off = 0; off < valid.size(); off += kChunk) {
        const size_t end = std::min(valid.size(), off + kChunk);
        std::vector<gsv_engine::ar_batch_item> items;
        items.reserve(end - off);
        for (size_t k = off; k < end; k++) {
            gsv_engine::ar_batch_item it;
            it.phones = g2p[valid[k]].phones;
            it.bert = berts[valid[k]];
            items.push_back(std::move(it));
        }
        int seed = p.seed;
        if (seed < 0) {
            std::random_device rd;
            seed = (int) (rd() & 0x7fffffff);
        }
        std::vector<std::vector<int32_t>> sems;
        const double t_ar = now_s();
        if (!eng->ar_generate_batch(items, prm->phones, prm->bert, ref->codes, p.max_steps,
                                    p.early_stop, (uint64_t) seed, sems, nullptr, &p.sampler)) {
            const double ms = (now_s() - t_ar) * 1000;
            for (size_t k = off; k < end; k++) {
                errs[valid[k]] = "ar_generate_batch failed";
                tms[valid[k]].ar_ms = ms / (double) (end - off);
            }
            ec = "engine";
            continue;
        }
        const double ms = (now_s() - t_ar) * 1000 / (double) (end - off);
        for (size_t k = off; k < end; k++) {
            const size_t i = valid[k];
            tms[i].ar_ms += ms;
            std::vector<float> wav;
            if (!eng->synthesize(sems[k - off], g2p[i].phones, ref->codes, prm->phones, ref->ge,
                                 ref->mel2, ref->mel2_T, p.steps, p.cfg, (uint64_t) seed, wav,
                                 p.speed)) {
                errs[i] = "synthesize failed";
                continue;
            }
            wavs[i] = std::move(wav);
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (wavs[i].empty() && errs[i].empty()) errs[i] = "empty output";
    }
    return true;
}

// 通用批量：任意切句方式（cut0/cut1..cut5）都先把文本拆成 segment，
// 所有 segment 一起做 AR（每 8 条一批），再逐条 synthesize，最后按 item 拼回。
// 与 repo 的 parallel_infer 语义一致（pad + mask）；每条 segment 有独立 seed。
struct BatchUnit {
    size_t item = 0;
    std::string text;
    G2POut g2p;
    std::vector<float> bert;
    std::vector<int32_t> sem;
    std::vector<float> wav;
    std::string err;
    uint64_t seed = 0;
    double ar_ms = 0, synth_ms = 0;
};

inline bool synth_batch_texts(State & st, const SidecarCfg & side, const SynthParams & base,
                              const std::vector<std::string> & texts,
                              std::vector<std::vector<float>> & wavs,
                              std::vector<SynthTiming> & tms,
                              std::vector<std::string> & errs,
                              std::string & err, std::string & ec) {
    const size_t n = texts.size();
    wavs.assign(n, {});
    tms.assign(n, SynthTiming{});
    errs.assign(n, std::string{});

    if (!base.voice.empty() && base.voice != st.voice) {
        double ms = 0;
        if (!switch_voice(st, base.voice, err, ms)) { ec = "engine"; return false; }
    }
    auto eng = snapshot(st);
    if (!eng) { err = "engine not ready"; ec = "engine"; return false; }
    if (st.waiters.fetch_add(1) >= 32) {
        st.waiters.fetch_sub(1);
        st.n_busy++;
        err = "server busy (too many waiting requests)";
        ec = "busy";
        return false;
    }
    std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
    const bool got = lk.try_lock_for(std::chrono::seconds(300));
    st.waiters.fetch_sub(1);
    if (!got) { st.n_timeouts++; err = "queue timeout"; ec = "timeout"; return false; }

    SynthParams p = base;
    std::shared_ptr<const RefPrep> ref;
    std::shared_ptr<const PromptPrep> prm;
    SynthTiming pre;
    if (!ensure_ref(st, st.voice, *eng, side, p, ref, prm, pre, err)) { ec = "sidecar"; return false; }

    std::vector<BatchUnit> units;
    std::vector<std::vector<size_t>> item_units(n);
    for (size_t i = 0; i < n; i++) {
        const auto segs = split_text(texts[i], p.split_method);
        tms[i].segments = (int) segs.size();
        for (const auto & seg : segs) {
            BatchUnit u;
            u.item = i;
            u.text = seg;
            if (!sidecar_g2p(side, seg, p.text_lang, u.g2p, u.err)) { errs[i] = u.err; continue; }
            if (u.g2p.phones.empty()) {
                errs[i] = "no phonemes produced for this text";
                continue;
            }
            if (!compute_bert(*eng, u.g2p, u.bert)) { errs[i] = "bert_feature failed"; continue; }
            item_units[i].push_back(units.size());
            units.push_back(std::move(u));
        }
        tms[i].g2p_ms = pre.g2p_ms;
        tms[i].prep_ms = pre.prep_ms;
    }

    static std::atomic<uint64_t> seed_seq{1};
    // 按音素数排序后分桶：同批长度接近，减少 pad 浪费
    std::vector<size_t> order;
    order.reserve(units.size());
    for (size_t k = 0; k < units.size(); k++) order.push_back(k);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return units[a].g2p.phones.size() < units[b].g2p.phones.size();
    });
    for (size_t bi = 0; bi < order.size();) {
        const double base_len = (double) units[order[bi]].g2p.phones.size();
        size_t bj = bi + 1;
        while (bj < order.size() && bj - bi < 8 &&
               (double) units[order[bj]].g2p.phones.size() <= base_len * 1.6 + 2.0)
            bj++;
        std::vector<gsv_engine::ar_batch_item> items;
        std::vector<uint64_t> seeds;
        std::vector<size_t> idx;
        items.reserve(bj - bi);
        seeds.reserve(bj - bi);
        idx.reserve(bj - bi);
        for (size_t k = bi; k < bj; k++) {
            const size_t u = order[k];
            gsv_engine::ar_batch_item it;
            it.phones = units[u].g2p.phones;
            it.bert = units[u].bert;
            items.push_back(std::move(it));
            int sd = p.seed;
            if (sd < 0) sd = (int) ((now_s() * 1000.0) + (double) seed_seq.fetch_add(1));
            seeds.push_back((uint64_t) (uint32_t) sd);
            idx.push_back(u);
        }
        std::vector<std::vector<int32_t>> sems;
        const double t_ar = now_s();
        if (!eng->ar_generate_batch(items, prm->phones, prm->bert, ref->codes, p.max_steps,
                                    p.early_stop, 0, sems, &seeds, &p.sampler)) {
            for (size_t u : idx) units[u].err = "ar_generate_batch failed";
            ec = "engine";
        } else {
            const double per = (now_s() - t_ar) * 1000.0 / (double) idx.size();
            for (size_t k = 0; k < idx.size(); k++) {
                units[idx[k]].sem = std::move(sems[k]);
                units[idx[k]].seed = seeds[k];
                units[idx[k]].ar_ms += per;
            }
        }
        bi = bj;
    }

    for (auto & u : units) {
        if (!u.err.empty() || u.sem.empty()) continue;
        std::vector<float> wav;
        const double ts = now_s();
        if (!eng->synthesize(u.sem, u.g2p.phones, ref->codes, prm->phones, ref->ge, ref->mel2,
                             ref->mel2_T, p.steps, p.cfg, u.seed, wav, p.speed, p.skip_steps)) {
            u.err = "synthesize failed";
            continue;
        }
        u.synth_ms += (now_s() - ts) * 1000.0;
        u.wav = std::move(wav);
    }

    for (size_t i = 0; i < n; i++) {
        if (!errs[i].empty()) continue;
        bool bad = false;
        for (size_t k : item_units[i]) {
            if (!units[k].err.empty()) { errs[i] = units[k].err; bad = true; break; }
        }
        if (bad) continue;
        std::vector<float> merged;
        for (size_t j = 0; j < item_units[i].size(); j++) {
            const auto & u = units[item_units[i][j]];
            merged.insert(merged.end(), u.wav.begin(), u.wav.end());
            if (j + 1 < item_units[i].size() && p.fragment_interval > 0.0f)
                merged.insert(merged.end(), (size_t) (p.fragment_interval * kSampleRate), 0.0f);
            tms[i].ar_ms += u.ar_ms;
            tms[i].synth_ms += u.synth_ms;
            tms[i].phones += (int) u.g2p.phones.size();
        }
        if (merged.empty()) errs[i] = "empty output";
        else wavs[i] = std::move(merged);
    }
    if (const char * dbg = std::getenv("GSV_DEBUG_BATCH")) {
        double ar = 0, sy = 0;
        for (const auto & t : tms) { ar += t.ar_ms; sy += t.synth_ms; }
        std::fprintf(stderr, "[batch] units=%zu ar=%.0fms synth=%.0fms\n", units.size(), ar, sy);
    }
    return true;
}

// ---------------------------------------------------------------- request log
class ReqLog {
public:
    void init(const std::string & dir) {
        std::lock_guard<std::mutex> lk(mu_);
        dir_ = dir;
        std::error_code ec;
        fs::create_directories(dir_, ec);
    }

    void write(const json & j) {
        std::lock_guard<std::mutex> lk(mu_);
        if (dir_.empty()) return;
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char day[16];
        std::snprintf(day, sizeof(day), "%04d%02d%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
        std::ofstream f(fs::path(dir_) / (std::string("requests-") + day + ".jsonl"), std::ios::app);
        if (f) f << j.dump() << "\n";
    }

    std::string new_id() {
        static std::atomic<uint64_t> c{0};
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%llx-%llu",
                      (unsigned long long) (now_s() * 1000), (unsigned long long) c++);
        return buf;
    }

private:
    std::mutex mu_;
    std::string dir_;
};

// ---------------------------------------------------------------- 跨请求 AR 聚合
// 所有 AR 请求进同一个队列；后台线程把同一 key（voice/参考/prompt/采样参数）的请求
// 合成一批（≤8 条、3 ms 攒批窗口），跑一次 ar_generate_batch，再各自取回。
struct ArTask {
    std::string key;
    std::vector<int32_t> phones;
    std::vector<float> bert;
    std::vector<int32_t> prompt_phones;
    std::vector<int32_t> prompt_codes;
    std::vector<float> prompt_bert;
    int max_steps = 1500, early_stop = 400;
    gsv_sampler_cfg sampler;
    uint64_t seed = 0;
    std::promise<std::vector<int32_t>> prom;
};

struct ArBatcher {
    State * st = nullptr;
    std::deque<std::shared_ptr<ArTask>> q;
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
    std::thread th;
    std::atomic<uint64_t> n_batches{0}, n_tasks{0};
    static constexpr size_t kMaxBatch = 8;
    static constexpr int kGraceMs = 20;

    void start(State * s) { st = s; th = std::thread([this] { run(); }); }
    void shutdown() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
        std::deque<std::shared_ptr<ArTask>> left;
        { std::lock_guard<std::mutex> lk(m); left.swap(q); }
        for (auto & t : left) t->prom.set_value({});
    }

    std::future<std::vector<int32_t>> enqueue(const std::shared_ptr<ArTask> & t) {
        auto fut = t->prom.get_future();
        {
            std::lock_guard<std::mutex> lk(m);
            q.push_back(t);
        }
        cv.notify_all();
        return fut;
    }

    void run() {
        for (;;) {
            std::vector<std::shared_ptr<ArTask>> group;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || !q.empty(); });
                if (stop) return;
                group.push_back(q.front());
                q.pop_front();
                const std::string key = group[0]->key;
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(kGraceMs);
                while (group.size() < kMaxBatch) {
                    if (!cv.wait_until(lk, deadline, [&] { return stop || !q.empty(); })) break;
                    if (stop) break;
                    auto it = std::find_if(q.begin(), q.end(), [&](const std::shared_ptr<ArTask> & x) {
                        return x->key == key;
                    });
                    if (it == q.end()) break;
                    group.push_back(*it);
                    q.erase(it);
                }
                if (stop) {
                    for (auto & t : group) t->prom.set_value({});
                    return;
                }
            }
            execute(group);
        }
    }

    void execute(std::vector<std::shared_ptr<ArTask>> & group) {
        auto eng = snapshot(*st);
        if (!eng) {
            for (auto & t : group) t->prom.set_value({});
            return;
        }
        std::unique_lock<std::timed_mutex> lk(st->compute_mu, std::defer_lock);
        if (!lk.try_lock_for(std::chrono::seconds(300))) {
            for (auto & t : group) t->prom.set_value({});
            return;
        }
        std::vector<gsv_engine::ar_batch_item> items;
        std::vector<uint64_t> seeds;
        items.reserve(group.size());
        seeds.reserve(group.size());
        for (auto & t : group) {
            gsv_engine::ar_batch_item it;
            it.phones = t->phones;
            it.bert = t->bert;
            items.push_back(std::move(it));
            seeds.push_back(t->seed);
        }
        std::vector<std::vector<int32_t>> sems;
        const bool ok = eng->ar_generate_batch(items, group[0]->prompt_phones,
                                               group[0]->prompt_bert, group[0]->prompt_codes,
                                               group[0]->max_steps, group[0]->early_stop, 0, sems,
                                               &seeds, &group[0]->sampler);
        n_batches++;
        n_tasks += group.size();
        for (size_t i = 0; i < group.size(); i++) {
            if (ok && i < sems.size()) group[i]->prom.set_value(std::move(sems[i]));
            else group[i]->prom.set_value({});
        }
    }
};

inline std::string ar_key(const std::string & voice, const SynthParams & p) {
    char b[640];
    std::snprintf(b, sizeof(b), "%s|%s|%s|%s|%d|%d|%d|%.4f|%.4f|%.4f|%s|%d|%.4f|%.4f|%.3f",
                  voice.c_str(), p.ref_path.c_str(), p.prompt_text.c_str(),
                  p.prompt_lang.c_str(), p.max_steps, p.early_stop, p.sampler.top_k,
                  (double) p.sampler.top_p, (double) p.sampler.temperature,
                  (double) p.sampler.repetition_penalty, p.split_method.c_str(), p.steps,
                  (double) p.cfg, p.speed, p.fragment_interval);
    return b;
}

inline std::shared_ptr<ArTask> ar_make_task(const std::string & key,
                                            const std::vector<int32_t> & phones,
                                            const std::vector<float> & bert,
                                            const std::vector<int32_t> & prompt_phones,
                                            const std::vector<int32_t> & prompt_codes,
                                            const std::vector<float> & prompt_bert, int max_steps,
                                            int early_stop, const gsv_sampler_cfg & sampler,
                                            uint64_t seed) {
    auto t = std::make_shared<ArTask>();
    t->key = key;
    t->phones = phones;
    t->bert = bert;
    t->prompt_phones = prompt_phones;
    t->prompt_codes = prompt_codes;
    t->prompt_bert = prompt_bert;
    t->max_steps = max_steps;
    t->early_stop = early_stop;
    t->sampler = sampler;
    t->seed = seed;
    return t;
}

inline bool ar_take(std::future<std::vector<int32_t>> & fut, std::vector<int32_t> & sem,
                    std::string & err) {
    if (fut.wait_for(std::chrono::seconds(300)) != std::future_status::ready) {
        err = "ar batch timeout";
        return false;
    }
    sem = fut.get();
    if (sem.empty()) {
        err = "ar_generate failed";
        return false;
    }
    return true;
}

// 短锁：仅包住一次 engine 调用
inline bool lock_engine(State & st, std::unique_lock<std::timed_mutex> & lk, std::string & err) {
    lk = std::unique_lock<std::timed_mutex>(st.compute_mu, std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(300))) {
        st.n_timeouts++;
        err = "engine lock timeout";
        return false;
    }
    return true;
}

inline bool ensure_ref_locked(State & st, const SidecarCfg & side, const SynthParams & p,
                              std::shared_ptr<const RefPrep> & ref,
                              std::shared_ptr<const PromptPrep> & prm, SynthTiming & pre,
                              std::string & err) {
    auto eng = snapshot(st);
    if (!eng) { err = "engine not ready"; return false; }
    std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(300))) { st.n_timeouts++; err = "engine lock timeout"; return false; }
    return ensure_ref(st, st.voice, *eng, side, p, ref, prm, pre, err);
}

// 单段：g2p -> bert(短锁) -> AR(队列) -> synth(短锁)
inline bool synth_segment_q(State & st, const SidecarCfg & side, const SynthParams & p,
                            const std::shared_ptr<const RefPrep> & ref,
                            const std::shared_ptr<const PromptPrep> & prm,
                            std::vector<float> & wav, SynthTiming & tm, std::string & err) {
    const double t0 = now_s();
    G2POut tgt;
    if (!sidecar_g2p(side, p.text, p.text_lang, tgt, err)) return false;
    tm.g2p_ms += (now_s() - t0) * 1000;
    tm.phones += (int) tgt.phones.size();
    if (tgt.phones.empty()) {
        err = "no phonemes produced for this text (unsupported characters or empty input)";
        return false;
    }

    std::vector<float> bert;
    if (!tgt.bert_segs.empty() || !tgt.bert_ids.empty()) {
        auto eng = snapshot(st);
        if (!eng) { err = "engine not ready"; return false; }
        std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
        if (!lk.try_lock_for(std::chrono::seconds(300))) { st.n_timeouts++; err = "engine lock timeout"; return false; }
        const double tb = now_s();
        if (!compute_bert(*eng, tgt, bert)) { err = "bert_feature(target) failed"; return false; }
        tm.bert_ms += (now_s() - tb) * 1000;
    }

    int seed = p.seed;
    if (seed < 0) {
        std::random_device rd;
        seed = (int) (rd() & 0x7fffffff);
    }
    const double t_ar = now_s();
    std::vector<int32_t> sem;
    if (st.ar_batcher) {
        auto task = ar_make_task(ar_key(st.voice, p), tgt.phones, bert, prm->phones, ref->codes,
                                 prm->bert, p.max_steps, p.early_stop, p.sampler, (uint64_t) seed);
        auto fut = st.ar_batcher->enqueue(task);
        if (!ar_take(fut, sem, err)) return false;
    } else {
        auto eng2 = snapshot(st);
        std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
        if (!lk.try_lock_for(std::chrono::seconds(300))) { st.n_timeouts++; err = "engine lock timeout"; return false; }
        if (!eng2->ar_generate(tgt.phones, bert, prm->phones, prm->bert, ref->codes, p.max_steps,
                               p.early_stop, (uint64_t) seed, sem, &p.sampler)) {
            err = "ar_generate failed";
            return false;
        }
    }
    tm.ar_ms += (now_s() - t_ar) * 1000;

    auto eng3 = snapshot(st);
    if (!eng3) { err = "engine not ready"; return false; }
    std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
    if (!lk.try_lock_for(std::chrono::seconds(300))) { st.n_timeouts++; err = "engine lock timeout"; return false; }
    const int dbn = p.dbcache_fn >= 0 ? p.dbcache_fn : (p.steps >= 16 ? 8 : 0);
    eng3->set_dbcache(dbn, p.dbcache_thr, p.dbcache_warmup, p.dbcache_ds);
    const double t_sy = now_s();
    if (!eng3->synthesize(sem, tgt.phones, ref->codes, prm->phones, ref->ge, ref->mel2, ref->mel2_T,
                          p.steps, p.cfg, (uint64_t) seed, wav, p.speed)) {
        err = "synthesize failed";
        return false;
    }
    tm.synth_ms += (now_s() - t_sy) * 1000;
    {
        int ch = 0, cm = 0;
        eng3->dbcache_stats(ch, cm);
        tm.cache_hits += ch;
        tm.cache_misses += cm;
    }
    return true;
}

// 整段：ensure_ref(短锁) + 逐 segment
inline bool synth_text_q(State & st, const SidecarCfg & side, const SynthParams & p,
                         std::vector<float> & out, SynthTiming & tm, std::string & err) {
    const auto segs = split_text(p.text, p.split_method);
    tm.segments = (int) segs.size();
    out.clear();
    std::shared_ptr<const RefPrep> ref;
    std::shared_ptr<const PromptPrep> prm;
    if (!ensure_ref_locked(st, side, p, ref, prm, tm, err)) return false;
    for (size_t i = 0; i < segs.size(); i++) {
        SynthParams sp = p;
        sp.text = segs[i];
        std::vector<float> wav;
        if (!synth_segment_q(st, side, sp, ref, prm, wav, tm, err)) return false;
        out.insert(out.end(), wav.begin(), wav.end());
        if (i + 1 < segs.size() && p.fragment_interval > 0.0f)
            out.insert(out.end(), (size_t) (p.fragment_interval * kSampleRate), 0.0f);
    }
    return true;
}

// 批量：所有 segment 的 AR 一次性入队（跨请求聚合），再逐段 synth
inline bool synth_batch_texts_q(State & st, const SidecarCfg & side, const SynthParams & base,
                                const std::vector<std::string> & texts,
                                std::vector<std::vector<float>> & wavs,
                                std::vector<SynthTiming> & tms, std::vector<std::string> & errs,
                                std::string & err, std::string & ec) {
    const size_t n = texts.size();
    wavs.assign(n, {});
    tms.assign(n, SynthTiming{});
    errs.assign(n, std::string{});
    if (st.mock_engine) {
        for (size_t i = 0; i < n; i++) {
            wavs[i].assign(12000, 0.1f);   // 0.25 s @ 48 kHz，确定性
            tms[i].segments = 1;
            tms[i].phones = 3;
        }
        return true;
    }
    if (!base.voice.empty() && base.voice != st.voice) {
        double ms = 0;
        if (!switch_voice(st, base.voice, err, ms)) { ec = "engine"; return false; }
    }
    SynthParams p = base;
    resolve_auto_params(st, p);
    std::shared_ptr<const RefPrep> ref;
    std::shared_ptr<const PromptPrep> prm;
    SynthTiming pre;
    if (!ensure_ref_locked(st, side, p, ref, prm, pre, err)) { ec = "sidecar"; return false; }

    std::vector<BatchUnit> units;
    std::vector<std::vector<size_t>> item_units(n);
    std::vector<double> t_g2p(n, 0), t_bert(n, 0), t_ar(n, 0), t_syn(n, 0), t_cond(n, 0), t_voc2(n, 0);
    for (size_t i = 0; i < n; i++) {
        const auto segs = split_text(texts[i], p.split_method);
        tms[i].segments = (int) segs.size();
        for (const auto & seg : segs) {
            BatchUnit u;
            u.item = i;
            u.text = seg;
            const double tg = now_s();
            if (!sidecar_g2p(side, seg, p.text_lang, u.g2p, u.err)) { errs[i] = u.err; continue; }
            t_g2p[i] += (now_s() - tg) * 1000.0;
            if (u.g2p.phones.empty()) { errs[i] = "no phonemes produced for this text"; continue; }
            if (const char * dbg = std::getenv("GSV_DEBUG_BATCH")) {
                std::fprintf(stderr, "[batch_q] unit i=%zu lang='%s' phones=%zu text='%.20s'\n", i,
                             p.text_lang.c_str(), u.g2p.phones.size(), seg.c_str());
            }
            item_units[i].push_back(units.size());
            units.push_back(std::move(u));
        }
        tms[i].prep_ms = pre.prep_ms;
    }

    static std::atomic<uint64_t> seed_seq{1};
    std::vector<std::future<std::vector<int32_t>>> futs(units.size());
    for (size_t k = 0; k < units.size(); k++) {
        if (units[k].err.empty() && !units[k].g2p.phones.empty()) {
            auto eng = snapshot(st);
            std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
            if (!lk.try_lock_for(std::chrono::seconds(300))) { units[k].err = "engine lock timeout"; continue; }
            const double tb = now_s();
            if (!compute_bert(*eng, units[k].g2p, units[k].bert)) { units[k].err = "bert_feature failed"; continue; }
            t_bert[units[k].item] += (now_s() - tb) * 1000.0;
        }
    }
    // 先全部入队（enqueue 很快），让批处理线程尽量攒满
    for (size_t k = 0; k < units.size(); k++) {
        if (!units[k].err.empty()) continue;
        int sd = p.seed;
        if (sd < 0) sd = (int) ((now_s() * 1000.0) + (double) seed_seq.fetch_add(1));
        units[k].seed = (uint64_t) (uint32_t) sd;
        if (st.ar_batcher) {
            auto task = ar_make_task(ar_key(st.voice, p), units[k].g2p.phones, units[k].bert,
                                     prm->phones, ref->codes, prm->bert, p.max_steps, p.early_stop,
                                     p.sampler, units[k].seed);
            futs[k] = st.ar_batcher->enqueue(task);
        }
    }
    for (size_t k = 0; k < units.size(); k++) {
        if (units[k].err.empty() && futs[k].valid()) {
            std::string e;
            const double ta = now_s();
            if (!ar_take(futs[k], units[k].sem, e)) units[k].err = e;
            t_ar[units[k].item] += (now_s() - ta) * 1000.0;
        }
    }
    // 逐段 synth
    // 逐段 synth：>=2 段时 DiT（主线程）与 vocoder（副线程）流水线重叠。
    // 数值与串行完全一致（同一 mel / 同一 vocoder），只改变执行顺序。
    const bool pipeline = units.size() >= 2 && std::getenv("GSV_NO_PIPELINE") == nullptr;
    std::vector<double> t_voc(n, 0);
    {
        std::mutex qm;
        std::condition_variable qcv;
        std::deque<std::pair<size_t, std::pair<std::vector<float>, int>>> melq;
        bool feed_done = false;
        auto eng = snapshot(st);
        std::thread voc_th;
        if (pipeline && eng) {
            voc_th = std::thread([&] {
                for (;;) {
                    std::pair<size_t, std::pair<std::vector<float>, int>> item;
                    {
                        std::unique_lock<std::mutex> lk(qm);
                        qcv.wait(lk, [&] { return feed_done || !melq.empty(); });
                        if (melq.empty() && feed_done) return;
                        item = std::move(melq.front());
                        melq.pop_front();
                    }
                    BatchUnit & u = units[item.first];
                    std::vector<float> wav;
                    const double tv = now_s();
                    if (eng->vocode_mel(item.second.first, item.second.second, wav))
                        u.wav = std::move(wav);
                    else
                        u.err = "vocoder failed";
                    t_voc[u.item] += (now_s() - tv) * 1000.0;
                }
            });
        }
        for (size_t k = 0; k < units.size(); k++) {
            BatchUnit & u = units[k];
            if (!u.err.empty() || u.sem.empty()) continue;
            auto eng2 = eng ? eng : snapshot(st);
            if (!eng2) { u.err = "engine not ready"; continue; }
            std::vector<float> mel;
            int melT = 0;
            {
                std::unique_lock<std::timed_mutex> lk(st.compute_mu, std::defer_lock);
                if (!lk.try_lock_for(std::chrono::seconds(300))) { u.err = "engine lock timeout"; continue; }
                const int dbn = p.dbcache_fn >= 0 ? p.dbcache_fn : (p.steps >= 16 ? 8 : 0);
                eng2->set_dbcache(dbn, p.dbcache_thr, p.dbcache_warmup, p.dbcache_ds);
                const double ts = now_s();
                try {
                    if (!eng2->synthesize_mel(u.sem, u.g2p.phones, ref->codes, prm->phones, ref->ge,
                                              ref->mel2, ref->mel2_T, p.steps, p.cfg, u.seed, mel, melT,
                                              p.speed, p.skip_steps)) {
                        u.err = "synthesize failed";
                        continue;
                    }
                } catch (const std::exception & e) {
                    std::fprintf(stderr,
                                 "[batch_q] synthesize_mel threw: phones=%zu mel2T=%d steps=%d: %s\n",
                                 u.g2p.phones.size(), ref->mel2_T, p.steps, e.what());
                    u.err = std::string("synthesize exception: ") + e.what();
                    continue;
                }
                t_syn[u.item] += (now_s() - ts) * 1000.0;
                {
                    double cm = 0, dm = 0;
                    eng2->last_stage_ms(cm, dm);
                    t_cond[u.item] += cm;
                    t_voc2[u.item] += dm;
                }
                int ch = 0, cm = 0;
                eng2->dbcache_stats(ch, cm);
                tms[u.item].cache_hits += ch;
                tms[u.item].cache_misses += cm;
            }
            if (voc_th.joinable()) {
                {
                    std::lock_guard<std::mutex> lk(qm);
                    melq.emplace_back(k, std::make_pair(std::move(mel), melT));
                }
                qcv.notify_one();
            } else {
                std::vector<float> wav;
                const double tv = now_s();
                if (eng2->vocode_mel(mel, melT, wav)) u.wav = std::move(wav);
                else u.err = "vocoder failed";
                t_voc[u.item] += (now_s() - tv) * 1000.0;
            }
        }
        if (voc_th.joinable()) {
            {
                std::lock_guard<std::mutex> lk(qm);
                feed_done = true;
            }
            qcv.notify_all();
            voc_th.join();
        }
    }
    for (size_t i = 0; i < n; i++) {
        tms[i].g2p_ms = t_g2p[i];
        tms[i].bert_ms = t_bert[i];
        tms[i].ar_ms = t_ar[i];
        tms[i].cond_ms = t_cond[i];
        tms[i].dit_ms = t_voc2[i];
        tms[i].voc_ms = t_voc[i];
        tms[i].synth_ms = t_syn[i] + t_voc[i];
    }
    for (size_t i = 0; i < n; i++) {
        if (!errs[i].empty()) continue;
        bool bad = false;
        for (size_t k : item_units[i])
            if (!units[k].err.empty()) { errs[i] = units[k].err; bad = true; break; }
        if (bad) continue;
        std::vector<float> merged;
        for (size_t j = 0; j < item_units[i].size(); j++) {
            const auto & u = units[item_units[i][j]];
            merged.insert(merged.end(), u.wav.begin(), u.wav.end());
            if (j + 1 < item_units[i].size() && p.fragment_interval > 0.0f)
                merged.insert(merged.end(), (size_t) (p.fragment_interval * kSampleRate), 0.0f);
            tms[i].phones += (int) u.g2p.phones.size();
        }
        if (merged.empty()) errs[i] = "empty output";
        else wavs[i] = std::move(merged);
    }
    return true;
}

// ---------------------------------------------------------------- 请求级调度
// /tts 与 /batch 的每个文本都作为一条请求进队列；同一 key（voice/参考/prompt/采样参数）
// 的请求攒成一组（≤8 条、25 ms 窗口），整组一起做 G2P/BERT/AR，再逐条 synth。
// 这样 AR 的批处理不再受各请求 G2P 完成时间分散的影响。
struct SynthResult {
    std::vector<float> wav;
    std::string err;
    SynthTiming tm;
};

struct SynthReq {
    SynthParams p;
    std::string text;
    std::promise<SynthResult> prom;
};

struct SynthScheduler {
    State * st = nullptr;
    SidecarCfg side;
    std::deque<std::shared_ptr<SynthReq>> q;
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::vector<std::shared_ptr<SynthReq>>> gq;   // groups ready to execute
    std::mutex gm;
    std::condition_variable gcv;
    bool stop = false;
    std::thread form;
    std::vector<std::thread> pool;
    int n_workers = 2;
    std::atomic<uint64_t> n_batches{0}, n_texts{0};
    static constexpr size_t kMaxTexts = 8;
    static constexpr int kGraceMs = 25;

    void start(State * s, const SidecarCfg & sc) {
        st = s;
        side = sc;
        if (const char * w = std::getenv("GSV_SCHED_WORKERS")) {
            const int n = std::atoi(w);
            if (n >= 1 && n <= 4) n_workers = n;
        }
        form = std::thread([this] { run(); });
        for (int i = 0; i < n_workers; i++) pool.emplace_back([this] { worker(); });
    }
    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        gcv.notify_all();
        if (form.joinable()) form.join();
        for (auto & t : pool) if (t.joinable()) t.join();
        std::deque<std::shared_ptr<SynthReq>> left;
        {
            std::lock_guard<std::mutex> lk(m);
            left.swap(q);
        }
        for (auto & r : left) r->prom.set_value(SynthResult{{}, "server shutting down"});
        std::deque<std::vector<std::shared_ptr<SynthReq>>> gleft;
        {
            std::lock_guard<std::mutex> lk(gm);
            gleft.swap(gq);
        }
        for (auto & g : gleft)
            for (auto & r : g) r->prom.set_value(SynthResult{{}, "server shutting down"});
    }
    std::future<SynthResult> submit(const SynthParams & p, const std::string & text) {
        auto r = std::make_shared<SynthReq>();
        r->p = p;
        r->text = text;
        auto fut = r->prom.get_future();
        {
            std::lock_guard<std::mutex> lk(m);
            q.push_back(r);
        }
        cv.notify_all();
        return fut;
    }

    // formation thread: group requests, hand groups to the worker pool
    void run() {
        for (;;) {
            std::vector<std::shared_ptr<SynthReq>> group;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || !q.empty(); });
                if (stop) return;
                group.push_back(q.front());
                q.pop_front();
                const std::string key = ar_key(st->voice, group[0]->p);
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(kGraceMs);
                while (group.size() < kMaxTexts) {
                    if (!cv.wait_until(lk, deadline, [&] { return stop || !q.empty(); })) break;
                    if (stop) break;
                    auto it = std::find_if(q.begin(), q.end(), [&](const std::shared_ptr<SynthReq> & x) {
                        return ar_key(st->voice, x->p) == key;
                    });
                    if (it == q.end()) break;
                    group.push_back(*it);
                    q.erase(it);
                }
                if (stop) {
                    for (auto & r : group) r->prom.set_value(SynthResult{{}, "server shutting down"});
                    return;
                }
            }
            {
                std::lock_guard<std::mutex> lk(gm);
                gq.push_back(std::move(group));
            }
            gcv.notify_one();
        }
    }

    // worker: runs one group at a time; several workers overlap vocoder of one group
    // with AR/DiT of the next (vocoder calls are serialized inside the engine).
    void worker() {
        for (;;) {
            std::vector<std::shared_ptr<SynthReq>> group;
            {
                std::unique_lock<std::mutex> lk(gm);
                gcv.wait(lk, [&] { return stop || !gq.empty(); });
                if (gq.empty()) return;
                group = std::move(gq.front());
                gq.pop_front();
            }
            execute(group);
        }
    }

    void execute(std::vector<std::shared_ptr<SynthReq>> & group) {
        std::vector<std::string> texts;
        texts.reserve(group.size());
        for (auto & r : group) texts.push_back(r->text);
        std::vector<std::vector<float>> wavs;
        std::vector<SynthTiming> tms;
        std::vector<std::string> errs;
        std::string err, ec;
        bool ok = false;
        try {
            ok = synth_batch_texts_q(*st, side, group[0]->p, texts, wavs, tms, errs, err, ec);
        } catch (const std::exception & e) {
            err = std::string("exception: ") + e.what();
        }
        n_batches++;
        n_texts += group.size();
        for (size_t i = 0; i < group.size(); i++) {
            if (ok && i < wavs.size() && !wavs[i].empty()) {
                group[i]->prom.set_value(SynthResult{std::move(wavs[i]), "",
                                                     (i < tms.size() ? tms[i] : SynthTiming{})});
            } else {
                const std::string e = (i < errs.size() && !errs[i].empty()) ? errs[i] : err;
                group[i]->prom.set_value(SynthResult{{}, e.empty() ? "synthesis failed" : e});
            }
        }
    }
};



}  // namespace svc
