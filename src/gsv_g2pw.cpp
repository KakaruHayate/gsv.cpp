#include "gsv_g2pw.h"

#include "onnxruntime_c_api.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static const int   G_LABELS = 1305;    // get_phoneme_labels(polyphonic_chars) 的输出数
static const int   G_NCHAR  = 3582;    // chars = sorted(char2phonemes.keys())
static const int   G_NPOS   = 11;      // num_pos_tags
static const float G_EPS    = 1e-6f;

// ============================ UTF-8 <-> code point ============================
// BERT 的 "字符" 是 unicode code point (中文一字一 cp), 而 repo 的 Python 侧按 str 索引
// (也是 cp)。故所有 query_id/char_id 都以 **code point** 为单位。
static bool utf8_next(const std::string & s, size_t & i, uint32_t & cp) {
    if (i >= s.size()) return false;
    const unsigned char c = (unsigned char) s[i];
    int n = 1;
    uint32_t v = 0;
    if (c < 0x80) { n = 1; v = c; }
    else if ((c & 0xE0) == 0xC0) { n = 2; v = c & 0x1Fu; }
    else if ((c & 0xF0) == 0xE0) { n = 3; v = c & 0x0Fu; }
    else if ((c & 0xF8) == 0xF0) { n = 4; v = c & 0x07u; }
    else { i++; cp = 0xFFFD; return true; }                       // 非法字节 -> 替换符
    if (i + n > s.size()) { i = s.size(); cp = 0xFFFD; return true; }
    for (int k = 1; k < n; k++) v = (v << 6) | ((unsigned char) s[i + k] & 0x3Fu);
    i += (size_t) n;
    cp = v;
    return true;
}

static void utf8_to_cps(const std::string & s, std::vector<uint32_t> & cps) {
    cps.clear();
    size_t i = 0;
    uint32_t cp = 0;
    while (utf8_next(s, i, cp)) cps.push_back(cp);
}

static void utf8_append(std::string & s, uint32_t cp) {
    if (cp < 0x80) s.push_back((char) cp);
    else if (cp < 0x800) { s.push_back((char) (0xC0 | (cp >> 6))); s.push_back((char) (0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { s.push_back((char) (0xE0 | (cp >> 12))); s.push_back((char) (0x80 | ((cp >> 6) & 0x3F))); s.push_back((char) (0x80 | (cp & 0x3F))); }
    else { s.push_back((char) (0xF0 | (cp >> 18))); s.push_back((char) (0x80 | ((cp >> 12) & 0x3F))); s.push_back((char) (0x80 | ((cp >> 6) & 0x3F))); s.push_back((char) (0x80 | (cp & 0x3F))); }
}

// ============================ BERT tokenizer ============================
// repo text/g2pw/utils.py::wordize_and_map + tokenize_and_map + HF BertTokenizer(fast)
// wordize_and_map: 连续 ASCII 字母数字成一个 "word", 其余**每个 code point 一个 word**。
// 每个 word 单独过 WordPiece; [UNK] 或空 -> 记一个 [UNK] token 且覆盖整个 word。
namespace {
struct g2pw_impl {
    // ---- ORT ----
    HMODULE dll = nullptr;
    const OrtApi * api = nullptr;
    const OrtApiBase * base = nullptr;
    OrtEnv * env = nullptr;
    OrtSessionOptions * so = nullptr;
    OrtSession * sess = nullptr;
    OrtMemoryInfo * mem = nullptr;
    char ep[64] = "CPUExecutionProvider";

    // ---- 词表/标签 ----
    std::map<std::string, int> tok2id;      // vocab.txt
    std::vector<std::string> labels;        // [1305] 注音符号
    std::vector<std::string> chars;         // [3582] sorted(char2phonemes) —— char2id 的下标
    std::map<std::string, int> char2id;
    std::vector<std::vector<int>> char_phoneme_masks;   // [n_chars][1305]
    int id_cls = -1, id_sep = -1, id_unk = -1, id_pad = -1;
    int n_threads = 0, max_len = 512;
    bool verbose = false;
    bool ready = false;

    ~g2pw_impl() {
        if (sess && api) api->ReleaseSession(sess);
        if (so && api)   api->ReleaseSessionOptions(so);
        if (env && api)  api->ReleaseEnv(env);
        if (mem && api)  api->ReleaseMemoryInfo(mem);
        if (dll)         FreeLibrary(dll);
    }
};
}  // namespace

// p 是 void* (头文件里刻意不暴露 impl); 每个成员函数入口转成 g2pw_impl&
static inline g2pw_impl & P(void * p) { return *static_cast<g2pw_impl *>(p); }
static inline const g2pw_impl & P(const void * p) { return *static_cast<const g2pw_impl *>(p); }

static void ort_check2(const OrtApi * api, OrtStatus * st, const char * what) {
    if (!st) return;
    fprintf(stderr, "[gsv_g2pw] %s failed: %s\n", what, api->GetErrorMessage(st));
    api->ReleaseStatus(st);
}

gsv_g2pw::gsv_g2pw() : p(new g2pw_impl()) {}
gsv_g2pw::~gsv_g2pw() { delete static_cast<g2pw_impl *>(p); }
int gsv_g2pw::n_labels() const { return G_LABELS; }
const char * gsv_g2pw::ep_name() const { return P(p).ep; }
const char * gsv_g2pw::label(int idx) const {
    const g2pw_impl & s = P(p);
    if (idx < 0 || idx >= (int) s.labels.size()) return nullptr;
    return s.labels[(size_t) idx].c_str();
}

// ---- repo text/g2pw/dataset.py::get_phoneme_labels ----
// labels = sorted(set(phoneme)); char2phonemes[char] = [labels.index(ph) for ph]
// chars  = sorted(char2phonemes.keys())
static bool load_g2pw_tables(const std::string & dir, g2pw_impl & s) {
    const std::string poly = dir + "/POLYPHONIC_CHARS.txt";
    std::ifstream f(poly.c_str());
    if (!f) { fprintf(stderr, "[gsv_g2pw] 打不开 %s\n", poly.c_str()); return false; }
    std::vector<std::pair<std::string, std::string>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        rows.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    f.close();
    if (rows.empty()) { fprintf(stderr, "[gsv_g2pw] %s 为空\n", poly.c_str()); return false; }

    std::vector<std::string> uniq;
    for (auto & r : rows) uniq.push_back(r.second);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    s.labels = uniq;
    if ((int) s.labels.size() != G_LABELS) {
        fprintf(stderr, "[gsv_g2pw] labels 数 = %zu, 期望 %d\n", s.labels.size(), G_LABELS);
        return false;
    }
    std::map<std::string, std::vector<int>> char2ph;
    for (auto & r : rows) {
        const int li = (int) (std::lower_bound(uniq.begin(), uniq.end(), r.second) - uniq.begin());
        char2ph[r.first].push_back(li);
    }
    for (auto & kv : char2ph) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }
    if ((int) char2ph.size() != G_NCHAR) {
        fprintf(stderr, "[gsv_g2pw] char2phonemes 数 = %zu, 期望 %d\n", char2ph.size(), G_NCHAR);
        return false;
    }
    s.chars.clear(); s.char2id.clear();
    for (auto & kv : char2ph) {                      // map 已按 key 排序 == sorted(...)
        s.char2id[kv.first] = (int) s.chars.size();
        s.chars.push_back(kv.first);
    }
    s.char_phoneme_masks.assign(s.chars.size(), std::vector<int>(G_LABELS, 0));
    for (size_t ci = 0; ci < s.chars.size(); ci++) {
        for (int li : char2ph[s.chars[ci]]) s.char_phoneme_masks[ci][li] = 1;
    }
    return true;
}

static bool load_vocab(const std::string & path, g2pw_impl & s) {
    std::ifstream f(path.c_str());
    if (!f) { fprintf(stderr, "[gsv_g2pw] 打不开 vocab %s\n", path.c_str()); return false; }
    std::string line;
    int idx = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (s.tok2id.find(line) == s.tok2id.end()) s.tok2id[line] = idx;
        if (line == "[CLS]") s.id_cls = idx;
        else if (line == "[SEP]") s.id_sep = idx;
        else if (line == "[UNK]") s.id_unk = idx;
        else if (line == "[PAD]") s.id_pad = idx;
        idx++;
    }
    if (s.id_cls < 0 || s.id_sep < 0 || s.id_unk < 0) {
        fprintf(stderr, "[gsv_g2pw] vocab 缺 [CLS]/[SEP]/[UNK] (cls=%d sep=%d unk=%d)\n", s.id_cls, s.id_sep, s.id_unk);
        return false;
    }
    return true;
}

// ---- WordPiece: 最长匹配, 最长 100 字符的剩余段 -> [UNK] (HF BasicTokenizer 的 do_lower_case
// 由 .lower() 在调用侧已做; BERT 对 CJK 是逐字查表, 天然一字一 token) ----
static void wordpiece(const std::string & w, const g2pw_impl & s, std::vector<int> & out) {
    out.clear();
    if (w.empty()) return;
    // 中文/非 ASCII: 逐 code point 查表 (bert-base-chinese 的中文都在词表里)
    size_t i = 0;
    std::vector<uint32_t> cps;
    utf8_to_cps(w, cps);
    bool all_ascii = true;
    for (uint32_t c : cps) if (c >= 0x80) { all_ascii = false; break; }

    if (!all_ascii) {
        for (uint32_t c : cps) {
            std::string t; utf8_append(t, c);
            auto it = s.tok2id.find(t);
            out.push_back(it != s.tok2id.end() ? it->second : s.id_unk);
        }
        return;
    }
    // ASCII 单词: 标准 WordPiece 最长匹配 + "##" 续接
    size_t start = 0;
    const size_t n = w.size();
    while (start < n) {
        size_t end = n;
        int found = s.id_unk;
        size_t cur_start = start;
        while (cur_start < end) {
            std::string sub = w.substr(cur_start, end - cur_start);
            if (cur_start > start) sub = "##" + sub;
            auto it = s.tok2id.find(sub);
            if (it != s.tok2id.end()) { found = it->second; break; }
            end--;
        }
        if (found == s.id_unk) {                    // 整段未命中 -> 一个 [UNK] 覆盖到 end
            out.push_back(s.id_unk);
            start = end > start ? end : start + 1;
            continue;
        }
        out.push_back(found);
        start = end;
    }
    (void) i;
}

// ---- repo utils.py::wordize_and_map + tokenize_and_map ----
// 返回 tokens 与 text2token (每个 code point -> token 下标)
static void tokenize_and_map(const std::string & text, const g2pw_impl & s,
                             std::vector<int> & tokens, std::vector<int> & text2token) {
    std::vector<uint32_t> cps;
    utf8_to_cps(text, cps);
    const size_t ncp = cps.size();
    tokens.clear();
    text2token.assign(ncp, 0);

    // 切成 "words": 连续 [a-zA-Z0-9] 一组, 其余逐 cp
    size_t i = 0;
    while (i < ncp) {
        const uint32_t c = cps[i];
        const bool is_alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        if (is_alnum) {
            size_t j = i;
            while (j < ncp) {
                const uint32_t d = cps[j];
                if ((d >= 'a' && d <= 'z') || (d >= 'A' && d <= 'Z') || (d >= '0' && d <= '9')) j++;
                else break;
            }
            std::string w;
            for (size_t k = i; k < j; k++) utf8_append(w, cps[k]);
            // ASCII 小写化 (repo 侧 texts[idx].lower())
            for (char & ch : w) if (ch >= 'A' && ch <= 'Z') ch = (char) (ch - 'A' + 'a');
            std::vector<int> wp;
            wordpiece(w, s, wp);
            if (wp.empty() || (wp.size() == 1 && wp[0] == s.id_unk)) {
                // 空或整词 UNK -> 一个 [UNK] 覆盖整个 word
                for (size_t k = i; k < j; k++) text2token[k] = (int) tokens.size();
                tokens.push_back(s.id_unk);
            } else {
                for (size_t k = i; k < j; k++) text2token[k] = (int) tokens.size();
                for (int t : wp) tokens.push_back(t);
            }
            i = j;
        } else {
            std::string t; utf8_append(t, c);
            for (size_t k = i; k <= i; k++) text2token[k] = (int) tokens.size();
            auto it = s.tok2id.find(t);
            tokens.push_back(it != s.tok2id.end() ? it->second : s.id_unk);
            i++;
        }
    }
}

gsv_g2pw * gsv_g2pw::load(const gsv_g2pw_cfg & cfg) {
    gsv_g2pw * m = new gsv_g2pw();
    g2pw_impl & s = P(m->p);
    s.verbose = cfg.verbose;
    s.n_threads = cfg.n_threads;
    s.max_len = cfg.max_len > 0 ? cfg.max_len : 512;

    if (!load_g2pw_tables(cfg.model_dir, s)) { delete m; return nullptr; }
    if (!load_vocab(cfg.vocab_txt, s)) { delete m; return nullptr; }

    // ---- ORT ----
    std::vector<std::string> cands;
    auto add_cand = [&](const std::string & d) {
        if (d.empty()) return;
        std::string full = d;
        const DWORD a = GetFileAttributesA(d.c_str());
        if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY)) full = d + "\\onnxruntime.dll";
        cands.push_back(full);
    };
    if (!cfg.ort_dll.empty()) add_cand(cfg.ort_dll);
    else {
        add_cand(".venv-wg\\Lib\\site-packages\\onnxruntime\\capi");
        add_cand(".venv-ort\\Lib\\site-packages\\onnxruntime\\capi");
    }
    for (const std::string & c : cands) { s.dll = LoadLibraryA(c.c_str()); if (s.dll) break; }
    if (!s.dll) s.dll = LoadLibraryA("onnxruntime.dll");
    if (!s.dll) { fprintf(stderr, "[gsv_g2pw] onnxruntime.dll 未找到\n"); delete m; return nullptr; }
    {
        char self[MAX_PATH] = {0};
        GetModuleFileNameA(s.dll, self, MAX_PATH);
        std::string dir(self);
        const size_t sl = dir.find_last_of("\\/");
        if (sl != std::string::npos)
            for (const char * dep : { "dxcompiler.dll", "dxil.dll" }) {
                const std::string dp = dir.substr(0, sl) + "\\" + dep;
                if (GetFileAttributesA(dp.c_str()) != INVALID_FILE_ATTRIBUTES) LoadLibraryA(dp.c_str());
            }
    }
    using GetApiBaseFn = const OrtApiBase * (ORT_API_CALL *)();
    auto gb = (GetApiBaseFn) GetProcAddress(s.dll, "OrtGetApiBase");
    if (!gb) { fprintf(stderr, "[gsv_g2pw] OrtGetApiBase 未导出\n"); delete m; return nullptr; }
    s.base = gb();
    s.api = s.base ? s.base->GetApi(ORT_API_VERSION) : nullptr;
    if (!s.api) { fprintf(stderr, "[gsv_g2pw] GetApi 失败\n"); delete m; return nullptr; }
    const OrtApi * A = s.api;

    ort_check2(A, A->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "gsv_g2pw", &s.env), "CreateEnv");
    ort_check2(A, A->CreateSessionOptions(&s.so), "CreateSessionOptions");
    if (!s.env || !s.so) { delete m; return nullptr; }
    A->SetSessionGraphOptimizationLevel(s.so, ORT_ENABLE_ALL);
    if (s.n_threads > 0) A->SetIntraOpNumThreads(s.so, s.n_threads);
    A->SetInterOpNumThreads(s.so, 1);
    A->SetSessionExecutionMode(s.so, ORT_SEQUENTIAL);

    const bool want_cpu = cfg.device == "cpu" || cfg.device == "CPU";
    std::vector<const char *> order;
    if (want_cpu) order = { "CPUExecutionProvider" };
    else if (cfg.device == "dml") order = { "DmlExecutionProvider", "CPUExecutionProvider" };
    else if (cfg.device == "webgpu") order = { "WebGpuExecutionProvider", "CPUExecutionProvider" };
    else order = { "WebGpuExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider" };
    bool first_ok = false;
    for (const char * n : order) {
        OrtStatus * st = A->SessionOptionsAppendExecutionProvider(s.so, n, nullptr, nullptr, 0);
        if (st) { A->ReleaseStatus(st); if (s.verbose) printf("[gsv_g2pw] EP %-24s unavailable\n", n); continue; }
        if (!first_ok) { snprintf(s.ep, sizeof(s.ep), "%s", n); first_ok = true; }
        if (s.verbose) printf("[gsv_g2pw] EP %-24s OK\n", n);
    }

    const std::string onnx = cfg.model_dir + "/g2pW.onnx";
    std::wstring wp(onnx.begin(), onnx.end());
    OrtStatus * st = A->CreateSession(s.env, wp.c_str(), s.so, &s.sess);
    if (st) { fprintf(stderr, "[gsv_g2pw] CreateSession failed: %s\n", A->GetErrorMessage(st)); A->ReleaseStatus(st); delete m; return nullptr; }
    A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s.mem);

    // 校验标签维
    {
        size_t n_out = 0;
        A->SessionGetOutputCount(s.sess, &n_out);
        OrtTypeInfo * ti = nullptr;
        if (n_out == 1 && A->SessionGetOutputTypeInfo(s.sess, 0, &ti) == nullptr && ti) {
            const OrtTensorTypeAndShapeInfo * si = nullptr;
            A->CastTypeInfoToTensorInfo(ti, &si);
            if (si) {
                int64_t d[2] = {0, 0};
                size_t nd = 0;
                A->GetDimensions(si, d, 2);
                A->GetDimensionsCount(si, &nd);
                if (nd >= 2 && d[1] != G_LABELS)
                    fprintf(stderr, "[gsv_g2pw] warning: 输出第二维 = %lld (期望 %d)\n", (long long) d[1], G_LABELS);
            }
            A->ReleaseTypeInfo(ti);
        }
    }
    s.ready = true;
    if (s.verbose)
        printf("[gsv_g2pw] loaded %s (labels=%d chars=%zu EP=%s)\n",
               onnx.c_str(), G_LABELS, s.chars.size(), s.ep);
    return m;
}

// ============================ batch 构造 + Run ============================
bool gsv_g2pw::run_raw(const std::vector<std::vector<int64_t>> & input_ids,
                       const std::vector<std::vector<int64_t>> & token_type_ids,
                       const std::vector<std::vector<int64_t>> & attention_masks,
                       const std::vector<std::vector<float>> & phoneme_masks,
                       const std::vector<int64_t> & char_ids,
                       const std::vector<int64_t> & position_ids,
                       std::vector<int> & label_idx,
                       std::vector<std::vector<float>> * probs_out) {
    g2pw_impl & s = P(p);
    const OrtApi * A = s.api;
    if (!s.ready) return false;
    const size_t N = input_ids.size();
    if (N == 0 || token_type_ids.size() != N || attention_masks.size() != N ||
        phoneme_masks.size() != N || char_ids.size() != N || position_ids.size() != N) {
        fprintf(stderr, "[gsv_g2pw] run_raw: batch 各输入长度不一致\n");
        return false;
    }
    label_idx.assign(N, -1);
    if (probs_out) probs_out->assign(N, {});

    // 右侧 pad 到 max_len (repo 的 _pad_sequences)
    size_t Lmax = 0;
    for (size_t i = 0; i < N; i++) {
        const size_t li = input_ids[i].size();
        if (li > Lmax) Lmax = li;
    }
    if (Lmax == 0 || Lmax > (size_t) s.max_len) {
        fprintf(stderr, "[gsv_g2pw] run_raw: Lmax=%zu (max_len=%d)\n", Lmax, s.max_len);
        return false;
    }
    std::vector<int64_t> ids((size_t) N * Lmax, 0), tts((size_t) N * Lmax, 0), ams((size_t) N * Lmax, 0);
    std::vector<float> pms((size_t) N * G_LABELS, 0.0f);
    std::vector<int64_t> cids(N), pids(N);
    for (size_t i = 0; i < N; i++) {
        if (input_ids[i].size() != token_type_ids[i].size() || input_ids[i].size() != attention_masks[i].size())
            { fprintf(stderr, "[gsv_g2pw] run_raw: 行 %zu 三序列长度不一致\n", i); return false; }
        for (size_t j = 0; j < input_ids[i].size(); j++) {
            ids[i * Lmax + j] = input_ids[i][j];
            tts[i * Lmax + j] = token_type_ids[i][j];
            ams[i * Lmax + j] = attention_masks[i][j];
        }
        for (int j = 0; j < G_LABELS; j++) pms[i * G_LABELS + j] = phoneme_masks[i][(size_t) j];
        cids[i] = char_ids[i];
        pids[i] = position_ids[i];
    }

    const int64_t sh_ids[2] = { (int64_t) N, (int64_t) Lmax };
    const int64_t sh_n[1]   = { (int64_t) N };
    const int64_t sh_pm[2]  = { (int64_t) N, G_LABELS };

    OrtValue * vals[6] = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};
    struct { const char * name; void * data; size_t bytes; const int64_t * shape; size_t nd; ONNXTensorElementDataType dt; } spec[6] = {
        { "input_ids",      ids.data(),  ids.size() * 8,  sh_ids, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
        { "token_type_ids", tts.data(),  tts.size() * 8,  sh_ids, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
        { "attention_mask", ams.data(),  ams.size() * 8,  sh_ids, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
        { "phoneme_mask",   pms.data(),  pms.size() * 4,  sh_pm,  2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT },
        { "char_ids",       cids.data(), cids.size() * 8, sh_n,   1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
        { "position_ids",   pids.data(), pids.size() * 8, sh_n,   1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 },
    };
    for (int k = 0; k < 6; k++) {
        OrtStatus * st = A->CreateTensorWithDataAsOrtValue(
            s.mem, spec[k].data, spec[k].bytes, spec[k].shape, spec[k].nd, spec[k].dt, &vals[k]);
        if (st) {
            fprintf(stderr, "[gsv_g2pw] CreateTensor(%s) failed: %s\n", spec[k].name, A->GetErrorMessage(st));
            A->ReleaseStatus(st);
            for (int q = 0; q < k; q++) A->ReleaseValue(vals[q]);
            return false;
        }
    }
    const char * in_names[]  = { "input_ids", "token_type_ids", "attention_mask", "phoneme_mask", "char_ids", "position_ids" };
    const char * out_names[] = { "probs" };
    OrtValue * out = nullptr;
    OrtStatus * st = A->Run(s.sess, nullptr, in_names, (const OrtValue * const *) vals, 6, out_names, 1, &out);
    for (int k = 0; k < 6; k++) A->ReleaseValue(vals[k]);
    if (st) { fprintf(stderr, "[gsv_g2pw] Run failed: %s\n", A->GetErrorMessage(st)); A->ReleaseStatus(st); return false; }
    if (!out) { fprintf(stderr, "[gsv_g2pw] Run 返回空\n"); return false; }

    float * probs = nullptr;
    st = A->GetTensorMutableData(out, (void **) &probs);
    if (st || !probs) { fprintf(stderr, "[gsv_g2pw] GetTensorMutableData 失败\n"); if (st) A->ReleaseStatus(st); return false; }

    for (size_t i = 0; i < N; i++) {
        const float * row = probs + i * G_LABELS;
        int best = 0;
        float bv = row[0];
        for (int j = 1; j < G_LABELS; j++) if (row[j] > bv) { bv = row[j]; best = j; }
        label_idx[i] = best;
        if (probs_out) (*probs_out)[i].assign(row, row + G_LABELS);
    }
    A->ReleaseValue(out);
    return true;
}

// ---- 单条: 自己 tokenize + batch=1 ----
bool gsv_g2pw::predict_one(const std::string & text, int query_id, int & label_idx, float * confidence) {
    g2pw_impl & s = P(p);
    if (!s.ready) return false;
    std::vector<uint32_t> cps;
    utf8_to_cps(text, cps);
    if (query_id < 0 || query_id >= (int) cps.size()) {
        fprintf(stderr, "[gsv_g2pw] query_id=%d 越界 (len=%zu)\n", query_id, cps.size());
        return false;
    }
    std::vector<int> tokens;
    std::vector<int> text2token;
    tokenize_and_map(text, s, tokens, text2token);
    if ((int) tokens.size() + 2 > s.max_len) {
        fprintf(stderr, "[gsv_g2pw] tokens=%zu 超过 max_len=%d (调用方需先按 repo 的 _truncate 截断)\n",
                tokens.size(), s.max_len);
        return false;
    }
    std::vector<int64_t> ids;
    ids.push_back(s.id_cls);
    for (int t : tokens) ids.push_back(t);
    ids.push_back(s.id_sep);
    const std::string qc = [&]{ std::string t; utf8_append(t, cps[(size_t) query_id]); return t; }();
    auto ci = s.char2id.find(qc);
    if (ci == s.char2id.end()) {
        fprintf(stderr, "[gsv_g2pw] query 字符 \"%s\" 不在多音字表 (非 polyphonic char)\n", qc.c_str());
        return false;
    }
    const int64_t pos = text2token[(size_t) query_id] + 1;    // [CLS] 占位置 0

    std::vector<std::vector<int64_t>> b_ids{ids}, b_tts{std::vector<int64_t>(ids.size(), 0)},
        b_ams{std::vector<int64_t>(ids.size(), 1)};
    std::vector<int64_t> b_cid{ci->second}, b_pid{pos};   // char_ids/position_ids 是扁平的 [N]
    std::vector<std::vector<float>> b_pm;
    {
        const std::vector<int> & m = s.char_phoneme_masks[(size_t) ci->second];
        std::vector<float> mf(m.size());
        for (size_t k = 0; k < m.size(); k++) mf[k] = (float) m[k];
        b_pm.push_back(mf);
    }

    std::vector<int> out;
    std::vector<std::vector<float>> probs;
    if (!run_raw(b_ids, b_tts, b_ams, b_pm, b_cid, b_pid, out, &probs)) return false;
    label_idx = out[0];
    if (confidence) *confidence = probs.empty() ? 0.0f : *std::max_element(probs[0].begin(), probs[0].end());
    return true;
}

bool gsv_g2pw::predict_batch(const std::vector<std::string> & texts,
                             const std::vector<int> & query_ids,
                             std::vector<int> & label_idx,
                             std::vector<float> * confidence) {
    g2pw_impl & s = P(p);
    if (!s.ready) return false;
    const size_t N = texts.size();
    if (N == 0 || query_ids.size() != N) { fprintf(stderr, "[gsv_g2pw] predict_batch: 参数不匹配\n"); return false; }
    std::vector<std::vector<int64_t>> b_ids, b_tts, b_ams;
    std::vector<int64_t> b_cid, b_pid;
    std::vector<std::vector<float>> b_pm;
    b_ids.reserve(N); b_tts.reserve(N); b_ams.reserve(N); b_cid.reserve(N); b_pid.reserve(N); b_pm.reserve(N);
    for (size_t i = 0; i < N; i++) {
        std::vector<uint32_t> cps;
        utf8_to_cps(texts[i], cps);
        if (query_ids[i] < 0 || query_ids[i] >= (int) cps.size()) {
            fprintf(stderr, "[gsv_g2pw] batch 行 %zu: query_id=%d 越界\n", i, query_ids[i]);
            return false;
        }
        std::vector<int> tokens;
        std::vector<int> text2token;
        tokenize_and_map(texts[i], s, tokens, text2token);
        if ((int) tokens.size() + 2 > s.max_len) {
            fprintf(stderr, "[gsv_g2pw] batch 行 %zu: tokens=%zu 超 max_len\n", i, tokens.size());
            return false;
        }
        std::vector<int64_t> ids;
        ids.push_back(s.id_cls);
        for (int t : tokens) ids.push_back(t);
        ids.push_back(s.id_sep);
        std::string qc; utf8_append(qc, cps[(size_t) query_ids[i]]);
        auto ci = s.char2id.find(qc);
        if (ci == s.char2id.end()) {
            fprintf(stderr, "[gsv_g2pw] batch 行 %zu: \"%s\" 非多音字\n", i, qc.c_str());
            return false;
        }
        b_ids.push_back(ids);
        b_tts.push_back(std::vector<int64_t>(ids.size(), 0));
        b_ams.push_back(std::vector<int64_t>(ids.size(), 1));
        b_cid.push_back(ci->second);
        b_pid.push_back(text2token[(size_t) query_ids[i]] + 1);
        {   // phoneme_mask: int[1305] -> float[1305]
            const std::vector<int> & m = s.char_phoneme_masks[(size_t) ci->second];
            std::vector<float> mf(m.size());
            for (size_t k = 0; k < m.size(); k++) mf[k] = (float) m[k];
            b_pm.push_back(mf);
        }
    }
    std::vector<std::vector<float>> probs;
    if (!run_raw(b_ids, b_tts, b_ams, b_pm, b_cid, b_pid, label_idx, confidence ? &probs : nullptr)) return false;
    if (confidence) {
        confidence->resize(N);
        for (size_t i = 0; i < N; i++)
            (*confidence)[i] = probs[i].empty() ? 0.0f : *std::max_element(probs[i].begin(), probs[i].end());
    }
    return true;
}
