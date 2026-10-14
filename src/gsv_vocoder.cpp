#include "gsv_vocoder.h"

#include "onnxruntime_c_api.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int MEL_DIM  = 100;
static const int UPSAMPLE = 480;    // 10*6*2*2*2 = prod(upsample_rates)

// ============================ 动态 ORT 加载 ============================
// onnxruntime.dll 只导出 OrtGetApiBase 一个符号; pip wheel 不带 import lib, 且我们想
// 在同一个可执行文件里同时支持 DML 包与 WebGPU 包 (NuGet 只有 DML, WebGPU 只发 pip wheel),
// 故运行时按路径/目录找 DLL 并 GetProcAddress。
struct gsv_vocoder::impl {
    HMODULE               dll = nullptr;
    const OrtApi *        api = nullptr;
    const OrtApiBase *    base = nullptr;
    OrtEnv *              env = nullptr;
    OrtSessionOptions *   so = nullptr;
    OrtSession *          sess = nullptr;
    OrtAllocator *        alloc = nullptr;   // OrtGetAllocatorWithDefaultOptions (无需释放)
    OrtMemoryInfo *       mem = nullptr;     // 需 Release
    char                  ep[64] = "CPUExecutionProvider";
    bool                  dml_ok = false, wgpu_ok = false, cpu_ok = false;
    int                   n_threads = 0;
    bool                  verbose = false;
    // 复用的输入/输出 OrtValue (形状变了就重建; 避免每次 malloc/free)
    OrtValue *            in_val = nullptr;
    std::vector<float>    in_buf;            // [100, T] 的 [1,100,T] 连续化
    OrtValue *            out_val = nullptr;
    size_t                out_len = 0;

    ~impl() {
        if (in_val && api)  api->ReleaseValue(in_val);
        if (out_val && api) api->ReleaseValue(out_val);
        if (mem && api)     api->ReleaseMemoryInfo(mem);
        if (sess && api)    api->ReleaseSession(sess);
        if (so && api)      api->ReleaseSessionOptions(so);
        if (env && api)     api->ReleaseEnv(env);
        if (dll)            FreeLibrary(dll);
    }
};

static void ort_check(const OrtApi * api, OrtStatus * st, const char * what) {
    if (!st) return;
    const char * msg = api->GetErrorMessage(st);
    fprintf(stderr, "[gsv_vocoder] %s failed: %s\n", what, msg ? msg : "(null)");
    api->ReleaseStatus(st);
}

gsv_vocoder::gsv_vocoder() : p(new impl) {}
gsv_vocoder::~gsv_vocoder() { delete p; }

int  gsv_vocoder::mel_dim()  const { return MEL_DIM; }
int  gsv_vocoder::upsample() const { return UPSAMPLE; }
const char * gsv_vocoder::ep_name() const { return p->ep; }

gsv_vocoder * gsv_vocoder::load(const std::string & onnx_path, const gsv_vocoder_cfg & cfg) {
    gsv_vocoder * m = new gsv_vocoder();
    impl & s = *m->p;
    s.verbose = cfg.verbose;
    s.n_threads = cfg.n_threads;

    // ---- 1) 找并加载 onnxruntime.dll ----
    std::vector<std::string> cands;
    auto add_cand = [&](const std::string & d) {
        if (d.empty()) return;
        std::string full = d;
        // 传的是目录 -> 拼 DLL 名
        const DWORD a = GetFileAttributesA(d.c_str());
        if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY)) full = d + "\\onnxruntime.dll";
        cands.push_back(full);
    };
    if (!cfg.ort_dll.empty()) add_cand(cfg.ort_dll);
    else {
        // 默认: 先找本仓库 venv 里的 WebGPU 版 (更快), 再系统路径 (可能是 DML 版)
        add_cand(".venv-wg\\Lib\\site-packages\\onnxruntime\\capi");
        add_cand(".venv-ort\\Lib\\site-packages\\onnxruntime\\capi");
        add_cand("tools");
    }
    for (const std::string & c : cands) {
        s.dll = LoadLibraryA(c.c_str());
        if (s.dll) { if (s.verbose) printf("[gsv_vocoder] loaded %s\n", c.c_str()); break; }
    }
    if (!s.dll) {
        // 最后退到系统搜索路径 (PATH / 当前目录)
        s.dll = LoadLibraryA("onnxruntime.dll");
        if (s.dll && s.verbose) printf("[gsv_vocoder] loaded onnxruntime.dll (from PATH)\n");
    }
    if (!s.dll) {
        fprintf(stderr, "[gsv_vocoder] onnxruntime.dll 未找到 (试过 %zu 个路径)\n", cands.size());
        for (const std::string & c : cands) fprintf(stderr, "    %s\n", c.c_str());
        delete m; return nullptr;
    }
    // WebGPU EP 的 dxcompiler.dll/dxil.dll 必须与 onnxruntime.dll 同目录 (或 PATH),
    // 从同一目录加载它们, 否则 WebGPU 初始化会静默失败
    {
        char self[MAX_PATH] = {0};
        GetModuleFileNameA(s.dll, self, MAX_PATH);
        std::string dir(self);
        const size_t slash = dir.find_last_of("\\/");
        if (slash != std::string::npos) {
            const std::string d = dir.substr(0, slash);
            for (const char * dep : { "dxcompiler.dll", "dxil.dll" }) {
                const std::string dp = d + "\\" + dep;
                if (GetFileAttributesA(dp.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    // 只影响后续 LoadLibrary 的同目录解析; 显式加载一次确保在位
                    HMODULE h = LoadLibraryA(dp.c_str());
                    if (!h && s.verbose)
                        fprintf(stderr, "[gsv_vocoder] note: %s load failed (WebGPU 可能不可用)\n", dep);
                }
            }
        }
    }

    using GetApiBaseFn = const OrtApiBase * (ORT_API_CALL *)();
    auto get_base = (GetApiBaseFn) GetProcAddress(s.dll, "OrtGetApiBase");
    if (!get_base) { fprintf(stderr, "[gsv_vocoder] OrtGetApiBase 未导出\n"); delete m; return nullptr; }
    s.base = get_base();
    if (!s.base) { fprintf(stderr, "[gsv_vocoder] OrtGetApiBase 返回 null\n"); delete m; return nullptr; }
    s.api = s.base->GetApi(ORT_API_VERSION);
    if (!s.api) { fprintf(stderr, "[gsv_vocoder] GetApi(%d) 返回 null\n", ORT_API_VERSION); delete m; return nullptr; }
    const OrtApi * A = s.api;

    // ---- 2) env + session options ----
    ort_check(A, A->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "gsv_vocoder", &s.env), "CreateEnv");
    if (!s.env) { delete m; return nullptr; }
    ort_check(A, A->CreateSessionOptions(&s.so), "CreateSessionOptions");
    if (!s.so) { delete m; return nullptr; }
    ort_check(A, A->SetSessionGraphOptimizationLevel(s.so, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
    if (s.n_threads > 0) A->SetIntraOpNumThreads(s.so, s.n_threads);
    A->SetInterOpNumThreads(s.so, 1);
    A->SetSessionExecutionMode(s.so, ORT_SEQUENTIAL);

    // ---- 3) EP 选择 ----
    // 盲试: 依次 append, 失败就忽略 (该 EP 未编入此 DLL 时 AppendExecutionProvider 返回错误)。
    // 顺序 = 用户意图优先, 否则 WebGPU > DML > CPU (WebGPU 在本机 vocoder 上最快, 见 docs)。
    const bool want_cpu = cfg.device == "cpu" || cfg.device == "CPU";
    const bool want_gpu = !cfg.device.empty() && !want_cpu;
    auto try_ep = [&](const char * name) -> bool {
        OrtStatus * st = A->SessionOptionsAppendExecutionProvider(s.so, name, nullptr, nullptr, 0);
        if (st) { A->ReleaseStatus(st); return false; }
        return true;
    };
    std::vector<const char *> order;
    if (want_cpu)                                     order = { "CPUExecutionProvider" };
    else if (cfg.device == "dml" || cfg.device == "DML")   order = { "DmlExecutionProvider", "CPUExecutionProvider" };
    else if (cfg.device == "webgpu" || cfg.device == "WebGPU") order = { "WebGpuExecutionProvider", "CPUExecutionProvider" };
    else if (want_gpu)                                order = { "WebGpuExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider" };
    else                                              order = { "WebGpuExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider" };
    for (const char * n : order) {
        const bool ok = try_ep(n);
        if (std::strcmp(n, "DmlExecutionProvider") == 0)       s.dml_ok = ok;
        else if (std::strcmp(n, "WebGpuExecutionProvider") == 0) s.wgpu_ok = ok;
        else if (std::strcmp(n, "CPUExecutionProvider") == 0)    s.cpu_ok = ok;
        if (s.verbose) printf("[gsv_vocoder] EP %-26s %s\n", n, ok ? "OK" : "unavailable");
    }
    // 生效 EP 名 = 第一个 append 成功的 (ORT 会按 append 顺序做 EP 分配)
    for (const char * n : order) {
        if (std::strcmp(n, "DmlExecutionProvider") == 0 && s.dml_ok) { snprintf(s.ep, sizeof(s.ep), "%s", n); break; }
        if (std::strcmp(n, "WebGpuExecutionProvider") == 0 && s.wgpu_ok) { snprintf(s.ep, sizeof(s.ep), "%s", n); break; }
        if (std::strcmp(n, "CPUExecutionProvider") == 0 && s.cpu_ok) { snprintf(s.ep, sizeof(s.ep), "%s", n); break; }
    }

    // ---- 4) session ----
    const ORTCHAR_T * mp = nullptr;
    std::wstring wpath;
#ifdef _WIN32
    {
        wpath.assign(onnx_path.begin(), onnx_path.end());
        mp = wpath.c_str();
    }
#else
    mp = onnx_path.c_str();
#endif
    OrtStatus * st = A->CreateSession(s.env, mp, s.so, &s.sess);
    if (st) {
        fprintf(stderr, "[gsv_vocoder] CreateSession(%s) failed: %s\n", onnx_path.c_str(), A->GetErrorMessage(st));
        A->ReleaseStatus(st);
        delete m; return nullptr;
    }
    A->GetAllocatorWithDefaultOptions(&s.alloc);
    A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s.mem);

    // ---- 5) 校验图接口 (input "mel" [B,100,T], output [B,1,T*480]) ----
    {
        size_t n_in = 0, n_out = 0;
        A->SessionGetInputCount(s.sess, &n_in);
        A->SessionGetOutputCount(s.sess, &n_out);
        if (n_in != 1 || n_out != 1) {
            fprintf(stderr, "[gsv_vocoder] 期望 1 进 1 出, 实际 %zu/%zu\n", n_in, n_out);
            delete m; return nullptr;
        }
        OrtTypeInfo * ti = nullptr;
        A->SessionGetInputTypeInfo(s.sess, 0, &ti);
        if (ti) {
            const OrtTensorTypeAndShapeInfo * si = nullptr;
            A->CastTypeInfoToTensorInfo(ti, &si);
            if (si) {
                size_t nd = 0;
                A->GetDimensionsCount(si, &nd);
                if (nd != 3)
                    fprintf(stderr, "[gsv_vocoder] warning: input 维度数 = %zu (期望 3)\n", nd);
            }
            A->ReleaseTypeInfo(ti);
        }
    }
    if (s.verbose) printf("[gsv_vocoder] loaded %s (EP=%s)\n", onnx_path.c_str(), s.ep);
    return m;
}

bool gsv_vocoder::run(const float * mel, int T, std::vector<float> & wav) {
    impl & s = *p;
    const OrtApi * A = s.api;
    if (!A || !s.sess || !mel || T <= 0) return false;
    wav.clear();
    if (T > 65536) { fprintf(stderr, "[gsv_vocoder] T=%d 过大\n", T); return false; }

    // 输入: [1, 100, T] (torch 布局的 mel 正好就是这个连续内存, 无需转置)
    const int64_t shape[3] = { 1, MEL_DIM, T };
    s.in_buf.assign(mel, mel + (size_t) MEL_DIM * T);
    OrtValue * nv = nullptr;
    OrtStatus * st = A->CreateTensorWithDataAsOrtValue(
        s.mem, s.in_buf.data(), s.in_buf.size() * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &nv);
    if (st) { fprintf(stderr, "[gsv_vocoder] CreateTensor(in) failed: %s\n", A->GetErrorMessage(st)); A->ReleaseStatus(st); return false; }
    if (s.in_val) A->ReleaseValue(s.in_val);
    s.in_val = nv;

    const char * in_names[]  = { "mel" };
    const char * out_names[] = { "wav" };
    if (s.out_val) { A->ReleaseValue(s.out_val); s.out_val = nullptr; }
    st = A->Run(s.sess, nullptr, in_names, (const OrtValue * const *) &s.in_val, 1,
                out_names, 1, &s.out_val);
    if (st) {
        fprintf(stderr, "[gsv_vocoder] Run failed: %s\n", A->GetErrorMessage(st));
        A->ReleaseStatus(st);
        return false;
    }
    if (!s.out_val) { fprintf(stderr, "[gsv_vocoder] Run 返回空输出\n"); return false; }

    float * out_data = nullptr;
    st = A->GetTensorMutableData(s.out_val, (void **) &out_data);
    if (st || !out_data) { fprintf(stderr, "[gsv_vocoder] GetTensorMutableData failed\n"); if (st) A->ReleaseStatus(st); return false; }
    const size_t n = (size_t) UPSAMPLE * (size_t) T;
    wav.assign(out_data, out_data + n);
    return true;
}

bool gsv_vocoder::run_ggml_mel(const float * mel, int T, std::vector<float> & wav) {
    if (T <= 0) return false;
    // ggml [100,T] (idx = m + t*100) -> torch [100,T] (idx = m*T + t)
    std::vector<float> tt((size_t) MEL_DIM * T);
    for (int m = 0; m < MEL_DIM; m++)
        for (int t = 0; t < T; t++) tt[(size_t) m * T + t] = mel[(size_t) m + (size_t) t * MEL_DIM];
    return run(tt.data(), T, wav);
}
