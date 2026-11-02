#include "gsv_vocoder.h"

#include "onnxruntime_c_api.h"

#ifdef _WIN32
#include <windows.h>
#define GSV_DLL_NAME "onnxruntime.dll"
#define GSV_PATH_SEP "\\"
#else
#include <dlfcn.h>
#include <unistd.h>
#include <limits.h>
#define GSV_DLL_NAME "libonnxruntime.so"
#define GSV_PATH_SEP "/"
#endif

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void * gsv_lib_open(const char * p) {
#ifdef _WIN32
    return (void *) LoadLibraryA(p);
#else
    return dlopen(p, RTLD_NOW | RTLD_LOCAL);
#endif
}
static void * gsv_lib_sym(void * h, const char * sym) {
#ifdef _WIN32
    return (void *) GetProcAddress((HMODULE) h, sym);
#else
    return dlsym(h, sym);
#endif
}
static void gsv_lib_close(void * h) {
    if (!h) return;
#ifdef _WIN32
    FreeLibrary((HMODULE) h);
#else
    dlclose(h);
#endif
}
static bool gsv_path_exists(const std::string & p) {
#ifdef _WIN32
    return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0;
#endif
}
static bool gsv_path_is_dir(const std::string & p) {
#ifdef _WIN32
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}
static std::string gsv_dirname(const std::string & p) {
    const size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}
static std::string gsv_join(const std::string & d, const std::string & f) {
    if (d.empty()) return f;
    const char last = d.back();
    if (last == '\\' || last == '/') return d + f;
    return d + GSV_PATH_SEP + f;
}
static std::string gsv_self_dir() {
#ifdef _WIN32
    char self[MAX_PATH] = {0};
    if (GetModuleFileNameA(nullptr, self, MAX_PATH)) return gsv_dirname(self);
#else
    char self[PATH_MAX] = {0};
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) { self[n] = '\0'; return gsv_dirname(self); }
#endif
    return ".";
}

static const int MEL_DIM  = 100;
static const int UPSAMPLE = 480;    // 10*6*2*2*2 = prod(upsample_rates)

// ============================ 动态 ORT 加载 ============================
// onnxruntime.dll 只导出 OrtGetApiBase 一个符号; pip wheel 不带 import lib, 且我们想
// 在同一个可执行文件里同时支持 DML 包与 WebGPU 包 (NuGet 只有 DML, WebGPU 只发 pip wheel),
// 故运行时按路径/目录找 DLL 并 GetProcAddress。
struct gsv_vocoder::impl {
    void *                dll = nullptr;
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
        if (dll)            gsv_lib_close(dll);
    }
};

static void ort_check(const OrtApi * api, OrtStatus * st, const char * what) {
    if (!st) return;
    const char * msg = api->GetErrorMessage(st);
    fprintf(stderr, "[gsv_vocoder] %s failed: %s\n", what, msg ? msg : "(null)");
    api->ReleaseStatus(st);
}

// plugin EP: onnxruntime_providers_webgpu.{dll,so} (env var / plugin dir / exe dir)
static std::string find_webgpu_plugin() {
    const char * env_dll = std::getenv("GSV_ORT_WEBGPU_DLL");
    if (env_dll && *env_dll && gsv_path_exists(env_dll)) return env_dll;
#ifdef _WIN32
    const char * name = "onnxruntime_providers_webgpu.dll";
#else
    const char * name = "onnxruntime_providers_webgpu.so";
#endif
    std::vector<std::string> dirs;
    if (const char * d = std::getenv("GSV_ORT_PLUGIN_DIR")) dirs.push_back(d);
    dirs.push_back(gsv_self_dir());
    for (const auto & d : dirs) {
        const std::string p = gsv_join(d, name);
        if (gsv_path_exists(p)) return p;
    }
    return "";
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

    // ---- 1) locate and load the onnxruntime library ----
    std::vector<std::string> cands;
    auto add_cand = [&](const std::string & d) {
        if (d.empty()) return;
        cands.push_back(gsv_path_is_dir(d) ? gsv_join(d, GSV_DLL_NAME) : d);
    };
    if (!cfg.ort_dll.empty()) add_cand(cfg.ort_dll);
    else {
        // prefer a local WebGPU build, then anything on the system search path
        add_cand(".venv-wg/Lib/site-packages/onnxruntime/capi");
        add_cand(".venv-ort/Lib/site-packages/onnxruntime/capi");
        add_cand("tools");
        add_cand(gsv_self_dir());
    }
    std::string loaded;
    for (const std::string & c : cands) {
        s.dll = gsv_lib_open(c.c_str());
        if (s.dll) { loaded = c; break; }
    }
    if (!s.dll) {
        s.dll = gsv_lib_open(GSV_DLL_NAME);
        if (s.dll) loaded = GSV_DLL_NAME;
    }
    if (!s.dll) {
        fprintf(stderr, "[gsv_vocoder] %s not found (tried %zu paths)\n", GSV_DLL_NAME, cands.size());
        for (const std::string & c : cands) fprintf(stderr, "    %s\n", c.c_str());
        delete m; return nullptr;
    }
    if (s.verbose) printf("[gsv_vocoder] loaded %s\n", loaded.c_str());
#ifdef _WIN32
    // WebGPU EP needs dxcompiler.dll/dxil.dll next to onnxruntime.dll (or on PATH)
    {
        const std::string d = gsv_dirname(loaded);
        for (const char * dep : { "dxcompiler.dll", "dxil.dll" }) {
            const std::string dp = gsv_join(d, dep);
            if (gsv_path_exists(dp)) {
                void * h = gsv_lib_open(dp.c_str());
                if (!h && s.verbose)
                    fprintf(stderr, "[gsv_vocoder] note: %s load failed (WebGPU may be unavailable)\n", dep);
            }
        }
    }
#endif

    using GetApiBaseFn = const OrtApiBase * (ORT_API_CALL *)();
    auto get_base = (GetApiBaseFn) gsv_lib_sym(s.dll, "OrtGetApiBase");
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
    // 3a) 插件 EP（新 API）：ORT >= 1.22 的 WebGPU 以 onnxruntime_providers_webgpu.dll 形式发布。
    //     先 RegisterExecutionProviderLibrary，再从 GetEpDevices 挑 WebGPU 设备，用 V2 API 挂载；
    //     找不到插件库时回退到内置 EP（旧行为）。
    bool plugin_ok = false;
    if (want_gpu && (cfg.device.empty() || cfg.device == "webgpu" || cfg.device == "WebGPU")) {
        const std::string plugin = find_webgpu_plugin();
        if (!plugin.empty()) {
#ifdef _WIN32
            std::wstring wplugin(plugin.begin(), plugin.end());
            OrtStatus * st = A->RegisterExecutionProviderLibrary(s.env, "webgpu", wplugin.c_str());
#else
            OrtStatus * st = A->RegisterExecutionProviderLibrary(s.env, "webgpu", plugin.c_str());
#endif
            if (st) {
                if (s.verbose)
                    fprintf(stderr, "[gsv_vocoder] register plugin EP failed: %s\n", A->GetErrorMessage(st));
                A->ReleaseStatus(st);
            } else {
                const OrtEpDevice * const * devs = nullptr;
                size_t ndev = 0;
                st = A->GetEpDevices(s.env, &devs, &ndev);
                if (st) {
                    A->ReleaseStatus(st);
                } else {
                    std::vector<const OrtEpDevice *> sel;
                    for (size_t i = 0; i < ndev; i++) {
                        const char * nm = A->EpDevice_EpName(devs[i]);
                        if (nm && (std::strcmp(nm, "WebGPU") == 0 ||
                                   std::strcmp(nm, "WebGpuExecutionProvider") == 0))
                            sel.push_back(devs[i]);
                    }
                    if (!sel.empty()) {
                        st = A->SessionOptionsAppendExecutionProvider_V2(s.so, s.env, sel.data(), sel.size(),
                                                                         nullptr, nullptr, 0);
                        if (st) {
                            A->ReleaseStatus(st);
                        } else {
                            plugin_ok = true;
                            snprintf(s.ep, sizeof(s.ep), "WebGPU(plugin)");
                            if (s.verbose) printf("[gsv_vocoder] EP WebGPU plugin OK (%s)\n", plugin.c_str());
                        }
                    }
                }
            }
        }
    }

    std::vector<const char *> order;
    if (plugin_ok) {
        // 已用插件 EP，不再 append 内置 EP
    } else if (want_cpu)                                     order = { "CPUExecutionProvider" };
    // 统一 WebGPU（DML 已放弃：本机实测反而更慢，见 known_issues KI-8）；
    // dml 请求也回落到 WebGPU，保证各平台行为一致。
    else                                                  order = { "WebGpuExecutionProvider", "CPUExecutionProvider" };
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
