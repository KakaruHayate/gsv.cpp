// 参考音频预处理: wav 读取 / 带限 sinc 重采样 / mel_fn_v4 / norm_spec
//
// 语义对齐 repo 的 GPT_SoVITS/module/mel_processing.py::mel_spectrogram_torch 与
// TTS_infer_pack/TTS.py 的 mel_fn_v4 / norm_spec:
//   - 无条件 reflect pad int((n_fft-hop)/2) 两侧对称, 再 center=False 的 stft
//     (即 frames = 1 + (n + 2p - n_fft)/hop, p = (n_fft-hop)/2)
//   - hann 周期窗 (torch.hann_window); win == n_fft
//   - |X| = sqrt(re^2 + im^2 + 1e-8)  (float32, 逐元素)
//   - mel = librosa Slaney filterbank (htk=False, norm='slaney', fmax=sr/2); log(clamp(x, 1e-5))
//   - norm_spec = (x + 12) / 14 * 2 - 1; denorm_spec 反之 (spec_min=-12, spec_max=2)
// mel 输出布局 = torch [1, n_mels, T] 行主序 (idx = m*T + t), 与 DiT 的 prompt mel 一致。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gsv_mel_cfg {
    int   n_fft = 1280, win = 1280, hop = 320, n_mels = 100, sr = 32000;
    float fmin  = 0.0f;
    float fmax  = -1.0f;          // < 0 = sr/2 (librosa 的 fmax=None)
    int   n_threads = 0;          // 0 = 硬件并发 (上限 8)
};

// wav 读取 (RIFF: PCM16/24/32 与 IEEE float32; 多声道取均值) -> mono float32 [-1,1]
bool gsv_wav_load(const std::string & path, std::vector<float> & out, int & sr);

// 带限 sinc 重采样, 参数与 torchaudio.functional.resample 一致:
//   sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99
void gsv_resample(const float * x, int64_t n, int sr_in, int sr_out, std::vector<float> & out);

// norm_spec / denorm_spec (spec_min=-12, spec_max=2)
inline float gsv_norm_spec(float x)   { return (x + 12.0f) / 14.0f * 2.0f - 1.0f; }
inline float gsv_denorm_spec(float x) { return (x + 1.0f) / 2.0f * 14.0f - 12.0f; }

// ref_enc (MelStyleEncoder) 的输入维度 (refer[:, :704]); 放在这里避免 gsv_refenc.h 的循环依赖
static const int gsv_refenc_in_dim = 704;

// ============================ 线性幅度谱 (spectrogram_torch) ============================
// 语义对齐 repo 的 GPT_SoVITS/module/mel_processing.py::spectrogram_torch —— 与 mel 的区别:
//   不做 mel filterbank, 不取 log; 输出 [n_fft/2+1, T] 的线性幅度 (idx = k*T + t)
// v5turbo 的参数来自 hps["data"]: filter_length=win_length=2048, hop_length=640, sr=32000
//   (s2Gv5turbo.pth 的 config.data, 已人工核对) -> 1025 个频点
struct gsv_spec_cfg {
    int   n_fft = 2048, win = 2048, hop = 640, sr = 32000;
    int   n_threads = 0;          // 0 = 硬件并发 (上限 8)
};

class gsv_spec {
public:
    explicit gsv_spec(const gsv_spec_cfg & cfg);
    ~gsv_spec();

    const gsv_spec_cfg & cfg() const;
    static int n_bins(int n_fft) { return n_fft / 2 + 1; }
    int n_bins() const;
    int frames(int64_t n_samples) const;             // 含 reflect pad 的帧数

    // wav [n] (mono, cfg.sr) -> spec [n_bins, T] 行主序 (idx = k*T + t)
    bool forward(const float * wav, int64_t n, std::vector<float> & spec, int & T) const;

private:
    struct impl;
    impl * p;
};

// ---- repo TTS.py::_get_ref_spec 的等价封装 (refer_audio_spec 的生产路径) ----
//   raw wav (任意 sr, 多声道取均值) -> torchaudio 重采样到 32k
//     -> maxx 归一化 (maxx>1 时 /= min(2, maxx))      <- 只作用于 spec 这条路, mel2 不做
//     -> spectrogram_torch(n_fft=2048, hop=640, win=2048, center=False)
//     -> refer[:, :704]  (ref_enc 的 in_dim = 704, 1025 个频点只用前 704 个)
//   输出 [704, T] torch 布局 (idx = c*T + t); gsv_refenc::encode 需要 ggml 布局
//   (idx = c + t*704), 调用方自行转置 (见 tests/test_refspec.cpp)
bool gsv_refspec_wav(const float * wav, int64_t n, int sr_in,
                     std::vector<float> & spec704, int & T, bool apply_maxx_norm = true);
bool gsv_refspec(const std::string & wav_path, std::vector<float> & spec704, int & T,
                 bool apply_maxx_norm = true, bool verbose = false);

class gsv_mel {
public:
    explicit gsv_mel(const gsv_mel_cfg & cfg);
    ~gsv_mel();

    const gsv_mel_cfg & cfg() const;
    int frames(int64_t n_samples) const;             // 含 reflect pad 的帧数

    // wav [n] (mono, cfg.sr) -> mel [n_mels, T] 行主序 (idx = m*T + t), 未归一化 (log 域)
    bool forward(const float * wav, int64_t n, std::vector<float> & mel, int & T) const;
    // 同上 + norm_spec
    bool forward_norm(const float * wav, int64_t n, std::vector<float> & mel, int & T) const;

private:
    struct impl;
    impl * p;
};
