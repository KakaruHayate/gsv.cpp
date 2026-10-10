#include "gsv_mel.h"

#include "pocketfft_hdronly.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>

static constexpr double kPi = 3.14159265358979323846;

static int gcd_i(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a < 0 ? -a : a; }

// ============================ Slaney mel (librosa, htk=False, norm='slaney') ============================

static float hz2mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f, min_log_hz = 1000.0f, min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (hz >= min_log_hz) return min_log_mel + std::log(hz / min_log_hz) / logstep;
    return hz / f_sp;
}
static float mel2hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f, min_log_hz = 1000.0f, min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (mel >= min_log_mel) return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    return mel * f_sp;
}

// [n_mels, n_bins] 行主序
static std::vector<float> make_filterbank(int n_fft, int n_mels, int sr, float fmin, float fmax) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> fb((size_t) n_mels * n_bins, 0.0f);
    if (fmax < 0.0f) fmax = (float) sr / 2.0f;
    std::vector<float> freq(n_bins);
    for (int k = 0; k < n_bins; k++) freq[k] = (float) k * sr / (float) n_fft;
    const float mel_min = hz2mel_slaney(fmin), mel_max = hz2mel_slaney(fmax);
    std::vector<float> hz((size_t) n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++)
        hz[i] = mel2hz_slaney(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    for (int m = 0; m < n_mels; m++) {
        const float lo = hz[m], ce = hz[m + 1], up = hz[m + 2];
        const float enorm = 2.0f / (up - lo);                       // norm='slaney'
        for (int k = 0; k < n_bins; k++) {
            const float f = freq[k];
            float w = 0.0f;
            if (f >= lo && f <= ce) w = (f - lo) / (ce - lo);
            else if (f > ce && f <= up) w = (up - f) / (up - ce);
            fb[(size_t) m * n_bins + k] = w * enorm;
        }
    }
    return fb;
}

// torch.hann_window(periodic=True)
static std::vector<float> make_hann(int n) {
    std::vector<float> w((size_t) n);
    for (int i = 0; i < n; i++) w[i] = 0.5f - 0.5f * std::cos(2.0f * (float) kPi * i / n);
    return w;
}

// F.pad(mode='reflect'): 不含边缘样本的镜像
static void reflect_pad(const float * src, int64_t n, int pad, std::vector<float> & dst) {
    dst.resize((size_t) (n + 2 * pad));
    for (int i = 0; i < pad; i++) dst[(size_t) i] = src[pad - i];
    std::memcpy(dst.data() + pad, src, (size_t) n * sizeof(float));
    for (int i = 0; i < pad; i++) dst[(size_t) (pad + n + i)] = src[n - 2 - i];
}

// ============================ wav 读取 ============================

bool gsv_wav_load(const std::string & path, std::vector<float> & out, int & sr) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char riff[4], wave[4];
    uint32_t rsz;
    if (fread(riff, 1, 4, f) != 4 || fread(&rsz, 4, 1, f) != 1 || fread(wave, 1, 4, f) != 4 ||
        memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) { fclose(f); return false; }

    uint16_t fmt = 0, ch = 0, bits = 0;
    int sample_rate = 0;
    std::vector<uint8_t> data;
    while (true) {
        char cid[4];
        uint32_t csz;
        if (fread(cid, 1, 4, f) != 4 || fread(&csz, 4, 1, f) != 1) break;
        if (memcmp(cid, "fmt ", 4) == 0) {
            uint8_t b[40] = {0};
            const uint32_t rd = std::min<uint32_t>(csz, 40);
            if (fread(b, 1, rd, f) != rd) { fclose(f); return false; }
            if (csz > rd) fseek(f, (long) (csz - rd), SEEK_CUR);
            std::memcpy(&fmt, b, 2);
            std::memcpy(&ch, b + 2, 2);
            std::memcpy(&sample_rate, b + 4, 4);
            std::memcpy(&bits, b + 14, 2);
            if (fmt == 0xFFFE && csz >= 26) std::memcpy(&fmt, b + 24, 2);   // WAVE_FORMAT_EXTENSIBLE
        } else if (memcmp(cid, "data", 4) == 0) {
            data.resize(csz);
            const size_t got = fread(data.data(), 1, csz, f);
            data.resize(got);
            if (csz & 1) fseek(f, 1, SEEK_CUR);
        } else {
            fseek(f, (long) csz + (csz & 1), SEEK_CUR);
        }
    }
    fclose(f);
    if (ch == 0 || sample_rate == 0 || data.empty()) return false;

    const int bytes_per = bits / 8;
    if (bytes_per <= 0) return false;
    const int64_t frames = (int64_t) data.size() / (bytes_per * ch);
    out.assign((size_t) frames, 0.0f);
    for (int64_t t = 0; t < frames; t++) {
        double acc = 0.0;
        for (int c = 0; c < ch; c++) {
            const uint8_t * q = data.data() + (t * ch + c) * bytes_per;
            if (fmt == 3) {                                     // IEEE float
                if (bits == 32) { float v; std::memcpy(&v, q, 4); acc += v; }
                else if (bits == 64) { double v; std::memcpy(&v, q, 8); acc += v; }
            } else if (bits == 16) {
                int16_t v; std::memcpy(&v, q, 2); acc += (double) v / 32768.0;
            } else if (bits == 24) {
                int32_t v = (int32_t) ((uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16));
                if (v & 0x800000) v -= 0x1000000;
                acc += (double) v / 8388608.0;
            } else if (bits == 32) {
                int32_t v; std::memcpy(&v, q, 4); acc += (double) v / 2147483648.0;
            } else if (bits == 8) {
                acc += ((double) q[0] - 128.0) / 128.0;
            }
        }
        out[(size_t) t] = (float) (acc / ch);
    }
    sr = sample_rate;
    return true;
}

// ============================ 带限 sinc 重采样 ============================
// torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99) 的等价直式:
//   y[o] = Σ_m x[m] · 2·fc·sinc(2·fc·(τ_o − m)) · hann((τ_o − m)/W),  τ_o = o·sr_in/sr_out, W = 6/(2·fc)
//   fc (每输入样本周期) = 0.5·min(1, sr_out/sr_in)·rolloff
// (用冲激响应反推核对: 16k→32k 中心 0.99 / 邻 0.6259; 32k→16k 中心 0.495 / 2 样本外 4.67e-3,
//  与 torchaudio 实测逐位吻合。核偶对称, 相位符号无关。)
void gsv_resample(const float * x, int64_t n, int sr_in, int sr_out, std::vector<float> & out) {
    if (n <= 0 || sr_in <= 0 || sr_out <= 0) { out.clear(); return; }
    if (sr_in == sr_out) { out.assign(x, x + n); return; }
    const int g = gcd_i(sr_in, sr_out);
    const int orig = sr_in / g, nw = sr_out / g;

    const double lowpass = 6.0, rolloff = 0.99;
    const double fc = 0.5 * std::min(1.0, (double) nw / (double) orig) * rolloff;   // 每输入样本周期
    const double W = lowpass / (2.0 * fc);                                          // hann 窗半宽 (输入样本单位)
    const int64_t n_out = (int64_t) ((double) n * (double) nw / (double) orig);
    out.assign((size_t) n_out, 0.0f);
    for (int64_t o = 0; o < n_out; o++) {
        const double tau = (double) o * (double) orig / (double) nw;               // 输出时刻 (输入样本单位)
        int64_t m0 = (int64_t) std::ceil(tau - W);
        int64_t m1 = (int64_t) std::floor(tau + W);
        m0 = std::max<int64_t>(m0, 0);
        m1 = std::min<int64_t>(m1, n - 1);
        double acc = 0.0;
        for (int64_t m = m0; m <= m1; m++) {
            const double u = tau - (double) m;
            const double t = 2.0 * fc * u;
            const double s = (u == 0.0) ? 1.0 : std::sin(kPi * t) / (kPi * t);
            const double w = 0.5 * (1.0 + std::cos(kPi * u / W));
            acc += (double) x[m] * (2.0 * fc) * s * w;
        }
        out[(size_t) o] = (float) acc;
    }
}

// ============================ mel ============================

struct gsv_mel::impl {
    gsv_mel_cfg cfg;
    std::vector<float> win, fb;
    int pad = 0, n_bins = 0;
};

gsv_mel::gsv_mel(const gsv_mel_cfg & cfg) : p(new impl) {
    p->cfg = cfg;
    p->win = make_hann(cfg.win);
    p->fb  = make_filterbank(cfg.n_fft, cfg.n_mels, cfg.sr, cfg.fmin, cfg.fmax);
    p->pad = (cfg.n_fft - cfg.hop) / 2;          // 两侧对称 (repo: int((n_fft-hop)/2))
    p->n_bins = cfg.n_fft / 2 + 1;
}
gsv_mel::~gsv_mel() { delete p; }
const gsv_mel_cfg & gsv_mel::cfg() const { return p->cfg; }

int gsv_mel::frames(int64_t n) const {
    const int64_t padded = n + 2 * (int64_t) p->pad;
    const int64_t T = (padded - p->cfg.n_fft) / p->cfg.hop + 1;
    return T > 0 ? (int) T : 0;
}

bool gsv_mel::forward(const float * wav, int64_t n, std::vector<float> & mel, int & T) const {
    const gsv_mel_cfg & c = p->cfg;
    const int n_bins = p->n_bins;
    T = frames(n);
    mel.clear();
    if (T <= 0) return false;

    std::vector<float> padded;
    reflect_pad(wav, n, p->pad, padded);

    mel.assign((size_t) c.n_mels * T, 0.0f);

    auto worker = [&](int t0, int t1) {
        std::vector<float> frame((size_t) c.n_fft);
        std::vector<std::complex<float>> spec((size_t) n_bins);
        pocketfft::shape_t  sh = { (size_t) c.n_fft };
        pocketfft::stride_t si = { (std::ptrdiff_t) sizeof(float) };
        pocketfft::stride_t so = { (std::ptrdiff_t) sizeof(std::complex<float>) };
        pocketfft::shape_t  ax = { 0 };
        for (int t = t0; t < t1; t++) {
            const float * src = padded.data() + (size_t) t * c.hop;
            for (int k = 0; k < c.win; k++) frame[(size_t) k] = src[k] * p->win[(size_t) k];
            for (int k = c.win; k < c.n_fft; k++) frame[(size_t) k] = 0.0f;
            pocketfft::r2c(sh, si, so, ax, pocketfft::FORWARD, frame.data(), spec.data(), 1.0f);
            for (int m = 0; m < c.n_mels; m++) {
                const float * row = p->fb.data() + (size_t) m * n_bins;
                float acc = 0.0f;
                for (int k = 0; k < n_bins; k++) {
                    const float re = spec[(size_t) k].real(), im = spec[(size_t) k].imag();
                    acc += row[k] * std::sqrt(re * re + im * im + 1e-8f);   // repo: sqrt(pow2.sum + 1e-8)
                }
                mel[(size_t) m * T + t] = std::log(std::max(acc, 1e-5f));   // repo: log(clamp(min=1e-5))
            }
        }
    };

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    int want = c.n_threads > 0 ? c.n_threads : (int) std::min<unsigned>(hw, 8u);
    const int n_thr = std::max(1, std::min(want, T / 64 + 1));
    if (n_thr == 1) { worker(0, T); return true; }
    const int per = (T + n_thr - 1) / n_thr;
    std::vector<std::thread> pool;
    for (int t0 = 0; t0 < T; t0 += per) pool.emplace_back(worker, t0, std::min(T, t0 + per));
    for (auto & th : pool) th.join();
    return true;
}

bool gsv_mel::forward_norm(const float * wav, int64_t n, std::vector<float> & mel, int & T) const {
    if (!forward(wav, n, mel, T)) return false;
    for (float & v : mel) v = gsv_norm_spec(v);
    return true;
}
