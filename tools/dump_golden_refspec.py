# 参考音频 -> refer_audio_spec (spectrogram_torch) -> ref_enc -> ge 的 golden dump
#   语义内联自 repo 的 GPT_SoVITS/TTS_infer_pack/TTS.py::_get_ref_spec +
#   module/mel_processing.py::spectrogram_torch + module/models.py::decode_encp 的 ge 分支
#
# 链 (v5turbo, hps["data"] = {sr 32000, filter_length 2048, hop 640, win 2048}):
#   raw wav (任意 sr, 多声道取均值) -> torchaudio resample 到 32k
#     -> maxx 归一化 (maxx>1 时 /= min(2, maxx))      <- 只作用于 spec, mel2 那条路不做
#     -> spectrogram_torch(2048, 640, 2048, center=False)  -> [1, 1025, T]
#     -> refer[:, :704] * refer_mask                  <- ref_enc 的 in_dim = 704
#     -> MelStyleEncoder -> ge [1, 512, 1]
# 落盘 float32 raw:
#   refspec.{tag}.wav32k_norm.bin   maxx 归一化后的 32k 波形 (C++ 端隔离重采样/归一化误差)
#   refspec.{tag}.spec.bin          spectrogram_torch 全 1025 频点 [1025, T] (torch 布局 k*T+t)
#   refspec.{tag}.spec704.bin       裁到前 704 频点 [704, T]
#   refspec.{tag}.ge.bin            ref_enc 输出 ge [512]
import argparse, os, sys
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, os.pardir, "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

from module.mel_processing import spectrogram_torch  # noqa: E402


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-26s %s" % (name, tuple(arr.shape)))


def load_audio(path):
    """避免 torchcodec (环境缺 DLL): soundfile 优先, 回退 stdlib wave (PCM16)"""
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=True)      # [n, C]
        return torch.from_numpy(a.T.copy()), sr                     # [C, n]
    except Exception:
        import wave
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        assert sw == 2, "stdlib fallback only supports PCM16 (sampwidth=%d)" % sw
        a = np.frombuffer(raw, dtype="<i2").astype("float32").reshape(-1, ch).T / 32768.0
        return torch.from_numpy(a.copy()), sr


# ---- MelStyleEncoder (语义内联自 repo/GPT_SoVITS/module/modules.py) ----
class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class Conv1dGLU(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.out_channels = out_ch
        self.conv1 = torch.nn.Module()
        self.conv1.conv = torch.nn.Conv1d(in_ch, 2 * out_ch, k, padding=(k - 1) // 2)

    def forward(self, x):
        residual = x
        y = self.conv1.conv(x)
        x1, x2 = torch.split(y, self.out_channels, dim=1)
        return residual + x1 * torch.sigmoid(x2)


class MHA(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.temperature = float(d_model) ** 0.5
        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        self.fc = torch.nn.Linear(n_head * d_v, d_model)

    def forward(self, x, mask=None):
        nh, dk, dv = self.n_head, self.d_k, self.d_v
        B, T, _ = x.size()
        residual = x
        q = self.w_qs(x).view(B, T, nh, dk).permute(2, 0, 1, 3).contiguous().view(-1, T, dk)
        k = self.w_ks(x).view(B, T, nh, dk).permute(2, 0, 1, 3).contiguous().view(-1, T, dk)
        v = self.w_vs(x).view(B, T, nh, dv).permute(2, 0, 1, 3).contiguous().view(-1, T, dv)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask.repeat(nh, 1, 1), -np.inf)
        attn = torch.softmax(attn, dim=2)
        out = torch.bmm(attn, v).view(nh, B, T, dv).permute(1, 2, 0, 3).contiguous().view(B, T, -1)
        return self.fc(out) + residual, attn


class MelStyleEncoder(torch.nn.Module):
    def __init__(self, n_mel=704, hidden=128, out=512, k=5, head=2):
        super().__init__()
        self.hidden_dim, self.out_dim, self.kernel_size, self.n_head = hidden, out, k, head
        self.spectral = torch.nn.Sequential(
            LinearNorm(n_mel, hidden), Mish(), torch.nn.Identity(),
            LinearNorm(hidden, hidden), Mish(), torch.nn.Identity())
        self.temporal = torch.nn.Sequential(
            Conv1dGLU(hidden, hidden, k), Conv1dGLU(hidden, hidden, k))
        self.slf_attn = MHA(head, hidden, hidden // head, hidden // head)
        self.fc = LinearNorm(hidden, out)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        len_ = (~mask).sum(dim=1).unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.float()
        x = torch.div(x, len_.unsqueeze(1))
        return x.sum(dim=1)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        x = self.spectral(x)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        x = self.fc(x)
        w = self.temporal_avg_pool(x, mask=mask)
        return w.unsqueeze(-1)


def run(out, tag, wav_path, ref_enc, verbose=False):
    a, sr = load_audio(wav_path)                       # [C, n]
    a = a.mean(0, keepdim=True)                        # mono
    if sr != 32000:
        a = torchaudio.transforms.Resample(sr, 32000)(a)
    maxx = a.abs().max()
    if maxx > 1:
        a = a / min(2, maxx)                           # TTS.py::_get_ref_spec 的归一化
    dump(out, "refspec." + tag + ".wav32k_norm", a[0])

    spec = spectrogram_torch(a, 2048, 32000, 640, 2048, center=False)   # [1, 1025, T]
    dump(out, "refspec." + tag + ".spec", spec[0])

    refer = spec[:, :704, :]                           # ref_enc 的 in_dim = 704
    dump(out, "refspec." + tag + ".spec704", refer[0])

    T = refer.shape[-1]
    refer_lengths = torch.LongTensor([T])
    refer_mask = torch.unsqueeze(
        torch.arange(T).unsqueeze(0) < refer_lengths.unsqueeze(1), 1).to(torch.float32)
    with torch.no_grad():
        ge = ref_enc(refer[:, :704] * refer_mask, refer_mask)          # [1, 512, 1]
    dump(out, "refspec." + tag + ".ge", ge)
    print("[%s] src_sr=%d -> 32k n=%d; spec T=%d (%.2fs); ge %s"
          % (tag, sr, a.shape[-1], T, T / 100.0, tuple(ge.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--audio", default="tests/audio")
    ap.add_argument("--T", type=int, default=0, help=">0 时追加一个合成输入用例 (T 帧)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    msd = {k[len("ref_enc."):]: v for k, v in sd.items() if k.startswith("ref_enc.")}
    m = MelStyleEncoder()
    m.load_state_dict(msd, strict=True)
    m = m.eval()

    for tag, fn in [("zh", "ref_zh_32k.wav"), ("en", "ref_en_16k.wav"),
                    ("zhs", "ref_zh_short_16k.wav")]:
        p = os.path.join(args.audio, fn)
        if not os.path.exists(p):
            print("missing", p, "-> skip")
            continue
        run(args.out, tag, p, m)

    if args.T > 0:
        # 合成输入: 固定 seed 的随机波形, 覆盖任意帧数 (与 test_mel 的合成用例同风格)
        g = torch.Generator().manual_seed(7)
        n = args.T * 640
        a = (torch.rand(1, n, generator=g) * 2 - 1) * 0.5
        maxx = a.abs().max()
        if maxx > 1:
            a = a / min(2, maxx)
        dump(args.out, "refspec.syn.wav32k_norm", a[0])
        spec = spectrogram_torch(a, 2048, 32000, 640, 2048, center=False)
        dump(args.out, "refspec.syn.spec", spec[0])
        refer = spec[:, :704, :]
        dump(args.out, "refspec.syn.spec704", refer[0])
        T = refer.shape[-1]
        refer_mask = torch.ones(1, 1, T)
        with torch.no_grad():
            ge = m(refer * refer_mask, refer_mask)
        dump(args.out, "refspec.syn.ge", ge)
        print("[syn] T=%d ge %s" % (T, tuple(ge.shape)))

    print("refspec golden done ->", args.out)


if __name__ == "__main__":
    main()
