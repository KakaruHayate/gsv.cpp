# 临时: HuBERT 分段 golden (真实语音) —— feature_projection / pos_conv / LN / 逐层, 用于定位变长偏差
import os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join("..", "repo"))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
from feature_extractor.cnhubert import CNHubert  # noqa: E402

OUT = "_dbg/stages"
os.makedirs(OUT, exist_ok=True)

def dump(name, t):
    a = t.detach().cpu().float().contiguous().numpy()
    a.tofile(os.path.join(OUT, name + ".bin"))
    print("  %-14s %s" % (name, tuple(a.shape)))

def load16(p):
    import wave
    with wave.open(p, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype("float32").reshape(-1, ch).mean(1) / 32768.0
    return torch.from_numpy(a), sr

wav, sr = load16("tests/audio/ref_zh_short_16k.wav")
zero = torch.zeros(int(32000 * 0.3), dtype=torch.float32)
fed = torch.cat([wav, zero])
hub = CNHubert("models/chinese-hubert-base").eval().float()
m = hub.model
with torch.no_grad():
    # 与 tests/test_refcode.cpp 的输入口径一致: 用同一 double 精度 z-score (与 HF 差 ~1e-7)
    m_ = float(fed.mean()); v_ = float(((fed - m_) ** 2).mean())
    x = ((fed - m_) / float(np.sqrt(v_ + 1e-7))).unsqueeze(0)
    print("fed", tuple(fed.shape), "zin head", np.round(x[0,:4].numpy(),6))
    fe = m.feature_extractor(x)                       # [1, 512, T']
    fp = m.feature_projection(fe.transpose(1, 2))     # (hidden, extract) or tensor
    feat = fp[0] if isinstance(fp, (tuple, list)) else fp     # [1, T, 768]
    pos = m.encoder.pos_conv_embed(feat)              # [1, T, 768]
    ln = m.encoder.layer_norm(feat + pos)             # [1, T, 768] = transformer 输入
    dump("cnn_out", fe[0])
    dump("feat_TD", feat[0])                          # [T,768] 行主 (给 GSV_HUBERT_ENCIN 用)
    dump("feat", feat[0].transpose(0,1))              # [768,T]
    dump("pos", pos[0].transpose(0,1))
    dump("ln", ln[0].transpose(0,1))
    h = ln
    for i, layer in enumerate(m.encoder.layers):
        o = layer(h)
        h = o[0] if isinstance(o, (tuple, list)) else o
        if i in (0, 1, 5, 11):
            dump("layer%02d" % (i+1), h[0].transpose(0,1))
    dump("ssl_final", h[0].transpose(0,1))
print("stages dumped ->", OUT)
