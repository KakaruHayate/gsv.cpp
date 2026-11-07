#!/usr/bin/env python3
"""Distilled HuBERT (student_v0.pt) -> 参考特征 dump, 用于和 src/gsv_dhubert.cpp 对拍。

    python3 tools/dump_ref_dhubert.py --wav in.wav --out ref.f32 \
        [--ckpt .../student_v0.pt] [--repo ~/GPT-SoVITS-CPUFast] [--dump prefix]

输出 (与 tools/dhubert_probe.cpp 完全同格式, 可直接用 numpy 比较):
    out      : f32, 字节序 == ggml [768, T] (idx = d + t*768), 即 last_hidden_state[0] (T,768) 原样
    --dump   : prefix.front.f32 / prefix.front{0..6}.f32 / prefix.conv0.f32 (通道主序 C x T)
               prefix.attn{0,1}.f32 (帧主序 T x C)  (分段定位数值偏差)

wav 读取与 src/gsv_mel.cpp::gsv_wav_load 一致 (PCM16/24/32 与 IEEE float32, 多声道取均值)。
"""
import argparse
import importlib.util
import json
import os
import sys
import wave

import numpy as np

REPO_DEFAULT = os.path.expanduser("~/GPT-SoVITS-CPUFast")


def read_wav(path):
    """-> (mono float32 [-1,1], sample_rate), 与 gsv_wav_load 逐样本一致。"""
    try:
        import soundfile as sf  # libsndfile: PCM16/32768 与 gsv_wav_load 相同
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(sr)
    except Exception:
        pass
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float64) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise SystemExit(f"unsupported sample width {sw}")
    x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def load_student_module(repo):
    path = os.path.join(repo, "GPT_SoVITS", "feature_extractor", "student_hubert.py")
    if not os.path.exists(path):
        raise SystemExit(f"student_hubert.py not found at {path} (pass --repo)")
    spec = importlib.util.spec_from_file_location("student_hubert_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="distilled HuBERT torch reference dump")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="student_v0.pt")
    ap.add_argument("--repo", default=os.environ.get("GSV_DHUBERT_REPO", REPO_DEFAULT))
    ap.add_argument("--dump", default=None, help="中间张量前缀 (与 dhubert_probe --dump 一致)")
    a = ap.parse_args()

    import torch

    mod = load_student_module(a.repo)
    ckpt = a.ckpt or mod.DEFAULT_CKPT
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    model = mod.DistilledHubertFront()
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    model.student.load_state_dict(state["student"])
    model.eval()

    wav, sr = read_wav(a.wav)
    if sr != 16000:
        raise SystemExit(f"wav is {sr} Hz; resample to 16 kHz first (gsv_resample / ffmpeg)")
    x = torch.from_numpy(wav)[None]  # (1, T)

    dumps = {}

    # 布局约定 (与 src/gsv_dhubert.cpp 的 ggml 内存序一致, 不做任何转置):
    #   front/tcn : torch (1, C, T) == ggml [T, C] 的字节序 (通道主序, 每通道 T 个连续值)
    #   attn      : torch (1, T, C) == ggml [C, T] 的字节序 (帧主序, 每帧 C 个连续值)
    #   out       : torch (1, T, 768) == ggml [768, T] 的字节序, 即 idx = d + t*768
    def hook(name):
        def fn(_m, _inp, out):
            dumps[name] = np.ascontiguousarray(out.detach().float().numpy()[0], dtype=np.float32)
        return fn

    if a.dump:
        for i, blk in enumerate(model.student.front):
            blk.register_forward_hook(hook(f"front{i}"))                   # (1,C,T), 逐层定位
        model.student.front.register_forward_hook(hook("front"))           # (1,C,T)
        model.student.tcn.register_forward_hook(hook("tcn"))               # (1,C,T)
        for i, layer in enumerate(model.student.attn.layers):
            layer.register_forward_hook(hook(f"attn{i}"))                   # (1,T,C)

    with torch.no_grad():
        out = model(x)["last_hidden_state"]  # (1, T50, 768)
    # (1, T, 768) 直接写: 字节序 == ggml [768, T] (idx = d + t*768), 与 probe 输出一致
    feats = np.ascontiguousarray(out[0].float().numpy(), dtype=np.float32)

    feats.tofile(a.out)
    if a.dump:
        for k, v in dumps.items():
            v.tofile(f"{a.dump}.{k}.f32")

    act = model.student.attn.layers[0].activation
    info = {
        "wav": a.wav,
        "samples": int(wav.shape[0]),
        "frames": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "out": a.out,
        "ffn_activation": str(act).replace("\n", " ")[:80],
        "norm_first": bool(model.student.attn.layers[0].norm_first),
        "dumps": sorted(dumps.keys()),
    }
    print(json.dumps(info))


if __name__ == "__main__":
    main()
