# 导出 resampy 的 "kaiser_best" 滤波表 (参考音频 16k 路径用的是 match_librosa 的重采样):
#   interp_win [nwin] f32 + interp_delta (f32 差分) -> models/resampy_kaiser_best.bin
# 布局: [int32 nwin][int32 precision][float32 rolloff][win(nwin f32)][delta(nwin f32)]
# C++ 侧 gsv_refcode 读它并复刻 TTS.py::resample(match_librosa=True) 的多相插值。
import argparse, os, struct
import numpy as np
import resampy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/resampy_kaiser_best.bin")
    args = ap.parse_args()

    w, precision, rolloff = resampy.filters.get_filter("kaiser_best")
    win = w.astype(np.float32)                                   # 与 repo 一致: 先转 f32 再求差分
    delta = np.empty_like(win)
    delta[:-1] = win[1:] - win[:-1]
    delta[-1] = 0.0
    nwin = win.size
    print("kaiser_best: nwin=%d precision=%d rolloff=%.16f" % (nwin, precision, rolloff))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<iii", nwin, int(precision), 0))
        f.write(struct.pack("<f", float(rolloff)))
        win.tofile(f)
        delta.tofile(f)
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
