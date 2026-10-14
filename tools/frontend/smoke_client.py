"""协议冒烟: 起 server -> ping -> prepRef -> 打印形状。不依赖 repo 的 text 前端。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.gsv_client import GsvClient  # noqa: E402


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exe = os.path.join(root, "tests", "relcpu", "gsv_ggml_cli.exe")
    ref = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "tests", "audio", "ref_zh_32k.wav")
    print("starting server...")
    c = GsvClient(autostart=True, exe=exe, verbose=True)
    print("connected, ready =", getattr(c, "ready", None))
    try:
        data, _ = c.request("ping")
        print("ping ->", data)

        data, t = c.request("prepRef", {"wavPath": ref})
        codes = np.frombuffer(t["codes"], dtype="<i4")
        ge = np.frombuffer(t["ge"], dtype="<f4")
        mel2 = np.frombuffer(t["mel2"], dtype="<f4")
        print("prepRef -> codes=%s ge=%s mel2=%s (T=%d)"
              % (codes.shape, ge.shape, mel2.shape, data["melT"]))
        print("  codes[:12] =", codes[:12].tolist())
        print("  ge[:6]     =", np.round(ge[:6], 4).tolist())
        print("  mel2 range = [%.3f, %.3f]" % (mel2.min(), mel2.max()))
    finally:
        c.close()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
