# BERT int8 参照: torch 动态量化 (quantize_dynamic, int8 Linear) vs fp32
#   - 用 golden 的 4 条文本 ids, 比较 hidden[-3]/feat 相对 fp32 的漂移 (这就是"BERT 可以 int8"说法的量尺)
#   - 顺带计时 CPU fp32 vs int8 (同线程数)
# 用法: <diffsinger py> tools/bench_bert_int8_torch.py [--model models/chinese-roberta-wwm-ext-large]
import argparse, os, time
import numpy as np
import torch
from transformers import AutoModelForMaskedLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_bin(p):
    return np.fromfile(p, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "chinese-roberta-wwm-ext-large"))
    ap.add_argument("--golden", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    D = 1024
    n = int(read_bin(os.path.join(args.golden, "bert.n_texts.bin"))[0] + 0.5)
    texts = []
    for i in range(n):
        ids = read_bin(os.path.join(args.golden, f"bert.{i}.ids.bin")).astype(np.int64)
        hid = read_bin(os.path.join(args.golden, f"bert.{i}.hidden.bin")).reshape(-1, D)  # [1, T, D]
        feat = read_bin(os.path.join(args.golden, f"bert.{i}.feat.bin"))
        texts.append((torch.from_numpy(ids)[None, :], hid, feat, ids.shape[0]))

    m = AutoModelForMaskedLM.from_pretrained(args.model).eval()
    mq = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8).eval()

    def run(mm, ids):
        with torch.no_grad():
            out = mm(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-3][0].numpy()          # [T, D] (torch 落盘 hidden 是 [1,T,D] 模型输出)
        f = h[1:-1].T.copy()                          # [D, T-2]
        return h, f

    print(f"== torch cpu int8 动态量化 vs fp32 (threads={args.threads}) ==")
    for i, (ids, hid_ref, feat_ref, T) in enumerate(texts):
        h32, f32 = run(m, ids)
        h8, f8 = run(mq, ids)
        fr = feat_ref.reshape(D, T - 2)               # golden feat 是 [D, T-2] 落盘
        d_h = np.abs(h8 - hid_ref.reshape(-1, D)).max()
        d_f = np.abs(f8 - fr).max()
        cos8 = float(np.dot(f8.ravel(), fr.ravel()) /
                     (np.linalg.norm(f8) * np.linalg.norm(fr) + 1e-12))
        # fp32 自检 (确认 golden 与我们加载的模型一致)
        d_self = np.abs(h32 - hid_ref.reshape(-1, D)).max()
        print(f"  text{i} T={T:2d}: fp32 自检 {d_self:.2e} | int8 max|d| hidden {d_h:.3e} feat {d_f:.3e} cos {cos8:.8f}")

    # 计时 (用最长文本)
    ids_max = max(texts, key=lambda t: t[3])[0]
    def timeit(mm, fn_n=args.iters):
        with torch.no_grad():
            for _ in range(3):
                mm(input_ids=ids_max, output_hidden_states=True)
            ts = []
            for _ in range(fn_n):
                t0 = time.perf_counter()
                mm(input_ids=ids_max, output_hidden_states=True)
                ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return sum(ts) / len(ts), ts[0]

    a32, m32 = timeit(m)
    a8, m8 = timeit(mq)
    print(f"  CPU fp32: avg {a32:8.2f} ms min {m32:8.2f}  |  int8: avg {a8:8.2f} ms min {m8:8.2f}  (T={ids_max.shape[1]})")


if __name__ == "__main__":
    main()
