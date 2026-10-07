# AR 基准 (Python 最快路径): GPT_SoVITS/Accel —— CUDA Graph (+ FlashAttention 若可用)
# 与 tests/bench_ar.cpp / tools/bench_ar.py 同 workload (greedy → top_k=1/rep=1.0)
# 用法: python tools/bench_ar_accel.py --n-gen 100
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("GSV_REPO", os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "repo")))
sys.path.insert(0, REPO)                       # for tools.acceleration
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))


def read_bin(gd, name):
    return np.fromfile(os.path.join(gd, name + ".bin"), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "s1v3.ckpt"))
    ap.add_argument("--golden", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from tools.acceleration import create_acceleration, resolve_acceleration

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "f16" else torch.float32
    backend, graph = resolve_acceleration(device, dtype, use_cuda_graph=True, use_flash_attention=True)
    print(f"[accel] device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)} "
          f"dtype={args.dtype} -> backend={backend!r} graph={graph}")
    if backend is None:
        print("[accel] 无可用加速后端")
        return

    prompt = torch.from_numpy(read_bin(args.golden, "ar.prompt").astype(np.int64)).unsqueeze(0)  # [1,24]
    base_ph, base_bert = [], []
    for b in range(3):
        ph = read_bin(args.golden, f"batch.phones{b}").astype(np.int64)
        be = read_bin(args.golden, f"batch.bert{b}").reshape(1024, -1)
        base_ph.append(ph)
        base_bert.append(be)

    golden_greedy = {}
    for b in range(3):
        golden_greedy[b] = read_bin(args.golden, f"batch.greedy.seq{b}.tokens").astype(np.int64)

    for bs in (1, 3, 8, 20):
        max_len = max(len(base_ph[i % 3]) for i in range(bs))
        x = torch.zeros((bs, max_len), dtype=torch.long)
        bert = torch.zeros((bs, 1024, max_len))
        for i in range(bs):
            j = i % 3
            x[i, :len(base_ph[j])] = torch.from_numpy(base_ph[j])
            bert[i, :, :len(base_ph[j])] = torch.from_numpy(base_bert[j])
        x_lens = torch.tensor([len(base_ph[i % 3]) for i in range(bs)], dtype=torch.long)
        prompts = prompt.expand(bs, -1)

        accel = create_acceleration(args.ckpt, device=device, dtype=dtype, max_batch_size=bs,
                                    use_cuda_graph=True, use_flash_attention=True)
        if accel is None or not accel.prepare(True, True):
            print(f"[accel] bs={bs}: prepare() 失败, 跳过")
            continue

        kw = dict(parallel_infer=(bs > 1), top_k=1, top_p=1.0, temperature=1.0,
                  early_stop_num=args.n_gen - 1, repetition_penalty=1.0)
        # 预热 (首次调用会捕获 CUDA Graph)
        t_w0 = time.perf_counter()
        tokens, lens = accel.infer_batch(x.to(device), x_lens, prompts.to(device), bert.to(device), **kw)
        torch.cuda.synchronize()
        t_warm = time.perf_counter() - t_w0

        # 计时
        t0 = time.perf_counter()
        tokens, lens = accel.infer_batch(x.to(device), x_lens, prompts.to(device), bert.to(device), **kw)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        steps = max(lens) + 1 if lens else 0
        per_step = t / max(1, steps) * 1000
        # 首步(prefill)近似: early_stop=0 只生成 1 个 token
        kw1 = dict(kw); kw1["early_stop_num"] = 0
        t0 = time.perf_counter()
        accel.infer_batch(x.to(device), x_lens, prompts.to(device), bert.to(device), **kw1)
        torch.cuda.synchronize()
        t_first = time.perf_counter() - t0
        print(f"[accel] bs={bs:2d}: first={t_first*1000:.1f}ms total={t*1000:.1f}ms steps={steps} "
              f"per_step={(t-t_first)/max(1,steps-1)*1000:.2f}ms {steps/t:.1f} tok/s(单序列) {bs*steps/t:.1f} tok/s(总)")

        # 正确性: 与 F32 golden greedy token 对比 (前 3 个序列)
        bad = 0
        for i in range(min(bs, 3)):
            ref = golden_greedy[i]
            got = tokens[i][0].to(torch.long).cpu().numpy() if tokens[i].dim() > 1 else tokens[i].cpu().numpy()
            n = min(len(ref), len(got))
            if len(ref) != len(got) or not np.array_equal(ref[:n], got[:n]):
                bad += 1
        print(f"[accel] bs={bs:2d}: 与 golden greedy token 对比 → 差异 {bad}/{min(bs,3)}"
              + ("  (fp16 路径存在轻微分歧属预期)" if bad else "  (完全一致)"))
        accel.close()


if __name__ == "__main__":
    main()
