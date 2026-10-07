# 量化验收 (分布级): 比较两次 100-token greedy 运行每步的采样概率分布
# 用法: python tools/compare_probs.py ref.bin test.bin
# 输出: 首步 TV / 共同前缀步的平均 TV / 最大 TV / top-1 概率比
import sys
import numpy as np

V = 1025


def load(p):
    a = np.fromfile(p, dtype=np.float32)
    assert a.size % V == 0, (p, a.size)
    return a.reshape(-1, V)


def main():
    ref = load(sys.argv[1])
    tst = load(sys.argv[2])
    n = min(len(ref), len(tst))
    tv = 0.5 * np.abs(ref[:n] - tst[:n]).sum(axis=1)          # 每步全变差距离
    top1 = np.array([ref[i].max() for i in range(n)])
    top1t = np.array([tst[i].max() for i in range(n)])
    # top-5 集合一致率
    agree5 = np.mean([len(set(np.argsort(-ref[i])[:5]) & set(np.argsort(-tst[i])[:5])) / 5 for i in range(n)])
    print(f"steps={n}  TV: first={tv[0]:.4f} mean={tv.mean():.4f} max={tv.max():.4f}  "
          f"p90={np.percentile(tv, 90):.4f}")
    print(f"top-1 prob: ref={top1.mean():.4f} test={top1t.mean():.4f}  |  与参考 top-5 平均重合 {agree5*100:.1f}%")
    # 前 20 步逐步
    print("per-step TV (first 12):", " ".join(f"{x:.3f}" for x in tv[:12]))


if __name__ == "__main__":
    main()
