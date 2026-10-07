# BERT 前向基准: torch (CPU fp32 / GPU fp32 / GPU fp16) vs ggml (CPU/Vulkan)
# 对齐 GPT-SoVITS 管线用法: output_hidden_states=True, 取 hidden_states[-3]
import argparse, os, sys, time
import torch

TEXTS = [
    "这是一个简单的示例，真没想到这么简单就完成了。",
    "然而，他红了20年以后，他竟退出了大家的视线。",
    "The King and His Stories. Once there was a king.",
    "短。",
]


def bench(fn, n=20):
    fn()  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/chinese-roberta-wwm-ext-large")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForMaskedLM.from_pretrained(args.model).eval()
    n_layer = m.config.num_hidden_layers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={device} layers={n_layer}")

    # 参考文本 + 合成长序列 (固定 ids, 不含 [CLS]/[SEP] 以外的语义)
    cases = []
    for t in TEXTS:
        enc = tok(t, return_tensors="pt")
        cases.append((f"text({enc['input_ids'].shape[1]})", enc["input_ids"]))
    for T in (64, 128, 256, 512):
        ids = torch.randint(1000, 20000, (1, T))
        ids[0, 0], ids[0, -1] = 101, 102
        cases.append((f"syn{T}", ids))

    def run(dev, half):
        mm = m.to(dev)
        if half:
            mm = mm.half()
        rows = []
        for name, ids in cases:
            ii = ids.to(dev)
            def f():
                with torch.no_grad():
                    out = mm(input_ids=ii, output_hidden_states=True)
                    _ = out.hidden_states[-3]
            rows.append((name, bench(f, args.n)))
        m.to("cpu")
        return rows

    res = {}
    res["cpu-fp32"] = run("cpu", False)
    if device == "cuda":
        res["gpu-fp32"] = run("cuda", False)
        res["gpu-fp16"] = run("cuda", True)

    print(f"{'case':10s} " + " ".join(f"{k:>10s}" for k in res))
    for i, (name, _) in enumerate(res["cpu-fp32"]):
        print(f"{name:10s} " + " ".join(f"{res[k][i][1]:10.2f}" for k in res) + "   ms/次")
    print("(注: torch 侧含 output_hidden_states=True 的全部 25 层; HF 跑满 24 层)")


if __name__ == "__main__":
    main()
