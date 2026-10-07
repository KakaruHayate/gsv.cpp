# AR 基准 (torch): 与 tests/bench_ar.cpp 同 workload
# - greedy 生成 (argmax), 与 C++ 的 top_k=1 等价
# - 测量: 首步耗时 / 100 步解码总耗时 / per-step / tok/s
# - CPU (与 C++ 同为 f32, 同线程数) 与 CUDA (f32/f16) 各一组
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("GSV_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from AR.models.t2s_lightning_module import Text2SemanticLightningModule  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_bin(gd, name):
    return np.fromfile(os.path.join(gd, name + ".bin"), dtype=np.float32)


def build_requests(gd, prompt, bs):
    reqs = []
    for b in range(3):
        ph = read_bin(gd, f"batch.phones{b}").astype(np.int64)
        be = read_bin(gd, f"batch.bert{b}").reshape(1024, -1)
        reqs.append((torch.from_numpy(ph).unsqueeze(0), torch.from_numpy(be).unsqueeze(0), prompt))
    return [reqs[i % 3] for i in range(bs)]


def gen_greedy(dec, reqs, n_gen, device, dtype):
    B = len(reqs)
    y_len = reqs[0][2].shape[1]
    x_lens = [r[0].shape[1] for r in reqs]
    max_len = max(x_lens)

    xs = []
    for ph, be, _ in reqs:
        xi = dec.ar_text_embedding(ph.to(device)) + dec.bert_proj(be.to(device).to(dtype).transpose(1, 2))
        xi = dec.ar_text_position(xi).to(dtype)
        xi = F.pad(xi, (0, 0, max_len - xi.shape[1], 0), value=0)
        xs.append(xi)
    x = torch.cat(xs, 0)
    y = dec.ar_audio_position(dec.ar_audio_embedding(reqs[0][2].to(device))).expand(B, -1, -1).to(dtype)
    S = max_len + y_len
    xy_pos = torch.cat([x, y], dim=1)

    nh = dec.num_head
    x_mask = F.pad(torch.zeros(max_len, max_len, dtype=torch.bool, device=device), (0, y_len), value=True)
    y_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool, device=device), diagonal=1), (max_len, 0), value=False)
    causal = torch.cat([x_mask, y_mask], dim=0)
    padding = torch.cat([
        __import__("AR.models.utils", fromlist=["make_pad_mask_left"]).make_pad_mask_left(torch.tensor(x_lens, device=device), max_len),
        __import__("AR.models.utils", fromlist=["make_pad_mask_left"]).make_pad_mask_left(torch.full((B,), y_len, device=device), y_len),
    ], dim=1)
    padding = padding.view(B, 1, S).repeat(1, S, 1)
    attn = (causal.view(1, S, S) | padding).bool()
    attn_h = attn.unsqueeze(1).expand(-1, nh, -1, -1).contiguous()

    yy = reqs[0][2].to(device).expand(B, -1).clone()
    with torch.no_grad():
        t0 = time.perf_counter()
        xy_dec, kc, vc = dec.t2s_transformer.process_prompt(xy_pos, attn_h, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_first = time.perf_counter() - t0

        ca = F.pad(attn_h[:, :, -1].unsqueeze(-2), (0, 1), value=False)
        t0 = time.perf_counter()
        for idx in range(n_gen):
            lg = dec.ar_predict_layer(xy_dec[:, -1])
            if idx < 11:
                lg = lg[:, :-1]
            samples = torch.argmax(lg, dim=-1, keepdim=True)
            yy = torch.cat([yy, samples], dim=1)
            if idx == n_gen - 1:
                break
            y_emb1 = dec.ar_audio_embedding(yy[:, -1:])
            x_in = y_emb1 * dec.ar_audio_position.x_scale + \
                dec.ar_audio_position.alpha * dec.ar_audio_position.pe[:, y_len + idx]
            xy_dec, kc, vc = dec.t2s_transformer.decode_next_token(x_in.to(dtype), kc, vc, ca)
            ca = F.pad(ca, (0, 1), value=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_steps = time.perf_counter() - t0
    return t_first, t_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "s1v3.ckpt"))
    ap.add_argument("--golden", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Text2SemanticLightningModule(ckpt["config"], "****", is_train=False)
    model.load_state_dict(ckpt["weight"])
    model = model.eval()
    dec = model.model
    prompt = torch.from_numpy(read_bin(args.golden, "ar.prompt").astype(np.int64)).unsqueeze(0)

    print(f"[bench-py] n_gen={args.n_gen} threads={args.threads} torch={torch.__version__}")
    for dev_name in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        device = torch.device(dev_name)
        for dt_name, dtype in (("f32", torch.float32), ("f16", torch.float16)):
            if dev_name == "cpu" and dtype == torch.float16:
                continue        # CPU f16 不具代表性
            dec.to(device).to(dtype)
            for bs in (1, 3, 8):
                reqs = build_requests(args.golden, prompt, bs)
                # warmup
                gen_greedy(dec, reqs, 5, device, dtype)
                t_first, t_steps = gen_greedy(dec, reqs, args.n_gen, device, dtype)
                per_step = t_steps / args.n_gen * 1000
                print(f"[bench-py] {dev_name}/{dt_name} bs={bs}: first={t_first*1000:.1f}ms "
                      f"steps={t_steps*1000:.1f}ms per_step={per_step:.2f}ms "
                      f"{args.n_gen/t_steps:.1f} tok/s(单序列) {bs*args.n_gen/t_steps:.1f} tok/s(总)")
        dec.to("cpu")


if __name__ == "__main__":
    main()
