# GSV AR 采样链 + batch 能力 golden 导出
# 1) 采样链: 用固定 logits + previous_tokens 导出各阶段 probs (含 exp-trick 的 q 注入用例)
# 2) batch: 3 条不同长度序列 (同 prompt) 的 batched 首步/解码步 logits + greedy 生成参考
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.environ.get("GSV_REPO", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

from AR.models.t2s_lightning_module import Text2SemanticLightningModule  # noqa: E402
from AR.models.utils import logits_to_probs, make_pad_mask_left  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dump(outdir, name, t):
    t = t.detach().cpu().float().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print(f"  dumped {name}: shape={t.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "s1v3.ckpt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--bs", type=int, default=3)
    ap.add_argument("--early-stop", type=int, default=6)
    args = ap.parse_args()
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = Text2SemanticLightningModule(ckpt["config"], "****", is_train=False)
    model.load_state_dict(ckpt["weight"])
    model = model.eval().float()
    dec = model.model
    EOS = dec.EOS
    D = dec.model_dim
    print(f"AR: D={D} nh={dec.num_head} L={dec.num_layers} vocab={dec.vocab_size} EOS={EOS}")

    torch.manual_seed(42)
    y_len = 24
    prompt = torch.randint(0, 1024, (1, y_len))

    # ---------------- 1) 采样链 golden ----------------
    # 复用 step0 logits (若存在), 否则用随机 logits
    logits_path = os.path.join(outdir, "ar.step0.logits.bin")
    if os.path.exists(logits_path):
        logits0 = torch.tensor(np.fromfile(logits_path, dtype=np.float32))
    else:
        logits0 = torch.randn(dec.vocab_size)

    prev_tokens = torch.cat([prompt[0], torch.tensor([100, 200, 100])]).long()  # 含重复 token, 测 gather/scatter 语义
    dump(outdir, "sample.logits", logits0.unsqueeze(0))
    dump(outdir, "sample.prev_tokens", prev_tokens.float().unsqueeze(0))

    cfgs = {
        "a_rep": dict(temperature=1.0, top_k=15, top_p=1.0, repetition_penalty=1.35),
        "b_full": dict(temperature=0.8, top_k=15, top_p=0.9, repetition_penalty=1.35),
        "c_topp": dict(temperature=1.0, top_k=None, top_p=0.95, repetition_penalty=1.0),
    }
    rng = np.random.RandomState(7)
    for name, cfg in cfgs.items():
        lg = logits0.clone().unsqueeze(0)   # logits_to_probs 会原地修改, 必须 clone
        probs = logits_to_probs(lg, prev_tokens.unsqueeze(0), **cfg)
        dump(outdir, f"sample.{name}.probs", probs)
        # exp-trick 采样规则: idx = argmax(probs / q), q 固定注入
        q = torch.tensor(rng.exponential(1.0, size=probs.shape).astype(np.float32))
        dump(outdir, f"sample.{name}.q", q)
        idx = torch.argmax(probs / q, dim=-1)
        dump(outdir, f"sample.{name}.idx", idx.float().reshape(1, 1))
        print(f"    {name}: idx={int(idx)} prob@{int(idx)}={float(probs[0, int(idx)]):.6g}")

    # ---------------- 2) batch golden ----------------
    B, y_len = args.bs, 24
    torch.manual_seed(1234)
    x_lens = [32 - 6 * i for i in range(B)]           # 32, 26, 20
    max_len = max(x_lens)
    bert_list, phones_list = [], []
    for i in range(B):
        phones_list.append(torch.randint(0, dec.phoneme_vocab_size, (1, x_lens[i])))
        bert_list.append(torch.randn(1, 1024, x_lens[i]))

    # x: 每序列 embedding+bert_proj+PE 后左 padding 到 max_len
    xs = []
    for i in range(B):
        xi = dec.ar_text_embedding(phones_list[i]) + dec.bert_proj(bert_list[i].transpose(1, 2))
        xi = dec.ar_text_position(xi)
        xi = F.pad(xi, (0, 0, max_len - xi.shape[1], 0), value=0)
        xs.append(xi)
    x = torch.cat(xs, 0)                                # [B, max_len, D]
    y = dec.ar_audio_position(dec.ar_audio_embedding(prompt)).expand(B, -1, -1)  # [B, y_len, D]
    S = max_len + y_len
    xy_pos = torch.cat([x, y], dim=1)                   # [B, S, D]

    x_mask = F.pad(torch.zeros(max_len, max_len, dtype=torch.bool), (0, y_len), value=True)
    y_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (max_len, 0), value=False)
    causal = torch.cat([x_mask, y_mask], dim=0)         # [S, S]: True=屏蔽
    padding = torch.cat([make_pad_mask_left(torch.tensor(x_lens), max_len),
                         make_pad_mask_left(torch.full((B,), y_len), y_len)], dim=1)  # [B, S]
    padding = padding.view(B, 1, S).repeat(1, S, 1)     # [B, S, S]
    attn = (causal.view(1, S, S) | padding).bool()      # [B, S, S]
    attn_f = torch.zeros(B, S, S).masked_fill_(attn, float("-inf"))
    attn_h = attn.unsqueeze(1).expand(-1, dec.num_head, -1, -1).contiguous()  # [B, nh, S, S] (torch 路径用)

    dump(outdir, "batch.x_len", torch.tensor(x_lens, dtype=torch.float32))
    # 引擎测试需要原始输入 (phones / bert)
    for i in range(B):
        dump(outdir, f"batch.phones{i}", phones_list[i].float())
        dump(outdir, f"batch.bert{i}", bert_list[i])
    dump(outdir, "batch.xy_pos", xy_pos)
    dump(outdir, "batch.attn_mask", attn_f)             # [B, sq, sk] -> ggml [sk, sq, 1, B]

    with torch.no_grad():
        xy_dec, k_cache, v_cache = dec.t2s_transformer.process_prompt(xy_pos, attn_h, None)
        logits = dec.ar_predict_layer(xy_dec[:, -1])    # [B, vocab]
        dump(outdir, "batch.step0.logits", logits)

        # 固定 token 流 [100,200,300] 的 batched 解码步 (每条序列同一个 token)
        cur_attn = F.pad(attn_h[:, :, -1].unsqueeze(-2), (0, 1), value=False)  # [B, nh, 1, S+1]
        for di, tok_v in enumerate([100, 200, 300]):
            tok = torch.tensor([[tok_v]] * B, dtype=torch.long)
            y_emb1 = dec.ar_audio_embedding(tok)                              # [B,1,D]
            x_in = y_emb1 * dec.ar_audio_position.x_scale + \
                dec.ar_audio_position.alpha * dec.ar_audio_position.pe[:, y_len + di]
            dump(outdir, f"batch.dec{di+1}.x_in", x_in)
            xy_dec1, k_cache, v_cache = dec.t2s_transformer.decode_next_token(x_in, k_cache, v_cache, cur_attn)
            dump(outdir, f"batch.dec{di+1}.logits", dec.ar_predict_layer(xy_dec1[:, -1]))
            cur_attn = F.pad(cur_attn, (0, 1), value=False)

        # ---------------- greedy 生成参考 (转录 infer_panel_batch_infer 的循环语义, 不做序列移除) ----------------
        def greedy_run(early_stop_num):
            xy_dec, kc, vc = dec.t2s_transformer.process_prompt(xy_pos, attn_h, None)
            yy = prompt.expand(B, -1).clone()
            idx_list = [None] * B
            y_list = [None] * B
            ca = F.pad(attn_h[:, :, -1].unsqueeze(-2), (0, 1), value=False)
            for idx in range(1500):
                lg = dec.ar_predict_layer(xy_dec[:, -1])
                if idx < 11:
                    lg = lg[:, :-1]
                samples = torch.argmax(lg, dim=-1, keepdim=True)
                yy = torch.cat([yy, samples], dim=1)
                tokens = torch.argmax(lg, dim=-1)
                for i in range(B):
                    if idx_list[i] is None and (samples[i, 0].item() == EOS or tokens[i].item() == EOS):
                        idx_list[i] = idx
                        y_list[i] = yy[i, :-1].clone()
                if early_stop_num != -1 and (yy.shape[1] - y_len) > early_stop_num:
                    for i in range(B):
                        if idx_list[i] is None:
                            idx_list[i] = idx
                            y_list[i] = yy[i, :-1].clone()
                if all(v is not None for v in idx_list):
                    break
                y_emb1 = dec.ar_audio_embedding(yy[:, -1:])
                x_in = y_emb1 * dec.ar_audio_position.x_scale + \
                    dec.ar_audio_position.alpha * dec.ar_audio_position.pe[:, y_len + idx]
                xy_dec, kc, vc = dec.t2s_transformer.decode_next_token(x_in, kc, vc, ca)
                ca = F.pad(ca, (0, 1), value=False)
            for i in range(B):
                if idx_list[i] is None:
                    idx_list[i] = 1499
            return y_list, idx_list

        y_list, idx_list = greedy_run(args.early_stop)
        dump(outdir, "batch.greedy.idx", torch.tensor(idx_list, dtype=torch.float32).reshape(1, B))
        for i in range(B):
            yi = y_list[i].float().reshape(1, -1)
            dump(outdir, f"batch.greedy.seq{i}.tokens", yi)
            print(f"    greedy seq{i}: idx={idx_list[i]} len(y)={yi.numel()} tail={yi[0, -4:].tolist()}")

        # 早期停止路径 (early_stop 很小) 的参考, 用于覆盖该分支语义
        y_list2, idx_list2 = greedy_run(3)
        dump(outdir, "batch.earlystop.idx", torch.tensor(idx_list2, dtype=torch.float32).reshape(1, B))
        for i in range(B):
            dump(outdir, f"batch.earlystop.seq{i}.tokens", y_list2[i].float().reshape(1, -1))

    print("sampling + batch golden done ->", outdir)


if __name__ == "__main__":
    main()
