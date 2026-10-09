# DiT (F5 式 CFM estimator, v5: use_step_embedding=False) golden dump
#   用 repo 真实实现 (f5_tts.model.backbones.dit.DiT) 加载 s2Gv5turbo.pth 的 cfm.estimator.*
#   (checkpoint 里 estimator 权重是 fp16 -> .float() 载入 fp32 模型 = 推理端 CPU 精度行为)
#
# 布局约定 (golden 一律 raw float32, 行主 = torch layout):
#   [1,C,T] 形 (x0/prompt_x/mu/vel)  = ggml ne{T,C} 字节序 (T 最快), 与 encode() 入口同约定
#   [1,T,C] 形 (text_embed/static/...) = ggml ne{C,T}, 即 DiT 图内部主干布局
#
# 关键语义 (2026-10-09 探针确认):
#   - RoPE 只作用在 q/k 的前 64 维 (= attention 的 head 0), 其余 15 头不旋转
#     (x_transformers.apply_rotary_pos_emb 的 rot_dim = freqs.shape[-1] = 64)
#   - inv_freq 用 checkpoint 里 fp16 存储值 (与 fp32 公式差 1.6e-4 量级, 必须用存档值)
#   - time_embed 正弦 = cat(sin, cos)(1000 * t * exp(-i*log(10000)/127)) — 与 C++ float 逐位一致
import argparse, os, sys, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, "..", "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

import GPT_SoVITS.f5_tts.model.modules as M                    # noqa: E402


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-24s %s" % (name, tuple(arr.shape)))


from tools_dit_common import build_dit   # noqa: E402


def run_case(dit, out, tag, T, prompt_len, x_lens, seed=7, t_val=0.25):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(1, 100, T, generator=g) * 0.875
    prompt_x = torch.zeros(1, 100, T)
    prompt_x[..., :prompt_len] = torch.randn(1, 100, prompt_len, generator=g)
    mu = torch.randn(1, 512, T, generator=g)
    x_lens_t = torch.LongTensor([x_lens])
    t_t = torch.tensor([t_val], dtype=torch.float32)

    cap = {}
    rope_orig = M.apply_rotary_pos_emb
    def rope_wrap(t_, freqs_, scale=1):
        o = rope_orig(t_, freqs_, scale)
        if "q0" not in cap:
            cap["q0"] = o; cap["q0_in"] = t_
        elif "k0" not in cap:
            cap["k0"] = o
        return o
    M.apply_rotary_pos_emb = rope_wrap
    # text 路分段 (engine: dbg_te_pos = pos emb 后, dbg_te_cn0..2 = 前 3 个 ConvNeXtV2 输出)
    te_hooks = []
    for ii, blk in enumerate(dit.text_embed.text_blocks):
        te_hooks.append(blk.register_forward_hook(
            lambda m_, i_, o_, idx=ii: cap.__setitem__("te_cn%d" % idx, o_)))
    te_hooks.append(dit.text_embed.text_blocks[0].register_forward_pre_hook(
        lambda m_, i_: cap.__setitem__("te_pos", i_[0])))
    hooks = [
        dit.text_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("text_embed", o)),
        dit.input_embed.conv_pos_embed.register_forward_hook(
            lambda m, i, o: cap.__setitem__("convpos_out", o)),
        dit.transformer_blocks[0].attn_norm.register_forward_hook(
            lambda m, i, o: cap.__setitem__("b0_norm", (o[0], o[1], o[2], o[3], o[4]))),
        dit.transformer_blocks[0].attn.register_forward_hook(
            lambda m, i, o: cap.__setitem__("b0_attn", o)),
        dit.transformer_blocks[0].ff.register_forward_hook(
            lambda m, i, o: cap.__setitem__("b0_ff", o)),
        dit.transformer_blocks[0].register_forward_hook(
            lambda m, i, o: cap.__setitem__("b0_out", o)),
        dit.transformer_blocks[21].register_forward_hook(
            lambda m, i, o: cap.__setitem__("b21_out", o)),
        dit.norm_out.register_forward_hook(
            lambda m, i, o: cap.__setitem__("norm_out", o)),
    ] + te_hooks
    try:
        with torch.no_grad():
            cache = dit.prepare_static_cache(prompt_x, x_lens_t, mu)
            vel, text_emb, dt = dit(x0, prompt_x, x_lens_t, t_t, text0=mu, infer=True, static_cache=cache)
    finally:
        M.apply_rotary_pos_emb = rope_orig
        for h in hooks:
            h.remove()

    # x_lin = F.linear(x, proj[:, :100]) + static  (复算, 用于中间量对拍)
    with torch.no_grad():
        x_lin = torch.nn.functional.linear(
            x0.transpose(2, 1), dit.input_embed.proj.weight[:, :100]) + cache["condition"]

    p = "dit." + tag + "."
    dump(out, p + "x0", x0)
    dump(out, p + "prompt_x", prompt_x)
    dump(out, p + "mu", mu)
    dump(out, p + "t", t_t)
    dump(out, p + "x_lens", x_lens_t.float())
    dump(out, p + "vel", vel)
    dump(out, p + "text_embed", cap["text_embed"])
    dump(out, p + "te_pos", cap["te_pos"])
    for ii in range(3):
        dump(out, p + "te_cn%d" % ii, cap["te_cn%d" % ii])
    dump(out, p + "static", cache["condition"])
    dump(out, p + "neg_static", cache["negative_condition"])
    dump(out, p + "rope", cache["rope"][0])
    dump(out, p + "x_lin", x_lin)
    dump(out, p + "convpos_out", cap["convpos_out"])
    dump(out, p + "b0_norm", cap["b0_norm"][0])
    dump(out, p + "b0_gate_msa", cap["b0_norm"][1])
    dump(out, p + "b0_shift_mlp", cap["b0_norm"][2])
    dump(out, p + "b0_scale_mlp", cap["b0_norm"][3])
    dump(out, p + "b0_gate_mlp", cap["b0_norm"][4])
    dump(out, p + "b0_attn", cap["b0_attn"])
    dump(out, p + "b0_ff", cap["b0_ff"])
    dump(out, p + "b0_out", cap["b0_out"])
    dump(out, p + "b21_out", cap["b21_out"])
    dump(out, p + "norm_out", cap["norm_out"])
    dump(out, p + "q0_roped", cap["q0"])
    dump(out, p + "k0_roped", cap["k0"])
    dump(out, p + "q0_raw", cap["q0_in"])
    for st in cap["b0_norm"][1:]:
        assert st.shape[-1] == 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dit = build_dit(args.ckpt)

    # 主用例: T=96, prompt 32, 全长度 (v5 推理的实际形态: x_lens = T)
    print("[main] T=96 prompt=32 full-length")
    run_case(dit, args.out, "t96", T=96, prompt_len=32, x_lens=96, t_val=0.25)
    # 掩码用例: 同尺寸但 x_lens = T-16 (检验 mask 语义: SDPA bool mask + conv_pos 双端置零)
    print("[pad] T=96 prompt=32 x_lens=80")
    run_case(dit, args.out, "pad", T=96, prompt_len=32, x_lens=80, t_val=0.75, seed=11)
    # t=0 用例 (Euler 首步)
    print("[t0] T=64 prompt=16 t=0")
    run_case(dit, args.out, "t0", T=64, prompt_len=16, x_lens=64, t_val=0.0, seed=3)
    print("dit golden done ->", args.out)


if __name__ == "__main__":
    main()
