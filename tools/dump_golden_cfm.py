# CFM (v5turbo: steps=4/cfg=0, v5dev: 32/cfg=1.30) 多步 Euler golden
#   用 repo 真实 CFMV5 (module.models_v5) + 真实 DiT, 捕获: 初始噪声 (randn 拦截) 与每步的
#   正/负分支速度 (estimator 前向 hook) —— host 循环只需: x += step*vel; x[..., :prompt_len] = 0
# 另含分块 (synthesize_v5_mel 调度) 用例: 小 chunk_frames 强制多块, 检验 rolling prompt 语义
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, "..", "repo"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))

from tools_dit_common import build_dit  # noqa: E402


def dump(outdir, name, t):
    arr = t.detach().cpu().float().contiguous().numpy()
    arr.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %-24s %s" % (name, tuple(arr.shape)))


def run_infer(cfm, out, tag, mu, prompt, steps, cfg, seed=1234, light=False):
    """在真实 CFMV5.inference 上拦截 randn + estimator, 逐步落盘
    light=True: 只落 mu/prompt/noise/out (T=1000 等大尺寸, per-step 落盘太重)"""
    cap = {"steps": []}
    orig_randn = torch.randn
    orig_forward = type(cfm.estimator).forward

    def rnd(*a, **k):
        t = orig_randn(*a, **k)
        cap["noise"] = t
        return t

    def fwd(self, x, cond0, x_lens, time, *a, **k):
        rec = {"x_in": x.detach().clone(),
               "time": float(time.reshape(-1)[0]),
               "drop_audio": bool(k.get("drop_audio_cond", False)),
               "static": k.get("static_cache") is not None}
        cap["steps"].append(rec)
        outp = orig_forward(self, x, cond0, x_lens, time, *a, **k)
        rec["vel"] = outp[0].detach().clone()      # [1, T, 100] (proj_out 输出)
        return outp

    torch.randn = rnd
    type(cfm.estimator).forward = fwd
    try:
        with torch.inference_mode():
            torch.manual_seed(seed)
            result = cfm.inference(mu, torch.LongTensor([mu.shape[1]]), prompt, steps,
                                   inference_cfg_rate=cfg)
    finally:
        torch.randn = orig_randn
        type(cfm.estimator).forward = orig_forward

    p = "cfm." + tag + "."
    dump(out, p + "mu", mu.transpose(2, 1))     # estimator 视角 [1,512,T] (= triton/text0 输入)
    dump(out, p + "prompt", prompt)
    dump(out, p + "noise", cap["noise"])
    dump(out, p + "t", torch.tensor([0.0]))
    dump(out, p + "steps", torch.tensor([float(steps)]))
    dump(out, p + "cfg", torch.tensor([float(cfg)]))
    dump(out, p + "out", result)
    n = 0
    for i, rec in enumerate(cap["steps"]):
        if light:
            break
        dump(out, p + "s%d.x_in" % i, rec["x_in"])
        dump(out, p + "s%d.vel" % i, rec["vel"])            # 原始 [1,T,100]; 无 cfg 分支则每步一条
        dump(out, p + "s%d.time" % i, torch.tensor([rec["time"]]))
        n = i + 1
    # 每步实际更新的速度 = vel.transpose(2,1) (+ cfg*(pos-neg)); 上面 dump 的是 estimator 原样输出
    print("  [%s] steps=%d cfg=%.2f captured=%d static=%s" %
          (tag, steps, cfg, len(cap["steps"]), [r["static"] for r in cap["steps"][:2]]))


def run_chunked(cfm, out, tag, fea_ref, fea_todo, mel_ref, block, steps=4, cfg=0.0, seed=99):
    """复刻 synthesize_v5_mel 的 rolling-prompt 分块调度 (参数化 block, 生产值 640)
    同时拦截 torch.randn 记录每块的初始噪声 (供 C++ 复算)"""
    noises = []
    orig_randn = torch.randn
    def rnd(*a, **k):
        t = orig_randn(*a, **k)
        noises.append(t.detach().clone())
        return t
    torch.randn = rnd
    try:
        return _run_chunked(cfm, out, tag, fea_ref, fea_todo, mel_ref, block, steps, cfg, seed, noises)
    finally:
        torch.randn = orig_randn


def _run_chunked(cfm, out, tag, fea_ref, fea_todo, mel_ref, block, steps, cfg, seed, noises):
    REF, TAIL, TOTAL = 500, 32, 1000
    reference_frames = min(mel_ref.shape[-1], fea_ref.shape[-1])
    mel_ref = mel_ref[..., :reference_frames][..., -REF:]
    fea_ref = fea_ref[..., :reference_frames][..., -REF:]
    reference_frames = mel_ref.shape[-1]
    chunk_frames = min(TOTAL - reference_frames, block)
    original_mel, original_features = mel_ref, fea_ref
    rolling_mel, rolling_features = mel_ref, fea_ref
    results = []
    torch.manual_seed(seed)
    for index, start in enumerate(range(0, fea_todo.shape[-1], chunk_frames)):
        target = fea_todo[..., start:start + chunk_frames]
        if index == 0:
            prompt, features = original_mel, original_features
        else:
            tail = min(TAIL, reference_frames, rolling_mel.shape[-1], rolling_features.shape[-1])
            prefix = reference_frames - tail
            prompt = torch.cat((original_mel[..., :prefix], rolling_mel[..., -tail:]), dim=-1)
            features = torch.cat((original_features[..., :prefix], rolling_features[..., -tail:]), dim=-1)
        mu = torch.cat((features, target), dim=-1).transpose(2, 1)
        lengths = torch.full((mu.shape[0],), mu.shape[1], dtype=torch.long)
        with torch.inference_mode():
            gen = cfm.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg)[..., prompt.shape[-1]:]
        results.append(gen)
        rolling_mel = gen[..., -reference_frames:]
        rolling_features = target[..., -reference_frames:]
    res = torch.cat(results, dim=-1)
    p = "cfm." + tag + "."
    dump(out, p + "fea_ref", fea_ref)
    dump(out, p + "fea_todo", fea_todo)
    dump(out, p + "mel_ref", mel_ref)
    dump(out, p + "block", torch.tensor([float(block)]))
    dump(out, p + "out", res)
    for i, nz in enumerate(noises):
        dump(out, p + "noise%d" % i, nz)
    print("  [%s] per-chunk noises dumped: %d" % (tag, len(noises)))
    print("  [%s] chunked: ref=%d todo=%d block=%d -> out %s (chunks=%d)" %
          (tag, reference_frames, fea_todo.shape[-1], chunk_frames, tuple(res.shape), len(results)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    from module.models_v5 import CFMV5
    dit = build_dit(args.ckpt)
    cfm = CFMV5(100, dit).eval()

    g = torch.Generator().manual_seed(2026)
    T = 96
    mu = torch.randn(1, 512, T, generator=g)
    prompt = torch.randn(1, 100, 32, generator=g) * 0.8
    # CFM.inference 的 mu 约定是 [b, T, dim] (内部 transpose 成 [b,dim,T] 给 estimator)
    mu_bt = mu.transpose(2, 1).contiguous()

    # v5turbo 生产形态: 4 步 cfg=0
    print("[cfg0] steps=4 cfg=0 (v5turbo)")
    run_infer(cfm, args.out, "s4c0", mu_bt, prompt, steps=4, cfg=0.0)
    # cfg 分支: 8 步 cfg=1.30 (每一步两个 forward: pos + neg)
    print("[cfg13] steps=8 cfg=1.30")
    run_infer(cfm, args.out, "s8c13", mu_bt, prompt, steps=8, cfg=1.30, seed=77)

    # v5dev 生产形态: 32 步 cfg=1.30 (cache-dit 消融的参考; 逐步落盘)
    print("[cfg13-32] steps=32 cfg=1.30 (v5dev)")
    run_infer(cfm, args.out, "s32c13", mu_bt, prompt, steps=32, cfg=1.30, seed=78)

    # 生产尺寸 (T=1000, prompt 500): 只落最终 mel + 噪声 (per-step 落盘太重)
    g3 = torch.Generator().manual_seed(7777)
    T2, P2 = 1000, 500
    mu2 = torch.randn(1, 512, T2, generator=g3)
    prompt2 = torch.randn(1, 100, P2, generator=g3) * 0.8
    mu2_bt = mu2.transpose(2, 1).contiguous()
    print("[s4c0_big] T=1000 steps=4 cfg=0")
    run_infer(cfm, args.out, "s4c0_big", mu2_bt, prompt2, steps=4, cfg=0.0, seed=202, light=True)
    print("[s32c13_big] T=1000 steps=32 cfg=1.30")
    run_infer(cfm, args.out, "s32c13_big", mu2_bt, prompt2, steps=32, cfg=1.30, seed=203, light=True)

    # 分块调度 (小 block 强制 3 块): fea_todo 长度 12, block=5, ref=4
    g2 = torch.Generator().manual_seed(4242)
    fea_ref = torch.randn(1, 512, 4, generator=g2)
    fea_todo = torch.randn(1, 512, 12, generator=g2)
    mel_ref = torch.randn(1, 100, 4, generator=g2)
    print("[chunk] block=5")
    run_chunked(cfm, args.out, "chunk", fea_ref, fea_todo, mel_ref, block=5)
    print("cfm golden done ->", args.out)


if __name__ == "__main__":
    main()
