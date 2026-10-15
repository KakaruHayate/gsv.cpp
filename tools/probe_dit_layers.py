"""逐层定位: 用我们 dump 的 DiT 输入在 torch 侧跑 estimator, 捕获与引擎探针同名的中间量。

输入 (dit_prompt_x.f32 / dit_mu.f32 / noise.f32) 由 server 用
GSV_ENGINE_DUMP + GSV_DIT_DUMP + GSV_DIT_DEBUG 落盘 —— 两侧是同一份输入。
输出存到 <dump>/torch_<probe>.npy, 与 <dump>/dbg_<probe>.f32 对比。
"""
import os, sys
import numpy as np
import torch

root = r'J:\GPT-SoVITS\gsv.cpp'
REPO = os.path.abspath(os.path.join(root, os.pardir, 'repo'))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'GPT_SoVITS')); sys.path.insert(0, os.path.join(root, 'tools'))
os.chdir(os.path.join(REPO, 'GPT_SoVITS'))

from tools_dit_common import build_dit


def main():
    D = os.path.join(root, 'tests', '_dump')
    px = np.fromfile(os.path.join(D, 'dit_prompt_x.f32'), dtype='<f4')
    mu = np.fromfile(os.path.join(D, 'dit_mu.f32'), dtype='<f4')
    nz = np.fromfile(os.path.join(D, 'noise.f32'), dtype='<f4')
    T = px.size // 100
    prompt_x = torch.tensor(px).reshape(1, 100, T)
    mu_t = torch.tensor(mu).reshape(1, 512, T)
    x0 = torch.tensor(nz).reshape(1, 100, T)     # 已含 0.875 (引擎侧就是这么多)
    print('prompt_x', tuple(prompt_x.shape), 'mu', tuple(mu_t.shape), 'x0', tuple(x0.shape))

    dit = build_dit(os.path.join(root, 'models', 's2Gv5turbo.pth')).eval()
    print('d_embed (step emb) =', dit.d_embed)

    lens = torch.LongTensor([T])
    t_t = torch.tensor([0.0], dtype=torch.float32)

    cap = {}
    hooks = [
        dit.input_embed.conv_pos_embed.register_forward_hook(
            lambda m, i, o: cap.__setitem__('convpos', o)),
        dit.transformer_blocks[0].attn_norm.register_forward_hook(
            lambda m, i, o: cap.__setitem__('b0_norm', o[0])),
        dit.transformer_blocks[0].attn.register_forward_hook(
            lambda m, i, o: cap.__setitem__('b0_attn', o)),
        dit.transformer_blocks[0].ff.register_forward_hook(
            lambda m, i, o: cap.__setitem__('b0_ff', o)),
        dit.transformer_blocks[0].register_forward_hook(
            lambda m, i, o: cap.__setitem__('b0_out', o)),
        dit.transformer_blocks[21].register_forward_hook(
            lambda m, i, o: cap.__setitem__('b21_out', o)),
        dit.norm_out.register_forward_hook(
            lambda m, i, o: cap.__setitem__('norm_out', o)),
    ]
    try:
        with torch.no_grad():
            cache = dit.prepare_static_cache(prompt_x, lens, mu_t)
            vel, _, _ = dit(x0, prompt_x, lens, t_t, text0=mu_t, infer=True, static_cache=cache)
            x_lin = torch.nn.functional.linear(
                x0.transpose(2, 1), dit.input_embed.proj.weight[:, :100]) + cache['condition']
    finally:
        for h in hooks:
            h.remove()

    print('vel', tuple(vel.shape), 'x_lin', tuple(x_lin.shape))
    for k, v in [('vel', vel)] + [(k, cap[k]) for k in cap] + [('x_lin', x_lin)]:
        a = v[0].float().numpy() if v.dim() == 3 else v.float().numpy()
        np.save(os.path.join(D, 'torch_%s.npy' % k), a)
    print('saved:', ['vel'] + list(cap) + ['x_lin'])


if __name__ == '__main__':
    main()
