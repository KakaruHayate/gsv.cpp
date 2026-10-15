"""逐层对拍: 用 dump 的输入构造 mu/prompt_x, 比较我们与 torch 的 estimator **单步**输出。

端到端输出与 torch 差 0.31 但单模块对拍只有 1.2e-3, 需要定位是哪一步开始偏的。
这里拦截 torch 侧 estimator 的每步输入/输出, 与 server 侧 dump 的同名张量对比。
"""
import os, sys
import numpy as np
import torch

root = r'J:\GPT-SoVITS\gsv.cpp'
REPO = os.path.abspath(os.path.join(root, os.pardir, 'repo'))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'GPT_SoVITS')); sys.path.insert(0, os.path.join(root, 'tools'))
os.chdir(os.path.join(REPO, 'GPT_SoVITS'))

from tools_dit_common import build_dit
from module.models_v5 import CFMV5

D = os.path.join(root, 'tests', '_dump')
REF = 500


def rb(n, dt='<f4'):
    return np.fromfile(os.path.join(D, n), dtype=dt)


def main():
    fr, ft, m2 = rb('fea_ref.f32'), rb('fea_todo.f32'), rb('mel2.f32')
    R, N, Tm = fr.size // 512, ft.size // 512, m2.size // 100
    rc = min(Tm, R); off = max(0, rc - REF); rf = rc - off
    fea_ref_s = fr.reshape(512, -1)[:, off:off + rf].copy()
    mel_ref_s = m2.reshape(100, -1)[:, off:off + rf].copy()
    # 本例只有一块 (N=340 <= chunk 640)
    T = rf + N
    mu = np.concatenate([fea_ref_s, ft.reshape(512, -1)], axis=1)          # [512, T]
    prompt_x = np.zeros((100, T), dtype=np.float32)
    prompt_x[:, :rf] = mel_ref_s
    print('mu', mu.shape, 'prompt_x', prompt_x.shape)

    dit = build_dit(os.path.join(root, 'models', 's2Gv5turbo.pth'))
    cfm = CFMV5(100, dit).eval()
    print('use_step_embedding =', cfm.use_step_embedding, ' noise_temperature =', cfm.noise_temperature)

    cap = []
    orig = type(cfm.estimator).forward

    def fwd(self, x, cond0, x_lens, time, *a, **k):
        rec = {'x': x.detach().clone(), 'cond0': cond0.detach().clone(),
               'time': float(time.reshape(-1)[0])}
        out = orig(self, x, cond0, x_lens, time, *a, **k)
        rec['vel'] = out[0].detach().clone()          # [1, T, 100]
        cap.append(rec)
        return out

    type(cfm.estimator).forward = fwd
    # noise.f32 是引擎已经乘过 0.875 的初始 x; 而 CFM.inference 内部还会做
    #   x = torch.randn(...) * temperature, temperature = noise_temperature = 0.875
    # 所以这里要**还原成未缩放的 randn**, 否则会被双重缩放。
    x0 = torch.tensor(rb("noise.f32")).reshape(1, 100, T) / 0.875
    orig_randn = torch.randn
    torch.randn = lambda *a, **k: x0.clone()
    try:
        with torch.inference_mode():
            # repo: mu = cat(features, target).transpose(2, 1) -> [B, T, C];
            #       CFM.inference 内部再 transpose(2, 1) 拿回 [B, C, T]
            mu_t = torch.tensor(mu.T).unsqueeze(0)                # [1, T, 512] (repo 的 transpose(2,1))
            prompt_t = torch.tensor(mel_ref_s).unsqueeze(0)       # [1, 100, rf] (mel 布局不转置)
            lens = torch.full((1,), T, dtype=torch.long)
            out = cfm.inference(mu_t, lens, prompt_t, 4, inference_cfg_rate=0.0)
    finally:
        torch.randn = orig_randn
        type(cfm.estimator).forward = orig
    print('steps captured:', len(cap), ' times:', [round(r['time'], 4) for r in cap])
    np.save(os.path.join(D, 'torch_vels.npy'), np.stack([r['vel'][0].float().numpy() for r in cap]))
    np.save(os.path.join(D, 'torch_mel_out.npy'), out[0].float().numpy())
    print('torch mel_out', tuple(out.shape))
    print('saved vels ->', os.path.join(D, 'torch_vels.npy'))


if __name__ == '__main__':
    main()
