"""验证我们 dump 的 fea_ref / fea_todo 是否等于 torch 的 decode_encp 输出。

复刻 tools/dump_golden_chain.py 的链 (codes -> RVQ/x2 -> enc_p -> bridge/x2 -> wns1),
输入用 server dump 出来的 sem / prompt codes / phones / ge, 与 dump 的 fea_ref / fea_todo 对比。
"""
import os, sys
import numpy as np
import torch

root = r'J:\GPT-SoVITS\gsv.cpp'
REPO = os.path.abspath(os.path.join(root, os.pardir, 'repo'))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'GPT_SoVITS')); sys.path.insert(0, os.path.join(root, 'tools'))
os.chdir(os.path.join(REPO, 'GPT_SoVITS'))

_h = {}
def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

dge = _load('tools/dump_golden_encp.py', 'dge')
dgw = _load('tools/dump_golden_wns1.py', 'dgw')

D = os.path.join(root, 'tests', '_dump')
def rb(n, dt):
    return np.fromfile(os.path.join(D, n), dtype=dt)

sem = rb('sem.i32', '<i4')
prompt = rb('prompt_codes.i32', '<i4') if os.path.exists(os.path.join(D, 'prompt_codes.i32')) else None
phones = rb('phones.i32', '<i4')
prompt_phones = rb('prompt_phones.i32', '<i4') if os.path.exists(os.path.join(D, 'prompt_phones.i32')) else None
ge = rb('ge.f32', '<f4') if os.path.exists(os.path.join(D, 'ge.f32')) else None
fea_ref = rb('fea_ref.f32', '<f4').reshape(512, -1)
fea_todo = rb('fea_todo.f32', '<f4').reshape(512, -1)
print('sem', sem.shape, 'phones', phones.shape, 'prompt_phones',
      None if prompt_phones is None else prompt_phones.shape,
      'fea_ref', fea_ref.shape, 'fea_todo', fea_todo.shape)
if prompt is None or ge is None or prompt_phones is None:
    print('缺 prompt_codes.i32 / prompt_phones.i32 / ge.f32 —— 需先在 server 侧 dump'); sys.exit(1)
print('prompt', prompt.shape, 'ge', ge.shape)

sd = torch.load(os.path.join(root, 'models', 's2Gv5turbo.pth'), map_location='cpu', weights_only=False)
sd = sd.get('weight', sd)

msd = {}
for k, v in sd.items():
    if not k.startswith('enc_p.'): continue
    kk = k[len('enc_p.'):].replace('norm_layers_1.', 'norm1.').replace('norm_layers_2.', 'norm2.')
    msd[kk] = v
encp = dge.TextEncoder().eval(); encp.load_state_dict(msd, strict=True)

raw = {k[5:]: v for k, v in sd.items() if k.startswith('wns1.')}
wsd = {}
for k, v in raw.items():
    if k.endswith('.weight_g'):
        b = k[:-len('_g')]; wsd[b] = dgw.wn_mat(v, raw[b + '_v'])
    elif k.endswith('.weight_v'): continue
    else: wsd[k] = v
wns1 = dgw.Encoder(512, 512, 512, 5, 8, 512).eval(); wns1.load_state_dict(wsd, strict=False)

cbw = sd['bridge.0.weight'].float(); cbb = sd['bridge.0.bias'].float()
codebook = sd['quantizer.vq.layers.0._codebook.embed'].float()
ge_t = torch.tensor(ge).reshape(1, 512, 1)
# 参考侧用**参考文本**的音素, 目标侧用目标文本的音素 (repo TTS.py 的两处 decode_encp)
text_ref = torch.tensor(prompt_phones.astype(np.int64)).unsqueeze(0)
text_tgt = torch.tensor(phones.astype(np.int64)).unsqueeze(0)

def decode_encp(codes, text):
    quant = torch.nn.functional.embedding(torch.tensor(codes.astype(np.int64)).unsqueeze(0), codebook).transpose(1, 2)
    quant = torch.nn.functional.interpolate(quant, scale_factor=2, mode='nearest')
    yl = torch.LongTensor([quant.size(2)]); tl = torch.LongTensor([text.size(-1)])
    x, m, logs, ymask = encp(quant, text, ge_t)
    br = torch.nn.functional.leaky_relu(torch.nn.functional.conv1d(x, cbw, cbb), 0.01)
    up = torch.nn.functional.interpolate(br, scale_factor=2, mode='nearest')
    fea, _ = wns1(up, torch.LongTensor([up.size(2)]), ge_t)
    return fea[0].float().numpy()          # [512, 4Tc]

with torch.no_grad():
    fr_t = decode_encp(prompt, text_ref)
    ft_t = decode_encp(sem, text_tgt)
print('torch fea_ref', fr_t.shape, ' fea_todo', ft_t.shape)
for name, ours, tt in (('fea_ref', fea_ref, fr_t), ('fea_todo', fea_todo, ft_t)):
    n = min(ours.shape[1], tt.shape[1])
    d = np.abs(ours[:, :n] - tt[:, :n])
    print('%s: ours %s torch %s -> %d cols max|d|=%.3e mean=%.3e refmax=%.3f corr=%.6f'
          % (name, ours.shape, tt.shape, n, d.max(), d.mean(), np.abs(tt).max(),
             np.corrcoef(ours[:, :n].ravel(), tt[:, :n].ravel())[0, 1]))
    per_col = d.max(axis=0)
    print('   per-100-col max:', np.round([per_col[i:i+100].max() for i in range(0, n, 100)], 3).tolist())
    print('   cols > 0.1: %d / %d' % ((per_col > 0.1).sum(), n))
    bad = np.where(per_col > 0.1)[0]
    if bad.size:
        print('   bad range: %d .. %d, segments=%d' % (bad.min(), bad.max(), 1 + int((np.diff(bad) > 1).sum())))

# 分段定位: 比较 decode_encp 的中间产物 (enc_p 的 y / bridge 输出), 看偏差从哪一级开始
with torch.no_grad():
    quant_r = torch.nn.functional.interpolate(
        torch.nn.functional.embedding(torch.tensor(prompt.astype(np.int64)).unsqueeze(0), codebook).transpose(1, 2),
        scale_factor=2, mode='nearest')
    yl = torch.LongTensor([quant_r.size(2)]); tl = torch.LongTensor([text_t.size(-1)])
    x_r, _, _, _ = encp(quant_r, text_ref, ge_t)        # [1, 192, 422] (proj 前的 y)
    br_r = torch.nn.functional.leaky_relu(torch.nn.functional.conv1d(x_r, cbw, cbb), 0.01)
    up_r = torch.nn.functional.interpolate(br_r, scale_factor=2, mode='nearest')
    print('torch 中间产物: quant %s  enc_p.y %s  bridge %s  up %s'
          % (tuple(quant_r.shape), tuple(x_r.shape), tuple(br_r.shape), tuple(up_r.shape)))
    np.save(os.path.join(D, 'torch_encp_y.npy'), x_r[0].float().numpy())
    np.save(os.path.join(D, 'torch_quant.npy'), quant_r[0].float().numpy())
    np.save(os.path.join(D, 'torch_bridge.npy'), br_r[0].float().numpy())
    np.save(os.path.join(D, 'torch_up.npy'), up_r[0].float().numpy())
