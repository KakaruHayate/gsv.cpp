# enc_p (TextEncoder + MRTE) golden dump —— 语义内联自
#   repo/GPT_SoVITS/module/models_onnx.py (TextEncoder) / module/attentions.py (Encoder/MHA/FFN)
#   / module/mrte_model.py (MRTE)
# 超参 (从张量形状反推): hidden=192, filter=768, heads=2 (dk=96), window=4, ffn_k=3;
#   encoder_ssl 3 层 / encoder_text 6 层 / encoder2 3 层; text_embedding 732 词; proj 384(=192*2)
# mask 语义: v5 推理路径 y_mask/text_mask 全 1 (无 padding, mask 恒等)
#   scores.masked_fill(mask==0, -1e4) 在 mask 全 1 时不生效
# rel-pos: window_size=4, emb_rel_k/v [1, 9, 96]; pad→slice→pad-to-absolute 流程照抄
import argparse, os, math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HID, FLT, HEADS, DK, WIN, K_FFN = 192, 768, 2, 96, 4, 3
N_SSL, N_TXT, N_E2 = 3, 6, 3


def mish_like(x):  # 未用, 留空
    return x


class MHA(nn.Module):
    def __init__(self, hidden=HID, heads=HEADS, has_rel=True):
        super().__init__()
        self.has_rel = has_rel
        self.heads, self.dk = heads, hidden // heads
        self.conv_q = nn.Conv1d(hidden, hidden, 1)
        self.conv_k = nn.Conv1d(hidden, hidden, 1)
        self.conv_v = nn.Conv1d(hidden, hidden, 1)
        self.conv_o = nn.Conv1d(hidden, hidden, 1)
        if has_rel:
            self.emb_rel_k = nn.Parameter(torch.randn(1, 2 * WIN + 1, hidden // heads) * 0.01)
            self.emb_rel_v = nn.Parameter(torch.randn(1, 2 * WIN + 1, hidden // heads) * 0.01)

    def _get_rel(self, emb, length):
        max_pos = 2 * WIN + 1
        pad_length = max(length - (WIN + 1), 0)
        start = max((WIN + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            emb = F.pad(emb, (0, 0, pad_length, pad_length))
        return emb[:, start:end]

    def _rel_to_abs(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, (0, 1))
        x = x.view(b, h, l * 2 * l)
        x = F.pad(x, (0, l - 1))
        return x.view(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1:]

    def _abs_to_rel(self, x):
        b, h, l, _ = x.size()
        x = F.pad(x, (0, l - 1))
        x = x.view(b, h, l ** 2 + l * (l - 1))
        x = F.pad(x, (l, 0))
        return x.view(b, h, l, 2 * l)[:, :, :, 1:]

    def forward(self, x, c, mask=None):
        heads, dk = self.heads, self.dk
        q = self.conv_q(x); k = self.conv_k(c); v = self.conv_v(c)
        b, d, t_s, t_t = *k.size(), q.size(2)
        q = q.view(b, heads, dk, t_t).transpose(2, 3)
        k = k.view(b, heads, dk, t_s).transpose(2, 3)
        v = v.view(b, heads, dk, t_s).transpose(2, 3)
        scores = torch.matmul(q / math.sqrt(dk), k.transpose(-2, -1))
        if self.has_rel:
            rel_k = self._get_rel(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(q / math.sqrt(dk), rel_k.unsqueeze(0).transpose(-2, -1))
            scores = scores + self._rel_to_abs(rel_logits)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p = F.softmax(scores, dim=-1)
        out = torch.matmul(p, v)
        if self.has_rel:
            rel_v = self._get_rel(self.emb_rel_v, t_s)
            rw = self._abs_to_rel(p)
            out = out + torch.matmul(rw, rel_v.unsqueeze(0))
        out = out.transpose(2, 3).contiguous().view(b, d, t_t)
        return self.conv_o(out)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(HID, FLT, K_FFN)
        self.conv_2 = nn.Conv1d(FLT, HID, K_FFN)

    def forward(self, x):
        x = F.pad(x, (K_FFN // 2, K_FFN // 2))
        x = F.relu(self.conv_1(x))
        x = self.conv_2(F.pad(x, (K_FFN // 2, K_FFN // 2)))
        return x


class LN(nn.Module):
    def __init__(self, c=HID):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        # x [B, C, T]: 沿通道 C 归一 (自定义 LN 与 nn.LayerNorm 不同)
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-5) * self.gamma[None, :, None] + self.beta[None, :, None]


class Encoder(nn.Module):
    def __init__(self, n_layers, hidden=HID):
        super().__init__()
        self.attn_layers = nn.ModuleList(MHA(hidden=hidden) for _ in range(n_layers))
        self.norm1 = nn.ModuleList(LN(hidden) for _ in range(n_layers))
        self.ffn_layers = nn.ModuleList(FFN() for _ in range(n_layers))
        self.norm2 = nn.ModuleList(LN(hidden) for _ in range(n_layers))

    def forward(self, x, hooks=None, seg=''):
        for i in range(len(self.attn_layers)):
            y = self.attn_layers[i](x, x)
            x = self.norm1[i](x + y)
            y = self.ffn_layers[i](x)
            x = self.norm2[i](x + y)
            if hooks is not None:
                hooks[f'{seg}{i}'] = x.clone()
        return x


class MRTE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = MHA(hidden=512, heads=4, has_rel=False)
        self.c_pre = nn.Conv1d(192, 512, 1)
        self.text_pre = nn.Conv1d(192, 512, 1)
        self.c_post = nn.Conv1d(512, 192, 1)

    def forward(self, ssl_enc, ssl_mask, text, text_mask, ge):
        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask) + ssl_enc + ge
        return self.c_post(x * ssl_mask)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_proj = nn.Conv1d(768, HID, 1)
        self.encoder_ssl = Encoder(N_SSL)
        self.encoder_text = Encoder(N_TXT)
        self.text_embedding = nn.Embedding(732, HID)
        self.mrte = MRTE()
        self.encoder2 = Encoder(N_E2)
        self.proj = nn.Conv1d(HID, 384, 1)

    def forward(self, y, text, ge, hooks=None):
        y_mask = torch.ones_like(y[:1, :1, :])
        y = self.ssl_proj(y * y_mask) * y_mask
        y = self.encoder_ssl(y)
        text_mask = torch.ones_like(text).unsqueeze(0).to(y.dtype)
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)
        y = self.encoder2(y * y_mask)
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, 192, dim=1)
        return y, m, logs, y_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/s2Gv5turbo.pth')
    ap.add_argument('--out', default='tests/golden')
    ap.add_argument('--T', type=int, default=120)   # 语义 token 帧
    ap.add_argument('--ntext', type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('weight', sd)
    msd = {}
    for k, v in sd.items():
        if not k.startswith('enc_p.'):
            continue
        kk = k[len('enc_p.'):]
        kk = kk.replace('norm_layers_1.', 'norm1.').replace('norm_layers_2.', 'norm2.')
        msd[kk] = v
    m = TextEncoder().eval()
    missing, unexpected = m.load_state_dict(msd, strict=True)

    g = torch.Generator().manual_seed(42)
    y = torch.randn(1, 768, args.T, generator=g)            # RVQ 输出 (量化后)
    text = torch.randint(0, 732, (1, args.ntext), generator=g)  # text token
    ge = torch.randn(1, 512, 1, generator=g)                # 全局风格向量

    hooks = {}
    orig_attn = m.encoder_ssl.attn_layers
    class Hk:
        pass
    with torch.no_grad():
        # 逐层 hook: encoder_ssl attention 输出
        for li, layer in enumerate(m.encoder_ssl.attn_layers):
            layer.register_forward_hook((lambda li: lambda mod, i, o: hooks.__setitem__(f'ssl_attn{li}', o.clone()))(li))
        # 阶段 hook: ssl 输出 / text 输出 / mrte 输出 / encoder2 输出
        y_mask = torch.ones_like(y[:1, :1, :])
        stage = {}
        y_in = m.ssl_proj(y * y_mask) * y_mask
        stage['ssl_proj'] = y_in.clone()
        y_in = m.encoder_ssl(y_in)
        stage['ssl'] = y_in.clone()
        tmask = torch.ones_like(text).unsqueeze(0).to(y.dtype)
        temb = m.text_embedding(text).transpose(1, 2)
        stage['temb'] = temb.clone()
        tenc = m.encoder_text(temb * tmask)
        stage['text'] = tenc.clone()
        ym = m.mrte(y_in, y_mask, tenc, tmask, ge)
        stage['mrte'] = ym.clone()
        ym = m.encoder2(ym * y_mask)
        stage['enc2'] = ym.clone()
        stats = m.proj(ym) * y_mask
        stage['stats'] = stats.clone()
        y_enc, mm, logs, y_mask2 = m(y, text, ge)
    def dump(name, t):
        t = t.detach().cpu().float().contiguous().numpy()
        t.tofile(os.path.join(args.out, name + '.bin'))
        print('  dumped %s: %s' % (name, tuple(t.shape)))
    dump('encp.input_y', y)
    dump('encp.text', text.to(torch.int32).float())
    dump('encp.ge', ge)
    dump('encp.y_enc', y_enc)
    dump('encp.m', mm)
    dump('encp.logs', logs)
    for kk, vv in hooks.items():
        dump('encp.hook_' + kk, vv)
    for kk, vv in stage.items():
        dump('encp.stage_' + kk, vv)
    print('enc_p golden done ->', args.out)


if __name__ == '__main__':
    main()
