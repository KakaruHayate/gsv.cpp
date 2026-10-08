# ref_enc (MelStyleEncoder) golden dump —— 语义内联自 repo/GPT_SoVITS/module/modules.py
#   输入 refer[:, :704] * refer_mask  [1, 704, T], mask [1,1,T]
#   spectral: Linear(704->128)+Mish -> Linear(128->128)+Mish
#   temporal: 2 x Conv1dGLU(k=5, pad=2): conv(128->256) -> split -> x1*sigmoid(x2) -> +residual
#   slf_attn: 2 heads d_k=d_v=64, temperature=sqrt(128), mask -> -inf  (fc + residual)
#   fc: Linear(128->512); temporal_avg_pool(mask) -> [1,512,1]
import argparse, os, math
import numpy as np
import torch
from torch import nn


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class Conv1dGLU(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.out_channels = out_ch
        self.conv1 = nn.Module()
        self.conv1.conv = nn.Conv1d(in_ch, 2 * out_ch, k, padding=(k - 1) // 2)

    def forward(self, x):
        residual = x
        y = self.conv1.conv(x)
        x1, x2 = torch.split(y, self.out_channels, dim=1)
        return residual + x1 * torch.sigmoid(x2)


class MHA(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.temperature = math.pow(d_model, 0.5)
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

    def forward(self, x, mask=None):
        nh, dk, dv = self.n_head, self.d_k, self.d_v
        B, T, _ = x.size()
        residual = x
        q = self.w_qs(x).view(B, T, nh, dk).permute(2, 0, 1, 3).contiguous().view(-1, T, dk)
        k = self.w_ks(x).view(B, T, nh, dk).permute(2, 0, 1, 3).contiguous().view(-1, T, dk)
        v = self.w_vs(x).view(B, T, nh, dv).permute(2, 0, 1, 3).contiguous().view(-1, T, dv)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask.repeat(nh, 1, 1), -np.inf)
        attn = torch.softmax(attn, dim=2)
        out = torch.bmm(attn, v).view(nh, B, T, dv).permute(1, 2, 0, 3).contiguous().view(B, T, -1)
        return self.fc(out) + residual, attn


class MelStyleEncoder(nn.Module):
    def __init__(self, n_mel=704, hidden=128, out=512, k=5, head=2):
        super().__init__()
        self.hidden_dim, self.out_dim, self.kernel_size, self.n_head = hidden, out, k, head
        self.spectral = nn.Sequential(
            LinearNorm(n_mel, hidden), Mish(), nn.Identity(),
            LinearNorm(hidden, hidden), Mish(), nn.Identity())
        self.temporal = nn.Sequential(Conv1dGLU(hidden, hidden, k), Conv1dGLU(hidden, hidden, k))
        self.slf_attn = MHA(head, hidden, hidden // head, hidden // head)
        self.fc = LinearNorm(hidden, out)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        len_ = (~mask).sum(dim=1).unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        return (x.float() / len_.unsqueeze(1)).sum(dim=1)

    def forward(self, x, mask=None, hooks=None):
        x = x.transpose(1, 2)
        if mask is not None:
            mask = (mask.int() == 0).squeeze(1)
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        x = self.spectral(x)
        if hooks is not None: hooks["spectral"] = x.clone()
        x = x.transpose(1, 2)
        x = self.temporal(x)
        if hooks is not None: hooks["temporal"] = x.clone()
        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, attn = self.slf_attn(x, mask=slf_attn_mask)
        if hooks is not None: hooks["attn"] = x.clone()
        x = self.fc(x)
        if hooks is not None: hooks["fc"] = x.clone()
        w = self.temporal_avg_pool(x, mask=mask)
        return w.unsqueeze(-1)


def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %s: %s" % (name, tuple(t.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--len", type=int, default=180)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    m = MelStyleEncoder()
    msd = {k[len("ref_enc."):]: v for k, v in sd.items() if k.startswith("ref_enc.")}
    missing, unexpected = m.load_state_dict(msd, strict=True)
    m = m.eval()
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 704, args.T, generator=g)
    mask = torch.zeros(1, 1, args.T)
    mask[:, :, :args.len] = 1.0
    hooks = {}
    with torch.no_grad():
        ge = m(x * mask, mask, hooks=hooks)
    dump(args.out, "refenc.input", x)
    dump(args.out, "refenc.mask", mask)
    dump(args.out, "refenc.spectral_out", hooks["spectral"])   # [1,T,128]
    dump(args.out, "refenc.temporal_out", hooks["temporal"])   # [1,128,T]
    dump(args.out, "refenc.attn_out", hooks["attn"])           # [1,T,128]
    dump(args.out, "refenc.fc_out", hooks["fc"])               # [1,T,512]
    dump(args.out, "refenc.out", ge)                           # [1,512,1]
    print("ref_enc golden done ->", args.out)


if __name__ == "__main__":
    main()
