# wns1 golden dump: standalone Encoder + WN, weight-norm materialized from ckpt
import argparse, os
import numpy as np
import torch
from torch import nn

def wn_mat(g, v, eps=1e-12):
    n = v.norm(dim=list(range(1, v.dim())), keepdim=True).clamp_min(eps)
    return g * v / n

def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %s: %s" % (name, tuple(t.shape)))

class WN(nn.Module):
    def __init__(self, hidden, k, n_layers, gin):
        super().__init__()
        self.hidden = hidden; self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = nn.Conv1d(gin, 2*hidden*n_layers, 1)
        for i in range(n_layers):
            self.in_layers.append(nn.Conv1d(hidden, 2*hidden, k, padding=(k-1)//2))
            rc = 2*hidden if i < n_layers-1 else hidden
            self.res_skip_layers.append(nn.Conv1d(hidden, rc, 1))
    def forward(self, x, x_mask, g):
        output = torch.zeros_like(x)
        g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            off = i * 2 * self.hidden
            g_l = g[:, off:off + 2*self.hidden, :]
            a = x_in + g_l
            acts = torch.tanh(a[:, :self.hidden, :]) * torch.sigmoid(a[:, self.hidden:, :])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + rs[:, :self.hidden, :]) * x_mask
                output = output + rs[:, self.hidden:, :]
            else:
                output = output + rs
        return output * x_mask

class Encoder(nn.Module):
    def __init__(self, cin, cout, hidden, k, n_layers, gin):
        super().__init__()
        self.pre = nn.Conv1d(cin, hidden, 1)
        self.enc = WN(hidden, k, n_layers, gin)
        self.proj = nn.Conv1d(hidden, cout, 1)
    def forward(self, x, x_lengths, g):
        g = g.detach()
        T = x.size(2)
        x_mask = torch.arange(T).unsqueeze(0) < x_lengths.unsqueeze(1)
        x_mask = x_mask.unsqueeze(1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--len", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)
    raw = {k[5:]: v for k, v in sd.items() if k.startswith("wns1.")}
    msd = {}
    for k, v in raw.items():
        if k.endswith(".weight_g"):
            b = k[:-len("_g")]
            msd[b] = wn_mat(v, raw[b + "_v"])
        elif k.endswith(".weight_v"):
            continue
        else:
            msd[k] = v
    m = Encoder(512, 512, 512, 5, 8, 512).eval()
    missing, unexpected = m.load_state_dict(msd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 512, args.T, generator=g)
    ge = torch.randn(1, 512, 1, generator=g)
    xlen = torch.LongTensor([args.len])
    cap = {}
    m.pre.register_forward_hook(lambda mod, i, o: cap.__setitem__("pre", o))
    m.enc.register_forward_hook(lambda mod, i, o: cap.__setitem__("enc", o))
    with torch.no_grad():
        stats, x_mask = m(x, xlen, ge)
    dump(args.out, "wns1.input", x)
    dump(args.out, "wns1.ge", ge)
    dump(args.out, "wns1.pre_out", cap["pre"] * x_mask)
    dump(args.out, "wns1.enc_out", cap["enc"])
    dump(args.out, "wns1.out", stats)
    dump(args.out, "wns1.mask", x_mask)
    print("wns1 golden done ->", args.out)

if __name__ == "__main__":
    main()
