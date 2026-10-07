# GSV HuBERT golden dump via official forward + hooks (version-robust).
import argparse
import os
import numpy as np
import torch

DEF_CKPT = "models/chinese-hubert-base"


def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %s: shape=%s" % (name, tuple(t.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--samples", type=int, default=16000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import HubertModel
    model = HubertModel.from_pretrained(args.ckpt, local_files_only=True)
    model = model.eval().float()

    g = torch.Generator().manual_seed(42)
    wav = torch.rand(args.samples, generator=g) * 2.0 - 1.0
    dump(args.out, "hubert.input", wav)
    x = (wav - wav.mean()) / (wav.std() + 1e-7)
    dump(args.out, "hubert.input_norm", x)
    xb = x.unsqueeze(0)

    cap = {}

    def mk_hook(name):
        def hook(mod, inp, out):
            cap[name] = out
        return hook

    model.feature_extractor.register_forward_hook(mk_hook("cnn"))
    model.feature_projection.register_forward_hook(mk_hook("proj"))

    def pre0(mod, inp):
        cap["enc_in"] = inp[0]

    model.encoder.layers[0].register_forward_pre_hook(pre0)

    with torch.no_grad():
        out = model(input_values=xb)
    last = out.last_hidden_state
    dump(args.out, "hubert.cnn_out", cap["cnn"])
    dump(args.out, "hubert.feat_proj", cap["proj"])
    dump(args.out, "hubert.enc_in", cap["enc_in"])
    dump(args.out, "hubert.out", last)
    dump(args.out, "hubert.out_t", last.transpose(1, 2))
    print("HuBERT golden done ->", args.out)


if __name__ == "__main__":
    main()
