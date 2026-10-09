# 条件链 golden dump (v5turbo decode_encp 链, 语义内联自 repo/GPT_SoVITS/module/models.py):
#   codes -> quantizer.decode (codebook gather) -> F.interpolate(x2 nearest) ->
#   enc_p (TextEncoder) -> x (proj 前 y) -> bridge (Conv1d 192->512 k=1 + LeakyReLU) ->
#   F.interpolate(x2 nearest) -> wns1 (Encoder: pre + WN 8 epoch + proj) -> fea
# 说明: v5turbo 走的是 SynthesizerTrnV3.decode_encp 的非 v3 分支
#   (semantic_frame_rate=25hz: x2; bridge 后 x2; y_lengths1 = int(T_c*4) = 长度一致)
import argparse, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_h1 = None
def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def dump(outdir, name, t):
    t = t.detach().cpu().float().contiguous().numpy()
    t.tofile(os.path.join(outdir, name + ".bin"))
    print("  dumped %s: %s" % (name, tuple(t.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="tests/golden")
    ap.add_argument("--ncodes", type=int, default=40)
    ap.add_argument("--ntext", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dge = _load("tools/dump_golden_encp.py", "dge")     # TextEncoder (enc_p)
    dgw = _load("tools/dump_golden_wns1.py", "dgw")     # wns1 Encoder (weight-norm)

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)

    # enc_p
    msd = {}
    for k, v in sd.items():
        if not k.startswith("enc_p."):
            continue
        kk = k[len("enc_p."):]
        kk = kk.replace("norm_layers_1.", "norm1.").replace("norm_layers_2.", "norm2.")
        msd[kk] = v
    encp = dge.TextEncoder().eval()
    encp.load_state_dict(msd, strict=True)

    # wns1 (weight_norm 物化)
    raw = {k[5:]: v for k, v in sd.items() if k.startswith("wns1.")}
    wsd = {}
    for k, v in raw.items():
        if k.endswith(".weight_g"):
            b = k[:-len("_g")]
            wsd[b] = dgw.wn_mat(v, raw[b + "_v"])
        elif k.endswith(".weight_v"):
            continue
        else:
            wsd[k] = v
    wns1 = dgw.Encoder(512, 512, 512, 5, 8, 512).eval()
    wns1.load_state_dict(wsd, strict=False)

    # bridge (checkpoint 里 bridge 权重是 fp16, 转 fp32 与其余模块一致)
    cbw = sd["bridge.0.weight"].float()   # [512,192,1]
    cbb = sd["bridge.0.bias"].float()

    codebook = sd["quantizer.vq.layers.0._codebook.embed"].float()   # [1024, 768]

    g = torch.Generator().manual_seed(42)
    codes = torch.randint(0, 1024, (1, 1, args.ncodes), generator=g)
    text  = torch.randint(0, 732, (1, args.ntext), generator=g)
    ge    = torch.randn(1, 512, 1, generator=g)

    with torch.no_grad():
        quant = torch.nn.functional.embedding(codes[:, 0, :], codebook).transpose(1, 2)   # [1,768,Tc]
        quant = torch.nn.functional.interpolate(quant, scale_factor=2, mode="nearest")    # [1,768,2Tc]
        yl = torch.LongTensor([quant.size(2)])
        tl = torch.LongTensor([text.size(-1)])
        x, m, logs, ymask = encp(quant, text, ge)                                         # x = proj 前 y
        y_ = x                                                                            # (speed=1 时 y_==x)
        br = torch.nn.functional.leaky_relu(
                torch.nn.functional.conv1d(x, cbw, cbb), 0.01)                            # [1,512,2Tc]
        up = torch.nn.functional.interpolate(br, scale_factor=2, mode="nearest")          # [1,512,4Tc]
        fea, _ = wns1(up, torch.LongTensor([up.size(2)]), ge)                             # [1,512,4Tc]

    dump(args.out, "chain.codes", codes.view(-1).to(torch.int32).float())
    dump(args.out, "chain.text", text.view(-1).to(torch.int32).float())
    dump(args.out, "chain.ge", ge)
    dump(args.out, "chain.y", y_)        # [1,192,2Tc] enc_p 的 y (speed=1 时 y_==x)
    dump(args.out, "chain.bridge_raw", br)   # [1,512,2Tc] bridge 输出 (×2 之前)
    dump(args.out, "chain.bridge", up)   # [1,512,4Tc] bridge ×2 nearest 之后 (= wns1 输入)
    dump(args.out, "chain.fea", fea)     # [1,512,4Tc]
    print("chain golden done ->", args.out)


if __name__ == "__main__":
    main()
