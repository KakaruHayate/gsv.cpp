# 参考音频 -> prompt semantic tokens 链路的权重 -> GGUF:
#   s2Gv5turbo.pth 里的 ssl_proj (Conv1d 768->768 k=2 stride=2) 与 quantizer codebook
#   -> models/gsv-refcode.gguf
# 布局约定: 2D/3D 权重 torch [out,in(,k)] 行主直写 = ggml ne{in,k,out}
#   - ssl_proj.weight [OC,IC,K=2] 拆成 w0/w1 两个 [IC,OC] (= ggml ne{IC,OC}, mul_mat src0 直用;
#     若存 3D ne{K,IC,OC} 则 ic 维被 k 打断, 切片 view 无法直接用)
#   - codebook embed [1024,768] -> ggml ne{768,1024}
import argparse, os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/s2Gv5turbo.pth")
    ap.add_argument("--out", default="models/gsv-refcode.gguf")
    args = ap.parse_args()

    import gguf
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("weight", sd)

    w = gguf.GGUFWriter(args.out, "gsv.refcode")
    w.add_architecture()
    w.add_string("general.name", "gsv-refcode")
    w.add_uint32("refcode.ssl_dim", 768)
    w.add_uint32("refcode.conv_kernel", 2)
    w.add_uint32("refcode.conv_stride", 2)
    w.add_uint32("refcode.bins", 1024)

    def add(name, t):
        arr = np.ascontiguousarray(t.detach().float().numpy())
        w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        print("  %-28s %s" % (name, tuple(arr.shape)))

    w3 = sd["ssl_proj.weight"].float()              # torch [OC, IC, K]
    assert tuple(w3.shape) == (768, 768, 2), w3.shape
    # 直接拆成两个 [IC, OC] (numpy [OC,IC] 行主 -> ggml ne{IC,OC}, 即 mul_mat 的 src0)
    add("refcode.ssl_proj.w0", w3[:, :, 0].contiguous())
    add("refcode.ssl_proj.w1", w3[:, :, 1].contiguous())
    add("refcode.ssl_proj.bias", sd["ssl_proj.bias"])
    add("refcode.codebook.embed", sd["quantizer.vq.layers.0._codebook.embed"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", args.out, os.path.getsize(args.out), "bytes")


if __name__ == "__main__":
    main()
