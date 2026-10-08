# torch CPU 分段: feature_extractor(CNN) / feature_projection / pos_conv / encoder
import time, numpy as np, torch
from transformers import HubertModel

m = HubertModel.from_pretrained('models/chinese-hubert-base', local_files_only=True).eval().float()
x = torch.from_numpy(np.fromfile('tests/golden/hubert.input_norm.bin', dtype=np.float32)).unsqueeze(0)

def timeit(fn, n=30, warm=5):
    with torch.inference_mode():
        for _ in range(warm): fn()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return sum(ts)/len(ts), ts[0], ts[len(ts)//2]

fe = m.feature_extractor
fp = m.feature_projection
pc = m.encoder.pos_conv_embed
ln = m.encoder.layer_norm
enc = m.encoder.layers

with torch.inference_mode():
    h = fe(x)                       # [1, 512, 49]
    ht = h.transpose(1, 2)          # [1, 49, 512]
    hp = fp(ht)                     # [1, 49, 768]

for name, fn in [("feature_extractor(cnn)", lambda: fe(x)),
                 ("feature_projection",    lambda: fp(ht)),
                 ("pos_conv",              lambda: pc(hp)),
                 ("encoder.layer_norm",    lambda: ln(hp)),
                 ("encoder 12 layers",     lambda: [l(hp)[0] for l in enc]),
                 ("full forward",          lambda: m(x))]:
    a, mn, med = timeit(fn)
    print(f"{name:22s} avg {a:7.2f}  min {mn:7.2f}  med {med:7.2f}  ms")
