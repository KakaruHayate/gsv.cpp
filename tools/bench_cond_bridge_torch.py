# bridge baseline: torch CPU / CUDA (Conv1d 192->512 k=1 + LeakyReLU), 同 golden 输入
import time, numpy as np, torch
from torch import nn

T = 60
x = torch.from_numpy(np.fromfile('tests/golden/bridge.input.bin', dtype=np.float32)).reshape(1, 192, T)
sd = torch.load('models/s2Gv5turbo.pth', map_location='cpu', weights_only=False)
sd = sd.get('weight', sd)
conv = nn.Conv1d(192, 512, 1).eval()
conv.weight.data.copy_(sd['bridge.0.weight']); conv.bias.data.copy_(sd['bridge.0.bias'])
act = nn.LeakyReLU()
m = nn.Sequential(conv, act).eval()

def bench(mm, xx, dev, n=200, warm=20):
    with torch.inference_mode():
        for _ in range(warm): mm(xx)
        if dev == 'cuda': torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); mm(xx)
            if dev == 'cuda': torch.cuda.synchronize()
            ts.append((time.perf_counter()-t0)*1e3)
    ts.sort(); return sum(ts)/len(ts), ts[0]

for dest in ('cuda', 'cpu'):
    if dest == 'cuda' and not torch.cuda.is_available(): continue
    mm = m.to(dest); xx = x.to(dest)
    a, mn = bench(mm, xx, dest)
    print(f'torch {dest:4s} fp32: avg {a:8.4f} ms  min {mn:8.4f} ms  (T={T})')
