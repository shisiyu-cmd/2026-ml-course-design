import time, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model

arch, enc, size, bs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dev = 'cuda'
m = build_model(arch, enc).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-4)
x = torch.randn(bs, 3, size, size, device=dev)
y = (torch.rand(bs, 1, size, size, device=dev) > 0.9).float()
crit = torch.nn.BCEWithLogitsLoss()
for _ in range(2):
    l = crit(m(x), y); opt.zero_grad(); l.backward(); opt.step()
torch.cuda.synchronize()
t = time.time()
for _ in range(5):
    l = crit(m(x), y); opt.zero_grad(); l.backward(); opt.step()
torch.cuda.synchronize()
dt = (time.time() - t) / 5
print(f'{arch}/{enc} size={size} bs={bs}: {dt*1000:.0f} ms/step  ({dt/bs*1000:.0f} ms/img)  '
      f'peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GiB', flush=True)
