"""Benchmark CPU training throughput for candidate FER models."""
import time, os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

torch.set_num_threads(os.cpu_count())
print("threads:", torch.get_num_threads(), "cpus:", os.cpu_count())

N = 28275
for name, size, bs in [("resnet18", 64, 128), ("resnet18", 96, 128),
                       ("resnet34", 96, 128), ("resnet18", 112, 96)]:
    if name == "resnet18":
        m = models.resnet18(weights=None)
    else:
        m = models.resnet34(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, size, size)
    y = torch.randint(0, 6, (bs,))
    # warmup
    for _ in range(2):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    n_iter = 6
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    dt = (time.time() - t0) / n_iter
    per_epoch = dt * (N / bs)
    print(f"{name} @{size}px bs{bs}: {dt*1000:.0f} ms/iter -> {per_epoch:.0f} s/epoch "
          f"({per_epoch/60:.1f} min) ; 5 folds x 20 ep = {per_epoch*100/60:.0f} min")
