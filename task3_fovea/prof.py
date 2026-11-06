# -*- coding: utf-8 -*-
"""训练速度剖析：区分 数据增强 / 前向 / 反向 的耗时。"""
import time, argparse
import numpy as np, torch, torch.nn.functional as F
import common as C

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="resnet18")
p.add_argument("--bs", type=int, default=8)
p.add_argument("--iters", type=int, default=20)
a = p.parse_args()

dev = torch.device("cuda")
meta = C.load_meta()
cache = C.build_cache(meta, verbose=False)
gt = C.load_gt(); gt_map = C.load_gt_map(gt)
keys = sorted(gt_map.keys())

print("GPU:", torch.cuda.get_device_name(0))
print("显存总量: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

# 1) 数据增强
ds = C.FoveaDataset(keys, cache, meta, gt_map, True, 2.0, seed=0)
t = time.time()
N = a.bs * 10
for i in range(N):
    ds[i % len(ds)]
print("数据增强: %.1f ms/样本 (batch=%d -> %.0f ms)" % ((time.time() - t) / N * 1000, a.bs,
                                                    (time.time() - t) / N * 1000 * a.bs))

# 2) 模型前向/反向
m = C.FoveaNet(a.backbone, pretrained=True).to(dev)
nparam = sum(x.numel() for x in m.parameters()) / 1e6
x = torch.randn(a.bs, 3, C.CANVAS, C.CANVAS, device=dev)
tgt = torch.rand(a.bs, 1, C.CANVAS // C.STRIDE, C.CANVAS // C.STRIDE, device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=True)
for _ in range(3):
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize()
t = time.time()
for _ in range(a.iters):
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
torch.cuda.synchronize()
dt = (time.time() - t) / a.iters
print("模型 训练一步: %.0f ms  (%.1f M 参数, 输出 %s)" % (dt * 1000, nparam, tuple(y.shape)))
print("峰值显存: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

m.eval()
with torch.no_grad():
    for _ in range(3):
        m(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(a.iters):
        m(x)
    torch.cuda.synchronize()
print("模型 推理一步: %.0f ms" % ((time.time() - t) / a.iters * 1000))