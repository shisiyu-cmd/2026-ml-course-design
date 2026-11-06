# -*- coding: utf-8 -*-
"""逐项定位「训练一步」耗时差异。"""
import time
import torch, torch.nn.functional as F
import common as C

dev = torch.device("cuda")
bs = 8
torch.manual_seed(0)
m = C.FoveaNet("resnet18", pretrained=True).to(dev)
x = torch.randn(bs, 3, C.CANVAS, C.CANVAS, device=dev)
tgt = torch.rand(bs, 1, C.CANVAS // C.STRIDE, C.CANVAS // C.STRIDE, device=dev)
gcn = torch.rand(bs, 2, device=dev) * C.CANVAS
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=True)


def bench(name, step_fn, iters=15, sync_each=False):
    for _ in range(3):
        step_fn()
    if sync_each:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        if sync_each:
            torch.cuda.synchronize()
        step_fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    print("  %-38s %6.0f ms" % (name, dt * 1000))
    return dt


def s_basic():
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()


def s_softargmax():
    with torch.amp.autocast("cuda"):
        y = m(x); prob = torch.sigmoid(y)
        loss = F.mse_loss(prob.float(), tgt)
        coord = C.soft_argmax(prob.float(), 20.0) * C.STRIDE
        loss = loss + 5.0 * F.l1_loss(coord / C.CANVAS, gcn / C.CANVAS)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()


def s_full_floatloss():
    with torch.amp.autocast("cuda"):
        y = m(x); prob = torch.sigmoid(y)
        loss = F.mse_loss(prob.float(), tgt)
        coord = C.soft_argmax(prob.float(), 20.0) * C.STRIDE
        loss = loss + 5.0 * F.l1_loss(coord / C.CANVAS, gcn / C.CANVAS)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    float(loss)          # 强制同步


def s_norm():
    u8 = torch.randint(0, 255, (bs, C.CANVAS, C.CANVAS, 3), dtype=torch.uint8, device=dev)
    img = C.normalize_batch(u8)
    with torch.amp.autocast("cuda"):
        y = m(img); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()


print("batch = %d, canvas = %d" % (bs, C.CANVAS))
print("GPU:", torch.cuda.get_device_name(0))
print("当前 GPU 利用率/时钟:")
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print()
bench("A 基础 (forward+backward, 异步流水)", s_basic)
bench("B + soft_argmax + L1 坐标损失", s_softargmax)
bench("C = B + float(loss) 逐迭代同步", s_full_floatloss)
bench("D = B + uint8 归一化", s_norm)
bench("E 基础 (每步同步)", s_basic, sync_each=True)
print()
print("nvidia-smi:")
print(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())