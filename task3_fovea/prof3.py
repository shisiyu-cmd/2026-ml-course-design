# -*- coding: utf-8 -*-
"""对比 uint8 NHWC -> float NCHW 归一化的几种实现。"""
import time
import numpy as np
import torch, torch.nn.functional as F
import common as C

dev = torch.device("cuda")
bs = 8
u8 = torch.randint(0, 255, (bs, C.CANVAS, C.CANVAS, 3), dtype=torch.uint8, device=dev)
mean = torch.tensor(C.NORM_MEAN, device=dev).view(1, 3, 1, 1)
std = torch.tensor(C.NORM_STD, device=dev).view(1, 3, 1, 1)
m = C.FoveaNet("resnet18", pretrained=True).to(dev)
mcl = C.FoveaNet("resnet18", pretrained=True).to(dev).to(memory_format=torch.channels_last)
tgt = torch.rand(bs, 1, C.CANVAS // C.STRIDE, C.CANVAS // C.STRIDE, device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
opt2 = torch.optim.AdamW(mcl.parameters(), lr=1e-4)


def bench(name, fn, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    print("  %-46s %6.0f ms" % (name, dt * 1000))


# --- 仅归一化 ---
def n_cur():
    x = u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean) / std


def n_contig():
    x = u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return (x - mean) / std


def n_cl():
    x = u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean) / std


def n_cpu_before():
    return None  # 见下方单独测量


print("仅归一化:")
bench("a) permute -> float (当前实现)", n_cur)
bench("b) permute -> contiguous -> float", n_contig)

u8cpu = u8.cpu()
t = time.time()
for _ in range(15):
    c = u8cpu.permute(0, 3, 1, 2).contiguous()
print("  c) CPU 上 permute+contiguous:            %6.0f ms" % ((time.time() - t) / 15 * 1000))
t = time.time()
for _ in range(15):
    a = u8cpu.numpy().transpose(0, 3, 1, 2)
    c2 = np.ascontiguousarray(a)
print("  d) numpy transpose+ascontiguousarray:    %6.0f ms" % ((time.time() - t) / 15 * 1000))

# --- 完整一步 ---
def step_cur():
    x = (u8.permute(0, 3, 1, 2).float().div_(255.0) - mean) / std
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def step_contig():
    x = (u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0) - mean) / std
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


def step_cl():
    x = (u8.permute(0, 3, 1, 2).float().div_(255.0) - mean) / std
    with torch.amp.autocast("cuda"):
        y = mcl(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()


def step_chw():
    x = u8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    x = (x - mean) / std
    with torch.amp.autocast("cuda"):
        y = m(x); loss = F.mse_loss(torch.sigmoid(y).float(), tgt)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


print()
print("完整训练一步 (归一化 + 前反向):")
bench("A) 当前实现", step_cur)
bench("B) permute->contiguous->float", step_contig)
bench("C) channels_last 模型 + 免转置", step_cl)
print()
print("说明: B 需要 CPU 侧额外做 CHW 拷贝, C 不需要。")