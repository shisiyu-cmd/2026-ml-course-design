"""
Single-fold recipe experiment: does higher res + longer schedule + EMA +
moderate augmentation beat the prior recipe? Reports fold-1 val accuracy
(with flip TTA) so it is directly comparable to prior_oof numbers.
"""
import os, time, random, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold

WORK = r"C:\Users\古恩宇\Downloads\2026task2"
CACHE = os.path.join(WORK, "cache", "data.npz")
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]


class DS(Dataset):
    def __init__(self, imgs, labels, size, train, aug="moderate"):
        self.imgs, self.labels, self.size, self.train, self.aug = imgs, labels, size, train, aug
        self.to_t = transforms.ToTensor(); self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = Image.fromarray(self.imgs[i]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        if self.train:
            if self.aug == "heavy":
                if random.random() < 0.8:
                    im = transforms.functional.affine(
                        im, angle=random.uniform(-15, 15), translate=(random.uniform(-.08, .08) * self.size,) * 2,
                        scale=random.uniform(0.9, 1.1), shear=random.uniform(-10, 10),
                        interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                if random.random() < 0.5:
                    im = transforms.functional.hflip(im)
                x = self.norm(self.to_t(im))
                if random.random() < 0.5:
                    x = (x * random.uniform(0.75, 1.3) + random.uniform(-.12, .12)).clamp(-2.2, 2.2)
                if random.random() < 0.4:
                    _, h, w = x.shape
                    eh, ew = random.randint(h // 10, h // 3), random.randint(w // 10, w // 3)
                    ey, ex = random.randint(0, h - eh), random.randint(0, w - ew)
                    x[:, ey:ey + eh, ex:ex + ew] = 0.
                return x, int(self.labels[i])
            else:  # moderate
                if random.random() < 0.6:
                    im = transforms.functional.affine(
                        im, angle=random.uniform(-10, 10), translate=(random.uniform(-.06, .06) * self.size,) * 2,
                        scale=random.uniform(0.94, 1.06), shear=random.uniform(-6, 6),
                        interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                if random.random() < 0.5:
                    im = transforms.functional.hflip(im)
                x = self.norm(self.to_t(im))
                if random.random() < 0.4:
                    x = (x * random.uniform(0.85, 1.18) + random.uniform(-.08, .08)).clamp(-2.2, 2.2)
                if random.random() < 0.25:
                    _, h, w = x.shape
                    eh, ew = random.randint(h // 12, h // 5), random.randint(w // 12, w // 5)
                    ey, ex = random.randint(0, h - eh), random.randint(0, w - ew)
                    x[:, ey:ey + eh, ex:ex + ew] = 0.
                return x, int(self.labels[i])
        return self.norm(self.to_t(im)), int(self.labels[i])


class EMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def build(arch):
    if arch == "r18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, 6)
    elif arch == "r34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, 6)
    elif arch == "dense":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1); m.classifier = nn.Linear(m.classifier.in_features, 6)
    elif arch == "mbv3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 6)
    else:
        raise ValueError(arch)
    return m


def mix(x, y, p=0.5):
    if random.random() > p:
        return x, y, y, 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < 0.5:
        lam = np.random.beta(0.4, 0.4)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam
    lam = np.random.beta(0.4, 0.4)
    H, W = x.size(2), x.size(3); cut = np.sqrt(1 - lam)
    cw, ch = int(W * cut), int(H * cut)
    cx, cy = np.random.randint(0, W), np.random.randint(0, H)
    x1, x2 = max(0, cx - cw // 2), min(W, cx + cw // 2)
    y1, y2 = max(0, cy - ch // 2), min(H, cy + ch // 2)
    xm = x.clone(); xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    return xm, y, y[idx], 1 - (x2 - x1) * (y2 - y1) / (W * H)


@torch.no_grad()
def ev(model, ld):
    model.eval(); out = []
    for x, _ in ld:
        x = x.cuda()
        out.append((0.5 * (model(x) + model(torch.flip(x, dims=[3])))).float().cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="r34"); ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--aug", default="moderate")
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_mix", type=float, default=0.5)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True

    d = np.load(CACHE, allow_pickle=True)
    Xtr, ytr = d["Xtr"], d["ytr"]
    tri, vai = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, ytr))[a.fold - 1]
    trl = DataLoader(DS(Xtr[tri], ytr[tri], a.size, True, a.aug), batch_size=a.bs, shuffle=True,
                     num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val = DataLoader(DS(Xtr[vai], ytr[vai], a.size, False, a.aug), batch_size=256, shuffle=False,
                     num_workers=8, pin_memory=True, persistent_workers=True)

    m = build(a.arch).cuda()
    head = [p for n, p in m.named_parameters() if "fc" in n or "classifier" in n]
    hid = {id(p) for p in head}
    base = [p for p in m.parameters() if id(p) not in hid]
    opt = optim.AdamW([{"params": base, "lr": a.lr}, {"params": head, "lr": a.lr * 8}], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.lr * 0.005)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    ema = EMA(m, 0.999)

    t0 = time.time(); best = 0
    for ep in range(1, a.epochs + 1):
        m.train(); tl = tn = 0
        for x, y in trl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x, ya, yb, lam = mix(x, y, a.p_mix)
            loss = lam * crit(m(x), ya) + (1 - lam) * crit(m(x), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ema.update(m); tl += loss.item() * x.size(0); tn += x.size(0)
        sch.step()
        if ep % 10 == 0 or ep >= a.epochs - 4:
            acc = (ev(m, val).argmax(1).numpy() == ytr[vai]).mean()
            eacc = (ev(ema.ema, val).argmax(1).numpy() == ytr[vai]).mean()
            best = max(best, eacc, acc)
            print(f"  ep{ep:02d} loss {tl/tn:.3f} raw {acc:.4f} ema {eacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"RESULT arch={a.arch} size={a.size} ep={a.epochs} aug={a.aug} fold{a.fold}: best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
