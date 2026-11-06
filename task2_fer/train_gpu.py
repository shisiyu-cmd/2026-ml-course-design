"""
Unified FER training script (GPU-ready).

Design:
  * Images cached in RAM as 48x48 uint8 grayscale, upscaled on the fly.
  * StratifiedKFold; every fold trains on all classes.
  * Produces per-fold checkpoints + out-of-fold (OOF) logits so we can measure
    ensemble quality *honestly* before touching the test set.
  * Test logits are saved per fold for later weighted ensembling.

Usage:
  python train_gpu.py --arch resnet18 --size 64 --epochs 40 --folds 5 --tag r18_64
"""
import os, time, random, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold

CACHE = r"C:\Users\古恩宇\Downloads\2026task2\cache\data.npz"
CKPT_ROOT = r"C:\Users\古恩宇\Downloads\2026task2\ckpt"
OUT_ROOT = r"C:\Users\古恩宇\Downloads\2026task2\preds"
os.makedirs(CKPT_ROOT, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class FERDataset(Dataset):
    """48x48 uint8 grayscale in RAM; augmentation done in torch/PIL."""

    def __init__(self, imgs, labels, size, train):
        self.imgs = imgs
        self.labels = labels
        self.size = size
        self.train = train
        self.to_t = transforms.ToTensor()
        self.norm = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = Image.fromarray(self.imgs[i]).convert("RGB")
        if self.train:
            # geometric augmentation at 48px is lossy, so do it after upscale
            im = im.resize((self.size, self.size), Image.BILINEAR)
            # random affine: rotation + translate + scale + shear
            if random.random() < 0.7:
                ang = random.uniform(-12, 12)
                sc = random.uniform(0.92, 1.08)
                tx = random.uniform(-0.07, 0.07) * self.size
                ty = random.uniform(-0.07, 0.07) * self.size
                sh = random.uniform(-8, 8)
                im = transforms.functional.affine(
                    im, angle=ang, translate=(tx, ty), scale=sc, shear=sh,
                    interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            if random.random() < 0.5:
                im = transforms.functional.hflip(im)
            x = self.to_t(im)
            # photometric
            if random.random() < 0.5:
                x = x * random.uniform(0.8, 1.25) + random.uniform(-0.1, 0.1)
                x = x.clamp(0, 1)
            if random.random() < 0.3:
                x = transforms.functional.gaussian_blur(x, kernel_size=3,
                                                        sigma=random.uniform(0.1, 1.0))
            x = self.norm(x)
            # cutout / random erasing
            if random.random() < 0.35:
                _, h, w = x.shape
                eh = random.randint(int(h * 0.1), int(h * 0.33))
                ew = random.randint(int(w * 0.1), int(w * 0.33))
                ey = random.randint(0, h - eh)
                ex = random.randint(0, w - ew)
                x[:, ey:ey + eh, ex:ex + ew] = 0.0
            return x, int(self.labels[i])
        else:
            im = im.resize((self.size, self.size), Image.BILINEAR)
            return self.norm(self.to_t(im)), int(self.labels[i])


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_model(arch, nc=6, drop=0.0):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, nc)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, nc)
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, nc)
    else:
        raise ValueError(arch)
    return m


# ----------------------------------------------------------------------------
# Mixup / CutMix
# ----------------------------------------------------------------------------
def mix_batch(x, y, alpha=0.3, p_mix=0.5, p_cut=0.5):
    """Return x, ya, yb, lam. Sometimes no mixing."""
    r = random.random()
    if r > p_mix:
        return x, y, y, 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    if random.random() < p_cut:  # CutMix
        lam = np.random.beta(alpha, alpha)
        H, W = x.size(2), x.size(3)
        cut = np.sqrt(1 - lam)
        cw, ch = int(W * cut), int(H * cut)
        cx, cy = np.random.randint(0, W), np.random.randint(0, H)
        x1, x2 = max(0, cx - cw // 2), min(W, cx + cw // 2)
        y1, y2 = max(0, cy - ch // 2), min(H, cy + ch // 2)
        xm = x.clone()
        xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        return xm, y, y[idx], lam
    else:  # Mixup
        lam = np.random.beta(alpha, alpha)
        xm = lam * x + (1 - lam) * x[idx]
        return xm, y, y[idx], lam


@torch.no_grad()
def predict_logits(model, loader, tta_flip=True):
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to("cuda", non_blocking=True)
        logits = model(x)
        if tta_flip:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
        outs.append(logits.float().cpu())
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only_fold", type=int, default=0, help="0 = all folds")
    ap.add_argument("--p_mix", type=float, default=0.5)
    ap.add_argument("--ls", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    d = np.load(CACHE, allow_pickle=True)
    Xtr, ytr, Xte = d["Xtr"], d["ytr"], d["Xte"]
    print(f"train {Xtr.shape}  test {Xte.shape}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    splits = list(skf.split(Xtr, ytr))

    ckdir = os.path.join(CKPT_ROOT, args.tag)
    os.makedirs(ckdir, exist_ok=True)
    fold_accs = []
    oof_logits = np.zeros((len(Xtr), 6), dtype=np.float32)
    test_logits_all = []

    te_ds = FERDataset(Xte, np.zeros(len(Xte), dtype=np.int64), args.size, False)
    te_ld = DataLoader(te_ds, batch_size=256, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    folds = range(1, args.folds + 1) if args.only_fold == 0 else [args.only_fold]

    for fold, (tri, vai) in enumerate(splits, 1):
        if fold not in folds:
            continue
        t0 = time.time()
        print(f"\n===== FOLD {fold}/{args.folds}  train={len(tri)} val={len(vai)} =====", flush=True)
        tr_ds = FERDataset(Xtr[tri], ytr[tri], args.size, True)
        va_ds = FERDataset(Xtr[vai], ytr[vai], args.size, False)
        tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True,
                           persistent_workers=True)
        va_ld = DataLoader(va_ds, batch_size=256, shuffle=False,
                           num_workers=args.workers, pin_memory=True, persistent_workers=True)

        model = build_model(args.arch).cuda()
        # differential LR: backbone lower, head higher
        head_params = [p for n, p in model.named_parameters() if "fc" in n or "classifier" in n]
        head_ids = {id(p) for p in head_params}
        base_params = [p for p in model.parameters() if id(p) not in head_ids]
        opt = optim.AdamW([
            {"params": base_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 8},
        ], weight_decay=args.wd)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

        best = 0.0
        ck = os.path.join(ckdir, f"fold{fold}.pt")
        for ep in range(1, args.epochs + 1):
            model.train()
            tl, tn = 0.0, 0
            for x, y in tr_ld:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                x, ya, yb, lam = mix_batch(x, y, alpha=0.3, p_mix=args.p_mix)
                logits = model(x)
                loss = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tl += loss.item() * x.size(0); tn += x.size(0)
            sched.step()

            if ep >= args.epochs - 12 or ep % 5 == 0:
                lg = predict_logits(model, va_ld, tta_flip=False)
                acc = (lg.argmax(1).numpy() == ytr[vai]).mean()
                if acc > best:
                    best = acc
                    torch.save(model.state_dict(), ck)
                if ep % 5 == 0 or ep >= args.epochs - 3:
                    print(f"  ep{ep:02d} loss {tl/tn:.3f} va {acc:.4f} best {best:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            elif ep % 5 == 0:
                print(f"  ep{ep:02d} loss {tl/tn:.3f} ({time.time()-t0:.0f}s)", flush=True)

        # reload best and produce OOF + test logits with flip TTA
        model.load_state_dict(torch.load(ck, map_location="cuda"))
        oof_logits[vai] = predict_logits(model, va_ld, tta_flip=True).numpy()
        acc_tta = (oof_logits[vai].argmax(1) == ytr[vai]).mean()
        fold_accs.append(float(acc_tta))
        print(f"  FOLD {fold} best={best:.4f}  TTA={acc_tta:.4f}", flush=True)

        test_logits_all.append(predict_logits(model, te_ld, tta_flip=True).numpy())
        np.save(os.path.join(OUT_ROOT, f"{args.tag}_test_fold{fold}.npy"),
                test_logits_all[-1])
        del model, tr_ld, va_ld
        torch.cuda.empty_cache()

    np.save(os.path.join(OUT_ROOT, f"{args.tag}_oof.npy"), oof_logits)
    np.save(os.path.join(OUT_ROOT, f"{args.tag}_ytest.npy"), ytr)
    acc = float((oof_logits.argmax(1) == ytr).mean())
    print(f"\n===== {args.tag}: OOF acc = {acc:.4f}  folds={[round(a,4) for a in fold_accs]} =====", flush=True)
    with open(os.path.join(OUT_ROOT, f"{args.tag}_meta.json"), "w") as fh:
        json.dump({"arch": args.arch, "size": args.size, "epochs": args.epochs,
                   "oof_acc": acc, "fold_accs": fold_accs}, fh, indent=2)


if __name__ == "__main__":
    main()
