"""Train retinal-vessel segmentation with k-fold CV.

Usage:
  python src/train.py --arch unetplusplus --encoder efficientnet-b4 --size 512 \
      --epochs 200 --folds 0,1,2,3 --tag upp_b4_512
"""
import argparse
import json
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TRAIN_IMG, WORK, VesselDataset, dice_score, ensure_dir,  # noqa: E402
                    list_external_pairs, list_images)
from models import build_loss, build_model  # noqa: E402

cv2.setNumThreads(0)


def get_aug(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.20), translate_percent=(-0.06, 0.06),
                 rotate=(-45, 45), shear=(-8, 8),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.75),
        A.RandomResizedCrop(size=(size, size), scale=(0.60, 1.0), ratio=(0.85, 1.18), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(pairs, size, train, batch_size, workers, aug=None):
    ds = VesselDataset(pairs, size=size, train=train, aug=aug, with_label=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True, drop_last=False,
                      persistent_workers=workers > 0)


@torch.no_grad()
def predict_probs(model, loader, device, size):
    """Return (probs at native resolution, gt at native resolution, stems)."""
    model.eval()
    probs, gts, stems = [], [], []
    for batch in loader:
        if len(batch) == 3:
            img, msk, st = batch
            gts.extend(msk.numpy()[:, 0])
        else:
            img, st = batch
        img = img.to(device, non_blocking=True)
        logits = model(img)
        if logits.shape[-2:] != (size, size):
            logits = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy()[:, 0])
        stems.extend(list(st))
    return probs, gts, stems


@torch.no_grad()
def evaluate(model, loader, device, size):
    probs, gts, _ = predict_probs(model, loader, device, size)
    return float(np.mean([dice_score(p > 0.5, g > 0.5) for p, g in zip(probs, gts)]))


def resize_prob(p, shape):
    return cv2.resize(p.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def save_oof(model, pairs, size, device, out_path, batch_size=4):
    """Save OOF probabilities and ground truth resized back to native resolution."""
    ds = VesselDataset(pairs, size=size, train=False, aug=None, with_label=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, gts, stems = predict_probs(model, loader, device, size)
    out_p, out_g = [], []
    for p, g, st in zip(probs, gts, stems):
        h, w = g.shape
        out_p.append(resize_prob(p, (h, w)))
        out_g.append(g)
    np.savez_compressed(out_path, probs=np.stack(out_p), gts=np.stack(out_g),
                        stems=np.array(stems))
    return np.mean([dice_score(p > 0.5, g > 0.5) for p, g in zip(out_p, out_g)])


def train_one_fold(args, fold, tr_pairs, va_pairs, device):
    size = args.size
    tag = f"{args.tag}_f{fold}"
    out_dir = ensure_dir(os.path.join(WORK, "checkpoints", tag))

    model = build_model(args.arch, args.encoder).to(device)
    if args.init_from:
        src = os.path.join(WORK, "checkpoints", f"{args.init_from}_f{fold}", "best.pth")
        if os.path.exists(src):
            ck = torch.load(src, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"[{tag}] initialised from {src} (val_dice={ck.get('val_dice', -1):.4f})", flush=True)
        else:
            print(f"[{tag}] WARNING init_from not found: {src}", flush=True)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    crit = build_loss(args.loss).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, int(np.ceil(len(tr_pairs) / args.batch_size)))
    total = args.epochs * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total, pct_start=0.12,
        div_factor=10.0, final_div_factor=100.0)

    tr_loader = make_loader(tr_pairs, size, True, args.batch_size, args.workers, get_aug(size))
    va_loader = make_loader(va_pairs, size, False, args.batch_size, args.workers)

    best_dice, best_ep, bad = 0.0, -1, 0
    hist = []
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for img, msk, _ in tr_loader:
            img = img.to(device, non_blocking=True)
            msk = msk.to(device, non_blocking=True)
            if args.channels_last:
                img = img.to(memory_format=torch.channels_last)
            logits = model(img)
            if logits.shape[-2:] != msk.shape[-2:]:
                logits = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
            loss = crit(logits, msk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sched.step()
            run.append(loss.item())

        vdice = evaluate(model, va_loader, device, size)
        hist.append(dict(epoch=ep, train_loss=float(np.mean(run)), val_dice=vdice))
        if (ep + 1) % args.print_every == 0 or ep == 0:
            print(f"[{tag}] ep {ep+1:3d}/{args.epochs} loss {np.mean(run):.4f} "
                  f"val_dice {vdice:.4f} lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"{time.time()-t0:.1f}s", flush=True)

        if vdice > best_dice:
            best_dice, best_ep, bad = vdice, ep, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "fold": fold,
                        "val_dice": vdice, "epoch": ep}, os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
        if ep >= args.min_epochs and bad >= args.patience:
            print(f"[{tag}] early stop ep {ep+1} (best {best_dice:.4f} @ep {best_ep+1})", flush=True)
            break

    # ---- save out-of-fold probabilities from the best checkpoint
    ck = torch.load(os.path.join(out_dir, "best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    oof_dice = save_oof(model, va_pairs, size, device, os.path.join(out_dir, "oof.npz"),
                        batch_size=args.batch_size)
    print(f"[{tag}] BEST val_dice {best_dice:.4f} @ epoch {best_ep+1} (oof@native {oof_dice:.4f})",
          flush=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return best_dice


def setup_logging(logfile):
    """Mirror stdout to a log file with immediate flushing (PowerShell redirection
    buffers, which makes background training impossible to monitor)."""
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    fh = open(logfile, "a", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="unetplusplus")
    p.add_argument("--encoder", default="efficientnet-b4")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--min_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--loss", default="combined")
    p.add_argument("--folds", default="0,1,2,3")
    p.add_argument("--n_splits", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--tag", default="exp")
    p.add_argument("--init_from", default="", help="tag of a previous run to initialise weights from")
    p.add_argument("--use_external", action="store_true", default=False,
                   help="add CHASE_DB1 + HRF as extra training data")
    p.add_argument("--channels_last", action="store_true", default=False)
    args = p.parse_args()

    ensure_dir(WORK)
    setup_logging(os.path.join(WORK, f"{args.tag}.log"))
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)

    pairs = list_images(TRAIN_IMG)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(pairs))

    # Optional extra training data from public retinal-vessel datasets. These are
    # added ONLY to the training split of every fold; validation stays pure
    # competition data so the CV numbers remain comparable.
    ext = list_external_pairs(verbose=True) if args.use_external else []
    if ext:
        print(f"using {len(ext)} external images as extra training data", flush=True)

    scores = {}
    for fold in [int(x) for x in args.folds.split(",") if x.strip() != ""]:
        tr_idx, va_idx = splits[fold]
        tr = [pairs[i] for i in tr_idx] + ext
        va = [pairs[i] for i in va_idx]
        print(f"fold {fold}: train {len(tr)} (ext {len(ext)}) val {len(va)} "
              f"val_ids={[s for s, _ in va]}", flush=True)
        scores[fold] = train_one_fold(args, fold, tr, va, device)

    print("fold scores:", scores, "mean:", float(np.mean(list(scores.values()))), flush=True)
    with open(os.path.join(WORK, f"{args.tag}_cv.json"), "w") as f:
        json.dump({"scores": scores, "mean": float(np.mean(list(scores.values()))),
                   "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
