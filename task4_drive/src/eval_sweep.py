"""Evaluate inference settings (input size / TTA / scales) on each fold's val images.

Uses the already-trained fold checkpoints, so it costs only inference time.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TRAIN_IMG, TRAIN_LBL, WORK, VesselDataset, dice_score,  # noqa: E402
                    ensure_dir, imread_unicode, list_images, load_label)
from models import build_model  # noqa: E402
from predict import tta_forward  # noqa: E402

cv2.setNumThreads(0)


def native_size(path):
    return imread_unicode(path, cv2.IMREAD_GRAYSCALE).shape[:2]


@torch.no_grad()
def infer(model, pairs, size, device, tta, scales, batch_size=2):
    """Return dict stem -> prob at native resolution."""
    ds = VesselDataset(pairs, size=size, train=False, with_label=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    nat = {s: native_size(p) for s, p in pairs}
    out = {}
    for img, stems in loader:
        img = img.to(device)
        acc = None
        for s in scales:
            xi = img if s == 1.0 else F.interpolate(
                img, scale_factor=s, mode="bilinear", align_corners=False, antialias=True)
            p = tta_forward(model, xi, tta)
            if s != 1.0:
                p = F.interpolate(p, size=img.shape[-2:], mode="bilinear", align_corners=False)
            acc = p if acc is None else acc + p
        acc = (acc / len(scales)).float().cpu().numpy()[:, 0]
        for st, pr in zip(stems, acc):
            h, w = nat[st]
            out[st] = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def best_dice(probs, gts, thresholds=np.arange(0.30, 0.81, 0.02)):
    """Best mean Dice over thresholds; also returns per-threshold curve."""
    res = []
    for t in thresholds:
        d = np.mean([dice_score((p >= t).astype(np.uint8), (g > 0.5).astype(np.uint8))
                     for p, g in zip(probs, gts)])
        res.append((float(t), float(d)))
    bt, bd = max(res, key=lambda x: x[1])
    return bd, bt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="upp_b4_512")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", default="512,576,640")
    ap.add_argument("--tta_list", default="i,i,h,v,hv")
    ap.add_argument("--scales", default="1.0,0.85+1.15")
    ap.add_argument("--out", default="work/infer_sweep.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs_all = list_images(TRAIN_IMG)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(pairs_all))

    sizes = [int(s) for s in args.sizes.split(",")]
    tta_cfgs = [tuple(t.split(",")) for t in args.tta_list.split("|")] if "|" in args.tta_list else None
    # format: "i,h,v,hv" single config ; multiple separated by "|"
    tta_cfgs = [tuple(x.strip() for x in c.split(",")) for c in args.tta_list.split("|")]
    scale_cfgs = []
    for sc in args.scales.split("|"):
        scale_cfgs.append(tuple(float(v) for v in sc.replace("+", ",").split(",")))

    results = {}
    for fold in [int(x) for x in args.folds.split(",")]:
        ckpt = os.path.join(WORK, "checkpoints", f"{args.tag}_f{fold}", "best.pth")
        if not os.path.exists(ckpt):
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        model = build_model(a.get("arch", "unetplusplus"), a.get("encoder", "efficientnet-b4"))
        model.load_state_dict(ck["model"])
        model.to(device).eval()

        _, va_idx = splits[fold]
        va = [pairs_all[i] for i in va_idx]
        gts = [load_label(os.path.join(TRAIN_LBL, s + ".jpg"), None) for s, _ in va]

        for size in sizes:
            for tta in tta_cfgs:
                for scales in scale_cfgs:
                    key = f"size{size}_tta{''.join(tta)}_sc{'+'.join(str(s) for s in scales)}"
                    try:
                        probs = [None] * len(va)
                        pd = infer(model, va, size, device, tta, scales)
                        probs = [pd[s] for s, _ in va]
                    except RuntimeError as e:
                        print(f"  fold{fold} {key}: OOM/skip ({str(e)[:60]})", flush=True)
                        torch.cuda.empty_cache()
                        continue
                    d, t, curve = best_dice(probs, gts)
                    results.setdefault(key, []).append(d)
                    print(f"fold{fold} {key}: dice {d:.4f} @thr {t:.2f}", flush=True)
                    torch.cuda.empty_cache()

    print("\n=== mean over folds ===", flush=True)
    summary = {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}
    for k, (m, s) in sorted(summary.items(), key=lambda x: -x[1][0]):
        print(f"{k:42s} {m:.4f} +- {s:.4f}", flush=True)
    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_fold": results}, f, indent=1)


if __name__ == "__main__":
    main()
