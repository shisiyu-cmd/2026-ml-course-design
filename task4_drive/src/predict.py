"""Predict probabilities for a set of images with one or more checkpoints (ensembled).

Saves an .npz with per-image probability maps at the original resolution.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TEST_IMG, TRAIN_IMG, WORK, VesselDataset, ensure_dir,  # noqa: E402
                    imread_unicode, list_images)
from models import build_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

cv2.setNumThreads(0)


def tta_forward(model, x, tta):
    """Test-time augmentation. Returns averaged sigmoid probabilities."""
    acc = None
    for k, tf in enumerate(tta):
        xi = x
        if tf == "h":
            xi = torch.flip(x, dims=[3])
        elif tf == "v":
            xi = torch.flip(x, dims=[2])
        elif tf == "hv":
            xi = torch.flip(x, dims=[2, 3])
        p = torch.sigmoid(model(xi))
        if tf == "h":
            p = torch.flip(p, dims=[3])
        elif tf == "v":
            p = torch.flip(p, dims=[2])
        elif tf == "hv":
            p = torch.flip(p, dims=[2, 3])
        acc = p if acc is None else acc + p
    return acc / len(tta)


@torch.no_grad()
def predict(ckpt_paths, pairs, size, device, batch_size=4, tta=("i",), scales=(1.0,)):
    """Ensemble over checkpoints, scales and TTA.

    Returns dict stem -> probability map at the ORIGINAL image resolution, matching
    the resolution used when tuning post-processing on out-of-fold predictions.
    """
    models = []
    for cp in ckpt_paths:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        m = build_model(a.get("arch", "unetplusplus"), a.get("encoder", "efficientnet-b4"))
        m.load_state_dict(ck["model"])
        m.to(device).eval()
        models.append(m)
        print(f"  loaded {os.path.basename(os.path.dirname(cp))} (val_dice={ck.get('val_dice', -1):.4f})",
              flush=True)

    ds = VesselDataset(pairs, size=size, train=False, with_label=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # native resolution of each source image
    native = {}
    for stem, path in pairs:
        h, w = imread_unicode(path, cv2.IMREAD_GRAYSCALE).shape[:2]
        native[stem] = (h, w)

    out = {}
    for img, stems in loader:
        img = img.to(device)
        acc = None
        for m in models:
            for s in scales:
                xi = img if s == 1.0 else F.interpolate(
                    img, scale_factor=s, mode="bilinear", align_corners=False, antialias=True)
                p = tta_forward(m, xi, tta)
                if s != 1.0:
                    p = F.interpolate(p, size=img.shape[-2:], mode="bilinear", align_corners=False)
                acc = p if acc is None else acc + p
        acc = (acc / (len(models) * len(scales))).float().cpu().numpy()[:, 0]
        for st, pr in zip(stems, acc):
            h, w = native[st]
            out[st] = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--tta", default="i,h,v,hv")
    ap.add_argument("--scales", default="1.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = list_images(TEST_IMG if args.split == "test" else TRAIN_IMG)
    tta = tuple(t.strip() for t in args.tta.split(",") if t.strip())
    scales = tuple(float(s) for s in args.scales.split(","))
    print(f"split={args.split} n={len(pairs)} tta={tta} scales={scales}", flush=True)

    probs = predict(args.ckpts, pairs, args.size, device, args.batch_size, tta, scales)
    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    np.savez_compressed(args.out,
                        probs=np.stack([probs[s] for s, _ in pairs]),
                        stems=np.array([s for s, _ in pairs]))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
