"""Produce validation-fold probabilities at a chosen inference size/TTA.

Saves work/valprobs_<tag>_s<size>.npz with probs (native resolution), gts, stems.
Used to tune the decision threshold for the exact inference configuration.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_IMG, TRAIN_LBL, WORK, ensure_dir, list_images, load_label  # noqa: E402
from eval_sweep import infer  # noqa: E402
from models import build_model  # noqa: E402

cv2.setNumThreads(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma separated training tags (ensemble)")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--tta", default="i,h,v,hv")
    ap.add_argument("--n_splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs_all = list_images(TRAIN_IMG)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(kf.split(pairs_all))
    tta = tuple(t.strip() for t in args.tta.split(","))

    tags = [t.strip() for t in args.tags.split(",")]
    all_p, all_g, all_s = [], [], []
    for fold in range(args.n_splits):
        models = []
        for tag in tags:
            ckpt = os.path.join(WORK, "checkpoints", f"{tag}_f{fold}", "best.pth")
            if not os.path.exists(ckpt):
                print("missing", ckpt, flush=True)
                continue
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            a = ck.get("args", {})
            m = build_model(a.get("arch", "unetplusplus"), a.get("encoder", "efficientnet-b4"))
            m.load_state_dict(ck["model"])
            m.to(device).eval()
            models.append(m)
        if not models:
            continue

        _, va_idx = splits[fold]
        va = [pairs_all[i] for i in va_idx]
        # average probabilities across all models for this fold's validation images
        acc = None
        for m in models:
            pd = infer(m, va, args.size, device, tta, (1.0,))
            pr = np.stack([pd[s] for s, _ in va])
            acc = pr if acc is None else acc + pr
            del m
            torch.cuda.empty_cache()
        acc /= len(models)
        gts = np.stack([load_label(os.path.join(TRAIN_LBL, s + ".jpg"), None) for s, _ in va])
        all_p.append(acc)
        all_g.append(gts)
        all_s.extend([s for s, _ in va])
        print(f"fold{fold}: n={len(va)} models={len(models)}", flush=True)

    order = np.argsort([int(s) for s in all_s])
    probs = np.concatenate(all_p, 0)[order]
    gts = np.concatenate(all_g, 0)[order]
    stems = np.array(all_s)[order]
    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    np.savez_compressed(args.out, probs=probs, gts=gts, stems=stems)
    print("wrote", args.out, probs.shape, flush=True)


if __name__ == "__main__":
    main()
