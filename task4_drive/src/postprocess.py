"""Search post-processing settings (threshold, FOV masking, small-object removal)
on out-of-fold predictions, then build the final submission.

Two modes:
  tune   : grid-search post-processing on OOF probs, report best Dice
  submit : apply the chosen settings to test probabilities -> submission.csv
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, TEST_IMG, TRAIN_IMG, TRAIN_LBL, WORK, check_rle,  # noqa: E402
                    dice_score, ensure_dir, fov_mask, imread_unicode, imwrite_unicode,
                    list_images, load_label, rle_encode)

# segmentation_to_csv.py always resizes the predicted image to 512x512 (BILINEAR)
# before building the RLE, so submissions are encoded on a 512x512 grid.
RLE_SIDE = 512


def postprocess(prob, fov, thr, min_size=0, fov_apply=True):
    m = (prob >= thr).astype(np.uint8)
    if fov_apply and fov is not None:
        if fov.shape != m.shape:
            fov = cv2.resize(fov, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m & (fov > 0).astype(np.uint8)
    if min_size > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                keep[i] = 1
        m = keep[lab]
    return m


def load_fovs(split):
    folder = TEST_IMG if split == "test" else TRAIN_IMG
    fovs = {}
    for stem, path in list_images(folder):
        rgb = imread_unicode(path, cv2.IMREAD_COLOR)
        fovs[stem] = fov_mask(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    return fovs


def cmd_tune(args):
    """Grid search threshold / fov / min_size on OOF predictions."""
    data = np.load(args.oof)
    probs, gts, stems = data["probs"], data["gts"], [str(s) for s in data["stems"]]
    fovs = load_fovs("train")
    print(f"OOF n={len(stems)} ids={stems}", flush=True)

    best = (-1, None)
    rows = []
    for fov_apply in (True, False):
        for thr in [round(t, 3) for t in np.arange(0.30, 0.76, 0.05)]:
            for ms in (0, 8, 20, 50):
                ds = []
                for p, g, st in zip(probs, gts, stems):
                    fov = fovs.get(st)
                    m = postprocess(p, fov, thr, ms, fov_apply)
                    ds.append(dice_score(m, (g > 0.5).astype(np.uint8)))
                d = float(np.mean(ds))
                rows.append(dict(fov=fov_apply, thr=thr, min_size=ms, dice=d))
                if d > best[0]:
                    best = (d, dict(fov=fov_apply, thr=thr, min_size=ms))
                print(f"fov={fov_apply} thr={thr:.2f} min_size={ms:3d} -> dice {d:.5f}", flush=True)

    print("\nBEST:", best, flush=True)
    ensure_dir(WORK)
    with open(os.path.join(WORK, "postprocess.json"), "w") as f:
        json.dump({"best_dice": best[0], "settings": best[1], "grid": rows}, f, indent=1)

    # per-image dice at the best setting for diagnostics
    s = best[1]
    per = {}
    for p, g, st in zip(probs, gts, stems):
        m = postprocess(p, fovs.get(st), s["thr"], s["min_size"], s["fov"])
        per[st] = dice_score(m, (g > 0.5).astype(np.uint8))
    print("per-image dice:", {k: round(v, 4) for k, v in sorted(per.items(), key=lambda x: int(x[0]))})
    with open(os.path.join(WORK, "postprocess_per_image.json"), "w") as f:
        json.dump(per, f, indent=1)


def cmd_submit(args):
    data = np.load(args.probs)
    probs, stems = data["probs"], [str(s) for s in data["stems"]]
    fovs = load_fovs("test")
    with open(os.path.join(WORK, "postprocess.json")) as f:
        s = json.load(f)["settings"]
    print("settings:", s, flush=True)

    out_dir = ensure_dir(os.path.join(ROOT, args.image_dir))
    ids, preds = [], []
    for p, st in zip(probs, stems):
        m = postprocess(p, fovs.get(st), s["thr"], s["min_size"], s["fov"])
        img = (m * 255).astype(np.uint8)
        assert imwrite_unicode(os.path.join(out_dir, f"{st}.png"), img), f"write failed {st}"
        ids.append(st)
        print(f"  {st}: vessel_frac={m.mean():.4f}", flush=True)

    # Encode with the official pipeline so the RLE is byte-identical to what
    # segmentation_to_csv.py would produce for these PNGs.
    sys.path.insert(0, ROOT)
    from segmentation_to_csv import turn_to_str  # noqa: E402
    paths = [os.path.join(out_dir, f"{st}.png") for st in ids]
    preds = turn_to_str(paths)
    for st, rle in zip(ids, preds):
        assert check_rle(rle, RLE_SIDE * RLE_SIDE), f"bad RLE for {st}"

    import pandas as pd
    df = pd.DataFrame({"Id": ids, "Predicted": preds})
    out_csv = os.path.join(ROOT, args.csv)
    df.to_csv(out_csv, index=False)
    print("wrote", out_csv, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tune")
    a.add_argument("--oof", required=True)
    a.set_defaults(func=cmd_tune)
    b = sub.add_parser("submit")
    b.add_argument("--probs", required=True)
    b.add_argument("--csv", default="submission.csv")
    b.add_argument("--image_dir", default="pred_image")
    b.set_defaults(func=cmd_submit)
    args = ap.parse_args()
    args.func(args)
