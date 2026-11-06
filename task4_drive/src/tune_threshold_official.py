"""Tune the decision threshold against the OFFICIAL evaluation metric.

Because segmentation_to_csv.py resizes each mask to 512x512 with BILINEAR and then
treats every non-zero pixel as foreground, the grader effectively sees a slightly
dilated version of both prediction and ground truth. Optimising the threshold on
the native-resolution Dice is therefore not optimal - this script optimises the
metric the leaderboard actually computes.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_LBL, WORK, dice_score, load_label  # noqa: E402

SIDE = 512


def to_official(mask):
    im = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
    im = im.resize((SIDE, SIDE), Image.Resampling.BILINEAR)
    return (np.asarray(im) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="work/oof_upp_b4_768_ft.npz")
    ap.add_argument("--out", default="work/threshold_official.json")
    args = ap.parse_args()

    d = np.load(args.probs)
    probs, stems = d["probs"], [str(s) for s in d["stems"]]

    gts_official = {}
    for st in stems:
        gt = (load_label(os.path.join(TRAIN_LBL, st + ".jpg"), None) > 0.5).astype(np.uint8)
        gts_official[st] = to_official(gt)

    rows = []
    for thr in np.arange(0.20, 0.86, 0.02):
        ds = []
        for p, st in zip(probs, stems):
            pred = (p >= thr).astype(np.uint8)
            ds.append(dice_score(to_official(pred), gts_official[st]))
        rows.append((float(thr), float(np.mean(ds))))

    best = max(rows, key=lambda r: r[1])
    print(f"BEST official Dice {best[1]:.4f} @ thr {best[0]:.2f}", flush=True)
    for t, v in rows:
        if abs(t * 10 - round(t * 10)) < 1e-9:
            print(f"  thr={t:.2f} -> {v:.4f}", flush=True)

    with open(os.path.join(WORK, "postprocess.json")) as f:
        pp = json.load(f)
    pp["settings"]["thr"] = best[0]
    pp["settings"]["fov"] = True
    pp["settings"]["min_size"] = 0
    pp["official_dice"] = best[1]
    pp["source"] = "tune_threshold_official.py"
    with open(os.path.join(WORK, "postprocess.json"), "w") as f:
        json.dump(pp, f, indent=1)

    with open(args.out, "w") as f:
        json.dump({"best_thr": best[0], "best_official_dice": best[1],
                   "grid": rows}, f, indent=1)
    print("updated work/postprocess.json ->", pp["settings"], flush=True)


if __name__ == "__main__":
    main()
