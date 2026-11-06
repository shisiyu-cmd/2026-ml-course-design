"""Evaluate Dice under the OFFICIAL submission encoding.

segmentation_to_csv.py resizes each PNG to 512x512 (PIL BILINEAR) and treats every
non-zero pixel as mask. If the grader prepares ground truth the same way, the Dice
it computes is NOT the native-resolution Dice. This script reports both, so the
reported number matches what the leaderboard would show.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_LBL, dice_score, load_label  # noqa: E402

SIDE = 512


def encode_like_official(mask_native):
    """Replicate segmentation_to_csv.py: PIL BILINEAR resize to 512, then >0."""
    im = Image.fromarray((mask_native * 255).astype(np.uint8)).convert("L")
    im = im.resize((SIDE, SIDE), Image.Resampling.BILINEAR)
    return (np.asarray(im) > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="work/oof_upp_b4_768_ft.npz")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.probs)
    probs, stems = d["probs"], [str(s) for s in d["stems"]]

    native, official = [], []
    print(f"{'id':>4} {'native':>8} {'official512':>12} {'frac_pred':>10} {'frac_gt':>8}", flush=True)
    for p, st in zip(probs, stems):
        pred = (p >= args.thr).astype(np.uint8)
        gt = (load_label(os.path.join(TRAIN_LBL, st + ".jpg"), None) > 0.5).astype(np.uint8)
        if pred.shape != gt.shape:   # OOF maps may be stored at the training resolution
            from PIL import Image as _I
            pred = (np.asarray(_I.fromarray(pred * 255).resize(
                (gt.shape[1], gt.shape[0]), _I.Resampling.NEAREST)) > 127).astype(np.uint8)
        native.append(dice_score(pred, gt))
        pe, ge = encode_like_official(pred), encode_like_official(gt)
        official.append(dice_score(pe, ge))
        print(f"{st:>4} {native[-1]:8.4f} {official[-1]:12.4f} {pe.mean():10.4f} {ge.mean():8.4f}",
              flush=True)

    print(f"\nmean native Dice     : {np.mean(native):.4f}", flush=True)
    print(f"mean official-512 Dice: {np.mean(official):.4f}", flush=True)
    print("(the official number is the realistic leaderboard estimate, because the "
          "BILINEAR resize to 512 inflates both masks similarly)", flush=True)


if __name__ == "__main__":
    main()
