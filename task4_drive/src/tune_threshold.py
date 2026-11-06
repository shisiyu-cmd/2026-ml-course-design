"""Tune the decision threshold on validation-fold probabilities produced by eval_sweep.

Writes work/threshold.json with the best global threshold (and FOV setting), which
postprocess.py submit then applies to the test probabilities.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_IMG, TRAIN_LBL, WORK, dice_score, ensure_dir, load_label  # noqa: E402
from common import fov_mask, imread_unicode, list_images  # noqa: E402
import cv2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True,
                    help="one or more .npz files with probs/gts/stems (e.g. per-fold val probs)")
    ap.add_argument("--out", default="work/threshold.json")
    args = ap.parse_args()

    probs, gts, stems = [], [], []
    for f in args.probs:
        d = np.load(f)
        probs.append(d["probs"])
        gts.append(d["gts"])
        stems.extend([str(s) for s in d["stems"]])
    probs = np.concatenate(probs, 0)
    gts = np.concatenate(gts, 0)

    fovs = {}
    for s, p in list_images(TRAIN_IMG):
        fovs[s] = fov_mask(cv2.cvtColor(imread_unicode(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

    rows = []
    for fov_apply in (True, False):
        for t in np.arange(0.30, 0.86, 0.02):
            ds = []
            for p, g, s in zip(probs, gts, stems):
                m = (p >= t).astype(np.uint8)
                if fov_apply:
                    f = fovs[s]
                    if f.shape != m.shape:
                        f = cv2.resize(f, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                    m = m & (f > 0).astype(np.uint8)
                ds.append(dice_score(m, (g > 0.5).astype(np.uint8)))
            rows.append(dict(thr=float(t), fov=bool(fov_apply), dice=float(np.mean(ds))))

    best = max(rows, key=lambda r: r["dice"])
    print(f"BEST thr={best['thr']:.2f} fov={best['fov']} dice={best['dice']:.4f}", flush=True)
    for r in rows:
        if abs(r["thr"] - round(r["thr"], 1)) < 1e-9 and r["fov"]:
            print(f"  thr={r['thr']:.2f} dice={r['dice']:.4f}", flush=True)

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w") as f:
        json.dump({"best": best, "grid": rows}, f, indent=1)

    # also emit the file postprocess.py submit reads
    pp = os.path.join(WORK, "postprocess.json")
    prev = {}
    if os.path.exists(pp):
        with open(pp) as f:
            prev = json.load(f)
    prev["settings"] = {"fov": best["fov"], "thr": best["thr"], "min_size": 0}
    prev["best_dice"] = best["dice"]
    prev["source"] = "tune_threshold.py"
    with open(pp, "w") as f:
        json.dump(prev, f, indent=1)
    print("wrote", pp, flush=True)


if __name__ == "__main__":
    main()
