"""Search ensemble weights + threshold over several saved val-prob files.

Each input .npz has probs/gts/stems for the same 20 validation images (each image
scored by the model trained without it). We grid-search non-negative weights.
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_IMG, WORK, dice_score, ensure_dir, fov_mask  # noqa: E402
from common import imread_unicode, list_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--out", default="work/ensemble_weights.json")
    ap.add_argument("--step", type=float, default=0.1)
    args = ap.parse_args()

    mats, gts, stems = [], None, None
    for f in args.probs:
        d = np.load(f)
        mats.append(d["probs"])
        if gts is None:
            gts = d["gts"]
            stems = [str(s) for s in d["stems"]]
        else:
            assert [str(s) for s in d["stems"]] == stems, "stem order mismatch"
    n = len(mats)
    print(f"{n} prob sets, {len(stems)} images", flush=True)

    fovs = {}
    for s, p in list_images(TRAIN_IMG):
        fovs[s] = fov_mask(cv2.cvtColor(imread_unicode(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

    steps = int(round(1.0 / args.step))
    best = (-1, None)
    results = []
    # weight grid summing to 1
    for combo in itertools.product(range(steps + 1), repeat=n):
        if sum(combo) == 0:
            continue
        w = np.array(combo, dtype=np.float64)
        w /= w.sum()
        p_ens = sum(wi * m for wi, m in zip(w, mats))
        for thr in np.arange(0.35, 0.71, 0.05):
            ds = []
            for p, g, s in zip(p_ens, gts, stems):
                m = (p >= thr).astype(np.uint8)
                f = fovs[s]
                if f.shape != m.shape:
                    f = cv2.resize(f, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = m & (f > 0).astype(np.uint8)
                ds.append(dice_score(m, (g > 0.5).astype(np.uint8)))
            d = float(np.mean(ds))
            results.append(dict(w=w.tolist(), thr=float(thr), dice=d))
            if d > best[0]:
                best = (d, dict(w=w.tolist(), thr=float(thr)))

    results.sort(key=lambda r: -r["dice"])
    print(f"\nBEST dice={best[0]:.4f} weights={[round(x,2) for x in best[1]['w']]} "
          f"thr={best[1]['thr']:.2f}", flush=True)
    print("top 8:", flush=True)
    for r in results[:8]:
        print(f"  dice={r['dice']:.4f} w={[round(x,2) for x in r['w']]} thr={r['thr']:.2f}", flush=True)

    ensure_dir(os.path.dirname(os.path.abspath(args.out)))
    with open(args.out, "w") as f:
        json.dump({"best": best[1], "best_dice": best[0],
                   "names": [os.path.basename(x) for x in args.probs],
                   "top": results[:20]}, f, indent=1)

    # write postprocess.json for the submit step
    pp = os.path.join(WORK, "postprocess.json")
    with open(pp, "w") as f:
        json.dump({"settings": {"fov": True, "thr": best[1]["thr"], "min_size": 0},
                   "best_dice": best[0], "weights": best[1]["w"],
                   "weight_files": [os.path.basename(x) for x in args.probs],
                   "source": "tune_ensemble.py"}, f, indent=1)
    print("wrote", pp, flush=True)


if __name__ == "__main__":
    main()
