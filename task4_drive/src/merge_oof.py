"""Merge per-fold OOF predictions into one file for post-processing search."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="0,1,2,3")
    args = ap.parse_args()

    probs, gts, stems = [], [], []
    for fold in [int(x) for x in args.folds.split(",") if x.strip() != ""]:
        p = os.path.join(WORK, "checkpoints", f"{args.tag}_f{fold}", "oof.npz")
        if not os.path.exists(p):
            print("missing", p, flush=True)
            continue
        d = np.load(p)
        probs.append(d["probs"])
        gts.append(d["gts"])
        stems.extend([str(s) for s in d["stems"]])
        print(f"fold {fold}: {len(d['stems'])} images", flush=True)

    order = np.argsort([int(s) for s in stems])
    probs = np.concatenate(probs, 0)[order]
    gts = np.concatenate(gts, 0)[order]
    stems = np.array(stems)[order]

    out = os.path.join(WORK, f"oof_{args.tag}.npz")
    np.savez_compressed(out, probs=probs, gts=gts, stems=stems)
    print(f"wrote {out}  n={len(stems)}  ids={list(stems)}", flush=True)


if __name__ == "__main__":
    main()
