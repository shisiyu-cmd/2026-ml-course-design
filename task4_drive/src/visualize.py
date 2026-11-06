"""Visualise predictions: original | ground truth | prediction | overlay."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (TRAIN_IMG, TRAIN_LBL, WORK, ensure_dir, imread_unicode,  # noqa: E402
                    imwrite_unicode, list_images, load_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="npz with probs (+ gts for train split)")
    ap.add_argument("--ids", default="1,2,3")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="work/viz")
    args = ap.parse_args()

    d = np.load(args.probs)
    probs = d["probs"]
    stems = [str(s) for s in d["stems"]]
    gts = d["gts"] if "gts" in d.files else None
    ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.out))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.out))

    img_dir = TRAIN_IMG if gts is not None else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "..", "test", "image")
    paths = dict(list_images(img_dir))

    for st in [x.strip() for x in args.ids.split(",")]:
        if st not in stems:
            print("skip", st, flush=True)
            continue
        p = probs[stems.index(st)]
        pred = (p >= args.thr).astype(np.uint8)
        rgb = cv2.cvtColor(imread_unicode(paths[st]), cv2.COLOR_BGR2RGB)
        if pred.shape != rgb.shape[:2]:
            pred = cv2.resize(pred, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        panels = [rgb, np.dstack([pred * 255] * 3)]
        if gts is not None:
            g = (gts[stems.index(st)] > 0.5).astype(np.uint8)
            if g.shape != rgb.shape[:2]:
                g = cv2.resize(g, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            panels = [rgb, np.dstack([g * 255] * 3), np.dstack([pred * 255] * 3)]
        ov = rgb.copy()
        ov[pred > 0] = [0, 255, 0]
        panels.append(cv2.addWeighted(rgb, 0.55, ov, 0.45, 0))
        side = np.concatenate(panels, axis=1)
        fp = os.path.join(out_dir, f"viz_{st}.png")
        imwrite_unicode(fp, cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        print("wrote", fp, flush=True)


if __name__ == "__main__":
    main()
