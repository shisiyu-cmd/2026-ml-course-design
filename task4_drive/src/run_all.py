"""End-to-end pipeline: train -> tune -> predict -> submit.

Run from the segmentation/ directory:
    python src/run_all.py --stage all

Stages are resumable: each writes into work/ and is skipped if its outputs exist
(unless --force).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(cmd, log=None):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r


def exists(*p):
    return os.path.exists(os.path.join(ROOT, *p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "train", "train768", "tune", "predict", "submit"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    W = os.path.join(ROOT, "work")
    os.makedirs(W, exist_ok=True)
    do = lambda s: args.stage in ("all", s)  # noqa: E731

    # ---------------------------------------------------------------- 1. base training
    if do("train") and (args.force or not exists("work", "upp_b4_512_cv.json")):
        run([PY, "-u", "src/train.py", "--arch", "unetplusplus", "--encoder", "efficientnet-b4",
             "--size", "512", "--epochs", "250", "--min_epochs", "120", "--patience", "45",
             "--batch_size", "4", "--lr", "3e-4", "--folds", "0,1,2,3",
             "--print_every", "10", "--tag", "upp_b4_512"])

    # ---------------------------------------------------------------- 2. 768 fine-tune
    if do("train768") and (args.force or not exists("work", "upp_b4_768_ft_cv.json")):
        run([PY, "-u", "src/train.py", "--arch", "unetplusplus", "--encoder", "efficientnet-b4",
             "--size", "768", "--epochs", "70", "--min_epochs", "70", "--patience", "200",
             "--batch_size", "2", "--lr", "1.2e-4", "--folds", "0,1,2,3",
             "--print_every", "5", "--tag", "upp_b4_768_ft", "--init_from", "upp_b4_512"])

    # ---------------------------------------------------------------- 3. threshold tuning
    if do("tune") and (args.force or not exists("work", "valprobs_upp_b4_768_ft_s768.npz")):
        run([PY, "-u", "src/val_probs.py", "--tags", "upp_b4_768_ft", "--size", "768",
             "--tta", "i,h,v,hv", "--out", "work/valprobs_upp_b4_768_ft_s768.npz"])
        run([PY, "-u", "src/tune_threshold.py",
             "--probs", "work/valprobs_upp_b4_768_ft_s768.npz",
             "--out", "work/threshold_final.json"])

    # ---------------------------------------------------------------- 4. test prediction
    if do("predict") and (args.force or not exists("work", "test_probs_final.npz")):
        ckpts = [f"work/checkpoints/upp_b4_768_ft_f{i}/best.pth" for i in range(4)]
        run([PY, "-u", "src/predict.py", "--ckpts", *ckpts, "--split", "test",
             "--size", "768", "--tta", "i,h,v,hv", "--out", "work/test_probs_final.npz"])

    # ---------------------------------------------------------------- 5. submission
    if do("submit"):
        run([PY, "-u", "src/postprocess.py", "submit",
             "--probs", "work/test_probs_final.npz", "--csv", "submission.csv",
             "--image_dir", "image"])
        with open(os.path.join(W, "postprocess.json")) as f:
            print("post-processing settings:", json.load(f)["settings"], flush=True)


if __name__ == "__main__":
    main()
