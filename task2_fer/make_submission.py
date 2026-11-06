"""
Build the final submission:
  1. weighted ensemble of test logits (weights from OOF greedy search)
  2. override with train-set near-duplicate labels (validated ~97% accurate)
  3. write in template ID order
"""
import os, json, csv
import numpy as np

PRED = r"C:\Users\古恩宇\Downloads\2026task2\preds"
WORK = r"C:\Users\古恩宇\Downloads\2026task2"
TPL_CSV = r"C:\Users\古恩宇\Downloads\submission.csv"
OUT_CSV = os.path.join(WORK, "submission.csv")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    with open(os.path.join(PRED, "ensemble_weights.json")) as fh:
        cfg = json.load(fh)
    weights = {k: v for k, v in cfg["weights"].items() if v > 0}
    print("weights:", {k: round(v, 3) for k, v in weights.items()})

    test_names = np.load(os.path.join(WORK, "cache", "data.npz"), allow_pickle=True)["test_names"]
    test_names = [str(x) for x in test_names]

    # weighted average of per-fold test logits (in probability space)
    acc = None
    tot_w = 0.0
    for tag, w in weights.items():
        files = sorted([f for f in os.listdir(PRED)
                        if f.startswith(tag + "_test_fold") and f.endswith(".npy")])
        if not files:
            print(f"  WARNING no test preds for {tag}")
            continue
        per_fold = [softmax(np.load(os.path.join(PRED, f))) for f in files]
        p = np.mean(per_fold, 0)  # average folds
        acc = p * w if acc is None else acc + p * w
        tot_w += w
        print(f"  {tag:24s} w={w:.3f} folds={len(files)}")
    prob = acc / tot_w
    preds = prob.argmax(1)
    print(f"\nensemble prob shape {prob.shape}")

    # ---- near-duplicate override ----
    td = np.load(os.path.join(WORK, "_test_nn_dist.npy"))
    tl = np.load(os.path.join(WORK, "_test_nn_lbl.npy"))
    with open(os.path.join(WORK, "_test_files_order.json")) as fh:
        nn_order = json.load(fh)
    assert nn_order == test_names, "test file order mismatch!"

    THR = 2.0
    mask = td < THR
    n_ovr = int(mask.sum())
    changed = int((tl[mask] != preds[mask]).sum())
    print(f"overriding {n_ovr} near-duplicates (dist<{THR}); {changed} predictions changed")
    preds = preds.copy()
    preds[mask] = tl[mask]

    # ---- write in template order ----
    tpl_ids = []
    with open(TPL_CSV) as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            tpl_ids.append(row[0])
    pm = dict(zip(test_names, preds.tolist()))
    missing = [i for i in tpl_ids if i not in pm]
    assert not missing, f"missing {len(missing)} ids"
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ID", "Emotion"])
        for fid in tpl_ids:
            w.writerow([fid, int(pm[fid])])

    from collections import Counter
    dist = Counter(pm[i] for i in tpl_ids)
    print(f"\nwrote {OUT_CSV}  rows={len(tpl_ids)}")
    print("distribution:", dict(sorted(dist.items())))
    np.save(os.path.join(PRED, "final_test_prob.npy"), prob)


if __name__ == "__main__":
    main()
