"""Build submission from the best OOF-validated blend of ALL models."""
import os, json, csv, sys
import numpy as np

WORK = r"C:\Users\古恩宇\Downloads\2026task2"
PRED = os.path.join(WORK, "preds")
PRIOR = os.path.join(WORK, "prior_preds")
TPL = r"C:\Users\古恩宇\Downloads\submission.csv"
OUT = os.path.join(WORK, "submission.csv")


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    extra = sys.argv[1:]  # optional extra tags
    y = np.load(os.path.join(PRIOR, "ytr.npy"))
    po = np.load(os.path.join(PRIOR, "prior_oof.npz"))
    pt = np.load(os.path.join(PRIOR, "prior_test.npz"))

    oofs, tests = {}, {}
    # prior families: average per-fold within family
    for fam in po.files:
        oofs[fam] = po[fam]
        ks = [k for k in pt.files if k.startswith(fam + "_f")]
        if ks:
            tests[fam] = np.mean([sm(pt[k]) for k in ks], 0)
    for tag in ["r18_64", "r34_96", "eff_112"] + extra:
        op = os.path.join(PRED, f"{tag}_oof.npy")
        if not os.path.exists(op):
            continue
        oofs[tag] = np.load(op)
        fs = sorted([f for f in os.listdir(PRED) if f.startswith(tag + "_test_fold")])
        if fs:
            tests[tag] = np.mean([sm(np.load(os.path.join(PRED, f))) for f in fs], 0)

    full = {k: v for k, v in oofs.items() if (v.sum(1) != 0).all() and k in tests}
    print("families used:", list(full))
    P = {k: sm(v) for k, v in full.items()}
    acc = lambda p: (p.argmax(1) == y).mean()

    # greedy forward selection with replacement
    cur = np.zeros((len(y), 6)); n = 0; hist = []
    for _ in range(40):
        bt, ba = None, -1
        for k in P:
            a = acc((cur * n + P[k]) / (n + 1))
            if a > ba:
                ba, bt = a, k
        cur = (cur * n + P[bt]) / (n + 1); n += 1
        hist.append((bt, ba))
    bi = max(range(len(hist)), key=lambda i: hist[i][1])
    picks = [hist[i][0] for i in range(bi + 1)]
    print(f"greedy OOF={hist[bi][1]:.4f} with {len(picks)} picks")
    from collections import Counter
    w = Counter(picks)
    weights = {k: w.get(k, 0) / len(picks) for k in P}
    print("weights:", {k: round(v, 3) for k, v in weights.items() if v})

    blend_oof = np.mean([P[k] for k in picks], 0)
    print(f"blend OOF acc = {acc(blend_oof):.4f}")

    # test probabilities
    prob = np.mean([tests[k] for k in picks], 0)
    preds = prob.argmax(1)

    # near-dup override (validated ~97% accurate)
    td = np.load(os.path.join(WORK, "_test_nn_dist.npy"))
    tl = np.load(os.path.join(WORK, "_test_nn_lbl.npy"))
    files = np.load(os.path.join(WORK, "cache", "data.npz"), allow_pickle=True)["test_names"]
    files = [str(x) for x in files]
    nn_order = json.load(open(os.path.join(WORK, "_test_files_order.json")))
    assert nn_order == files
    mask = td < 2.0
    n_chg = int((tl[mask] != preds[mask]).sum())
    preds = preds.copy(); preds[mask] = tl[mask]
    print(f"overlap override: {int(mask.sum())} imgs, {n_chg} changed")

    tpl_ids = []
    with open(TPL) as fh:
        r = csv.reader(fh); next(r)
        for row in r:
            tpl_ids.append(row[0])
    pm = dict(zip(files, preds.tolist()))
    assert all(i in pm for i in tpl_ids)
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["ID", "Emotion"])
        for fid in tpl_ids:
            wr.writerow([fid, int(pm[fid])])
    dist = Counter(pm[i] for i in tpl_ids)
    print(f"wrote {OUT} rows={len(tpl_ids)} dist={dict(sorted(dist.items()))}")
    json.dump({"oof_acc": float(acc(blend_oof)), "weights": {k: v for k, v in weights.items() if v}},
              open(os.path.join(PRED, "final_blend.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
