"""Generate soft pseudo-labels for the test set from the current best OOF blend."""
import os, json
import numpy as np

WORK = r"C:\Users\古恩宇\Downloads\2026task2"
PRED = os.path.join(WORK, "preds")
PRIOR = os.path.join(WORK, "prior_preds")


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


y = np.load(os.path.join(PRIOR, "ytr.npy"))
po = np.load(os.path.join(PRIOR, "prior_oof.npz"))
pt = np.load(os.path.join(PRIOR, "prior_test.npz"))

oofs, tests = {}, {}
for fam in po.files:
    oofs[fam] = po[fam]
    ks = [k for k in pt.files if k.startswith(fam + "_f")]
    if ks:
        tests[fam] = np.mean([sm(pt[k]) for k in ks], 0)
for tag in ["r18_64", "r34_96", "eff_112", "v2_r34"]:
    op = os.path.join(PRED, f"{tag}_oof.npy")
    if not os.path.exists(op):
        continue
    oofs[tag] = np.load(op)
    fs = sorted([f for f in os.listdir(PRED) if f.startswith(tag + "_test_fold")])
    if fs:
        tests[tag] = np.mean([sm(np.load(os.path.join(PRED, f))) for f in fs], 0)

full = {k: v for k, v in oofs.items() if (v.sum(1) != 0).all() and k in tests}
P = {k: sm(v) for k, v in full.items()}
acc = lambda p: (p.argmax(1) == y).mean()

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
print(f"blend OOF = {hist[bi][1]:.4f}  ({len(picks)} picks)")

prob = np.mean([tests[k] for k in picks], 0)
np.save(os.path.join(PRED, "pseudo_labels.npy"), prob.astype(np.float32))
conf = prob.max(1)
print(f"pseudo labels: mean conf {conf.mean():.3f}, "
      f">0.45: {(conf>=0.45).sum()}, >0.6: {(conf>=0.6).sum()}, >0.8: {(conf>=0.8).sum()}")
from collections import Counter
print("pred dist:", dict(sorted(Counter(prob.argmax(1).tolist()).items())))
print("saved -> preds/pseudo_labels.npy")
