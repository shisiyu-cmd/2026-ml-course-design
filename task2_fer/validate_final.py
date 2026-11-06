"""Honest evaluation of the final blend: split-half validation of the greedy
procedure, so we know whether 0.7293 is real or selection noise."""
import os, json
import numpy as np
from collections import Counter

WORK = r"C:\Users\古恩宇\Downloads\2026task2"
PRED = os.path.join(WORK, "preds")
PRIOR = os.path.join(WORK, "prior_preds")


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


y = np.load(os.path.join(PRIOR, "ytr.npy"))
po = np.load(os.path.join(PRIOR, "prior_oof.npz"))
oofs = {k: po[k] for k in po.files}
for tag in ["r18_64", "r34_96", "eff_112", "v2_r34", "p2_r34"]:
    p = os.path.join(PRED, f"{tag}_oof.npy")
    if os.path.exists(p):
        oofs[tag] = np.load(p)
full = {k: v for k, v in oofs.items() if (v.sum(1) != 0).all()}
P = {k: sm(v) for k, v in full.items()}
print("models:", list(P))


def greedy(P, yy, iters=30):
    keys = list(P)
    cur = np.zeros((len(yy), 6)); n = 0; hist = []
    for _ in range(iters):
        bt, ba = None, -1
        for k in keys:
            a = (((cur * n + P[k]) / (n + 1)).argmax(1) == yy).mean()
            if a > ba:
                ba, bt = a, k
        cur = (cur * n + P[bt]) / (n + 1); n += 1
        hist.append((bt, ba))
    bi = max(range(len(hist)), key=lambda i: hist[i][1])
    return [hist[i][0] for i in range(bi + 1)], hist[bi][1]


# full-data blend
picks, acc_full = greedy(P, y)
print(f"\nfull-data greedy OOF = {acc_full:.4f} ({len(picks)} picks)")
print("weights:", {k: round(v, 3) for k, v in Counter(picks).items()})

# ---- honest 5x random split-half check ----
print("\n=== split-half validation (select on A, score on B) ===")
res = []
for seed in range(5):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    h1, h2 = idx[:len(y) // 2], idx[len(y) // 2:]
    P1 = {k: P[k][h1] for k in P}
    pk, a1 = greedy(P1, y[h1])
    blend = np.mean([P[k] for k in pk], 0)
    a2 = (blend[h2].argmax(1) == y[h2]).mean()
    res.append((a1, a2))
    print(f"  seed{seed}: select-half={a1:.4f}  heldout-half={a2:.4f}  gap={a1-a2:+.4f}")
gaps = [a - b for a, b in res]
print(f"\nmean held-out = {np.mean([b for _, b in res]):.4f}  (vs full-data {acc_full:.4f})")
print(f"mean selection gap = {np.mean(gaps):+.4f}")

# ---- compare against simple baselines on same held-out halves ----
print("\n=== baselines on the same held-out halves ===")
eq = np.mean(list(P.values()), 0)
best_single = max(P, key=lambda k: (P[k].argmax(1) == y).mean())
print(f"  equal-weight all ({len(P)}): {(eq.argmax(1)==y).mean():.4f}")
print(f"  best single ({best_single}): {(P[best_single].argmax(1)==y).mean():.4f}")
print(f"  greedy blend: {acc_full:.4f}")
