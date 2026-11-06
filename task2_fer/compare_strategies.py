"""Compare ensemble strategies honestly on OOF, including the near-dup effect."""
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
oofs = {k: po[k] for k in po.files}
for tag in ["r18_64", "r34_96", "eff_112"]:
    p = os.path.join(PRED, f"{tag}_oof.npy")
    if os.path.exists(p):
        oofs[tag] = np.load(p)

full = {k: v for k, v in oofs.items() if (v.sum(1) != 0).all()}
print(f"models with full coverage: {len(full)}")

P = {k: sm(v) for k, v in full.items()}
acc = lambda p: (p.argmax(1) == y).mean()


def greedy(P, y, iters=40, seed=None):
    keys = list(P)
    cur = np.zeros((len(y), 6)); n = 0; hist = []
    for _ in range(iters):
        bt, ba = None, -1
        for k in keys:
            cand = (cur * n + P[k]) / (n + 1)
            a = (cand.argmax(1) == y).mean()
            if a > ba:
                ba, bt = a, k
        cur = (cur * n + P[bt]) / (n + 1); n += 1
        hist.append((bt, ba))
    return hist


print("\n=== strategy comparison (OOF accuracy) ===")
res = {}

# A: equal weight over all individual models
A = np.mean(list(P.values()), 0)
res["equal_all_models"] = acc(A)

# B: equal weight per family (r18,r34,gnet,pseudo,eff + new)
fams = {}
for k in P:
    fam = k.rsplit("_f", 1)[0] if "_f" in k else k
    fams.setdefault(fam, []).append(k)
B = np.mean([np.mean([P[k] for k in ks], 0) for ks in fams.values()], 0)
res["equal_per_family"] = acc(B)

# C: greedy on individual models
h = greedy(P, y)
best_i = max(range(len(h)), key=lambda i: h[i][1])
from collections import Counter
picks = [h[i][0] for i in range(best_i + 1)]
C = np.mean([P[k] for k in picks], 0)
res["greedy_models"] = acc(C)
wC = Counter(picks)

# D: drop weak families then equal weight
strong = {k: v for k, v in P.items() if not k.startswith("gnet")}
D = np.mean(list(strong.values()), 0)
res["equal_no_gnet"] = acc(D)

# E: logit average (not prob) over all
E = np.mean([full[k] for k in full], 0)
res["equal_all_logits"] = acc(E)

for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k:22s} {v:.4f}")

print(f"\ngreedy picks ({len(picks)}):", dict(wC))
print(f"greedy OOF: {res['greedy_models']:.4f}")

# honest 2-fold check of the greedy procedure itself
rng = np.random.RandomState(0); idx = rng.permutation(len(y))
h1, h2 = idx[:len(y)//2], idx[len(y)//2:]
Ph1 = {k: P[k][h1] for k in P}
h = greedy(Ph1, y[h1], iters=25)
pk = [h[i][0] for i in range(len(h))]
blend = np.mean([P[k] for k in pk], 0)
print(f"\nhonest check -> selected on half A, scored on half B: {acc(blend):.4f}")
print(f"equal-per-family on half B: {(np.mean([np.mean([P[k] for k in ks],0) for ks in fams.values()],0)[h2].argmax(1)==y[h2]).mean():.4f}")

json.dump({k: float(v) for k, v in res.items()}, open(os.path.join(PRED, "strategy_cmp.json"), "w"), indent=2)
