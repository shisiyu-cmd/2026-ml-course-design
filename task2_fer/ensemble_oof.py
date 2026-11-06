"""Ensemble OOF predictions to find optimal weights, then apply to test."""
import os, json, glob, argparse
import numpy as np
from itertools import product

PRED = r"C:\Users\古恩宇\Downloads\2026task2\preds"


def load_oof(tags):
    """Return dict tag -> (oof_logits, y)."""
    out = {}
    for t in tags:
        p = os.path.join(PRED, f"{t}_oof.npy")
        if not os.path.exists(p):
            print(f"  MISSING {t}")
            continue
        oof = np.load(p)
        y = np.load(os.path.join(PRED, f"{t}_ytest.npy"))
        out[t] = (oof, y)
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--test_only", action="store_true")
    args = ap.parse_args()

    oofs = load_oof(args.tags)
    if not oofs:
        print("no OOF found"); return
    y = list(oofs.values())[0][1]
    for t, (o, yy) in oofs.items():
        assert (yy == y).all(), f"label mismatch for {t}"
        print(f"{t:24s} OOF acc = {(o.argmax(1)==y).mean():.4f}")

    tags = list(oofs.keys())
    # normalize each model's logits to probabilities (better for mixing)
    P = {t: softmax(oofs[t][0]) for t in tags}

    print("\n=== equal-weight ensemble ===")
    eq = np.mean([P[t] for t in tags], 0)
    print(f"  acc = {(eq.argmax(1)==y).mean():.4f}")

    # greedy forward selection with replacement (Caruana)
    print("\n=== greedy ensemble selection ===")
    cur = np.zeros_like(eq)
    chosen = []
    best_acc = 0.0
    for step in range(30):
        best_t, best_a = None, -1
        for t in tags:
            cand = (cur * len(chosen) + P[t]) / (len(chosen) + 1)
            a = (cand.argmax(1) == y).mean()
            if a > best_a:
                best_a, best_t = a, t
        cur = (cur * len(chosen) + P[best_t]) / (len(chosen) + 1)
        chosen.append(best_t)
        if best_a > best_acc:
            best_acc = best_a
        if step % 3 == 0:
            print(f"  step {step+1:2d} +{best_t:22s} acc={best_a:.4f}")
    print(f"  greedy best acc = {best_acc:.4f}")
    from collections import Counter
    w = Counter(chosen)
    print("  weights:", dict(w))
    final_w = {t: w.get(t, 0) / len(chosen) for t in tags}
    print("  normalized:", {k: round(v, 3) for k, v in final_w.items()})

    # save weights
    with open(os.path.join(PRED, "ensemble_weights.json"), "w") as fh:
        json.dump({"weights": final_w, "greedy_oof_acc": float(best_acc),
                   "equal_oof_acc": float((eq.argmax(1) == y).mean())}, fh, indent=2)
    print("\nsaved ensemble_weights.json")


if __name__ == "__main__":
    main()
