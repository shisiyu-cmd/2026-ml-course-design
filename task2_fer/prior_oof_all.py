"""Generate OOF + test logits for ALL prior checkpoints (22 models)."""
import os, json, time
import numpy as np
import torch
from prior_oof import build, load_sd, infer, Xtr, ytr, Xte, splits, WORK, DEV

MODELS = []
for f in range(1, 6):
    MODELS.append(("r18_f%d" % f, "r18", 96, os.path.join(WORK, "folds", f"fold{f}.pt")))
for f in range(1, 6):
    MODELS.append(("r34_f%d" % f, "r34", 112, os.path.join(WORK, "folds_r34", f"fold{f}.pt")))
for f in range(1, 6):
    MODELS.append(("gnet_f%d" % f, "gnet", 112, os.path.join(WORK, "folds_googlenet", f"fold{f}.pt")))
for f in range(1, 6):
    MODELS.append(("pseudo_f%d" % f, "r34", 112, os.path.join(WORK, "folds_pseudo", f"fold{f}.pt")))
for f in [1, 2]:
    MODELS.append(("eff_f%d" % f, "eff", 112, os.path.join(WORK, "folds_eff", f"fold{f}.pt")))
for f in range(1, 5):
    MODELS.append(("align_f%d" % f, "r34", 112, os.path.join(WORK, "folds_aligned", f"fold{f}.pt")))

OUT = os.path.join(WORK, "prior_preds")
os.makedirs(OUT, exist_ok=True)

# group members by model family so each "tag" = one fold-ensemble
FAMILY = {}
for name, arch, size, path in MODELS:
    fam = name.rsplit("_f", 1)[0]
    FAMILY.setdefault(fam, []).append((name, arch, size, path))

print("families:", {k: len(v) for k, v in FAMILY.items()}, flush=True)

oof_all = {}
test_all = {}
for fam, members in FAMILY.items():
    oof = np.zeros((len(Xtr), 6), dtype=np.float32)
    covered = np.zeros(len(Xtr), dtype=bool)
    tacc = []
    for name, arch, size, path in members:
        if not os.path.exists(path):
            print(f"  skip {name} (missing)", flush=True); continue
        fold = int(name.rsplit("_f", 1)[1])
        tri, vai = splits[fold - 1]
        t0 = time.time()
        m = build(arch); load_sd(m, path); m = m.to(DEV)
        lg = infer(m, Xtr[vai], size)
        oof[vai] = lg; covered[vai] = True
        tacc.append(float((lg.argmax(1) == ytr[vai]).mean()))
        te = infer(m, Xte, size)
        test_all.setdefault(name, te)
        print(f"  {name:12s} val_acc={tacc[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del m; torch.cuda.empty_cache()
    oof_all[fam] = oof
    print(f"{fam}: fold-avg val acc = {np.mean(tacc):.4f}  covered={covered.mean():.2%}", flush=True)

np.savez(os.path.join(OUT, "prior_oof.npz"), **oof_all)
np.savez(os.path.join(OUT, "prior_test.npz"), **test_all)
np.save(os.path.join(OUT, "ytr.npy"), ytr)
print("\nsaved prior OOF/test to", OUT, flush=True)
