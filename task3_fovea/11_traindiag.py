# -*- coding: utf-8 -*-
"""诊断：模型在「自己训练过的图」上的误差。

- 若训练误差也很大(≈ OOF) -> 瓶颈是表达能力/目标设计/分辨率，不是泛化
- 若训练误差很小而 OOF 很大 -> 瓶颈是泛化(数据太少)
"""
import os, argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tag", default="base200")
p.add_argument("--beta", type=float, default=20.0)
a = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = C.load_meta()
gt = C.load_gt(); gt_map = C.load_gt_map(gt)
keys = sorted(gt_map.keys())
strat = np.array([0 if meta[k]["W"] < 2500 else 1 for k in keys])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
folds = list(skf.split(keys, strat))

caches = {}
default_norm = (list(C.NORM_MEAN), list(C.NORM_STD))
print("%-6s %-8s %10s %10s %10s %10s" % ("fold", "n", "训练RMSE", "训练中位", "OOF RMSE", "OOF中位"))
all_tr, all_va = [], []
for fold, (ti, vi) in enumerate(folds):
    pth = os.path.join("runs", "%s_f%d.pt" % (a.tag, fold))
    if not os.path.exists(pth):
        continue
    d = torch.load(pth, map_location="cpu", weights_only=False)
    ar = d.get("args", {})
    cl = bool(ar.get("clahe", False))
    C.set_norm(ar.get("norm_mean", default_norm[0]), ar.get("norm_std", default_norm[1]))
    if cl not in caches:
        caches[cl] = C.build_cache(meta, verbose=False, clahe=cl)
    m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                   mid=ar.get("mid", 64)).to(dev)
    m.load_state_dict(d["state"]); m.eval()
    res = {}
    for name, idxs in (("train", ti), ("oof", vi)):
        P, Y = [], []
        with torch.no_grad():
            for i in idxs:
                k = keys[i]
                x = C.normalize_batch(torch.from_numpy(
                    np.ascontiguousarray(caches[cl][k])).unsqueeze(0).to(dev))
                prob = torch.sigmoid(m(x))
                c = C.soft_argmax(prob, a.beta)[0].cpu().numpy() * C.STRIDE
                P.append(C.to_orig(meta[k], c[0], c[1])); Y.append(gt_map[k])
        P, Y = np.array(P), np.array(Y)
        e = np.sqrt(((P - Y) ** 2).sum(1))
        res[name] = (float(np.sqrt(((P - Y) ** 2).mean())), float(np.median(e)), len(idxs))
    all_tr += [res["train"][0]] * 1
    all_va += [res["oof"][0]] * 1
    print("%-6d %-8d %10.2f %10.2f %10.2f %10.2f"
          % (fold, res["train"][2], res["train"][0], res["train"][1],
             res["oof"][0], res["oof"][1]))
print()
print("训练集平均 RMSE = %.2f px" % np.mean(all_tr))
print("OOF   平均 RMSE = %.2f px" % np.mean(all_va))
print()
print("判读: 若训练 RMSE 远小于 OOF(例如 2px vs 30px)，说明模型有精度能力、")
print("      瓶颈在泛化；若两者接近，说明目标/分辨率本身限制了精度。")