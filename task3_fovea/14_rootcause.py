# -*- coding: utf-8 -*-
"""定位「向圆心收缩」偏差的根因：对比三种解码方式的收缩斜率。

假设: 全局 softmax 的 soft_argmax 会被整张热力图的背景响应拉向背景重心，
      形成系统性收缩。若改用「argmax 附近窗口内的 soft-argmax」(局部解码)，
      收缩应消失。
"""
import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["base200", "seed1", "clahe1"])
p.add_argument("--beta", type=float, default=20.0)
p.add_argument("--win", type=int, default=16, help="局部窗口半径(热力图格子)")
a = p.parse_args()


def soft_argmax_local(heat, beta=20.0, win=16):
    """argmax 邻域窗口内的 soft-argmax。"""
    B, _, h, w = heat.shape
    flat = heat.view(B, -1)
    idx = flat.argmax(1)
    cy = (idx // w).long()
    cx = (idx % w).long()
    ys = torch.arange(h, device=heat.device).view(1, h, 1)
    xs = torch.arange(w, device=heat.device).view(1, 1, w)
    mask = ((ys - cy.view(B, 1, 1)).abs() <= win) & ((xs - cx.view(B, 1, 1)).abs() <= win)
    z = (heat * beta).masked_fill(~mask, -1e4).view(B, -1)
    pr = torch.softmax(z, dim=-1).view(B, 1, h, w)
    ex = (pr.sum(2) * torch.arange(w, device=heat.device).float()).sum(-1).squeeze(-1)
    ey = (pr.sum(3) * torch.arange(h, device=heat.device).float()).sum(-1).squeeze(-1)
    return torch.stack([ex, ey], 1)


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = C.load_meta()
gt = C.load_gt(); gt_map = C.load_gt_map(gt)
keys = sorted(gt_map.keys())
strat = np.array([0 if meta[k]["W"] < 2500 else 1 for k in keys])
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=100).split(keys, strat))

default_norm = (list(C.NORM_MEAN), list(C.NORM_STD))
out = {}   # (tag, decoder) -> (N,2) 原图像素
for tag in a.tags:
    preds = {d: [] for d in ("global", "local", "hard")}
    keyset = []
    for fold, (ti, vi) in enumerate(folds):
        pth = os.path.join("runs", "%s_f%d.pt" % (tag, fold))
        if not os.path.exists(pth):
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        ar = d.get("args", {})
        C.set_norm(ar.get("norm_mean", default_norm[0]), ar.get("norm_std", default_norm[1]))
        cl = bool(ar.get("clahe", False))
        cache = C.build_cache(meta, verbose=False, clahe=cl)
        m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                       mid=ar.get("mid", 64)).to(dev)
        m.load_state_dict(d["state"]); m.eval()
        with torch.no_grad():
            for i in vi:
                k = keys[i]
                x = C.normalize_batch(torch.from_numpy(
                    np.ascontiguousarray(cache[k])).unsqueeze(0).to(dev))
                prob = torch.sigmoid(m(x))
                cs = {
                    "global": C.soft_argmax(prob, a.beta),
                    "local": soft_argmax_local(prob, a.beta, a.win),
                    "hard": C.hard_argmax_refine(prob),
                }
                for name, c in cs.items():
                    cc = c[0].cpu().numpy() * C.STRIDE
                    preds[name].append(C.to_orig(meta[k], cc[0], cc[1]))
                keyset.append(k)
    for name, P in preds.items():
        out[(tag, name)] = np.array(P)
    out[("_keys", tag)] = keyset

K = out[("_keys", a.tags[0])]
GT = np.array([gt_map[k] for k in K])
M = np.array([[meta[k]["cx"], meta[k]["cy"], meta[k]["r"]] for k in K])
G = np.stack([(GT[:, 0] - M[:, 0]) / M[:, 2], (GT[:, 1] - M[:, 1]) / M[:, 2]], 1)


def report(P, label):
    R = np.stack([(P[:, 0] - M[:, 0]) / M[:, 2], (P[:, 1] - M[:, 1]) / M[:, 2]], 1)
    sx = np.polyfit(G[:, 0], R[:, 0], 1)[0]
    sy = np.polyfit(G[:, 1], R[:, 1], 1)[0]
    rmse = float(np.sqrt(((P - GT) ** 2).mean()))
    med = float(np.median(np.sqrt(((P - GT) ** 2).sum(1))))
    # 反向回归(贝叶斯最优, 保证降低平方误差)的留一法增益
    fx = np.array([np.polyval(np.polyfit(R[np.arange(len(R)) != i, 0], G[np.arange(len(R)) != i, 0], 1),
                              R[i, 0]) for i in range(len(R))])
    fy = np.array([np.polyval(np.polyfit(R[np.arange(len(R)) != i, 1], G[np.arange(len(R)) != i, 1], 1),
                              R[i, 1]) for i in range(len(R))])
    Pc = np.stack([M[:, 0] + fx * M[:, 2], M[:, 1] + fy * M[:, 2]], 1)
    print("%-26s %8.3f %8.2f   X斜率=%6.3f  Y斜率=%6.3f   反向回归后=%6.3f"
          % (label, rmse, med, sx, sy, float(np.sqrt(((Pc - GT) ** 2).mean()))))


print("=" * 100)
print("%-26s %8s %8s   %s" % ("解码方式", "RMSE", "中位", "收缩斜率(1.0=无收缩)"))
print("=" * 100)
for tag in a.tags:
    for name in ("global", "local", "hard"):
        if (tag, name) in out:
            report(out[(tag, name)], "%s / %s" % (tag, name))
    ens = np.mean([out[(tag, n)] for n in ("global", "local", "hard")], 0)
    report(ens, "%s / 三者均值" % tag)
    print()
ens_all = np.mean([out[(t, n)] for t in a.tags for n in ("global", "local", "hard")], 0)
report(ens_all, "全部模型+全部解码 均值")
report(np.mean([out[(t, "local")] for t in a.tags], 0), "全部模型 / local")
report(np.mean([out[(t, "global")] for t in a.tags], 0), "全部模型 / global")