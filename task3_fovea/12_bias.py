# -*- coding: utf-8 -*-
"""检查 OOF 误差是否存在可修正的系统性成分：
1) 全局偏移   2) 相对视野圆心的缩放   3) 与位置/尺寸的相关性
"""
import os, argparse
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["base200", "seed1", "clahe1"])
p.add_argument("--variant", default="plain")
a = p.parse_args()

meta = C.load_meta()
P, Y, K, PK = [], [], [], []
for t in a.tags:
    f = os.path.join("runs", "tta_%s.npz" % t)
    if not os.path.exists(f):
        continue
    d = np.load(f, allow_pickle=True)
    P.append(d["soft_" + a.variant]); Y.append(d["gt"]); PK.append(d["peak_mean"])
    if not K:
        K = list(d["keys"])
    else:
        assert K == list(d["keys"])
P, Y, PK = np.stack(P), np.stack(Y), np.stack(PK)
pred, gt = P.mean(0), Y[0]
rmse = lambda Q: float(np.sqrt(((Q - gt) ** 2).mean()))
print("集成(%d 轮) OOF RMSE = %.3f" % (len(a.tags), rmse(pred)))

err = pred - gt
print()
print("误差均值(系统偏移) = (%.2f, %.2f) px" % (err[:, 0].mean(), err[:, 1].mean()))
print("误差标准差         = (%.2f, %.2f) px" % (err[:, 0].std(), err[:, 1].std()))
print()
print("去掉系统偏移后的 RMSE = %.3f  (改善 %.2f px)"
      % (rmse(pred - err.mean(0)), rmse(pred) - rmse(pred - err.mean(0))))

# --- 相对视野圆心的缩放修正 ---
M = np.array([[meta[k]["cx"], meta[k]["cy"], meta[k]["r"]] for k in K])
gx = (gt[:, 0] - M[:, 0]) / M[:, 2]
gy = (gt[:, 1] - M[:, 1]) / M[:, 2]
px = (pred[:, 0] - M[:, 0]) / M[:, 2]
py = (pred[:, 1] - M[:, 1]) / M[:, 2]
print()
print("相对视野圆心的归一化坐标:")
print("  真值 std = (%.5f, %.5f)" % (gx.std(), gy.std()))
print("  预测 std = (%.5f, %.5f)" % (px.std(), py.std()))
print("  -> 预测的离散度是真值的 %.3f 倍 (若明显 <1 说明模型在向均值收缩)"
      % (np.hypot(px.std(), py.std()) / np.hypot(gx.std(), gy.std())))

# 最优线性缩放 pred_rel = a * gt_rel + b
A = np.stack([gx, np.ones_like(gx)], 1)
coef_x, *_ = np.linalg.lstsq(A, px, rcond=None)
A2 = np.stack([gy, np.ones_like(gy)], 1)
coef_y, *_ = np.linalg.lstsq(A2, py, rcond=None)
print("  用真值回归预测: px = %.3f*gx + %.5f ; py = %.3f*gy + %.5f"
      % (coef_x[0], coef_x[1], coef_y[0], coef_y[1]))
corr = np.stack([coef_x[0] * gx + coef_x[1], coef_y[0] * gy + coef_y[1]], 1)
Pc = np.stack([M[:, 0] + corr[:, 0] * M[:, 2], M[:, 1] + corr[:, 1] * M[:, 2]], 1)
print("  用该线性修正(在 OOF 上拟合, 乐观估计) RMSE = %.3f" % rmse(Pc))

# --- 误差与位置/尺寸的相关性 ---
df = pd.DataFrame(dict(err_x=err[:, 0], err_y=err[:, 1],
                       dist=np.sqrt((err ** 2).sum(1)),
                       gx=gx, gy=gy, r=M[:, 2], W=[meta[k]["W"] for k in K],
                       peak=PK.mean(0)))
print()
print("误差与各量的相关性 (Pearson):")
for c in ["gx", "gy", "r", "W", "peak"]:
    print("   err_x~%-5s %+.3f   err_y~%-5s %+.3f   dist~%-5s %+.3f"
          % (c, df.err_x.corr(df[c]), c, df.err_y.corr(df[c]), c, df.dist.corr(df[c])))

# --- 按误差大小分组看是否有共同模式 ---
print()
print("误差 >40px 的图 (%d 张):" % (df.dist > 40).sum())
bad = df[df.dist > 40].sort_values("dist", ascending=False)
print(bad[["dist", "gx", "gy", "r", "peak"]].to_string(
    index=False, float_format=lambda v: "%.3f" % v))