# -*- coding: utf-8 -*-
"""严格交叉验证的"去收缩校正"评估，并按离心度分层。

背景: 中心凹位置分布偏向周边的样本在测试集中占比很高，而模型预测存在向中心
收缩的倾向（回归稀释）。若校正能降低周边组的误差，则可直接用于测试集预测。

关键: 拟合必须用交叉验证，绝不能把拟合式套回同一批数据（13_recalib.py 犯过的泄漏错误）。
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["long800", "clahelong"])
p.add_argument("--folds", type=int, default=5)
a = p.parse_args()

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())


def fit_apply(P, G, nfold, seed=0):
    """P,G: (n,2)。返回交叉验证后的校正预测。"""
    out = np.empty_like(P)
    kf = KFold(nfold, shuffle=True, random_state=seed)
    for tri, tei in kf.split(P):
        for c in range(2):
            A = np.polyfit(P[tri, c], G[tri, c], 1)      # 反向回归 G ~ P
            out[tei, c] = np.polyval(A, P[tei, c])
    return out


for tag in a.tags:
    try:
        z = np.load("runs/%s_oof.npz" % tag, allow_pickle=True)
    except FileNotFoundError:
        print("跳过", tag); continue
    soft, orig, keys = z["soft"], z["orig"], [str(k) for k in z["keys"]]

    ecc = np.array([np.hypot((gt[k][0] - META[k]["cx"]) / META[k]["r"],
                             (gt[k][1] - META[k]["cy"]) / META[k]["r"]) for k in keys])
    rmse_raw = float(np.sqrt(((soft - orig) ** 2).mean()))
    corr = fit_apply(soft, orig, a.folds)
    rmse_cor = float(np.sqrt(((corr - orig) ** 2).mean()))

    # 每轴的收缩斜率（全量拟合，仅用于展示）
    sl = [np.polyfit(soft[:, c], orig[:, c], 1)[0] for c in range(2)]

    print("=" * 88)
    print("%s  n=%d   原始 RMSE=%.2f   交叉验证校正后=%.2f   (%+.2f, %+.1f%%)"
          % (tag, len(keys), rmse_raw, rmse_cor, rmse_cor - rmse_raw,
             100 * (rmse_cor - rmse_raw) / rmse_raw))
    print("   反向回归斜率: X=%.3f  Y=%.3f   (<1 表示模型输出被压缩)" % (sl[0], sl[1]))

    q1, q2 = np.quantile(ecc, [1 / 3, 2 / 3])
    lab = np.where(ecc <= q1, "近中心", np.where(ecc <= q2, "中", "偏周边"))
    print("   %-8s %5s %10s %10s %9s" % ("分组", "n", "原始RMSE", "校正RMSE", "变化"))
    for L in ["近中心", "中", "偏周边"]:
        m = lab == L
        r0 = float(np.sqrt(((soft[m] - orig[m]) ** 2).mean()))
        r1 = float(np.sqrt(((corr[m] - orig[m]) ** 2).mean()))
        print("   %-8s %5d %10.2f %10.2f %+9.2f" % (L, m.sum(), r0, r1, r1 - r0))
    print()

    # 逐图误差: 校正是否改善了尾部?
    e0 = np.sqrt(((soft - orig) ** 2).sum(1))
    e1 = np.sqrt(((corr - orig) ** 2).sum(1))
    print("   逐图误差 p90: 原始 %.1f -> 校正 %.1f | max: %.1f -> %.1f"
          % (np.percentile(e0, 90), np.percentile(e1, 90), e0.max(), e1.max()))
    imp = e1 < e0
    print("   变好的图 %d/%d；变差的图 %d" % (imp.sum(), len(imp), (~imp).sum()))
    print()