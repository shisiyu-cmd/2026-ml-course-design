# -*- coding: utf-8 -*-
"""分析「置信度 -> 误差」关系，并搜索低置信度时的兜底策略。

背景: OOF 中位误差仅 7px, 但约 9/80 张灾难性失败(70~210px)主导了 RMSE,
且这些图的预测热力图峰值明显偏低。这里检验能否用峰值做兜底/收缩。
"""
import os, argparse
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tta", default="runs/tta_base200.npz")
p.add_argument("--oof", default="runs/base200_oof.npz")
p.add_argument("--variant", default="plain")
a = p.parse_args()

d = np.load(a.tta, allow_pickle=True)
keys = list(d["keys"])
gt = d["gt"]
pred = d["soft_" + a.variant]
peak = d["peak_mean"]
fold = d["fold"]
err = np.sqrt(((pred - gt) ** 2).sum(1))
rmse = lambda P: float(np.sqrt(((P - gt) ** 2).mean()))
print("基准 (variant=%s, soft-argmax): RMSE=%.3f  中位欧氏=%.2f  最大=%.1f"
      % (a.variant, rmse(pred), np.median(err), err.max()))
print()

# ---------- 1. 峰值与误差的关系 ----------
df = pd.DataFrame(dict(key=keys, peak=peak, err=err, fold=fold,
                       ex=pred[:, 0] - gt[:, 0], ey=pred[:, 1] - gt[:, 1]))
print("按峰值分箱统计:")
bins = [0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
df["bin"] = pd.cut(df.peak, bins)
print(df.groupby("bin", observed=True).agg(
    n=("err", "size"), 欧氏中位=("err", "median"), 欧氏均值=("err", "mean"),
    欧氏最大=("err", "max")).to_string())
print()
print("峰值与误差的相关系数 (Spearman): %.3f"
      % df[["peak", "err"]].corr(method="spearman").iloc[0, 1])

# ---------- 2. 先验位置 ----------
meta = C.load_meta()
# 每折的训练集均值(画布坐标) —— 只用该折训练样本, 避免信息泄漏
prior_c = {}
for f in sorted(set(fold)):
    tr = [i for i in range(len(keys)) if fold[i] != f]
    cs = np.array([C.to_canvas(meta[keys[i]], *gt[i]) for i in tr])
    prior_c[f] = cs.mean(0)
prior = np.array([C.to_orig(meta[keys[i]], *prior_c[fold[i]]) for i in range(len(keys))])
print()
print("先验(每折训练集均值)单独使用的 RMSE=%.3f" % rmse(prior))

# ---------- 3. 兜底 / 收缩策略 ----------
print()
print("=" * 74)
print("策略 A: 峰值 < 阈值 时改用先验")
print("%-10s %-8s %-10s %-10s" % ("阈值", "触发数", "RMSE", "触发图平均误差"))
best_a = (1e9, None)
for t in np.arange(0.30, 0.75, 0.05):
    m = peak < t
    P = pred.copy()
    P[m] = prior[m]
    r = rmse(P)
    print("%-10.2f %-8d %-10.3f %-10.1f" % (t, m.sum(), r, err[m].mean() if m.sum() else 0))
    if r < best_a[0]:
        best_a = (r, t)

print()
print("策略 B: 向先验收缩, 权重随峰值变化  P = prior + w*(pred-prior)")
print("%-10s %-10s %-10s" % ("w_min", "w_max", "RMSE"))
best_b = (1e9, None)
for wlo in (0.0, 0.1, 0.2, 0.3, 0.5):
    for whi in (0.9, 1.0):
        # 峰值 0.3 以下取 wlo, 0.75 以上取 whi, 中间线性
        w = np.clip((peak - 0.30) / (0.75 - 0.30), 0, 1) * (whi - wlo) + wlo
        P = prior + w[:, None] * (pred - prior)
        r = rmse(P)
        if r < best_b[0]:
            best_b = (r, (wlo, whi))
print("  最优 w: 峰值0.3处=%.1f, 峰值0.75处=%.1f -> RMSE=%.3f"
      % (best_b[1][0], best_b[1][1], best_b[0]) if best_b[1] else "  (无)")
for wlo, whi in [(0.0, 1.0), (0.2, 1.0), (0.3, 1.0), (0.5, 1.0), (0.0, 0.9)]:
    w = np.clip((peak - 0.30) / (0.75 - 0.30), 0, 1) * (whi - wlo) + wlo
    P = prior + w[:, None] * (pred - prior)
    print("  w=(%.1f->%.1f) RMSE=%.3f" % (wlo, whi, rmse(P)))

print()
print("策略 C: 直接向先验固定收缩  P = prior + w*(pred-prior)")
for w in (0.7, 0.8, 0.9, 0.95, 1.0):
    print("  w=%.2f -> RMSE=%.3f" % (w, rmse(prior + w * (pred - prior))))

print()
print("结论参考: 若把全部 9 张最差图换成先验, RMSE=%.3f"
      % rmse(np.where((err > 60)[:, None], prior, pred)))
print("若这些图的误差能被压到 60px, RMSE=%.3f"
      % float(np.sqrt((np.minimum(err, 60.0) ** 2).mean())))

# ---------- 4. 误差 vs 峰值 明细 ----------
print()
print("最差 15 张明细:")
print(df.nlargest(15, "err")[["key", "peak", "err", "fold"]].to_string(
    index=False, float_format=lambda v: "%.3f" % v))