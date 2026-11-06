# -*- coding: utf-8 -*-
"""OOF 结果分析：误差分布、按图像尺寸/来源分组、最差样本可视化。"""
import os, sys, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tag", default="base200")
p.add_argument("--which", default=None, choices=["soft", "hard", "blend"],
               help="使用哪种解码；默认自动选较优者")
p.add_argument("--alpha", type=float, default=0.5, help="blend 时 soft 的权重")
p.add_argument("--topn", type=int, default=12)
a = p.parse_args()

d = np.load(os.path.join("runs", "%s_oof.npz" % a.tag), allow_pickle=True)
soft, hard, orig, keys = d["soft"], d["hard"], d["orig"], list(d["keys"])
blend = a.alpha * soft + (1 - a.alpha) * hard
cands = {"soft": soft, "hard": hard, "blend": blend}
scores = {k: np.sqrt((((v - orig) ** 2)).mean()) for k, v in cands.items()}
which = a.which or min(scores, key=scores.get)
pred = cands[which]
print("解码方式 RMSE: " + "  ".join("%s=%.2f" % (k, v) for k, v in scores.items()))
print("采用: %s" % which)

meta = C.load_meta()
rows = []
for i, k in enumerate(keys):
    m = meta[k]
    e = pred[i] - orig[i]
    rows.append(dict(key=k, W=m["W"], H=m["H"], r=m["r"],
                     px=pred[i, 0], py=pred[i, 1], gx=orig[i, 0], gy=orig[i, 1],
                     ex=e[0], ey=e[1], dist=float(np.hypot(e[0], e[1])),
                     size="1956x1934" if m["W"] < 2500 else "2992x2000",
                     keynum=int(k.split("/")[1])))
df = pd.DataFrame(rows)

rmse = float(np.sqrt((((pred - orig) ** 2)).mean()))
print()
print("=" * 70)
print("总体 OOF RMSE = %.3f px   (中位欧氏误差 %.2f, 平均 %.2f)"
      % (rmse, df.dist.median(), df.dist.mean()))
print("=" * 70)
print()
print("按原图尺寸分组:")
g = df.groupby("size").apply(
    lambda s: pd.Series({
        "n": len(s),
        "RMSE_px": np.sqrt((np.concatenate([s.ex, s.ey]) ** 2).mean()),
        "RMSE相对宽": np.sqrt(((np.concatenate([s.ex, s.ey]) / s.W.iloc[0]) ** 2).mean()),
        "欧氏均值": s.dist.mean(), "欧氏中位": s.dist.median(),
        "最差": s.dist.max()}), include_groups=False)
print(g.to_string())
print()
print("欧氏误差分位数: " + "  ".join("p%d=%.1f" % (q, np.percentile(df.dist, q))
                                for q in (50, 75, 90, 95, 100)))
print()
print("误差最大的 %d 张:" % a.topn)
print(df.nlargest(a.topn, "dist")[["key", "size", "gx", "gy", "px", "py", "ex", "ey", "dist"]]
      .to_string(index=False, float_format=lambda v: "%.1f" % v))
print()
print("误差最小的 5 张:")
print(df.nsmallest(5, "dist")[["key", "size", "dist"]].to_string(index=False,
                                                                 float_format=lambda v: "%.2f" % v))

# ---- 可视化 ----
cache = C.build_cache(meta)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ax, r in zip(axes.ravel(), df.nlargest(a.topn, "dist").itertuples()):
    m = meta[r.key]
    im = cache[r.key]
    ax.imshow(im)
    cx, cy = C.to_canvas(m, r.gx, r.gy)
    px, py = C.to_canvas(m, r.px, r.py)
    ax.plot(cx, cy, "o", mfc="none", mec="lime", ms=20, mew=2)
    ax.plot(cx, cy, "+", c="lime", ms=14, mew=2)
    ax.plot(px, py, "x", c="red", ms=14, mew=2.5)
    ax.set_title("%s  err=%.1fpx" % (r.key, r.dist), fontsize=10)
    ax.axis("off")
plt.suptitle("绿=真值  红=预测  (%s, RMSE=%.2f px)" % (which, rmse), fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(C.HERE, "viz", "oof_worst_%s.png" % a.tag), dpi=70)
plt.close()

# 误差方向散点
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
axes[0].scatter(df.ex, df.ey, s=30, alpha=.75)
axes[0].axhline(0, c="k", lw=.6); axes[0].axvline(0, c="k", lw=.6)
axes[0].set_xlabel("X 误差 (px)"); axes[0].set_ylabel("Y 误差 (px)")
axes[0].set_title("误差散点 (含系统偏差)  均值=(%.1f, %.1f)" % (df.ex.mean(), df.ey.mean()))
axes[0].grid(alpha=.3); axes[0].axis("equal")

axes[1].hist(df.dist, bins=25)
axes[1].set_xlabel("欧氏误差 (px)"); axes[1].set_title("误差分布 (中位 %.1f)" % df.dist.median())
axes[1].grid(alpha=.3)

for sz, sub in df.groupby("size"):
    axes[2].scatter(sub.gx / sub.W, sub.gy / sub.H, s=30, label=sz, alpha=.8)
axes[2].set_xlabel("真值 x/W"); axes[2].set_ylabel("真值 y/H")
axes[2].set_title("样本分布 (检查划分是否有偏)"); axes[2].legend(); axes[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(C.HERE, "viz", "oof_stats_%s.png" % a.tag), dpi=90)
plt.close()
print()
print("已保存 viz/oof_worst_%s.png 与 viz/oof_stats_%s.png" % (a.tag, a.tag))

# ---- 划分检查：小图在训练/测试中的分布 ----
print()
print("图像尺寸在 train 中的编号分布:")
small = sorted([int(k.split("/")[1]) for k in meta if k.startswith("train/") and meta[k]["W"] < 2500])
print("  1956x1934 训练样本编号:", small)
print("  测试集中小图编号:", sorted([int(k.split("/")[1]) for k in meta
                                if k.startswith("test/") and meta[k]["W"] < 2500]))
big = [int(k.split("/")[1]) for k in meta if k.startswith("train/") and meta[k]["W"] >= 2500]
print("  2992x2000 训练编号范围: %d~%d (共 %d)" % (min(big), max(big), len(big)))