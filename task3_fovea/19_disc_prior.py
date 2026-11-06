# -*- coding: utf-8 -*-
"""用解剖学先验裁决分歧：中心凹相对视盘的位置分布。

中心凹位于视盘颞侧、大致与视盘中心等高。先在训练集上统计
「中心凹 - 视盘中心」的分布（按视野半径归一化），再用它评估测试图上
各模型族的候选位置哪个更符合解剖学。
"""
import os, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

p = argparse.ArgumentParser()
p.add_argument("--preds", nargs="+",
               default=["runs/test_pred_clahe1.csv", "runs/test_pred_long800.csv"])
p.add_argument("--out", default="viz/disc_prior.png")
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)
gt = C.load_gt(); gt_map = C.load_gt_map(gt)


def detect_disc(canvas):
    """视盘 = 画布中最亮的大连通域。返回 (cx, cy, radius) 或 None。"""
    g = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = g.shape[0]
    yy, xx = np.mgrid[0:S, 0:S]
    inside = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) < (S * 0.46) ** 2
    vals = g[inside]
    thr = np.percentile(vals, 97)
    m = ((g >= thr) & inside).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = st[i, cv2.CC_STAT_AREA]
    if area < 0.0005 * S * S:      # 太小, 多半是噪声
        return None
    r = float(np.sqrt(area / np.pi))
    return float(cen[i][0]), float(cen[i][1]), r


# ---------- 训练集: 中心凹相对视盘的分布 ----------
rows = []
for k, (gx, gy) in gt_map.items():
    d = detect_disc(cache[k])
    if d is None:
        rows.append(dict(key=k, ok=False)); continue
    m = meta[k]
    fx, fy = C.to_canvas(m, gx, gy)
    rows.append(dict(key=k, ok=True, dx=(fx - d[0]) / m["r"], dy=(fy - d[1]) / m["r"],
                     dr=np.hypot(fx - d[0], fy - d[1]) / m["r"],
                     disc_r=d[2] / m["r"], cx=d[0], cy=d[1], fx=fx, fy=fy))
T = pd.DataFrame(rows)
print("视盘检测成功率: %d/%d" % (T.ok.sum(), len(T)))
ok = T[T.ok]
print()
print("中心凹相对视盘中心 (按视野半径 r 归一化):")
print(ok[["dx", "dy", "dr", "disc_r"]].describe().to_string())
print()
print("dx 分布: mean=%.4f std=%.4f  |  dy 分布: mean=%.4f std=%.4f"
      % (ok.dx.mean(), ok.dx.std(), ok.dy.mean(), ok.dy.std()))
print("=> 中心凹在视盘水平方向 %.2f±%.2f r 处, 垂直方向 %.2f±%.2f r"
      % (ok.dx.mean(), ok.dx.std(), ok.dy.mean(), ok.dy.std()))

# ---------- 测试图评估 ----------
print()
print("=" * 88)
print("测试图: 各模型族候选位置 vs 解剖学先验")
print("=" * 88)
dfs = [(os.path.basename(f).replace("test_pred_", "").replace(".csv", ""),
        pd.read_csv(f).set_index("key")) for f in a.preds]
test_keys = sorted([k for k in meta if k.startswith("test/")])
res = []
for k in test_keys:
    m = meta[k]
    d = detect_disc(cache[k])
    if d is None:
        continue
    exp_fx = d[0] + ok.dx.mean() * m["r"]
    exp_fy = d[1] + ok.dy.mean() * m["r"]
    row = dict(key=k, disc=(round(d[0]), round(d[1])), exp=(round(exp_fx), round(exp_fy)))
    for nm, df in dfs:
        px, py = float(df.loc[k, "canvas_x"]), float(df.loc[k, "canvas_y"])
        row[nm] = round(float(np.hypot(px - exp_fx, py - exp_fy)), 1)
        row[nm + "_xy"] = (round(px), round(py))
    res.append(row)
R = pd.DataFrame(res)
cols = ["key", "disc", "exp"] + [c for nm, _ in dfs for c in (nm + "_xy", nm)]
print(R[cols].to_string(index=False))
print()
print("(数字为候选点与解剖学期望位置的距离, 单位: 画布像素; 越小越符合解剖学)")
print()
for nm, _ in dfs:
    print("  %-14s 与先验的平均偏差 = %.1f px" % (nm, R[nm].mean()))

# ---------- 可视化 ----------
sel = [k for k in test_keys if k in set(R.key)][:20]
n = len(sel)
cols_n = 5
rows_n = (n + cols_n - 1) // cols_n
fig, axes = plt.subplots(rows_n, cols_n, figsize=(4.6 * cols_n, 4.6 * rows_n))
axes = np.atleast_2d(axes)
colors = ["red", "deepskyblue", "orange", "magenta"]
for ax, k in zip(axes.ravel(), sel):
    m = meta[k]
    im = cache[k]
    ax.imshow(im)
    d = detect_disc(im)
    if d:
        ax.plot(d[0], d[1], "o", mfc="none", mec="yellow", ms=26, mew=2)
        ax.plot(d[0], d[1], "+", c="yellow", ms=14, mew=2.4)
    rr = R[R.key == k]
    if len(rr):
        ex, ey = rr.iloc[0]["exp"]
        ax.plot(ex, ey, "P", c="lime", ms=11, mew=2)
    for (nm, df), col in zip(dfs, colors):
        px, py = float(df.loc[k, "canvas_x"]), float(df.loc[k, "canvas_y"])
        ax.plot(px, py, "x", c=col, ms=12, mew=2.4, label=nm)
    ax.set_title(k, fontsize=9); ax.axis("off")
axes.ravel()[0].legend(fontsize=6, loc="upper left")
plt.suptitle("黄=视盘  绿P=解剖学期望中心凹位置  各色X=模型预测", fontsize=13)
plt.tight_layout()
plt.savefig(a.out, dpi=68)
plt.close()
print("\n已保存", a.out)