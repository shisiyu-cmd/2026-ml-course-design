# -*- coding: utf-8 -*-
"""排查 OOF 与榜单脱节的可能原因。

1) 训练集 vs 测试集 的图像统计/视野参数是否同分布
2) 去除「按验证集选权重」的选择偏差后，OOF 真实水平是多少
3) 若测试集抽到 N 张难图，榜单 RMSE 会是多少（模拟小样本方差）
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import cv2
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["long800"])
a = p.parse_args()

meta = C.load_meta()
cache = C.build_cache(meta)


def stats_of(k):
    m = meta[k]
    im = cache[k]
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    S = g.shape[0]
    yy, xx = np.mgrid[0:S, 0:S]
    disk = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) < (S * 0.45) ** 2
    r = g[disk]
    lap = cv2.Laplacian(g, cv2.CV_32F)[disk]
    return dict(key=k, W=m["W"], r_fov=m["r"],
                mean=float(r.mean()), std=float(r.std()),
                p05=float(np.percentile(r, 5)), p95=float(np.percentile(r, 95)),
                sharp=float(lap.std()),
                dark_frac=float((r < 0.6 * np.median(r)).mean()))


rows = [stats_of(k) for k in sorted(meta)]
df = pd.DataFrame(rows)
df["split"] = df.key.str.split("/").str[0]
print("=" * 92)
print("训练集 vs 测试集 图像统计（画布内视网膜区域）")
print("=" * 92)
print(df.groupby("split")[["W", "r_fov", "mean", "std", "p05", "p95", "sharp", "dark_frac"]]
      .agg(["mean", "std"]).T.to_string())
print()
big = df[df.W > 2500]
print("仅 2992x2000 大图对比:")
print(big.groupby("split")[["r_fov", "mean", "std", "sharp", "dark_frac"]]
      .agg(["mean", "std"]).T.to_string())

# ---- 2. 选择偏差: 用最后一轮 vs 按验证集选 ----
print()
print("=" * 92)
print("「按验证集选权重」带来的乐观偏差")
print("=" * 92)
for tag in a.tags:
    f = os.path.join("runs", "%s.log" % tag)
    if not os.path.exists(f):
        continue
    lines = [l for l in open(f, encoding="utf-8") if "ep" in l and "val RMSE" in l]
    per_fold = {}
    for l in lines:
        try:
            fold = int(l.split("f")[1].split("]")[0])
            ep = int(l.split("ep")[1].split()[0])
            v = float(l.split("val RMSE soft=")[1].split()[0])
        except Exception:
            continue
        per_fold.setdefault(fold, []).append((ep, v))
    best, final = [], []
    for fold, arr in sorted(per_fold.items()):
        arr.sort()
        best.append(min(v for _, v in arr))
        final.append(arr[-1][1])
    print("%-10s 各折最佳(被选中) 均值=%.2f  各折末轮 均值=%.2f  -> 乐观偏差约 %.0f%%"
          % (tag, np.mean(best), np.mean(final),
             100 * (1 - np.mean(best) / np.mean(final))))
    print("          最佳: %s" % np.round(best, 1).tolist())
    print("          末轮: %s" % np.round(final, 1).tolist())

# ---- 3. 小样本方差模拟 ----
print()
print("=" * 92)
print("测试集只抽 20 张 -> 榜单 RMSE 的抽样波动 (用 OOF 误差分布重采样)")
print("=" * 92)
d = np.load("runs/tta_%s.npz" % a.tags[0], allow_pickle=True)
P, Y = d["soft_plain"], d["gt"]
err = np.sqrt(((P - Y) ** 2).sum(1))          # 每张图的欧氏误差(仅供观察)
sq = ((P - Y) ** 2).reshape(-1)               # 逐值平方误差, 40 个/20 张
per_img_sq = ((P - Y) ** 2)
rng = np.random.default_rng(0)
sims = []
for _ in range(20000):
    idx = rng.choice(len(P), 20, replace=False)
    sims.append(np.sqrt(per_img_sq[idx].mean()))
sims = np.array(sims)
print("真实 OOF(80 张) RMSE = %.2f" % np.sqrt(per_img_sq.mean()))
print("随机抽 20 张时的 RMSE 分布: p5=%.1f p25=%.1f 中位=%.1f p75=%.1f p95=%.1f 最大=%.1f"
      % tuple(np.percentile(sims, [5, 25, 50, 75, 95, 100])))
print()
print("若抽到 2 张 >100px 的难图会怎样:")
for n_bad in range(0, 6):
    rest = np.sort(per_img_sq.sum(1))[:-n_bad] if n_bad else per_img_sq.sum(1)
    # 用最差的 n_bad 张 + 随机 20-n_bad 张
    bad = np.sort(per_img_sq.sum(1))[-n_bad:] if n_bad else np.array([])
    good_pool = np.sort(per_img_sq.sum(1))[:-n_bad] if n_bad else per_img_sq.sum(1)
    est = (bad.sum() + good_pool.mean() * (20 - n_bad)) / 40
    print("   难图 %d 张 -> 预期 RMSE ≈ %.1f px" % (n_bad, np.sqrt(max(est, 0))))
print()
print("OOF 中误差 >60px 的图占比: %.1f%% (即 20 张里期望 %.1f 张)"
      % (100 * (err > 60).mean(), 20 * (err > 60).mean()))