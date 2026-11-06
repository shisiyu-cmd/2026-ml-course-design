# -*- coding: utf-8 -*-
"""量化外部数据(IDRiD)与本赛题(GAMMA)在画布上的域差距。

若色彩/亮度分布差异大，则"域匹配"（把外部数据的统计量对齐到本赛题）
可能显著提升迁移效果 —— 这是纯 CPU 分析，不占 GPU。
"""
import numpy as np
import cv2
import common as C

META = C.load_meta()
_, ext_gt = C.load_ext()
gam_keys = sorted([k for k in META if k.startswith("train/") or k.startswith("test/")])
ext_keys = sorted(ext_gt.keys())
cache = C.build_cache(META, verbose=False)


def stats(keys):
    rows = []
    for k in keys:
        im = cache[k]
        rgb = im.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        # 只统计视野内部（避免黑边干扰）
        m = g > 12
        rows.append(dict(
            R=rgb[..., 0][m].mean(), G=rgb[..., 1][m].mean(), B=rgb[..., 2][m].mean(),
            S=hsv[..., 1][m].mean(), V=hsv[..., 2][m].mean(),
            gray=g[m].mean(), gray_std=g[m].std(),
            sharp=float(cv2.Laplacian(g, cv2.CV_32F)[m].var()),
        ))
    import pandas as pd
    return pd.DataFrame(rows)


A = stats(gam_keys)
B = stats(ext_keys)
print("样本数: 本赛题 %d, 外部(IDRiD) %d" % (len(A), len(B)))
print()
print("%-10s %12s %12s %10s   %s" % ("统计量", "本赛题", "IDRiD", "相对差", "判断"))
for c in ["R", "G", "B", "S", "V", "gray", "gray_std", "sharp"]:
    a, b = A[c].median(), B[c].median()
    rel = (b - a) / (abs(a) + 1e-9)
    tag = "*** 差异大" if abs(rel) > 0.25 else ("* 中等" if abs(rel) > 0.10 else "接近")
    print("%-10s %12.3f %12.3f %9.1f%%   %s" % (c, a, b, 100 * rel, tag))
print()
print("按 10/90 分位看分布重叠度（灰度均值）:")
for nm, D in [("本赛题", A), ("IDRiD", B)]:
    q = np.percentile(D.gray, [10, 50, 90])
    print("   %-8s p10=%.1f p50=%.1f p90=%.1f" % (nm, q[0], q[1], q[2]))
print()
# 逐通道直方图距离
print("逐通道分布距离 (1D Wasserstein, 单位: 像素值 0-255):")
for i, c in enumerate("RGB"):
    av = (cache[gam_keys[0]][..., i]).ravel()
    a_all = np.concatenate([cache[k][..., i].ravel() for k in gam_keys[::4]])
    b_all = np.concatenate([cache[k][..., i].ravel() for k in ext_keys[::16]])
    from scipy.stats import wasserstein_distance
    d = wasserstein_distance(a_all, b_all)
    print("   %s: %.2f  (相对 %.1f%%)" % (c, d, 100 * d / 255))
print()
print("结论提示: 若 R/G/B 的相对差都在 10% 以内, 域差距主要不在色彩;")
print("          若某个通道超过 25%, 值得做逐通道均值/方差对齐(Reinhard 色彩迁移)。")