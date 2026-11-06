# -*- coding: utf-8 -*-
"""鲁棒定位眼底视野圆(FOV) -> fov_meta.json

方法：以「圆内视网膜亮度均值 − 圆外环带亮度均值」为目标函数。
用 disk 卷积核在整幅图上一次性算出所有圆心位置的得分，因此可做全局搜索，
且只用边界处的明暗跃变信息，对暗视野/病灶/眩光都稳健。

输出统一裁剪纸: 以圆心为中心, 边长 = 2r*MARGIN 的正方形。
"""
import os, glob, json
import numpy as np
import pandas as pd
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "2026task3", "detection")
MARGIN = 1.06
BRIGHT_THR = 8


def _disk(r):
    d = 2 * int(np.ceil(r)) + 1
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32) - (d - 1) / 2.0
    return ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)


def _score_maps(gray_s, thr, radii):
    """返回 {r: (Sb, Sv)}，Sb=圆内亮像素数, Sv=圆内有效像素数(图像范围内)。"""
    bright = (gray_s > thr).astype(np.float32)
    ones = np.ones_like(bright)
    out = {}
    for r in radii:
        k = _disk(r)
        Sb = cv2.filter2D(bright, -1, k, borderType=cv2.BORDER_CONSTANT)
        Sv = cv2.filter2D(ones, -1, k, borderType=cv2.BORDER_CONSTANT)
        out[float(r)] = (Sb, Sv)
    return out


def _best_in(maps, radii, cx_lo, cx_hi, cy_lo, cy_hi, r_ratio=1.045):
    best = None
    for r in radii:
        Sb, Sv = maps[r]
        r_out = r * r_ratio
        rk = min(radii, key=lambda x: abs(x - r_out))
        if abs(rk - r_out) > 1.5:
            continue
        Sb2, Sv2 = maps[rk]
        sub = lambda A: A[cy_lo:cy_hi, cx_lo:cx_hi]
        s_in, v_in = sub(Sb), sub(Sv)
        s_out, v_out = sub(Sb2), sub(Sv2)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_mean = np.where(v_in > 0.5 * v_in.max(), s_in / np.maximum(v_in, 1), 0.0)
            ring_s = np.maximum(s_out - s_in, 0)
            ring_v = np.maximum(v_out - v_in, 0)
            ring_ok = ring_v > 0.5 * np.percentile(ring_v, 95)
            r_mean = np.where(ring_ok, ring_s / np.maximum(ring_v, 1), 1.0)
            J = d_mean - r_mean
        J = np.where(v_in > 0.55 * (np.pi * r * r), J, -9.0)
        iy, ix = np.unravel_index(np.argmax(J), J.shape)
        val = J[iy, ix]
        if best is None or val > best[0]:
            best = (float(val), float(ix + cx_lo), float(iy + cy_lo), float(r))
    return best


def detect_fov(gray, size_hint):
    H, W = gray.shape
    # 粗尺度：原图尺寸决定初始半径猜测
    if size_hint == "small":
        r0 = 0.474 * W          # 1956x1934 -> ~927
    else:
        r0 = 0.324 * W          # 2992x2000 -> ~969
    cx0, cy0 = W / 2.0, H / 2.0

    # ---- Pass 1: 1/8 尺度全局搜索 ----
    s1 = 8.0
    g1 = cv2.resize(gray, (int(W / s1), int(H / s1)), interpolation=cv2.INTER_AREA)
    radii1 = [r / s1 for r in np.arange(r0 * 0.90, r0 * 1.10 + 1, 10.0)]
    maps1 = _score_maps(g1, BRIGHT_THR, radii1)
    b = _best_in(maps1, radii1, 0, g1.shape[1], 0, g1.shape[0])
    if b is None:
        return None
    _, cx, cy, r = b
    cx *= s1; cy *= s1; r *= s1
    span = 40.0  # 原图像素

    # ---- Pass 2/3: 逐步精细化 ----
    for s in (4.0, 2.0):
        g = cv2.resize(gray, (int(W / s), int(H / s)), interpolation=cv2.INTER_AREA)
        radii = [rr / s for rr in np.arange(r - span, r + span + 1, 2.0)]
        x0 = max(0, int((cx - span) / s)); x1 = min(g.shape[1], int((cx + span) / s) + 1)
        y0 = max(0, int((cy - span) / s)); y1 = min(g.shape[0], int((cy + span) / s) + 1)
        maps = _score_maps(g, BRIGHT_THR, radii)
        b = _best_in(maps, radii, x0, x1, y0, y1)
        if b is None:
            break
        _, bx, by, br = b
        cx, cy, r = bx * s, by * s, br * s
        span = 10.0
    return float(cx), float(cy), float(r)


def run(split):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, split, "*.jpg"))):
        key = int(os.path.splitext(os.path.basename(f))[0])
        with Image.open(f) as im:
            W, H = im.size
            arr = np.asarray(im.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hint = "small" if W < 2500 else "big"
        res = detect_fov(gray, hint)
        cx, cy, r = res if res else (np.nan,) * 3
        rows.append(dict(key=key, split=split, W=W, H=H, cx=cx, cy=cy, r=r, d=2 * r))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tr, te = run("train"), run("test")
    allr = pd.concat([tr, te], ignore_index=True)
    print("失败数:", allr.r.isna().sum(), "/", len(allr))
    allr["size"] = allr.W.astype(str) + "x" + allr.H.astype(str)
    print()
    print("FOV 圆拟合结果（按原图尺寸分组）:")
    print(allr.groupby("size")[["cx", "cy", "r"]].agg(["size", "mean", "std", "min", "max"]).to_string())
    print()
    print("同一尺寸内的离散度 (std):")
    for sz, sub in allr.groupby("size"):
        print("  %-12s n=%2d  cx std=%.2f  cy std=%.2f  r std=%.2f   直径 %.1f~%.1f"
              % (sz, len(sub), sub.cx.std(), sub.cy.std(), sub.r.std(), sub.d.min(), sub.d.max()))

    gt = pd.read_csv(os.path.join(ROOT, "fovea_localization_train_GT.csv"))
    gt["key"] = gt["data"].astype(int)
    m = gt.merge(tr, on="key")
    m["fx"] = (m.Fovea_X - (m.cx - m.r)) / (2 * m.r)
    m["fy"] = (m.Fovea_Y - (m.cy - m.r)) / (2 * m.r)
    m["size"] = m.W.astype(str) + "x" + m.H.astype(str)
    print()
    print("=" * 70)
    print("相对 FOV 框的归一化中心凹坐标 (fx, fy), 0.5=圆心")
    print("=" * 70)
    print(m[["fx", "fy"]].describe().to_string())
    print()
    print(m.groupby("size")[["fx", "fy"]].agg(["size", "mean", "std", "min", "max"]).to_string())

    meta = {}
    for _, r_ in allr.iterrows():
        meta["%s/%04d" % (r_.split, r_.key)] = dict(W=int(r_.W), H=int(r_.H),
                                                    cx=float(r_.cx), cy=float(r_.cy), r=float(r_.r))
    with open(os.path.join(HERE, "fov_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("\n已保存 fov_meta.json (%d 条)" % len(meta))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = ["train/0001", "train/0003", "train/0004", "train/0005", "train/0007",
           "test/0081", "test/0082", "test/0090", "test/0095", "test/0100"]
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, s in zip(axes.ravel(), sel):
        sp, kk = s.split("/"); k = int(kk)
        with Image.open(os.path.join(ROOT, sp, "%04d.jpg" % k)) as im:
            ax.imshow(np.asarray(im.convert("RGB")))
        md = meta[s]
        ax.add_patch(plt.Circle((md["cx"], md["cy"]), md["r"], fill=False, ec="cyan", lw=1.6))
        side = 2 * md["r"] * MARGIN
        ax.add_patch(plt.Rectangle((md["cx"] - side / 2, md["cy"] - side / 2), side, side,
                                   fill=False, ec="lime", lw=1.6))
        t = m[m.key == k]
        if len(t) and sp == "train":
            ax.plot(t.iloc[0].Fovea_X, t.iloc[0].Fovea_Y, "r+", ms=15, mew=2.2)
        ax.set_title("%s  W=%d r=%.1f" % (s, md["W"], md["r"]), fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "viz", "fov_circle.png"), dpi=75)
    plt.close()
    print("已保存 viz/fov_circle.png")