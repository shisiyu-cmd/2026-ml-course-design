# -*- coding: utf-8 -*-
"""验证预训练 Messidor 模型在本地数据上的可用性。

先在训练图上跑（真值已知），确认 PyTorch 复刻正确；
若正确，再看它在测试集上的表现（尤其 0081 / 0086 两个锚点）。
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import cv2
sys.path.insert(0, "ext/odfov")
import odfov_torch as O
import common as C

p = argparse.ArgumentParser()
p.add_argument("--keys", nargs="*", default=None)
p.add_argument("--n", type=int, default=10)
p.add_argument("--mode", default="global", choices=["global", "perchan"])
p.add_argument("--crop", action="store_true", help="先做 FOV 裁剪再送入")
p.add_argument("--dump", default=None)
a = p.parse_args()

dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
model = O.load_pretrained(dev=dev) if False else O.load_pretrained()
model = model.to(dev)

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
if a.keys:
    keys = a.keys
else:
    keys = sorted([k for k in gt])[:a.n]

imgs, metas = [], []
for k in keys:
    img = C.imread_u(os.path.join(C.ROOT, k + ".jpg"))
    if img is None:
        raise SystemExit("读不到 %s" % k)
    if a.crop:
        img = C.fov_crop(img, META[k])
    imgs.append(img)
    metas.append(META[k])

dmaps = O.predict_maps(model, imgs, mode=a.mode, dev=dev)
print("距离图: shape=%s 范围=[%.3f, %.3f]" % (dmaps.shape, dmaps.min(), dmaps.max()))

rows = []
for i, k in enumerate(keys):
    H, W = imgs[i].shape[:2]
    if a.crop:
        m = metas[i]
        # fov_crop 的输出边长
        H, W = imgs[i].shape[:2]
    pk = O.find_two_peaks(dmaps[i])
    gx, gy = gt.get(k, (np.nan, np.nan))
    rec = {"key": k, "npeaks": len(pk)}
    for j, (py, px) in enumerate(pk[:2]):
        # 512 -> 原图
        X = px * W / 512.0
        Y = py * H / 512.0
        if a.crop:
            # 映射回原图坐标
            m = metas[i]
            x0, y0 = m["x0"], m["y0"]
            si = m["si"]
            X = x0 + X * si / W
            Y = y0 + Y * si / H
        rec["p%d" % j] = (X, Y)
        rec["e%d" % j] = float(np.hypot(X - gx, Y - gy)) if np.isfinite(gx) else np.nan
    rows.append(rec)

print()
print("%-12s %-22s %-22s %8s %8s" % ("key", "peak0(原图)", "peak1(原图)", "err0", "err1"))
errs = []
for r in rows:
    s0 = "(%.0f,%.0f)" % r["p0"] if "p0" in r else "-"
    s1 = "(%.0f,%.0f)" % r["p1"] if "p1" in r else "-"
    e0 = r.get("e0", np.nan); e1 = r.get("e1", np.nan)
    print("%-12s %-22s %-22s %8.1f %8.1f" % (r["key"], s0, s1, e0, e1))
    if np.isfinite(e0): errs.append(e0)
    if np.isfinite(e1): errs.append(e1)
errs = np.array(errs)
if len(errs):
    print()
    print("把两个峰都当作候选时的误差分布: med=%.1f p90=%.1f (n=%d)" % (np.median(errs), np.percentile(errs, 90), len(errs)))
    # 取每图较近的那个峰（模拟"选对哪个是黄斑"的乐观上界）
    best = [min([r.get("e0", 1e9), r.get("e1", 1e9)]) for r in rows]
    best = np.array(best)
    print("每图取较近峰(乐观上界): mean=%.1f med=%.1f max=%.1f" % (best.mean(), np.median(best), best.max()))