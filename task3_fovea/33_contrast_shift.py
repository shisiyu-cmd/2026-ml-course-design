# -*- coding: utf-8 -*-
"""比较训练集与测试集在中心凹周围的局部对比度分布。

若测试集系统性更低对比度，则"对比度增强/增强数据"就是对症的修复。
"""
import numpy as np
import pandas as pd
import cv2
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
td = pd.read_csv("runs/test_pred_alldata.csv").set_index("key")


def local_stats(img, cx, cy, frac=0.04):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    r = int(frac * W)
    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r)
    y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r)
    win = g[y0:y1, x0:x1]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    gw = grad[y0:y1, x0:x1]
    # 相对对比度: 局部标准差 / 全图标准差
    return dict(loc_mean=float(win.mean()), loc_std=float(win.std()),
                grad=float(gw.mean()), gstd=float(g.std()),
                rel_std=float(win.std() / (g.std() + 1e-6)),
                rel_grad=float(gw.mean() / (grad.mean() + 1e-6)),
                img_std=float(g.std()), img_mean=float(g.mean()))


rows = []
for k in sorted(META):
    img = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
    if k.startswith("train/"):
        cx, cy = gt[k]
        split = "train"
    else:
        cx, cy = td.x[k], td.y[k]
        split = "test"
    m = META[k]
    st = local_stats(img, cx, cy)
    st.update(key=k, split=split,
              ecc=float(np.hypot((cx - m["cx"]) / m["r"], (cy - m["cy"]) / m["r"])))
    rows.append(st)
df = pd.DataFrame(rows)

print("%-8s %5s %10s %10s %10s %10s %10s" % ("集合", "n", "局部均值", "局部std", "相对std", "梯度", "整图std"))
for s, g in df.groupby("split"):
    print("%-8s %5d %10.1f %10.1f %10.3f %10.1f %10.1f"
          % (s, len(g), g.loc_mean.median(), g.loc_std.median(),
             g.rel_std.median(), g.grad.median(), g.img_std.median()))
print()
print("分位数对比 (局部标准差):")
for s, g in df.groupby("split"):
    q = np.percentile(g.loc_std, [10, 25, 50, 75, 90])
    print("   %-6s p10=%.1f p25=%.1f p50=%.1f p75=%.1f p90=%.1f"
          % (s, q[0], q[1], q[2], q[3], q[4]))
print()
print("分位数对比 (相对对比度 rel_std):")
for s, g in df.groupby("split"):
    q = np.percentile(g.rel_std, [10, 25, 50, 75, 90])
    print("   %-6s p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f"
          % (s, q[0], q[1], q[2], q[3], q[4]))
print()
te = df[df.split == "test"]
print("测试集中局部对比度最低的 6 张:")
print(te.nsmallest(6, "loc_std")[["key", "loc_std", "rel_std", "ecc"]]
      .to_string(index=False, float_format=lambda v: "%.3f" % v))