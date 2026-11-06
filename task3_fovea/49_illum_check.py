# -*- coding: utf-8 -*-
"""检验: 测试图中失败的那些，其亮度/对比度是否落在训练分布之外。

若落在分布外，说明增强范围或数据覆盖不足 —— 那才是可干预的点。
"""
import numpy as np
import pandas as pd
import cv2
import common as C

META = C.load_meta()
GT = C.load_gt_map(C.load_gt())
cache = C.build_cache(META, verbose=False)

E5 = pd.read_csv("runs/test_pred_ext11.csv").set_index("key")
z = np.load("runs/long800_oof.npz", allow_pickle=True)
oof = {str(k): p for k, p in zip(z["keys"], z["soft"])}


def stat(k):
    im = cache[k]
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = g > 12
    lo = cv2.GaussianBlur(g, (0, 0), 3)
    hi = g - lo
    return (g[m].mean(), g[m].std(), hi[m].std())


tr = sorted(GT)
rows_tr = [stat(k) for k in tr]
A = np.array(rows_tr)
print("训练集 (%d 张) 分布:" % len(tr))
for i, nm in enumerate(["平均亮度", "全局对比度(std)", "局部对比度(高通std)"]):
    q = np.percentile(A[:, i], [1, 5, 50, 95, 99])
    print("   %-18s p1=%7.2f  p5=%7.2f  中位=%7.2f  p95=%7.2f  p99=%7.2f"
          % (nm, *q))

te = sorted([k for k in META if k.startswith("test/")])
rows_te = [stat(k) for k in te]
B = np.array(rows_te)
print()
print("测试图逐张（按 11.35 版本的预测；★=落在训练 p1~p99 之外）:")
print("%-6s %9s %9s %9s   %s" % ("图", "亮度", "全局对比", "局部对比", "落点"))
outside = []
for i, k in enumerate(te):
    marks = []
    for j, nm in enumerate(["亮度", "全局对比", "局部对比"]):
        q1, q99 = np.percentile(A[:, j], [1, 99])
        if B[i, j] < q1 or B[i, j] > q99:
            marks.append(nm)
    tag = ("★" + "/".join(marks)) if marks else ""
    if marks:
        outside.append((k, marks))
    print("%-6s %9.2f %9.2f %9.2f   %s" % (k.split("/")[1], B[i, 0], B[i, 1], B[i, 2], tag))

print()
print("落在训练分布 (p1~p99) 之外的测试图: %d / %d" % (len(outside), len(te)))
for k, m in outside:
    print("   %-10s %s" % (k, "/".join(m)))
print()
print("判读: 若 0081/0086 出现在这张表里，说明它们的光照条件训练时从未见过，")
print("      对策是把 photo_aug 的范围放宽到覆盖它们（或对训练图做同类退化）。")