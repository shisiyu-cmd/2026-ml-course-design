# -*- coding: utf-8 -*-
"""逐图对比 extf0（+外部数据）与 long800（基线）在同一批留出图上的误差。

目的: 验证"外部数据的收益来自尾部收缩"这一解释 —— 若收益集中在原本最差的图上，
则说明数据量确实解决了"难图崩溃"，而不是让容易的图更准。
"""
import numpy as np
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())

za = np.load("runs/long800_oof.npz", allow_pickle=True)
zb = np.load("runs/extf0_oof.npz", allow_pickle=True)

ka = [str(k) for k in za["keys"]]
kb = [str(k) for k in zb["keys"]]
ea = np.sqrt(((za["soft"] - za["orig"]) ** 2).sum(1))
eb = np.sqrt(((zb["soft"] - zb["orig"]) ** 2).sum(1))
da = dict(zip(ka, ea))
db = dict(zip(kb, eb))

common = [k for k in kb if k in da]
print("可对比的图: %d 张（extf0 的留出集）" % len(common))
print()
A = np.array([da[k] for k in common])
B = np.array([db[k] for k in common])
print("整体:  基线 RMSE=%.2f 中位=%.2f p90=%.2f max=%.1f"
      % (np.sqrt((A ** 2).mean()), np.median(A), np.percentile(A, 90), A.max()))
print("       extf0 RMSE=%.2f 中位=%.2f p90=%.2f max=%.1f"
      % (np.sqrt((B ** 2).mean()), np.median(B), np.percentile(B, 90), B.max()))
print()
order = np.argsort(-A)
print("%-12s %10s %10s %9s" % ("key", "基线误差", "extf0", "变化"))
for j in order:
    k = common[j]
    print("%-12s %10.1f %10.1f %+9.1f" % (k, A[j], B[j], B[j] - A[j]))
print()
# 分档看收益来源
for lo, hi, nm in [(0, 10, "原本很好 (<10px)"), (10, 30, "中等 (10-30px)"),
                   (30, 1e9, "原本很差 (>30px)")]:
    m = (A >= lo) & (A < hi)
    if m.sum() == 0:
        continue
    print("%-20s n=%2d  基线均值=%7.2f  extf0均值=%7.2f  变化=%+7.2f"
          % (nm, m.sum(), A[m].mean(), B[m].mean(), (B[m] - A[m]).mean()))
print()
print("解读: 若收益主要集中在 '原本很差' 一档，则证实尾部收缩；")
print("      若各档均匀改善，则说明是整体精度提升。")