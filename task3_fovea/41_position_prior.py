# -*- coding: utf-8 -*-
"""检验"外部数据把位置先验带偏"这一假设。

IDRiD 的中心凹离心度中位 0.297，而本赛题只有 0.118。若模型被它带偏，
则 allExt 的测试集预测应系统性比 v4 更偏周边。
"""
import numpy as np
import pandas as pd
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
_, ext_gt = C.load_ext()


def ecc_of(keys, pts):
    return np.array([np.hypot((pts[k][0] - META[k]["cx"]) / META[k]["r"],
                              (pts[k][1] - META[k]["cy"]) / META[k]["r"])
                     for k in keys])


gam = ecc_of(sorted(gt), {k: gt[k] for k in sorted(gt)})
ext = ecc_of(sorted(ext_gt), {k: ext_gt[k] for k in sorted(ext_gt)})
print("真实标注的离心度分布:")
print("   本赛题 (n=%3d): 中位 %.3f  p25 %.3f  p75 %.3f  >0.25 占 %.0f%%"
      % (len(gam), np.median(gam), np.percentile(gam, 25), np.percentile(gam, 75), 100 * (gam > .25).mean()))
print("   IDRiD  (n=%3d): 中位 %.3f  p25 %.3f  p75 %.3f  >0.25 占 %.0f%%"
      % (len(ext), np.median(ext), np.percentile(ext, 25), np.percentile(ext, 75), 100 * (ext > .25).mean()))
print()
print("测试集【预测】的离心度分布（若被带偏，allExt 应显著更大）:")
rows = {}
for t in ["allA", "allB", "allJ", "allM", "allExt"]:
    try:
        d = pd.read_csv("runs/test_pred_%s.csv" % t).set_index("key")
    except FileNotFoundError:
        continue
    ks = sorted([k for k in META if k.startswith("test/")])
    e = ecc_of(ks, {k: (d.x[k], d.y[k]) for k in ks})
    rows[t] = e
    print("   %-8s 中位 %.3f  均值 %.3f  >0.25 占 %.0f%%" % (t, np.median(e), e.mean(), 100 * (e > .25).mean()))
print()
if "allExt" in rows and "allA" in rows:
    d = rows["allExt"] - rows["allB"] if "allB" in rows else rows["allExt"] - rows["allA"]
    print("allExt 减 allB 的逐图离心度差: 均值 %+.4f  中位 %+.4f  (正=更偏周边)"
          % (d.mean(), np.median(d)))
    print("   最大 3 个: %s" % np.round(np.sort(d)[-3:], 3))
    print("   最小 3 个: %s" % np.round(np.sort(d)[:3], 3))
    print()
    print("判读: 若均值在 ±0.02 以内，说明位置先验基本没被带偏；")
    print("      若 allExt 系统性更偏周边（均值 > +0.05），则需要对外部数据做位置重加权。")