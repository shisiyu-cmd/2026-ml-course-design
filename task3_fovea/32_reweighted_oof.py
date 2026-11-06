# -*- coding: utf-8 -*-
"""按测试集的离心度分布对 OOF 做重要性重加权，得到更贴近榜单的估计。

动机: 测试集的中心凹位置分布显著偏向周边（中位 ecc 0.220 vs 训练 0.118）。
原始 OOF RMSE 是"训练分布下"的期望，会系统性低估榜单误差。
按 ecc 分箱做重要性加权后，估计值应更接近真实榜单分。

用法: python 32_reweighted_oof.py --tags long800 eccup ...
"""
import argparse
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--test_pred", default="runs/test_pred_alldata.csv")
p.add_argument("--bins", type=int, default=5)
a = p.parse_args()

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())

# 测试集的 ecc 分布（用已验证模型的预测近似真值）
td = pd.read_csv(a.test_pred).set_index("key")
te = np.array([np.hypot((td.x[k] - META[k]["cx"]) / META[k]["r"],
                        (td.y[k] - META[k]["cy"]) / META[k]["r"]) for k in td.index])

edges = np.quantile(np.concatenate([te, [0.0]]), np.linspace(0, 1, a.bins + 1))
edges[0], edges[-1] = -1e-9, 1e9
te_frac = np.histogram(te, bins=edges)[0] / len(te)

print("测试集 ecc 分箱占比 (%d 箱):" % a.bins)
for i in range(a.bins):
    print("   箱%d [%.3f, %.3f): %.1f%%" % (i, max(edges[i], 0), edges[i + 1], 100 * te_frac[i]))
print()

rows = []
for tag in a.tags:
    try:
        z = np.load("runs/%s_oof.npz" % tag, allow_pickle=True)
    except FileNotFoundError:
        print("跳过", tag); continue
    soft, orig, keys = z["soft"], z["orig"], [str(k) for k in z["keys"]]
    ecc = np.array([np.hypot((gt[k][0] - META[k]["cx"]) / META[k]["r"],
                             (gt[k][1] - META[k]["cy"]) / META[k]["r"]) for k in keys])
    se = ((soft - orig) ** 2).sum(1)          # 每图平方误差(欧氏距离的平方)
    plain = float(np.sqrt(se.mean()))
    b = np.clip(np.digitize(ecc, edges) - 1, 0, a.bins - 1)
    tr_frac = np.array([(b == i).mean() for i in range(a.bins)])
    per_img = te_frac[b] / np.maximum(tr_frac[b], 1e-9)   # 每张图的重要性权重
    w = np.where(tr_frac[b] > 0, per_img, 0.0)
    w = w / w.sum() * len(w)
    re_w = float(np.sqrt((w * se).sum() / w.sum()))
    rows.append(dict(tag=tag, n=len(keys), plain_oof=plain, reweighted=re_w,
                     ratio=re_w / plain))
    print("%-12s n=%d  原始OOF=%.2f  按测试分布重加权=%.2f  (×%.2f)"
          % (tag, len(keys), plain, re_w, re_w / plain))

print()
print("解读: '重加权'列才是与榜单可比的量。若某方案的原始 OOF 更好但重加权更差，")
print("      说明它的改善集中在近中心区域，对本次测试集没有帮助。")
df = pd.DataFrame(rows)
if len(df) > 1:
    print()
    print(df.to_string(index=False, float_format=lambda v: "%.2f" % v))