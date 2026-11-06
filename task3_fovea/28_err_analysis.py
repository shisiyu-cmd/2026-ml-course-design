# -*- coding: utf-8 -*-
"""分析 OOF 误差与图像属性的关系，寻找系统性误差来源。"""
import numpy as np
import pandas as pd
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
rows = []
for tag in ["long800", "clahelong", "base200"]:
    try:
        z = np.load("runs/%s_oof.npz" % tag, allow_pickle=True)
    except FileNotFoundError:
        continue
    soft, orig, keys = z["soft"], z["orig"], [str(k) for k in z["keys"]]
    eu = np.sqrt(((soft - orig) ** 2).sum(1))
    for i, k in enumerate(keys):
        m = META[k]
        gx, gy = gt[k]
        # 中心凹相对视野圆心的归一化偏移
        ex = (gx - m["cx"]) / m["r"]
        ey = (gy - m["cy"]) / m["r"]
        rows.append(dict(tag=tag, key=k, err=eu[i],
                         W=m["W"], H=m["H"], r=m["r"], small=int(m["W"] < 2500),
                         ecc=np.hypot(ex, ey), ex=ex, ey=ey,
                         npx=m["W"] * m["H"]))
df = pd.DataFrame(rows)
print("总样本:", len(df), " 实验:", df.tag.unique().tolist())

for tag, g in df.groupby("tag"):
    print()
    print("=" * 70)
    print("%s  OOF: n=%d  mean=%.2f med=%.2f p90=%.2f max=%.2f"
          % (tag, len(g), g.err.mean(), g.err.median(),
             g.err.quantile(.9), g.err.max()))
    print("  按图像尺寸分组:")
    for s, gg in g.groupby("small"):
        print("    %s (n=%2d): mean=%.2f med=%.2f p90=%.2f"
              % ("1956宽" if s else "2992宽", len(gg), gg.err.mean(),
                 gg.err.median(), gg.err.quantile(.9)))
    print("  误差与各属性的 Spearman 相关:")
    for col in ["ecc", "r", "W", "npx"]:
        r = g[["err", col]].corr(method="spearman").iloc[0, 1]
        print("    %-6s rho=%+.3f" % (col, r))
    # 按离心度分组
    g2 = g.copy()
    g2["eccbin"] = pd.qcut(g2.ecc, 3, labels=["近中心", "中", "偏周边"])
    print("  按中心凹离心度三分位:")
    for b, gg in g2.groupby("eccbin", observed=True):
        print("    %-6s (n=%2d): ecc=%.2f~%.2f  err mean=%.2f med=%.2f p90=%.2f"
              % (b, len(gg), gg.ecc.min(), gg.ecc.max(),
                 gg.err.mean(), gg.err.median(), gg.err.quantile(.9)))
    # 最差 6 张
    print("  最差 6 张:")
    for _, rr in g.nlargest(6, "err").iterrows():
        print("    %-12s err=%6.1f  ecc=%.2f  W=%d" % (rr.key, rr.err, rr.ecc, rr.W))

print()
print("=" * 70)
print("跨实验汇总: 最难(平均误差最大)的训练图")
piv = df.pivot_table(index="key", columns="tag", values="err")
if piv.shape[1] > 1:
    piv["mean"] = piv.mean(1)
    print(piv.nlargest(10, "mean").to_string(float_format=lambda v: "%.1f" % v))