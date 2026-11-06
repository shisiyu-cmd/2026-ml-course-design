# -*- coding: utf-8 -*-
"""多轮实验的 OOF 集成评估。

每个 runs/tta_<tag>.npz 都由「该图所属折的模型」预测得到（真正的 out-of-fold），
因此把它们平均得到的 RMSE 是对集成效果的无偏估计。

用法: python 09_ensemble.py --tags base200 seed1
"""
import os, argparse, itertools
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--variant", default="plain")
p.add_argument("--decode", default="soft", choices=["soft", "hard"])
a = p.parse_args()

data = {}
for t in a.tags:
    f = os.path.join("runs", "tta_%s.npz" % t)
    if not os.path.exists(f):
        print("缺少 %s，跳过" % f); continue
    d = np.load(f, allow_pickle=True)
    data[t] = dict(keys=list(d["keys"]), gt=d["gt"],
                   pred=d["%s_%s" % (a.decode, a.variant)], peak=d["peak_mean"])
if not data:
    raise SystemExit("没有可用的 OOF 文件")

tags = list(data)
base_keys = data[tags[0]]["keys"]
for t in tags[1:]:
    assert data[t]["keys"] == base_keys, "各轮实验的图片顺序不一致: %s" % t

gt = data[tags[0]]["gt"]
rmse = lambda P: float(np.sqrt(((P - gt) ** 2).mean()))
eucl = lambda P: np.sqrt(((P - gt) ** 2).sum(1))

print("=" * 74)
print("单轮结果 (variant=%s, decode=%s)" % (a.variant, a.decode))
print("=" * 74)
print("%-14s %10s %10s %10s %10s" % ("tag", "RMSE", "中位欧氏", "均值欧氏", "最大"))
for t in tags:
    e = eucl(data[t]["pred"])
    print("%-14s %10.3f %10.2f %10.2f %10.1f" % (t, rmse(data[t]["pred"]),
                                                 np.median(e), e.mean(), e.max()))

# 相关性: 各轮在同一张图上的误差是否同向
if len(tags) >= 2:
    print()
    print("各轮误差的相关性 (Pearson/Spearman):")
    E = np.stack([eucl(data[t]["pred"]) for t in tags])
    for i, j in itertools.combinations(range(len(tags)), 2):
        r = np.corrcoef(E[i], E[j])[0, 1]
        rs = pd.Series(E[i]).corr(pd.Series(E[j]), method="spearman")
        print("   %-12s vs %-12s  pearson=%.3f  spearman=%.3f"
              % (tags[i], tags[j], r, rs))

# 集成
print()
print("=" * 74)
print("集成结果")
print("=" * 74)
P = np.stack([data[t]["pred"] for t in tags])
print("%-24s %10s %10s %10s" % ("方式", "RMSE", "中位欧氏", "最大"))
ens = {}
ens["均值(全部)"] = P.mean(0)
ens["中位数(全部)"] = np.median(P, 0)
if len(tags) >= 3:
    Ps = np.sort(P, 0)
    ens["截尾均值(去极值)"] = Ps[1:-1].mean(0)
# 按峰值加权
PK = np.stack([data[t]["peak"] for t in tags])
w = np.clip(PK, 1e-3, None)
ens["峰值加权均值"] = (P * w[:, :, None]).sum(0) / w.sum(0)[:, None]

for name, Q in ens.items():
    e = eucl(Q)
    print("%-24s %10.3f %10.2f %10.1f" % (name, rmse(Q), np.median(e), e.max()))

best = min(ens, key=lambda k: rmse(ens[k]))
print()
print("最佳集成方式: %s  RMSE=%.3f  (相对单轮最好 %.3f 改善 %.1f%%)"
      % (best, rmse(ens[best]),
         min(rmse(data[t]["pred"]) for t in tags),
         100 * (1 - rmse(ens[best]) / min(rmse(data[t]["pred"]) for t in tags))))

# 最差样本上集成是否有效
print()
print("最差 12 张在各方案下的误差:")
worse = np.argsort(-eucl(ens["均值(全部)"]))[:12]
rows = []
for i in worse:
    row = dict(key=base_keys[i])
    for t in tags:
        row[t] = eucl(data[t]["pred"])[i]
    row["均值集成"] = eucl(ens["均值(全部)"])[i]
    row["中位集成"] = eucl(ens["中位数(全部)"])[i]
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.1f" % v))