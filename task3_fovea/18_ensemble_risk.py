# -*- coding: utf-8 -*-
"""风险感知的集成评估。

榜单只评测 20 张图，RMSE 被最差的那张主导，所以只看 OOF 均值会误导。
本脚本对每个集成方案同时给出：
  1) 全 80 张的 OOF RMSE（偏差）
  2) 随机抽 20 张时的 RMSE 分布（方差/风险）—— 这才是榜单的真实预期
  3) 各方案在「最难的那几张」上的表现
"""
import os, argparse, itertools
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", required=True)
p.add_argument("--variant", default="plain")
p.add_argument("--decode", default="soft", choices=["soft", "hard", "blend"])
p.add_argument("--n_sim", type=int, default=20000)
p.add_argument("--max_combo", type=int, default=4)
a = p.parse_args()

D, K = {}, None
for t in a.tags:
    f = os.path.join("runs", "tta_%s.npz" % t)
    if not os.path.exists(f):
        print("缺少 %s，跳过" % f); continue
    d = np.load(f, allow_pickle=True)
    ks = list(d["keys"])
    if K is None:
        K = ks
    assert K == ks, "顺序不一致"
    D[t] = d
if not D:
    raise SystemExit("没有可用的 OOF 文件")

tags = list(D)
GT = D[tags[0]]["gt"]


def decode_of(d, tag):
    if a.decode == "soft":
        return d["soft_" + a.variant]
    if a.decode == "hard":
        return d["hard_" + a.variant]
    return 0.5 * (d["soft_" + a.variant] + d["hard_" + a.variant])


def per_img_sq(P):
    return ((P - GT) ** 2)


rng = np.random.default_rng(0)
IDX = np.stack([rng.choice(len(K), 20, replace=False) for _ in range(a.n_sim)])


def risk(P):
    """返回 (OOF RMSE, 20 张抽样的 RMSE 分布分位数)。"""
    sq = per_img_sq(P).mean(1)                 # 每张图的均方误差(含 X,Y)
    oof = float(np.sqrt(sq.mean()))
    sims = np.sqrt(sq[IDX].mean(1))
    return oof, np.percentile(sims, [25, 50, 75, 90, 95])


def show(name, P):
    oof, q = risk(P)
    print("%-34s %7.2f   %6.1f %6.1f %6.1f %6.1f %6.1f"
          % (name, oof, q[0], q[1], q[2], q[3], q[4]))
    return oof, q


print("=" * 104)
print("单轮结果")
print("=" * 104)
print("%-34s %7s   %s" % ("方案", "OOF", "随机20张 RMSE:  p25    p50    p75    p90    p95"))
singles = {}
for t in tags:
    singles[t] = show(t, decode_of(D[t], t))

# 预测之间的分歧（用于判断互补性）
if len(tags) > 1:
    print()
    print("模型间误差相关性:")
    E = np.stack([np.sqrt(per_img_sq(decode_of(D[t], t)).sum(1)) for t in tags])
    for i, j in itertools.combinations(range(len(tags)), 2):
        print("   %-12s vs %-12s  pearson=%.3f" % (tags[i], tags[j],
                                                   np.corrcoef(E[i], E[j])[0, 1]))

print()
print("=" * 104)
print("集成")
print("=" * 104)
print("%-34s %7s   %s" % ("方案", "OOF", "随机20张 RMSE:  p25    p50    p75    p90    p95"))
results = {}
P_all = np.stack([decode_of(D[t], t) for t in tags])
results["均值(全部)"] = P_all.mean(0)
if len(tags) >= 3:
    results["中位数(全部)"] = np.median(P_all, 0)
PK = np.stack([D[t]["peak_mean"] for t in tags])
w = np.clip(PK, 1e-3, None)
results["峰值加权均值"] = (P_all * w[:, :, None]).sum(0) / w.sum(0)[:, None]

# 穷举子集
if len(tags) >= 2:
    for r in range(2, min(a.max_combo, len(tags)) + 1):
        for combo in itertools.combinations(range(len(tags)), r):
            nm = "+".join(tags[i] for i in combo)
            results["均值[%s]" % nm] = P_all[list(combo)].mean(0)

best = None
for nm, P in results.items():
    oof, q = show(nm, P)
    # 以「榜单预期」为主排序: p50 与 p90 的折中
    score = 0.5 * q[1] + 0.5 * q[3]
    if best is None or score < best[0]:
        best = (score, nm, oof, q)

print()
print("=" * 104)
print("按「榜单预期」(0.5*p50 + 0.5*p90) 最优: %s" % best[1])
print("   OOF=%.2f  p50=%.1f  p90=%.1f" % (best[2], best[3][1], best[3][3]))
print("=" * 104)
print()
print("最难图上的表现 (long800 误差最大的 6 张):")
worst = np.argsort(-per_img_sq(decode_of(D[tags[-1]], tags[-1])).sum(1))[:6]
rows = []
for i in worst:
    row = {"key": K[i]}
    for t in tags:
        row[t] = float(np.sqrt(per_img_sq(decode_of(D[t], t))[i].sum()))
    row["均值集成"] = float(np.sqrt(per_img_sq(results["均值(全部)"])[i].sum()))
    if "中位数(全部)" in results:
        row["中位数"] = float(np.sqrt(per_img_sq(results["中位数(全部)"])[i].sum()))
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: "%.0f" % v))