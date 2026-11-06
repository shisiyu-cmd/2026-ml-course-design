# -*- coding: utf-8 -*-
"""分层比较两个 5 折 OOF 实验：按中心凹离心度分组，看尾部是否改善。

教训来源: 总体均值会被掩盖，决定榜单的是尾部。故必须分层比较。

用法: python 29_oof_compare.py --a long800 --transbig
"""
import argparse
import numpy as np
import pandas as pd
import common as C

p = argparse.ArgumentParser()
p.add_argument("tags", nargs="+")
p.add_argument("--baseline", default=None, help="指定基线 tag（默认取第一个）")
a = p.parse_args()

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())

D = {}
for tag in a.tags:
    try:
        z = np.load("runs/%s_oof.npz" % tag, allow_pickle=True)
    except FileNotFoundError:
        print("跳过 %s（无 OOF）" % tag); continue
    soft, orig, keys = z["soft"], z["orig"], [str(k) for k in z["keys"]]
    eu = np.sqrt(((soft - orig) ** 2).sum(1))
    rows = []
    for i, k in enumerate(keys):
        m = META[k]; gx, gy = gt[k]
        rows.append(dict(key=k, err=eu[i],
                         ecc=np.hypot((gx - m["cx"]) / m["r"], (gy - m["cy"]) / m["r"])))
    D[tag] = pd.DataFrame(rows).set_index("key")

if not D:
    raise SystemExit("没有任何 OOF 可用")

tags = list(D)
print("=" * 96)
print("总体对比 (n=%d 张)" % len(D[tags[0]]))
print("=" * 96)
print("%-12s %8s %8s %8s %8s %10s %10s" % ("tag", "RMSE", "均值", "中位", "p90", "max", "(RMSE口径)"))
for t in tags:
    g = D[t]
    # 官方口径: X/Y 拆成 40 行后的 RMSE
    e = g.err.values
    rmse = float(np.sqrt(np.mean(e ** 2 / 2 * 2)))  # 每图两点, err 是欧氏距离
    print("%-12s %8.2f %8.2f %8.2f %8.2f %10.2f" % (t, np.sqrt((e ** 2).mean()), e.mean(),
                                                    np.median(e), np.percentile(e, 90), e.max()))

if len(tags) >= 2:
    base = a.baseline or tags[0]
    print()
    print("=" * 96)
    print("分层对比（离心度三分位, 以 %s 为基线）" % base)
    print("=" * 96)
    b = D[base]
    q1, q2 = b.ecc.quantile([1 / 3, 2 / 3])
    bins = pd.cut(b.ecc, [-1, q1, q2, 9], labels=["近中心", "中", "偏周边"])
    print("分位边界: %.3f / %.3f" % (q1, q2))
    print()
    hdr = "%-8s %5s" % ("分组", "n")
    for t in tags:
        hdr += " %18s" % t
    print(hdr)
    for lab in ["近中心", "中", "偏周边"]:
        idx = b.index[bins == lab]
        line = "%-8s %5d" % (lab, len(idx))
        for t in tags:
            g = D[t].loc[idx]
            line += " %18s" % ("med %5.1f/p90 %6.1f" % (np.median(g.err), np.percentile(g.err, 90)))
        print(line)
    print()
    print("相对基线的变化 (p90):")
    for t in tags[1:]:
        line = "  %-12s" % t
        for lab in ["近中心", "中", "偏周边", "全部"]:
            idx = b.index if lab == "全部" else b.index[bins == lab]
            pb = np.percentile(D[base].loc[idx].err, 90)
            pt = np.percentile(D[t].loc[idx].err, 90)
            line += "  %s %+6.1f (%+.0f%%)" % (lab, pt - pb, 100 * (pt - pb) / pb)
        print(line)
    print()
    print("最难图的变化 (基线中误差最大的 6 张):")
    worst = b.nlargest(6, "err").index
    for k in worst:
        line = "  %-12s ecc=%.3f" % (k, b.loc[k, "ecc"])
        for t in tags:
            line += "  %s=%6.1f" % (t, D[t].loc[k, "err"])
        print(line)