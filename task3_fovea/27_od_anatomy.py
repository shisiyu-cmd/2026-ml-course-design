# -*- coding: utf-8 -*-
"""用预训练 Messidor 模型的视盘峰做解剖学校验。

思路: 中心凹距视盘的归一化距离 dp = dist/r 在训练集上近似正态(0.297±0.071)。
先在训练集上验证"预训练模型的视盘峰"是否可靠（用已知的中心凹真值做基准），
再据此审查测试集预测是否解剖学上可疑。
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import cv2
import torch
sys.path.insert(0, "ext/odfov")
import odfov_torch as O
import common as C

p = argparse.ArgumentParser()
p.add_argument("--split", default="train", choices=["train", "test"])
p.add_argument("--n", type=int, default=0)
p.add_argument("--dump", default=None)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = O.load_pretrained().to(dev)
META = C.load_meta()
gt = C.load_gt_map(C.load_gt())

keys = sorted([k for k in META if k.startswith(a.split + "/")])
if a.n:
    keys = keys[:a.n]

rows = []
for k in keys:
    img = C.imread_u(os.path.join(C.ROOT, k + ".jpg"))
    m = META[k]
    dm = O.predict_maps(model, [img], mode="global", dev=dev)[0]
    pk = O.find_two_peaks(dm)
    H, W = img.shape[:2]
    # 512 网格 -> 原图
    cands = [(px * W / 512.0, py * H / 512.0) for (py, px) in pk[:2]]
    ref = gt.get(k)
    if ref is not None:
        # 训练图: 以已知中心凹为基准, 选出距离在合理范围内的峰当视盘
        d = [np.hypot(c[0] - ref[0], c[1] - ref[1]) / m["r"] for c in cands]
        # 视盘应在 0.15~0.6 r 之间, 取最接近 0.3r 的
        score = [abs(x - 0.3) if 0.12 < x < 0.65 else 9 for x in d]
        j = int(np.argmin(score))
        od = cands[j]
        dp = d[j]
        # 方向: 视盘 -> 中心凹 的向量与"鼻侧"无关, 只统计夹角是否约 180 度(即中心凹在视盘的反向)
        vx, vy = ref[0] - od[0], ref[1] - od[1]
        ang = np.degrees(np.arctan2(vy, vx))
        rows.append(dict(key=k, odx=od[0], ody=od[1], dp=dp, ang=ang,
                         fd=np.hypot(ref[0] - od[0], ref[1] - od[1]) / m["r"],
                         ok=bool(score[j] < 9)))
    else:
        od = cands[0]
        rows.append(dict(key=k, odx=od[0], ody=od[1], dp=np.nan, ang=np.nan,
                         fd=np.nan, ok=True))
    del dm

df = pd.DataFrame(rows)
if a.split == "train":
    ok = df[df.ok]
    print("训练集视盘峰校验 (n=%d, 有效 %d)" % (len(df), len(ok)))
    print("  中心凹-视盘归一化距离 dp: mean=%.3f med=%.3f std=%.3f  (5%%~95%%: %.3f~%.3f)"
          % (ok.dp.mean(), ok.dp.median(), ok.dp.std(),
             ok.dp.quantile(.05), ok.dp.quantile(.95)))
    print("  参考: 用 XML 框中心算的同类统计 mean≈0.297 std≈0.071")
    bad = df[~df.ok]
    if len(bad):
        print("  无法判定视盘的图 (%d): %s" % (len(bad), list(bad.key)[:8]))
    print()
    # 方向: 中心凹相对视盘的角度分布（用圆形统计）
    ang = np.radians(ok.ang.values)
    mx, my = np.cos(ang).mean(), np.sin(ang).mean()
    R = np.hypot(mx, my)
    mean_ang = np.degrees(np.arctan2(my, mx))
    print("  方向(视盘->中心凹) 圆形均值=%.1f°  集中度 R=%.3f (R=1 完全一致)" % (mean_ang, R))
    sd = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-6))))
    print("  方向标准差≈%.1f°" % sd)
else:
    print("测试集视盘峰:")
    print(df[["key", "odx", "ody"]].to_string(index=False, float_format=lambda v: "%.0f" % v))

if a.dump:
    df.to_csv(a.dump, index=False)
    print("\n已保存", a.dump)