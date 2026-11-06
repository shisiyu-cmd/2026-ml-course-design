# -*- coding: utf-8 -*-
"""跟踪全量模型快照对关键测试图的预测轨迹。

若某模型对 test/0081 的预测在各训练阶段稳定收敛到同一位置，
说明它是"想清楚了"而不是碰巧；若一路漂移，则不可信。
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import torch
import common as C

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["allA", "allB"])
p.add_argument("--keys", nargs="*", default=["test/0081", "test/0086", "test/0090"])
p.add_argument("--beta", type=float, default=20.0)
a = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = C.load_meta()
caches = {}
out = {}
for tag in a.tags:
    snaps = sorted(glob.glob("runs/%s_snap*.pt" % tag))
    final = "runs/%s_all.pt" % tag
    if os.path.exists(final):
        snaps.append(final)
    if not snaps:
        print("跳过 %s（无快照）" % tag); continue
    rows = []
    for f in snaps:
        d = torch.load(f, map_location="cpu", weights_only=False)
        ar = d.get("args", {})
        cl = bool(ar.get("clahe", False))
        # 必须先恢复画布边长再建缓存: 否则 canvas!=1024 的模型会拿到错误尺寸的输入，
        # 预测值会带上 1024/CANVAS 的系统性偏差（此 bug 曾让 allM 显得漂移很大）。
        C.set_canvas(ar.get("canvas", 1024))
        C.set_norm(ar.get("norm_mean", C.NORM_MEAN), ar.get("norm_std", C.NORM_STD))
        ckey = (cl, C.CANVAS)
        if ckey not in caches:
            caches[ckey] = C.build_cache(meta, verbose=False, clahe=cl,
                                         clahe_clip=ar.get("clahe_clip", 3.0))
        m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                       mid=ar.get("mid", 64)).to(dev)
        m.load_state_dict(d["state"]); m.eval()
        rec = {"tag": tag, "ckpt": os.path.basename(f),
               "ep": d.get("epoch", 800 if "all.pt" in f else -1)}
        with torch.no_grad():
            for k in a.keys:
                x = C.normalize_batch(torch.from_numpy(
                    np.ascontiguousarray(caches[ckey][k])).unsqueeze(0).to(dev))
                prob = torch.sigmoid(m(x))
                c = C.soft_argmax(prob, a.beta)[0].cpu().numpy() * C.STRIDE
                rec[k + "_x"] = float(c[0]); rec[k + "_y"] = float(c[1])
                rec[k + "_pk"] = float(prob.max().cpu())
        rows.append(rec)
        del m
        torch.cuda.empty_cache()
    out[tag] = pd.DataFrame(rows)

for tag, df in out.items():
    print("=" * 96)
    print("%s —— 各训练阶段快照的预测 (画布坐标)" % tag)
    print("=" * 96)
    cols = ["ckpt", "ep"]
    for k in a.keys:
        cols += [k + "_x", k + "_y", k + "_pk"]
    print(df[cols].to_string(index=False, float_format=lambda v: "%.1f" % v))
    print()
    for k in a.keys:
        xs = df[k + "_x"].values; ys = df[k + "_y"].values
        spread = np.hypot(xs - xs.mean(), ys - ys.mean()).max()
        print("  %-12s 快照间最大漂移 = %6.1f 画布px = %6.1f 原图px   终值=(%.0f, %.0f)"
              % (k, spread, spread * meta[k]["si"] / C.CANVAS, xs[-1], ys[-1]))
    print()