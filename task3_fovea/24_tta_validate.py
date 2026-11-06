# -*- coding: utf-8 -*-
"""在折模型的验证集上诚实地重新检验 TTA 与解码策略。

背景：现有的 TTA 结论（翻转有用、多尺度有害）是用 200 轮模型得出的，
从未在 800 轮模型上复核。且历史上解码方式是靠榜单反馈间接推断的。
本脚本直接用留出集给出可比较的 OOF 数字。

用法:
  python 24_tta_validate.py --tags long800 clahelong
"""
import os, glob, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import common as C
from sklearn.model_selection import StratifiedKFold

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["long800"])
p.add_argument("--beta", type=float, default=20.0)
p.add_argument("--rot", type=float, default=10.0, help="旋转 TTA 角度")
p.add_argument("--scales", nargs="*", type=float, default=[1.15, 0.87])
args = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
META = C.load_meta()
gt = C.load_gt()
gt_map = C.load_gt_map(gt)
keys = sorted(gt_map.keys())
strat = np.array([0 if META[k]["W"] < 2500 else 1 for k in keys])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
folds = list(skf.split(keys, strat))


def warp_pt(M, pt):
    x, y = pt
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


def rot_mat(S, deg):
    c = (S - 1) / 2.0
    a, b = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[a, b, (1 - a) * c - b * c],
                     [-b, a, b * c + (1 - a) * c]], np.float32)


def scale_mat(S, s):
    c = (S - 1) / 2.0
    return np.array([[s, 0, (1 - s) * c], [0, s, (1 - s) * c]], np.float32)


def cell_from_canvas(canvas_xy, S):
    """画布坐标 -> 热力图格子坐标。"""
    return canvas_xy / C.STRIDE


@torch.no_grad()
def predict_variants(model, u8, S, beta):
    """返回 {变体名: (soft, hard)} 画布坐标。输入 u8 为 HWC uint8。"""
    def prep(arr):
        return C.normalize_batch(
            torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(dev))

    out = {}
    variants = [("plain", None), ("hflip", "flip")]
    for deg in (-args.rot, args.rot):
        variants.append(("rot%+d" % deg, ("rot", deg)))
    for s in args.scales:
        variants.append(("scale%.2f" % s, ("scale", s)))

    base = {}
    for name, op in variants:
        if op is None:
            img = prep(u8); Minv = None
        elif op == "flip":
            img = prep(u8[:, ::-1]); Minv = None
        elif op[0] == "rot":
            M = rot_mat(S, op[1])
            w = cv2.warpAffine(u8, M, (S, S), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img = prep(w); Minv = rot_mat(S, -op[1])
        else:
            M = scale_mat(S, op[1])
            w = cv2.warpAffine(u8, M, (S, S), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            img = prep(w); Minv = scale_mat(S, 1.0 / op[1])
        prob = torch.sigmoid(model(img))
        soft = C.soft_argmax(prob, beta)[0].cpu().numpy() * C.STRIDE
        hard = C.hard_argmax_refine(prob)[0].cpu().numpy() * C.STRIDE
        if op == "flip":
            soft[0] = (S - 1) - soft[0]; hard[0] = (S - 1) - hard[0]
        elif op is not None and op[0] in ("rot", "scale"):
            soft = np.array(warp_pt(Minv, soft)); hard = np.array(warp_pt(Minv, hard))
        base[name] = (soft, hard)
    return base


def main():
    allres = {}
    for tag in args.tags:
        ckpts = sorted(glob.glob("runs/%s_f*.pt" % tag))
        if not ckpts:
            print("跳过 %s（无折权重）" % tag); continue
        rows = []
        for fold, (ti, vi) in enumerate(folds):
            f = "runs/%s_f%d.pt" % (tag, fold)
            if not os.path.exists(f):
                continue
            d = torch.load(f, map_location="cpu", weights_only=False)
            ar = d.get("args", {})
            C.set_canvas(ar.get("canvas", 1024))
            C.set_norm(ar.get("norm_mean", C.NORM_MEAN), ar.get("norm_std", C.NORM_STD))
            cache = C.build_cache(META, verbose=False, clahe=bool(ar.get("clahe", False)),
                                  clahe_clip=ar.get("clahe_clip", 3.0))
            m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                           mid=ar.get("mid", 64),
                           box_head=bool(ar.get("box", False))).to(dev)
            m.load_state_dict(d["state"]); m.eval()
            S = C.CANVAS
            for k in [keys[i] for i in vi]:
                base = predict_variants(m, cache[k], S, args.beta)
                rec = {"tag": tag, "fold": fold, "key": k,
                       "gt": gt_map[k], "meta": META[k]}
                for name, (s, h) in base.items():
                    rec[name + "_s"] = s; rec[name + "_h"] = h
                rows.append(rec)
            del m; torch.cuda.empty_cache()
        allres[tag] = rows

    for tag, rows in allres.items():
        print("=" * 92)
        print("实验 %s —— 各 TTA/解码组合在验证集上的 OOF RMSE (n=%d)" % (tag, len(rows)))
        print("=" * 92)
        names = [k[:-2] for k in rows[0] if k.endswith("_s")]

        def rmse(getter):
            e = []
            for r in rows:
                p = np.array(getter(r))
                po = C.to_orig(r["meta"], p[0], p[1])
                e.append((po[0] - r["gt"][0]) ** 2 + (po[1] - r["gt"][1]) ** 2)
            return float(np.sqrt(np.mean(e)))

        res = {}
        for nm in names:
            res[nm + "/soft"] = rmse(lambda r, nm=nm: r[nm + "_s"])
            res[nm + "/hard"] = rmse(lambda r, nm=nm: r[nm + "_h"])
            res[nm + "/blend"] = rmse(lambda r, nm=nm: 0.5 * (r[nm + "_s"] + r[nm + "_h"]))

        # 组合：对 4 个 TTA 变体先平均，再按 alpha 混合 soft/hard
        FLIPROT = ["plain", "hflip", "rot-10", "rot+10"]

        def combo(r, use, kind):
            acc = np.zeros(2)
            for u in use:
                acc += r[u + "_" + kind]
            return acc / len(use)

        res["TTA4/soft"] = rmse(lambda r: combo(r, FLIPROT, "s"))
        res["TTA4/hard"] = rmse(lambda r: combo(r, FLIPROT, "h"))
        for al in (0.25, 0.5, 0.75):
            res["TTA4/blend%.2f" % al] = rmse(
                lambda r, al=al: al * combo(r, FLIPROT, "s") + (1 - al) * combo(r, FLIPROT, "h"))
        res["TTA2(flip)/blend0.50"] = rmse(
            lambda r: 0.5 * combo(r, ["plain", "hflip"], "s") +
                      0.5 * combo(r, ["plain", "hflip"], "h"))

        for nm, v in sorted(res.items(), key=lambda x: x[1]):
            print("   %-34s %.3f" % (nm, v))
        print()
        print("   相对当前方案 hflip/blend 的改善:")
        b = res.get("TTA2(flip)/blend0.50", res["plain/soft"])
        for nm in ["flip/blend0.50" if False else "TTA4/blend0.50", "TTA4/blend0.75", "TTA4/soft", "TTA4/hard"]:
            if nm in res:
                print("     %-34s %+.3f (%+.1f%%)" % (nm, res[nm] - b, 100 * (res[nm] - b) / b))
        print()


if __name__ == "__main__":
    main()