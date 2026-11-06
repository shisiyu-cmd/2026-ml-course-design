# -*- coding: utf-8 -*-
"""在 OOF 上搜索最佳 TTA 组合与解码方式。

必须用「该图所属折的模型」来预测，否则会用到见过该图的模型，导致评估虚高。
折的划分与 train.py 完全一致（StratifiedKFold, random_state=100+rep）。
"""
import os, argparse, itertools
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

import common as C
from predict import predict_one


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base200")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def predict_canvas(model, imgs_t, device, beta):
    """imgs_t: (N,3,S,S) 已归一化张量 -> 热力图格子坐标 (N,2)。"""
    out = model(imgs_t)
    prob = torch.sigmoid(out)
    return C.soft_argmax(prob, beta), C.hard_argmax_refine(prob), prob


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = C.load_meta()
    gt = C.load_gt()
    gt_map = C.load_gt_map(gt)
    keys = sorted(gt_map.keys())
    strat = np.array([0 if meta[k]["W"] < 2500 else 1 for k in keys])
    caches = {}          # 按是否 CLAHE 分别缓存画布，必须与权重的训练配置一致

    # 还原折划分
    fold_of = {}
    for rep in range(args.repeats):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=100 + rep)
        for fold, (ti, vi) in enumerate(skf.split(keys, strat)):
            for i in vi:
                fold_of[keys[i]] = fold

    # 收集每张图在若干 TTA 变体下的原始预测
    variants = {
        "plain": dict(flip=False, scale=1.0),
        "hflip": dict(flip=True, scale=1.0),
        "s125": dict(flip=False, scale=1.25),
        "s150": dict(flip=False, scale=1.5),
        "hflip_s125": dict(flip=True, scale=1.25),
    }
    S = C.CANVAS
    raw = {k: {} for k in keys}
    peaks = {}
    models = {}
    default_norm = (list(C.NORM_MEAN), list(C.NORM_STD))
    for fold in range(args.folds):
        pth = os.path.join("runs", "%s_f%d.pt" % (args.tag, fold))
        if not os.path.exists(pth):
            print("缺少 %s，跳过该折" % pth)
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        a = d.get("args", {})
        m = C.FoveaNet(a.get("backbone", "resnet18"), pretrained=False,
                       mid=a.get("mid", 64)).to(device)
        m.load_state_dict(d["state"]); m.eval()
        # 必须用该权重训练时的归一化统计量（CLAHE 与非 CLAHE 分布不同）
        C.set_norm(a.get("norm_mean", default_norm[0]), a.get("norm_std", default_norm[1]))
        cl = bool(a.get("clahe", False))
        if cl not in caches:
            caches[cl] = C.build_cache(meta, verbose=False, clahe=cl,
                                       clahe_clip=a.get("clahe_clip", 3.0))
        models[fold] = m
        vk = [k for k in keys if fold_of[k] == fold]
        for k in vk:
            canvas = caches[cl][k]
            for name, v in variants.items():
                sc = v["scale"]
                n = int(round(S / sc))
                img = canvas if n == S else C.cv2.resize(canvas, (n, n),
                                                         interpolation=C.cv2.INTER_AREA)
                if n != S:
                    img = C.cv2.copyMakeBorder(img, 0, S - n, 0, S - n,
                                               C.cv2.BORDER_CONSTANT, value=(0, 0, 0))
                x = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device)
                x = C.normalize_batch(x)
                if v["flip"]:
                    x = torch.flip(x, dims=[3])
                s, h, prob = predict_canvas(m, x, device, args.beta)
                off = prob.shape[3] - 0.25
                if v["flip"]:
                    s[:, 0] = off - s[:, 0]; h[:, 0] = off - h[:, 0]
                s = s[0].cpu().numpy() * C.STRIDE * sc
                h = h[0].cpu().numpy() * C.STRIDE * sc
                raw[k][name] = (s, h)
                peaks.setdefault(k, []).append(float(prob.max().cpu()))

    # 各组合的 OOF RMSE
    def oof_rmse(combo, decode, alpha=0.5):
        P, Y = [], []
        for k in keys:
            if not raw[k]:
                continue
            ss = np.mean([raw[k][c][0] for c in combo], 0)
            hh = np.mean([raw[k][c][1] for c in combo], 0)
            if decode == "soft":
                c = ss
            elif decode == "hard":
                c = hh
            else:
                c = alpha * ss + (1 - alpha) * hh
            P.append(C.to_orig(meta[k], c[0], c[1]))
            Y.append(gt_map[k])
        P, Y = np.array(P), np.array(Y)
        return float(np.sqrt(((P - Y) ** 2).mean())), P, Y

    combos = [("plain",), ("hflip",), ("plain", "hflip"), ("plain", "hflip", "s125"),
              ("plain", "hflip", "s125", "s150"), ("plain", "s125"),
              ("hflip", "s125", "hflip_s125")]
    rows = []
    print("%-40s %10s %10s %10s" % ("TTA 组合", "soft", "hard", "blend"))
    for combo in combos:
        vals = []
        for dec in ("soft", "hard", "blend"):
            r, _, _ = oof_rmse(combo, dec)
            vals.append(r)
        rows.append((("+".join(combo)), *vals))
        print("%-40s %10.3f %10.3f %10.3f" % (rows[-1][0], vals[0], vals[1], vals[2]))

    best = min(rows, key=lambda r: min(r[1:]))
    print()
    print("最优: TTA=%s  解码=%s  RMSE=%.3f"
          % (best[0], ["soft", "hard", "blend"][int(np.argmin(best[1:]))], min(best[1:])))

    # 单模型(不带 TTA) vs 最优
    r0, _, _ = oof_rmse(("plain",), "soft")
    print("对照: 无 TTA + soft-argmax = %.3f" % r0)

    # blend 权重扫描
    print()
    print("blend 权重扫描 (TTA=%s):" % best[0])
    combo = tuple(best[0].split("+"))
    for al in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        r, _, _ = oof_rmse(combo, "blend", al)
        print("   alpha(soft)=%.1f -> RMSE=%.3f" % (al, r))

    # 低置信度样本
    print()
    conf = sorted(((k, float(np.mean(v))) for k, v in peaks.items()), key=lambda x: x[1])
    print("OOF 中置信度最低的 10 张:")
    for k, pk in conf[:10]:
        print("   %s  peak=%.3f" % (k, pk))

    if args.out:
        # 保存逐图原始预测（原图像素）与峰值，供后续策略分析使用
        save = {"keys": np.array(keys), "fold": np.array([fold_of[k] for k in keys])}
        for name in variants:
            save["soft_" + name] = np.array(
                [C.to_orig(meta[k], *raw[k][name][0]) for k in keys])
            save["hard_" + name] = np.array(
                [C.to_orig(meta[k], *raw[k][name][1]) for k in keys])
        save["peak_mean"] = np.array([float(np.mean(peaks[k])) for k in keys])
        save["gt"] = np.array([gt_map[k] for k in keys])
        np.savez(args.out, **save)
        print("\n已保存逐图预测: %s" % args.out)


if __name__ == "__main__":
    main()