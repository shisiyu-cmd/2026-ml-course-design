# -*- coding: utf-8 -*-
"""推理并生成 submission.csv

- 支持多折 / 多种子 / 多主干 / 是否 CLAHE 的混合集成
  （每个权重按自身训练配置取对应的画布，CLAHE 与非 CLAHE 模型不会互相污染）
- TTA: 水平翻转与多尺度，默认关闭（OOF 实验表明 plain 最优）
- 集成方式: 各模型预测取均值（对离群模型更稳健），可选按峰值加权

用法:
  python predict.py --tags base200 seed1 --out submission.csv
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd
import torch

import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="*", default=None)
    p.add_argument("--tags", nargs="*", default=None, help="按 runs/<tag>_f*.pt 收集权重")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--scales", nargs="*", type=float, default=[1.0])
    p.add_argument("--flip", action="store_true", help="启用水平翻转 TTA")
    p.add_argument("--decode", default="soft", choices=["soft", "hard", "blend"])
    p.add_argument("--alpha", type=float, default=1.0, help="blend 时 soft-argmax 的权重")
    p.add_argument("--weight", default="uniform", choices=["uniform", "peak"])
    p.add_argument("--mindist", type=float, default=0.0,
                   help=">0 时距圆心超过 mindist*r 判为不可见并输出 (0,0)")
    p.add_argument("--dump", default="runs/test_pred.csv")
    return p.parse_args()


@torch.no_grad()
def predict_one(model, canvas, device, beta, scales, flip_tta):
    """多尺度(+翻转) TTA。返回 (soft 画布坐标, hard 画布坐标, 峰值)。"""
    S = canvas.shape[0]
    acc_s, acc_h, wsum, peak = np.zeros(2), np.zeros(2), 0.0, 0.0
    for sc in scales:
        sc = max(1.0, float(sc))
        n = int(round(S / sc))
        if n == S:
            img = canvas
        else:
            img = C.cv2.resize(canvas, (n, n), interpolation=C.cv2.INTER_AREA)
            img = C.cv2.copyMakeBorder(img, 0, S - n, 0, S - n,
                                       C.cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x = C.normalize_batch(torch.from_numpy(np.ascontiguousarray(img))
                              .unsqueeze(0).to(device))
        flips = (False, True) if flip_tta else (False,)
        for flip in flips:
            xi = torch.flip(x, dims=[3]) if flip else x
            prob = torch.sigmoid(model(xi))
            cs = C.soft_argmax(prob, beta)[0].cpu().numpy() * C.STRIDE
            ch = C.hard_argmax_refine(prob)[0].cpu().numpy() * C.STRIDE
            if flip:
                # 翻转发生在 n×n 有效区域内（缩略图贴左上、右下补黑边）
                cs[0] = (n - 1) - cs[0]
                ch[0] = (n - 1) - ch[0]
            cs = np.clip(cs, 0, n - 1) * sc
            ch = np.clip(ch, 0, n - 1) * sc
            pk = float(prob.max().cpu())
            acc_s += cs; acc_h += ch; wsum += 1.0
            peak = max(peak, pk)
    return acc_s / max(wsum, 1e-9), acc_h / max(wsum, 1e-9), peak


def decode(soft, hard, mode, alpha):
    if mode == "soft":
        return soft
    if mode == "hard":
        return hard
    return alpha * soft + (1 - alpha) * hard


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = list(args.ckpts or [])
    for t in (args.tags or []):
        ckpts += sorted(glob.glob(os.path.join("runs", "%s_f*.pt" % t)))
    if not ckpts:
        raise SystemExit("没有找到权重文件")
    print("集成 %d 个模型:" % len(ckpts))

    meta = C.load_meta()
    default_norm = (list(C.NORM_MEAN), list(C.NORM_STD))
    models, caches = [], {}
    canvas_seen = set()
    for c in ckpts:
        d = torch.load(c, map_location="cpu", weights_only=False)
        a = d.get("args", {})
        cl = bool(a.get("clahe", False))
        # 每个权重用它自己训练时的归一化统计量（旧权重没有该字段则用默认值）
        nm = a.get("norm_mean", default_norm[0])
        ns = a.get("norm_std", default_norm[1])
        # 框监督(--box)训练的权重含 box_head.* 参数，而预测只需要中心凹输出，
        # 故这里不带框头构建，并用 strict=False 忽略多出来的键。
        # 注意: 若用 strict=True 会直接报 "Unexpected key(s) in state_dict: box_head.*"，
        # 导致所有框监督模型无法用于预测。
        m = C.FoveaNet(a.get("backbone", "resnet18"), pretrained=False,
                       mid=a.get("mid", 64)).to(device)
        missing, unexpected = m.load_state_dict(d["state"], strict=False)
        missing = [k for k in missing if not k.startswith("box_head")]
        if missing:
            raise RuntimeError("权重缺少必要参数: %s" % missing[:5])
        if unexpected:
            extra = [k for k in unexpected if not k.startswith("box_head")]
            if extra:
                raise RuntimeError("权重含意外参数: %s" % extra[:5])
        m.eval()
        models.append((m, cl, a, nm, ns))
        cv = int(a.get("canvas", 1024))
        canvas_seen.add(cv)
        if (cl, cv) not in caches:
            C.set_canvas(cv)
            caches[(cl, cv)] = C.build_cache(meta, verbose=False, clahe=cl,
                                             clahe_clip=a.get("clahe_clip", 3.0))
        print("   %-34s backbone=%-12s clahe=%-5s canvas=%-5d norm=%s"
              % (c, a.get("backbone", "?"), cl, cv, np.round(nm, 3).tolist()))
    if len(canvas_seen) > 1:
        raise SystemExit("集成混用了不同画布尺寸 %s" % sorted(canvas_seen))

    test_keys = sorted([k for k in meta if k.startswith("test/")])
    rows = []
    for k in test_keys:
        cs, hs, ps = [], [], []
        for (m, cl, a, nm, ns) in models:
            cv = int(a.get("canvas", 1024))
            C.set_canvas(cv)
            C.set_norm(nm, ns)
            s, h, pk = predict_one(m, caches[(cl, cv)][k], device, args.beta,
                                   args.scales, args.flip)
            cs.append(s); hs.append(h); ps.append(pk)
        cs, hs, ps = np.array(cs), np.array(hs), np.array(ps)
        if args.weight == "peak":
            w = np.clip(ps, 1e-3, None); w = w / w.sum()
        else:
            w = np.full(len(cs), 1.0 / len(cs))
        canvas_xy = (w[:, None] * decode(cs, hs, args.decode, args.alpha)).sum(0)
        x_orig, y_orig = C.to_orig(meta[k], canvas_xy[0], canvas_xy[1])
        md = meta[k]
        dist_r = float(np.hypot(x_orig - md["cx"], y_orig - md["cy"]) / md["r"])
        rows.append(dict(key=k, keynum=int(k.split("/")[1]), x=x_orig, y=y_orig,
                         canvas_x=canvas_xy[0], canvas_y=canvas_xy[1],
                         peak=float(ps.mean()), peak_min=float(ps.min()),
                         agree=float(np.hypot(*(cs - cs.mean(0)).T).mean()),
                         dist_r=dist_r, W=md["W"]))
        print("  %s  预测=(%8.1f,%8.1f)  画布=(%6.1f,%6.1f)  峰值=%.3f(最低%.3f)  "
              "模型分歧=%.1fpx  距圆心=%.3fr"
              % (k, x_orig, y_orig, canvas_xy[0], canvas_xy[1],
                 ps.mean(), ps.min(), rows[-1]["agree"], dist_r))

    df = pd.DataFrame(rows).sort_values("keynum")
    os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
    df.to_csv(args.dump, index=False)
    print("\n已保存逐图预测: %s" % args.dump)
    print("峰值: min=%.3f median=%.3f | 模型分歧: median=%.1fpx max=%.1fpx"
          % (df.peak.min(), df.peak.median(), df.agree.median(), df.agree.max()))

    # ---- 组装提交文件（严格照 sample_submission.csv 的顺序）----
    sub = pd.read_csv(os.path.join(C.ROOT, "sample_submission.csv"))
    pred, n_inv = {}, 0
    for r in df.itertuples():
        if args.mindist > 0 and r.dist_r > args.mindist:
            x, y = 0.0, 0.0; n_inv += 1
        else:
            x, y = r.x, r.y
        pred["%d_Fovea_X" % r.keynum] = x
        pred["%d_Fovea_Y" % r.keynum] = y
    missing = [i for i in sub.ImageID if i not in pred]
    if missing:
        raise SystemExit("提交缺少条目: %s" % missing[:5])
    sub["value"] = sub.ImageID.map(pred)
    assert sub.value.notna().all(), "存在空值"
    sub.to_csv(args.out, index=False)
    print("已写出提交文件: %s (%d 行, 判为不可见 %d 张)" % (args.out, len(sub), n_inv))


if __name__ == "__main__":
    main()