# -*- coding: utf-8 -*-
"""调试 canvas=768 模型的定位：直接看热力图峰值与目标的相对位置。"""
import numpy as np
import torch
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for tag, ckpt in [("allM(768)", "runs/allM_snap0200.pt"), ("allL(1024)", "runs/allL_snap0200.pt")]:
    try:
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print("跳过", ckpt); continue
    ar = d["args"]
    S = ar.get("canvas", 1024)
    C.set_canvas(S)
    C.set_norm(ar.get("norm_mean", C.NORM_MEAN), ar.get("norm_std", C.NORM_STD))
    cache = C.build_cache(META, verbose=False, clahe=bool(ar.get("clahe", False)))
    m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                   mid=ar.get("mid", 64)).to(dev)
    m.load_state_dict(d["state"]); m.eval()
    print("=" * 76)
    print("%s  canvas=%d  快照轮数=%s" % (tag, S, d.get("epoch")))
    hs = S // C.STRIDE
    errs = []
    for k in ["train/0001", "train/0010", "train/0020", "train/0030", "train/0040"]:
        img = cache[k]
        gx, gy = gt[k]
        tcx, tcy = C.to_canvas(META[k], gx, gy)
        with torch.no_grad():
            x = C.normalize_batch(torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(dev))
            prob = torch.sigmoid(m(x))[0, 0].cpu()
        sm = C.soft_argmax(prob[None, None], ar.get("beta", 20.0))[0] * C.STRIDE
        hm = C.hard_argmax_refine(prob[None, None])[0] * C.STRIDE
        pk = float(prob.max())
        # 预测落点 -> 原图, 与真值比较
        px, py = C.to_orig(META[k], float(sm[0]), float(sm[1]))
        e = float(np.hypot(px - gx, py - gy))
        errs.append(e)
        print("  %-12s 真值画布=(%6.1f,%6.1f)  soft=(%6.1f,%6.1f) hard=(%6.1f,%6.1f)"
              "  峰值=%.2f  误差=%6.1f px" % (k, tcx, tcy, sm[0], sm[1], hm[0], hm[1], pk, e))
    print("  平均误差: %.1f px   (画布边长 %d, 热力图 %dx%d)" % (np.mean(errs), S, hs, hs))
    del m
    torch.cuda.empty_cache()