# -*- coding: utf-8 -*-
"""复现 train.py 评估路径的 canvas bug：直接调用其 evaluate()。"""
import numpy as np
import torch
import common as C
import train as T

META = T.META
gt_map = C.load_gt_map(C.load_gt())
keys = sorted(gt_map.keys())
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for ckpt in ["runs/allM_snap0200.pt", "runs/allL_snap0200.pt"]:
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    ar = d["args"]
    S = ar.get("canvas", 1024)
    C.set_canvas(S)
    C.set_norm(ar.get("norm_mean", C.NORM_MEAN), ar.get("norm_std", C.NORM_STD))
    cache = C.build_cache(META, verbose=False, clahe=bool(ar.get("clahe", False)))
    m = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                   mid=ar.get("mid", 64)).to(dev)
    m.load_state_dict(d["state"]); m.eval()

    ds = C.FoveaDataset(keys, cache, META, gt_map, False, ar.get("sigma", 2.0),
                        clahe=bool(ar.get("clahe", False)))
    coll = C.HeatmapCollate(ar.get("sigma", 2.0), "cpu")
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                     num_workers=0, collate_fn=coll)

    sp, hp, bp, ks, orig = T.evaluate(m, dl, dev, ar.get("beta", 20.0))
    r_soft = C.report_rmse(sp, orig, ks, verbose=False)
    r_hard = C.report_rmse(hp, orig, ks, verbose=False)
    print("canvas=%4d  经 train.evaluate:  soft=%.2f  hard=%.2f   (n=%d)"
          % (S, r_soft, r_hard, len(ks)))
    # 直接前向（不经 evaluate）作为对照
    raw = []
    with torch.no_grad():
        for k in keys:
            x = C.normalize_batch(torch.from_numpy(
                np.ascontiguousarray(cache[k])).unsqueeze(0).to(dev))
            prob = torch.sigmoid(m(x))
            c = C.soft_argmax(prob, ar.get("beta", 20.0))[0].cpu().numpy() * C.STRIDE
            ox, oy = C.to_orig(META[k], c[0], c[1])
            gx, gy = gt_map[k]
            raw.append((ox - gx) ** 2 + (oy - gy) ** 2)
    print("canvas=%4d  直接前向(无TTA):      soft=%.2f" % (S, np.sqrt(np.mean(raw))))
    print()
    del m
    torch.cuda.empty_cache()