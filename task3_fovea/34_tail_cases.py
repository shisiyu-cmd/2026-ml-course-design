# -*- coding: utf-8 -*-
"""直视 long800 的尾部失败案例：图 + 真值 + OOF 预测 + 模型热力图。"""
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
z = np.load("runs/long800_oof.npz", allow_pickle=True)
soft, keys = z["soft"], [str(k) for k in z["keys"]]
eu = np.sqrt(((soft - z["orig"]) ** 2).sum(1))
order = np.argsort(-eu)

# 每个 fold 的权重文件里存的热力图是训练时的；这里直接用 allA（全量）看它的热力图
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("runs/allA_all.pt", map_location="cpu", weights_only=False)
ar = ck["args"]
C.set_canvas(ar.get("canvas", 1024))
C.set_norm(ar.get("norm_mean", C.NORM_MEAN), ar.get("norm_std", C.NORM_STD))
cache = C.build_cache(META, verbose=False)
model = C.FoveaNet(ar.get("backbone", "resnet18"), pretrained=False,
                   mid=ar.get("mid", 64)).to(dev)
model.load_state_dict(ck["state"]); model.eval()

N = 6
fig, axes = plt.subplots(N, 3, figsize=(15, 5.0 * N))
for r in range(N):
    j = order[r]
    k = keys[j]
    m = META[k]
    img = cache[k]
    S = img.shape[0]
    gx, gy = gt[k]
    tcx, tcy = C.to_canvas(m, gx, gy)
    # 模型热力图
    with torch.no_grad():
        x = C.normalize_batch(torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(dev))
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    pk = C.soft_argmax(torch.from_numpy(prob)[None, None], ar.get("beta", 20.0))[0].numpy() * C.STRIDE
    # 原图
    oimg = C.imread_u("%s/%s/%s.jpg" % (C.ROOT, k.split("/")[0], k.split("/")[1]))
    axes[r, 0].imshow(oimg)
    axes[r, 0].plot(gx, gy, "g+", ms=20, mew=3.5, label="真值")
    op = soft[j]
    axes[r, 0].plot(op[0], op[1], "rx", ms=16, mew=3.5, label="OOF预测")
    axes[r, 0].set_title("%s  OOF误差=%.1f px" % (k, eu[j]), fontsize=11)
    axes[r, 0].legend(fontsize=9)
    axes[r, 0].axis("off")

    axes[r, 1].imshow(img)
    axes[r, 1].plot(tcx, tcy, "g+", ms=20, mew=3.5)
    axes[r, 1].plot(pk[0], pk[1], "rx", ms=16, mew=3.5)
    axes[r, 1].set_title("画布 + allA预测(红)", fontsize=11)
    axes[r, 1].axis("off")

    im = axes[r, 2].imshow(prob, cmap="inferno")
    axes[r, 2].plot(tcx / C.STRIDE, tcy / C.STRIDE, "g+", ms=20, mew=3.5)
    axes[r, 2].plot(pk[0] / C.STRIDE, pk[1] / C.STRIDE, "rx", ms=16, mew=3.5)
    axes[r, 2].set_title("allA 热力图 (峰值 %.2f)" % prob.max(), fontsize=11)
    axes[r, 2].axis("off")
    plt.colorbar(im, ax=axes[r, 2], fraction=0.046)

plt.suptitle("long800 尾部失败案例（OOF 误差最大的 6 张）", fontsize=14)
plt.tight_layout()
plt.savefig("viz/tail_cases.png", dpi=70)
print("已保存 viz/tail_cases.png")
for r in range(N):
    j = order[r]; k = keys[j]; m = META[k]
    print("%-12s OOFerr=%6.1f  ecc=%.3f  W=%d  真值=(%.0f,%.0f) 预测=(%.0f,%.0f)"
          % (k, eu[j], np.hypot((gt[k][0] - m["cx"]) / m["r"], (gt[k][1] - m["cy"]) / m["r"]),
             m["W"], gt[k][0], gt[k][1], soft[j][0], soft[j][1]))