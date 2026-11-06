# -*- coding: utf-8 -*-
"""可视化预训练模型的输出距离图，定位复刻问题。"""
import sys, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "ext/odfov")
import odfov_torch as O
import common as C

META = C.load_meta()
gt = C.load_gt_map(C.load_gt())
model = O.load_pretrained().to("cuda" if __import__("torch").cuda.is_available() else "cpu")

keys = ["train/0001", "train/0005", "train/0010"]
fig, axes = plt.subplots(len(keys), 4, figsize=(19, 4.8 * len(keys)))
for r, k in enumerate(keys):
    img = C.imread_u(os.path.join(C.ROOT, k + ".jpg"))
    H, W = img.shape[:2]
    dm = O.predict_maps(model, [img], mode="global")[0]
    pk = O.find_two_peaks(dm)
    gx, gy = gt[k]
    # GT 映射到 512 网格
    g512 = (gx * 512.0 / W, gy * 512.0 / H)

    axes[r, 0].imshow(cv2.resize(img, (512, 512)))
    axes[r, 0].plot(g512[0], g512[1], "g+", ms=18, mew=3, label="GT 黄斑")
    for j, (py, px) in enumerate(pk[:2]):
        axes[r, 0].plot(px, py, "ro", ms=11, mfc="none", mew=2.5)
        axes[r, 0].text(px + 8, py, "p%d" % j, color="red", fontsize=11)
    axes[r, 0].set_title("%s  输入512 + GT(绿) + 预测峰(红)" % k, fontsize=10)
    axes[r, 0].legend(fontsize=8, loc="upper right")

    im1 = axes[r, 1].imshow(dm, cmap="inferno")
    axes[r, 1].set_title("距离图 (max=%.3f)" % dm.max(), fontsize=10)
    plt.colorbar(im1, ax=axes[r, 1], fraction=0.046)

    axes[r, 2].imshow(dm > 0.5, cmap="gray")
    axes[r, 2].set_title("阈值>0.5 区域 (%.2f%% 像素)" % (100 * (dm > 0.5).mean()), fontsize=10)

    # 输入原图的通道统计，确认归一化合理
    x = C.imread_u(os.path.join(C.ROOT, k + ".jpg")).astype(np.float32)
    axes[r, 3].hist(x.ravel(), bins=60, color="gray")
    axes[r, 3].set_title("像素直方图 mean=%.1f std=%.1f" % (x.mean(), x.std()), fontsize=10)
    for c in range(4):
        axes[r, c].axis("off") if c < 3 else None
plt.tight_layout()
plt.savefig("viz/ext_model_debug.png", dpi=80)
print("已保存 viz/ext_model_debug.png")
for r, k in enumerate(keys):
    img = C.imread_u(os.path.join(C.ROOT, k + ".jpg"))
    H, W = img.shape[:2]
    dm = O.predict_maps(model, [img], mode="global")[0]
    pk = O.find_two_peaks(dm)
    gx, gy = gt[k]
    print("%s GT512=(%.0f,%.0f)  峰值=%s  dm>0.5占%.2f%%  峰值个数=%d"
          % (k, gx * 512 / W, gy * 512 / H,
             [(round(p[1]), round(p[0])) for p in pk[:2]], 100 * (dm > 0.5).mean(), len(pk)))