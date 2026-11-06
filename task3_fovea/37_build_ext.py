# -*- coding: utf-8 -*-
"""把外部数据集（IDRiD 等）接入现有管线。

产出:
  ext/<key>.jpg      统一命名的图像（build_cache 按 ROOT/<split>/<key>.jpg 读取）
  ext/meta.json      视野圆参数，schema 与 fov_meta.json 一致
  ext/gt.csv         黄斑坐标真值

设计要点: 复用 02_fov.detect_fov 做视野圆定位，使外部图像与本赛题数据
经过**完全相同**的裁剪/归一化流程，避免域间预处理不一致。
"""
import os, io, csv, json, zipfile, argparse
import numpy as np
import cv2
import common as C
import importlib.util

spec = importlib.util.spec_from_file_location("fov", "02_fov.py")
fov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fov)

ap = argparse.ArgumentParser()
ap.add_argument("--idrid_zip", default="ext/data/IDRiD_Localization.zip")
ap.add_argument("--out", default="ext")
ap.add_argument("--min_r_frac", type=float, default=0.30,
                help="视野半径下限（相对短边），过小的检测判为失败")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
z = zipfile.ZipFile(a.idrid_zip)
names = z.namelist()


def find(sub):
    return [n for n in names if sub in n]


# ---- 1. 读真值（只用训练集；测试集文件名与训练集重名，混用会互相覆盖）----
gt = {}
f = find("IDRiD_Fovea_Center_Training Set_Markups.csv")
if not f:
    raise SystemExit("找不到 IDRiD 黄斑真值 CSV")
txt = z.read(f[0]).decode("utf-8", "replace")
for row in csv.reader(io.StringIO(txt)):
    if len(row) < 3 or not row[0].strip().startswith("IDRiD_"):
        continue
    try:
        gt[row[0].strip()] = (float(row[1]), float(row[2]))
    except ValueError:
        continue
print("读入 IDRiD 训练集黄斑真值: %d 张" % len(gt))

# ---- 2. 找图像（只取 Training Set；注意 basename 要去掉扩展名再截取）----
imgs = {}
for n in names:
    stem = os.path.splitext(os.path.basename(n))[0]
    if (n.lower().endswith(".jpg") and "Training Set" in n and stem in gt):
        imgs[stem] = n
print("匹配到图像: %d 张" % len(imgs))

meta, gtout, failed = {}, [], []
for key in sorted(imgs):
    n = imgs[key]
    buf = np.frombuffer(z.read(n), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        failed.append((key, "解码失败")); continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 降采样 2x 再做视野圆检测: detect_fov 内部会在 1/8、1/4、1/2 三个尺度上搜索，
    # 最贵的是 1/2 尺度（半径上千的卷积核全图卷积）。先降 2x 即可把该步降到 1/4 尺度，
    # 精度损失约 ±4 原图像素，对圆形检测可忽略。
    DS = 2
    g2 = cv2.resize(gray, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    try:
        det = fov.detect_fov(g2, min(g2.shape))
    except Exception as e:
        failed.append((key, "检测异常 %s" % type(e).__name__)); continue
    if det is None:
        failed.append((key, "检测失败")); continue
    cx, cy, r = det[0] * DS, det[1] * DS, det[2] * DS
    if r < a.min_r_frac * min(H, W) or not (0 < cx < W and 0 < cy < H):
        failed.append((key, "圆不合理 c=(%.0f,%.0f) r=%.0f" % (cx, cy, r))); continue
    ekey = "ext/%s" % key.lower()
    cv2.imwrite(os.path.join(a.out, "%s.jpg" % key.lower()),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    meta[ekey] = dict(W=int(W), H=int(H), cx=float(cx), cy=float(cy), r=float(r))
    gx, gy = gt[key]
    # 真值须落在视野内，否则标注与图像不匹配
    if np.hypot(gx - cx, gy - cy) > 0.62 * r:
        failed.append((key, "真值远离视野中心 %.2fr" % (np.hypot(gx-cx, gy-cy)/r)))
        continue
    gtout.append(dict(key=ekey, Fovea_X=gx, Fovea_Y=gy))

with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(a.out, "gt.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["key", "Fovea_X", "Fovea_Y"])
    w.writeheader()
    for row in gtout:
        w.writerow(row)

print()
print("成功: %d 张 -> %s/" % (len(gtout), a.out))
print("失败: %d 张" % len(failed))
for k, why in failed[:12]:
    print("   %s: %s" % (k, why))
if gtout:
    rr = np.array([meta[g["key"]]["r"] for g in gtout])
    cc = np.array([[meta[g["key"]]["cx"], meta[g["key"]]["cy"]] for g in gtout])
    ee = np.array([np.hypot(g["Fovea_X"] - m[0], g["Fovea_Y"] - m[1]) / r
                   for g, m, r in zip(gtout, cc, rr)])
    print("视野半径: 中位 %.0f (%.0f~%.0f)" % (np.median(rr), rr.min(), rr.max()))
    print("中心凹离心度: 中位 %.3f (%.3f~%.3f)" % (np.median(ee), ee.min(), ee.max()))