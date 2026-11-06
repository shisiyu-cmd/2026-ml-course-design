# -*- coding: utf-8 -*-
"""把 IDRiD 测试集（103 张，带黄斑坐标）追加进 ext/。

先前 37_build_ext.py 只用训练集，原因是测试集文件名与训练集重名
（都是 IDRiD_001…），混在一起会互相覆盖。这里用 `idt_` 前缀区分。
"""
import os, io, csv, json, zipfile
import numpy as np
import cv2
import common as C
import importlib.util

spec = importlib.util.spec_from_file_location("fov", "02_fov.py")
fov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fov)

EXT = os.path.join(C.ROOT, "ext")
ZIP = "ext/data/IDRiD_Localization.zip"
z = zipfile.ZipFile(ZIP)
names = z.namelist()

# 真值
gt = {}
f = [n for n in names if "IDRiD_Fovea_Center_Testing Set_Markups.csv" in n]
for row in csv.reader(io.StringIO(z.read(f[0]).decode("utf-8", "replace"))):
    if len(row) < 3 or not row[0].strip().startswith("IDRiD_"):
        continue
    try:
        gt[row[0].strip()] = (float(row[1]), float(row[2]))
    except ValueError:
        pass
print("IDRiD 测试集真值: %d 张" % len(gt))

imgs = {}
for n in names:
    stem = os.path.splitext(os.path.basename(n))[0]
    if n.lower().endswith(".jpg") and "Testing Set" in n and stem in gt:
        imgs[stem] = n
print("匹配图像: %d 张" % len(imgs))

meta = json.load(open(os.path.join(EXT, "meta.json")))
rows = list(csv.DictReader(open(os.path.join(EXT, "gt.csv"))))
have = set(m["key"] for m in rows)
print("现有 ext 条目: %d" % len(have))

added, failed = 0, []
for stem in sorted(imgs):
    key = "ext/idt_%s" % stem.split("_")[-1]
    if key in have:
        continue
    buf = np.frombuffer(z.read(imgs[stem]), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        failed.append((stem, "解码失败")); continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    DS = 2
    g2 = cv2.resize(gray, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    try:
        det = fov.detect_fov(g2, min(g2.shape))
    except Exception as e:
        failed.append((stem, "检测异常")); continue
    if det is None:
        failed.append((stem, "检测失败")); continue
    cx, cy, r = det[0] * DS, det[1] * DS, det[2] * DS
    if r < 0.30 * min(H, W):
        failed.append((stem, "圆过小")); continue
    gx, gy = gt[stem]
    if np.hypot(gx - cx, gy - cy) > 0.62 * r:
        failed.append((stem, "真值远离中心")); continue
    # cv2.imwrite 对含中文的绝对路径会**静默失败**（只返回 False，不抛异常），
    # 必须用 imencode + ndarray.tofile 这种对非 ASCII 路径安全的写法。
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        failed.append((stem, "编码失败")); continue
    buf.tofile(os.path.join(EXT, "%s.jpg" % key.split("/")[1]))
    if not os.path.exists(os.path.join(EXT, "%s.jpg" % key.split("/")[1])):
        failed.append((stem, "写盘失败")); continue
    meta[key] = dict(W=int(W), H=int(H), cx=float(cx), cy=float(cy), r=float(r))
    rows.append(dict(key=key, Fovea_X=gx, Fovea_Y=gy))
    added += 1

json.dump(meta, open(os.path.join(EXT, "meta.json"), "w"), indent=1)
with open(os.path.join(EXT, "gt.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["key", "Fovea_X", "Fovea_Y"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print()
print("新增: %d 张   失败: %d 张" % (added, len(failed)))
for k, why in failed[:8]:
    print("   %s: %s" % (k, why))
print("ext 总数: meta %d, gt %d" % (len(meta), len(rows)))