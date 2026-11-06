# -*- coding: utf-8 -*-
"""回收被过严的离心度阈值(0.62r)误删的 IDRiD 图像。

视觉核查已确认: 这些图的黄斑标注是正确的，被删只是因为拍摄时以视盘为中心
取景，导致中心凹离视野圆心较远。放宽到 0.80r（画布半宽为 1.06r，仍留 25% 边距）。
只对尚未采用的图重跑视野圆检测，已采用的不重算。
"""
import os, io, csv, json, zipfile
import numpy as np
import cv2
import common as C
import importlib.util

spec = importlib.util.spec_from_file_location("fov", "02_fov.py")
fov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fov)

THR = 0.80
ZIP = "ext/data/IDRiD_Localization.zip"
EXT = os.path.join(C.ROOT, "ext")
z = zipfile.ZipFile(ZIP)
names = z.namelist()

# 两个集合都要（训练集 idrid_xxx，测试集 idt_xxx）
gt = {}
for sub, pre in [("IDRiD_Fovea_Center_Training Set_Markups.csv", "idrid_"),
                 ("IDRiD_Fovea_Center_Testing Set_Markups.csv", "idt_")]:
    f = [n for n in names if sub in n][0]
    for row in csv.reader(io.StringIO(z.read(f).decode("utf-8", "replace"))):
        if len(row) < 3 or not row[0].strip().startswith("IDRiD_"):
            continue
        try:
            gt[pre + row[0].strip().split("_")[-1]] = (float(row[1]), float(row[2]),
                                                       "Training Set" in sub)
        except ValueError:
            pass

imgs = {}
for n in names:
    stem = os.path.splitext(os.path.basename(n))[0]
    if not n.lower().endswith(".jpg"):
        continue
    tag = None
    if "Training Set" in n:
        tag = "idrid_" + stem.split("_")[-1]
    elif "Testing Set" in n:
        tag = "idt_" + stem.split("_")[-1]
    if tag and tag in gt:
        imgs[tag] = n

meta = json.load(open(os.path.join(EXT, "meta.json")))
rows = list(csv.DictReader(open(os.path.join(EXT, "gt.csv"))))
have = set(r["key"] for r in rows)
todo = [t for t in sorted(imgs) if "ext/%s" % t not in have]
print("图像总数 %d，已采用 %d，待回收 %d" % (len(imgs), len(have), len(todo)))

added, failed = 0, []
for tag in todo:
    gx, gy, _ = gt[tag]
    buf = np.frombuffer(z.read(imgs[tag]), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        failed.append((tag, "解码失败")); continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    DS = 2
    g2 = cv2.resize(gray, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    try:
        det = fov.detect_fov(g2, min(g2.shape))
    except Exception:
        failed.append((tag, "检测异常")); continue
    if det is None:
        failed.append((tag, "检测失败")); continue
    cx, cy, r = det[0] * DS, det[1] * DS, det[2] * DS
    if r < 0.30 * min(H, W):
        failed.append((tag, "圆过小")); continue
    ecc = np.hypot(gx - cx, gy - cy) / r
    if ecc > THR:
        failed.append((tag, "离心度 %.2f 仍超阈值" % ecc)); continue
    # 真值必须落在画布内（to_canvas 需要 _finalize_meta 补齐的 x0/y0/si）
    S = C.CANVAS
    mm = C._finalize_meta({"t": dict(W=int(W), H=int(H), cx=cx, cy=cy, r=r)})["t"]
    tcx, tcy = C.to_canvas(mm, gx, gy)
    if not (0.02 * S < tcx < 0.98 * S and 0.02 * S < tcy < 0.98 * S):
        failed.append((tag, "真值出画布")); continue
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        failed.append((tag, "编码失败")); continue
    enc.tofile(os.path.join(EXT, "%s.jpg" % tag))
    if not os.path.exists(os.path.join(EXT, "%s.jpg" % tag)):
        failed.append((tag, "写盘失败")); continue
    key = "ext/%s" % tag
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
print("回收: %d 张   仍失败: %d 张" % (added, len(failed)))
for k, why in failed[:10]:
    print("   %-14s %s" % (k, why))
print("ext 总数: %d 图 / %d 真值" % (len(meta), len(rows)))