# -*- coding: utf-8 -*-
"""数据侦察：GT 坐标、检测框、图像尺寸、是否存在不可见中心凹 (0,0)。"""
import os, sys, glob, importlib
import xml.etree.ElementTree as ET

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026task3", "detection")

print("=" * 70)
print("库可用性")
print("=" * 70)
for m in ["numpy", "PIL", "pandas", "sklearn", "matplotlib", "tqdm", "scipy", "timm",
          "albumentations", "cv2", "torch", "torchvision"]:
    try:
        mod = importlib.import_module(m)
        print("  %-15s OK   %s" % (m, getattr(mod, "__version__", "?")))
    except Exception:
        print("  %-15s MISSING" % m)

print()
print("=" * 70)
print("目录结构")
print("=" * 70)
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    exts = {}
    for f in filenames:
        e = os.path.splitext(f)[1].lower()
        exts[e] = exts.get(e, 0) + 1
    print("  %-30s files=%d  %s" % (rel, len(filenames), exts))

# ---------- GT ----------
print()
print("=" * 70)
print("fovea_localization_train_GT.csv")
print("=" * 70)
import pandas as pd
gt = pd.read_csv(os.path.join(ROOT, "fovea_localization_train_GT.csv"))
print(gt.head(10).to_string())
print("shape:", gt.shape, "| columns:", list(gt.columns))
print("dtypes:\n", gt.dtypes)
print("describe:\n", gt.describe().to_string())
zero = gt[(gt.Fovea_X == 0) & (gt.Fovea_Y == 0)]
print("(0,0) 行数 (中心凹不可见):", len(zero))
print("近零行 (X<1 或 Y<1):")
print(gt[(gt.Fovea_X < 1) | (gt.Fovea_Y < 1)].to_string())
print("NaN 数量:\n", gt.isna().sum().to_string())

# ---------- 检测框 ----------
print()
print("=" * 70)
print("train_location XML 检测框")
print("=" * 70)
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "train_location", "*.xml"))):
    t = ET.parse(p)
    r = t.getroot()
    size = r.find("size")
    W = float(size.find("width").text); H = float(size.find("height").text)
    fname = r.find("filename").text
    objs = r.findall("object")
    names = [o.find("name").text for o in objs]
    for o in objs:
        bb = o.find("bndbox")
        xmin = float(bb.find("xmin").text); ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text); ymax = float(bb.find("ymax").text)
        rows.append(dict(xml=os.path.basename(p), img=fname, W=W, H=H,
                         cls=o.find("name").text,
                         xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                         cx=(xmin + xmax) / 2, cy=(ymin + ymax) / 2,
                         bw=xmax - xmin, bh=ymax - ymin))
det = pd.DataFrame(rows)
print("框总数:", len(det), "| 文件数:", det.xml.nunique())
print("类别分布:\n", det.cls.value_counts().to_string())
print("每文件框数分布:\n", det.groupby("xml").size().value_counts().to_string())
print("XML 内尺寸 vs 宽高样例:\n", det[["img", "W", "H"]].head(5).to_string())
print("框宽高 describe:\n", det[["bw", "bh"]].describe().to_string())

# 关键：框中心 vs GT 中心凹 距离
print()
print("--- 框中心 与 GT中心凹 的偏差 (原始像素) ---")
gt["idx"] = gt["data"].astype(str)
det["idx"] = det["xml"].str.replace(".xml", "", regex=False).str.lstrip("0")
m = det.merge(gt, on="idx", how="inner")
print("成功匹配:", len(m), "/", len(det))
if len(m):
    d = ((m.cx - m.Fovea_X) ** 2 + (m.cy - m.Fovea_Y) ** 2) ** 0.5
    print("欧氏距离 describe:\n", d.describe().to_string())
    print("最大距离的 5 个:\n",
          m.assign(dist=d).nlargest(5, "dist")[["idx", "cls", "cx", "cy", "Fovea_X", "Fovea_Y", "dist"]].to_string())
    print("距离 <30px 的比例: %.1f%%" % (100 * (d < 30).mean()))
    # 归一化坐标(以图像宽高)
    m = m.assign(nx=m.Fovea_X / m.W, ny=m.Fovea_Y / m.H,
                 bcx=m.cx / m.W, bcy=m.cy / m.H)
    print("\nGT 归一化坐标 describe:\n", m[["nx", "ny"]].describe().to_string())

# ---------- 图像尺寸 ----------
print()
print("=" * 70)
print("图像尺寸")
print("=" * 70)
from PIL import Image
for split in ["train", "test"]:
    files = sorted(glob.glob(os.path.join(ROOT, split, "*.jpg")))
    sizes = {}
    for f in files:
        with Image.open(f) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
    print("%s (%d 张):" % (split, len(files)))
    for s, c in sorted(sizes.items(), key=lambda x: -x[1]):
        print("    %s x %s  ->  %d 张" % (s[0], s[1], c))

# ---------- submission ----------
print()
print("=" * 70)
print("sample_submission.csv")
print("=" * 70)
sub = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
print(sub.head(8).to_string())
print("shape:", sub.shape)
print("value 非空数:", sub.value.notna().sum())
ids = sub.ImageID.tolist()
print("前 6 个 ID:", ids[:6])
print("ImageID 唯一图片数:", len(set(i.rsplit("_", 1)[0] for i in ids)))
print("后缀分布:", pd.Series([i.rsplit("_", 1)[1] for i in ids]).value_counts().to_dict())