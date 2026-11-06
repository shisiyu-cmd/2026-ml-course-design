# -*- coding: utf-8 -*-
"""黄斑中心凹定位 —— 公共模块

核心设计
--------
1. FOV 归一化: 按拟合出的视网膜圆裁剪正方形并缩放到固定画布 S×S，
   消除 2992x2000 与 1956x1934 两种分辨率的差异，使视网膜尺度恒定。
2. 热力图回归: 模型输出 stride=4 的高斯热力图，用 soft-argmax 得到亚像素坐标。
3. 坐标反算: 画布坐标 -> 原图像素坐标（用与预处理完全一致的取整参数）。
"""
import os, json, math, random, glob
import xml.etree.ElementTree as ET
# HuggingFace 直连在本机不可用，改用镜像下载 timm 预训练权重（必须在 import timm 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "2026task3", "detection")
FOV_META = os.path.join(HERE, "fov_meta.json")

MARGIN = 1.06          # 裁剪框 = 2r * MARGIN
CANVAS = 1024          # 画布边长
STRIDE = 4             # 热力图下采样倍率

# 画布图像的均值/标准差（由 03_stats.py 在全部 100 张画布上统计得到）
NORM_MEAN = (0.3563, 0.1761, 0.0516)
NORM_STD = (0.2875, 0.1595, 0.0811)


def set_canvas(S):
    """切换画布边长。注意: 改变后必须重建画布缓存，且坐标换算随之改变。"""
    global CANVAS
    CANVAS = int(S)
    return CANVAS


# ----------------------------------------------------------------------------
# 元数据 / 预处理
# ----------------------------------------------------------------------------
def load_ext():
    """读取 ext/ 下的外部数据集（若已构建）。返回 (meta 增量, gt 增量)。

    外部数据经 37_build_ext.py 用**相同的视野圆检测与裁剪流程**处理后落盘，
    因此与本赛题数据同构，可直接并入训练。
    """
    mp = os.path.join(ROOT, "ext", "meta.json")
    gp = os.path.join(ROOT, "ext", "gt.csv")
    if not (os.path.exists(mp) and os.path.exists(gp)):
        return {}, {}
    with open(mp, "r") as f:
        m = json.load(f)
    g = {}
    import csv as _csv
    with open(gp, "r", newline="") as f:
        for r in _csv.DictReader(f):
            try:
                g[r["key"]] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))
            except (KeyError, ValueError):
                continue
    keys = set(m) & set(g)
    return ({k: m[k] for k in keys}, {k: g[k] for k in keys})


def load_meta():
    with open(FOV_META, "r") as f:
        m = json.load(f)
    _finalize_meta(m)
    ext, _ = load_ext()
    _finalize_meta(ext)
    m.update(ext)
    return m


def _finalize_meta(m):
    """补齐 side / x0 / y0 / si（外部数据与本赛题数据共用同一套字段）。"""
    for k, v in m.items():
        side = 2 * v["r"] * MARGIN
        v["side"] = side
        v["x0"] = math.floor(v["cx"] - side / 2)   # 整数裁剪原点
        v["y0"] = math.floor(v["cy"] - side / 2)
        v["si"] = int(round(side))                 # 整数裁剪边长
    return m


def fov_crop(img, m):
    """按 FOV 元数据裁剪并缩放为 S×S 画布（越界用黑色填充）。"""
    H, W = img.shape[:2]
    x0, y0, si = m["x0"], m["y0"], m["si"]
    out = np.zeros((si, si, 3), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + si), min(H, y0 + si)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    if si != CANVAS:
        out = cv2.resize(out, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
    return out


def to_canvas(m, x, y):
    """原图像素坐标 -> 画布坐标（标量或 np 数组）。"""
    return (x - m["x0"]) * CANVAS / m["si"], (y - m["y0"]) * CANVAS / m["si"]


def to_orig(m, cx, cy):
    """画布坐标 -> 原图像素坐标。"""
    return m["x0"] + cx * m["si"] / CANVAS, m["y0"] + cy * m["si"] / CANVAS


def imread_u(path):
    """读取图像，支持中文等非 ASCII 路径（cv2.imread 在 Windows 上不支持）。"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("无法读取图像: %s" % path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_clahe(img, clip=3.0, grid=8):
    """在 LAB 的 L 通道做限制对比度自适应直方图均衡，增强低对比眼底图的局部结构。"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def build_cache(meta, verbose=True, clahe=False, clahe_clip=3.0):
    """把所有图读入并转成画布，缓存在内存中。

    clahe=True 时先对原图做 CLAHE 再裁剪；训练与推理必须保持一致。
    """
    cache = {}
    for k in sorted(meta.keys()):
        split, key = k.split("/")
        p = os.path.join(ROOT, split, "%s.jpg" % key)
        img = imread_u(p)
        if clahe:
            img = apply_clahe(img, clip=clahe_clip)
        cache[k] = fov_crop(img, meta[k])
    if verbose:
        print("已缓存 %d 张画布图像 (%dx%d)" % (len(cache), CANVAS, CANVAS))
    return cache


_CACHE_STORE = {}


def get_cache(meta, clahe=False, clahe_clip=3.0):
    """进程内单例缓存。

    DataLoader 用多进程时，缓存不能随 Dataset 一起 pickle（几百 MB），
    因此让每个 worker 首次取数时按同一 key 自行构建，之后常驻复用。
    """
    key = (bool(clahe), float(clahe_clip), CANVAS)
    if key not in _CACHE_STORE:
        _CACHE_STORE[key] = build_cache(meta, verbose=False, clahe=clahe,
                                        clahe_clip=clahe_clip)
    return _CACHE_STORE[key]


def load_gt():
    import pandas as pd
    gt = pd.read_csv(os.path.join(ROOT, "fovea_localization_train_GT.csv"))
    gt["key"] = gt["data"].astype(int)
    return gt


# ----------------------------------------------------------------------------
# 数据增强
# ----------------------------------------------------------------------------
def affine_aug(img, pt, rng, max_rot=20.0, scale=(0.90, 1.10), trans=0.04, flip=True,
               wh=None):
    """在画布空间做 旋转/缩放/平移/水平翻转，同步变换关键点。

    wh=(w,h) 时同步缩放框边长（框近正方，旋转影响可忽略，只缩放）。
    返回 (img, pt, wh_out)；wh 为 None 时 wh_out 也是 None。
    """
    S = img.shape[0]
    c = (S - 1) / 2.0
    if flip and rng.random() < 0.5:
        img = img[:, ::-1].copy()
        pt = (S - 1 - pt[0], pt[1])
    ang = rng.uniform(-max_rot, max_rot) * math.pi / 180.0
    sc = rng.uniform(*scale)
    tx = rng.uniform(-trans, trans) * S
    ty = rng.uniform(-trans, trans) * S
    a, b = sc * math.cos(ang), sc * math.sin(ang)
    M = np.array([[a, b, (1 - a) * c - b * c + tx],
                  [-b, a, b * c + (1 - a) * c + ty]], np.float32)
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    x, y = pt
    px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    wh_out = None if wh is None else (wh[0] * sc, wh[1] * sc)
    return img, (px, py), wh_out


def photo_aug(img, rng):
    """光度增强：亮度/对比度/gamma/通道增益/模糊/噪声。

    用 256 级查找表一次性完成 gamma+亮度+对比度+颜色增益，避免多次全图浮点运算。
    """
    v = np.arange(256, dtype=np.float32) / 255.0
    g = rng.uniform(0.75, 1.35) if rng.random() < 0.6 else 1.0
    base = np.power(v, g) * 255.0
    if rng.random() < 0.9:
        base = base * rng.uniform(0.75, 1.30)                  # 亮度
    if rng.random() < 0.9:
        mv = base.mean()
        base = (base - mv) * rng.uniform(0.75, 1.30) + mv       # 对比度
    lut = np.empty((256, 1, 3), np.uint8)
    for c in range(3):
        gain = rng.uniform(0.88, 1.14) if rng.random() < 0.7 else 1.0
        lut[:, 0, c] = np.clip(base * gain, 0, 255).astype(np.uint8)
    out = cv2.LUT(img, lut)
    if rng.random() < 0.30:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.30:
        f = out.astype(np.float32)
        f += np.random.randn(*f.shape).astype(np.float32) * rng.uniform(1.0, 8.0)
        out = np.clip(f, 0, 255).astype(np.uint8)
    return out


_MEAN_T = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(NORM_STD).view(1, 3, 1, 1)
_NORM_DEV = {}


def set_norm(mean, std):
    """切换归一化统计量（CLAHE 与非 CLAHE 画布的分布不同，必须各用各的）。"""
    global NORM_MEAN, NORM_STD, _MEAN_T, _STD_T
    NORM_MEAN = tuple(float(v) for v in mean)
    NORM_STD = tuple(float(v) for v in std)
    _MEAN_T = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
    _STD_T = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    _NORM_DEV.clear()


def cache_stats(cache):
    """统计画布集合的逐通道均值/标准差。"""
    acc = np.zeros(3); acc2 = np.zeros(3); n = 0
    for img in cache.values():
        x = img.astype(np.float64) / 255.0
        acc += x.reshape(-1, 3).mean(0) * x[..., 0].size
        acc2 += (x ** 2).reshape(-1, 3).mean(0) * x[..., 0].size
        n += x[..., 0].size
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-8))
    return tuple(mean), tuple(std)


def normalize_batch(u8_hwc):
    """(B,H,W,3) uint8 GPU 张量 -> 归一化后的 (B,3,H,W) float32。

    注意: permute 后必须 .contiguous()。否则张量带 NHWC 步长，
    cuDNN 会退回慢速算法，训练一步从 ~195ms 恶化到 ~757ms。
    """
    key = u8_hwc.device
    if key not in _NORM_DEV:
        _NORM_DEV[key] = (_MEAN_T.to(key), _STD_T.to(key))
    mean, std = _NORM_DEV[key]
    x = u8_hwc.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return (x - mean) / std


def make_heatmap_target(cx, cy, hs, ws, sigma, device):
    yy = torch.arange(hs, device=device, dtype=torch.float32).view(-1, 1)
    xx = torch.arange(ws, device=device, dtype=torch.float32).view(1, -1)
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))


# ----------------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------------
class FoveaNet(nn.Module):
    """timm 主干 + 轻量自顶向下 FPN -> stride-4 单通道热力图。

    stride-4 的精细层改用深度可分离卷积，避免在 256x256 分辨率上做大通道 3x3 卷积。
    """

    def __init__(self, backbone="resnet18", pretrained=True, mid=64, drop=0.0,
                 box_head=False):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained,
                                     features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()          # strides 4,8,16,32
        self.lat = nn.ModuleList([nn.Conv2d(c, mid, 1, bias=False) for c in chs])
        self.sm = nn.ModuleList([
            nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for _ in chs])
        self.fine = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.head = nn.Conv2d(mid, 1, 1)
        nn.init.constant_(self.head.bias, -2.0)
        # 黄斑框回归头（赛题提示的两阶段路线）：输出 (cx,cy,log w,log h)，按画布边长归一化
        self.box_head = None
        if box_head:
            self.box_head = nn.Sequential(
                nn.Linear(mid, mid), nn.ReLU(inplace=True),
                nn.Linear(mid, 4))
            nn.init.zeros_(self.box_head[-1].weight)
            nn.init.zeros_(self.box_head[-1].bias)

    def forward(self, x, return_box=False):
        f = self.enc(x)                                 # strides 4,8,16,32
        p = self.sm[3](self.lat[3](f[3]))
        for i in (2, 1):
            p = F.interpolate(p, size=f[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.sm[i](p + self.lat[i](f[i]))
        p = F.interpolate(p, size=f[0].shape[-2:], mode="bilinear", align_corners=False)
        p = self.fine(p + self.lat[0](f[0]))
        heat = self.head(self.drop(p))                   # stride 4, 1 通道 logits
        if not return_box:
            return heat
        return heat, self.box_head(p.mean(dim=(2, 3)))


def box_loss(pred, target, w_xy=1.0, w_wh=1.0):
    """pred/target: (B,4) 归一化 (cx,cy,log w,log h)。"""
    l_xy = F.l1_loss(pred[:, :2], target[:, :2])
    l_wh = F.l1_loss(pred[:, 2:], target[:, 2:])
    return w_xy * l_xy + w_wh * l_wh


def box_decode(box, S=None):
    """(B,4) 归一化 -> (B,2) 画布坐标的中心。"""
    S = float(S if S is not None else CANVAS)
    return torch.stack([box[:, 0] * S, box[:, 1] * S], dim=1)


def soft_argmax(heat, beta=20.0):
    """heat: (B,1,h,w) 经 sigmoid 的概率图 -> (B,2) 热力图格子坐标。"""
    B, _, h, w = heat.shape
    p = torch.softmax((heat * beta).view(B, -1), dim=-1).view(B, 1, h, w)
    xs = torch.arange(w, device=heat.device, dtype=torch.float32)
    ys = torch.arange(h, device=heat.device, dtype=torch.float32)
    ex = (p.sum(dim=2) * xs).sum(dim=-1).squeeze(-1)     # (B,)
    ey = (p.sum(dim=3) * ys).sum(dim=-1).squeeze(-1)     # (B,)
    return torch.stack([ex, ey], dim=1)


def hard_argmax_refine(heat):
    """argmax + 局部质心细化（推理期的稳健替代方案）。"""
    B, _, h, w = heat.shape
    hm = heat.detach()
    flat = hm.view(B, -1)
    idx = flat.argmax(dim=1)
    iy = (idx // w).float()
    ix = (idx % w).float()
    out = torch.stack([ix, iy], dim=1)
    r = 3
    for b in range(B):
        cx, cy = int(ix[b].item()), int(iy[b].item())
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        win = hm[b, 0, y0:y1, x0:x1].clamp(min=0)
        s = win.sum()
        if s > 1e-6:
            xs = torch.arange(x0, x1, device=heat.device, dtype=torch.float32)
            ys = torch.arange(y0, y1, device=heat.device, dtype=torch.float32)
            out[b, 0] = (win.sum(0) * xs).sum() / s
            out[b, 1] = (win.sum(1) * ys).sum() / s
    return out


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class FoveaDataset(torch.utils.data.Dataset):
    def __init__(self, keys, cache, meta, gt_map, train, sigma, seed=0,
                 max_rot=20.0, scale=(0.90, 1.10), trans=0.04, no_aug=False,
                 clahe=False, clahe_clip=3.0, box=None):
        self.keys = list(keys)
        self.cache = cache          # None -> 取数时按 clahe 配置惰性构建(供 worker 用)
        self.clahe = bool(clahe)
        self.clahe_clip = float(clahe_clip)
        self.meta = meta
        self.gt = gt_map
        self.box = box          # {key: (x0,y0,x1,y1)} 原图坐标；None 表示不用框监督
        # 记录画布边长：Windows 上 DataLoader 用 spawn 启动子进程，子进程会重新导入
        # common.py，CANVAS 回到默认 1024，导致 worker 构建出错误尺寸的画布缓存。
        # 把画布随 Dataset 一起 pickle，并在取数时恢复，才能保证 canvas != 1024 可用。
        self.canvas = int(CANVAS)
        self.train = train
        self.no_aug = no_aug
        self.sigma = sigma
        self.max_rot = max_rot
        self.scale = scale
        self.trans = trans
        self.rng = random.Random(seed)
        self.hs = CANVAS // STRIDE

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if CANVAS != self.canvas:          # spawn 出来的 worker 里恢复画布边长
            set_canvas(self.canvas)
        if self.cache is None:
            self.cache = get_cache(self.meta, self.clahe, self.clahe_clip)
        k = self.keys[i]
        img = self.cache[k]
        m = self.meta[k]
        gx, gy = self.gt[k]
        cx, cy = to_canvas(m, gx, gy)
        # 该图的框（画布坐标）
        bwh = None
        bcx, bcy = cx, cy
        if self.box is not None:
            if k in self.box:
                bx0, by0, bx1, by1 = self.box[k]
                bcx, bcy = to_canvas(m, 0.5 * (bx0 + bx1), 0.5 * (by0 + by1))
                sc = CANVAS / float(m["si"])
                bwh = ((bx1 - bx0) * sc, (by1 - by0) * sc)
            else:
                bwh = (0.19 * CANVAS, 0.19 * CANVAS)
        if self.train and not self.no_aug:
            img, (cx, cy), bwh2 = affine_aug(img, (cx, cy), self.rng,
                                             max_rot=self.max_rot, scale=self.scale,
                                             trans=self.trans, wh=bwh)
            if self.box is not None:
                # 框中心跟随增强后的关键点；边长随缩放变化
                # （旋转对近正方框影响可忽略，故只缩放）
                if bwh is not None:
                    bcx, bcy = cx, cy
                    bwh = bwh2
                else:
                    bcx, bcy = cx, cy
            img = photo_aug(img, self.rng)
        # 以 uint8 HWC 返回，归一化放到 GPU 上做，减少 CPU 开销
        t = torch.from_numpy(np.ascontiguousarray(img))
        hx, hy = cx / STRIDE, cy / STRIDE
        out = dict(img=t, hm=(hx, hy), key=k,
                   canvas=(cx, cy), orig=(gx, gy))
        if self.box is not None:
            out["box"] = (bcx, bcy, bwh[0], bwh[1])
        return out


class HeatmapCollate:
    """可 pickle 的 collate（闭包不能 pickle，DataLoader 多进程会失败）。"""

    def __init__(self, sigma, device="cpu"):
        self.sigma = float(sigma)
        self.device = device if isinstance(device, str) else str(device)

    def __call__(self, batch):
        imgs = torch.stack([b["img"] for b in batch])
        hms = []
        for b in batch:
            hx, hy = b["hm"]
            hms.append(make_heatmap_target(hx, hy, CANVAS // STRIDE, CANVAS // STRIDE,
                                           self.sigma, self.device))
        hm = torch.stack(hms).unsqueeze(1)
        out = dict(img=imgs, hm=hm,
                   key=[b["key"] for b in batch],
                   canvas=torch.from_numpy(np.array([b["canvas"] for b in batch], np.float32)),
                   orig=np.array([b["orig"] for b in batch], np.float32))
        if "box" in batch[0]:
            S = float(CANVAS)
            bt = []
            for b in batch:
                bcx, bcy, bw, bh = b["box"]
                bt.append([bcx / S, bcy / S, math.log(max(bw, 1e-3) / S),
                           math.log(max(bh, 1e-3) / S)])
            out["box"] = torch.tensor(bt, dtype=torch.float32)
        return out


def make_collate(sigma, device):
    return HeatmapCollate(sigma, device)


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def load_gt_map(gt):
    return {("train/%04d" % r.key): (float(r.Fovea_X), float(r.Fovea_Y))
            for r in gt.itertuples()}


def load_box_map(meta=None):
    """读 train_location/*.xml 的 Fovea 框 -> {key: (x0,y0,x1,y1)}（原图坐标）。

    赛题提示"先检测黄斑区域"，且框中心与坐标真值的一致性达 2.4 px，
    故把框作为额外监督信号。
    """
    meta = meta if meta is not None else load_meta()
    d = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "train_location", "*.xml"))):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        for obj in root.findall("object"):
            if obj.findtext("name", "").strip().lower() != "fovea":
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            x0 = float(bb.findtext("xmin")); y0 = float(bb.findtext("ymin"))
            x1 = float(bb.findtext("xmax")); y1 = float(bb.findtext("ymax"))
            base = os.path.splitext(os.path.basename(f))[0]
            k = "train/%s" % base
            if k in meta:
                d[k] = (x0, y0, x1, y1)
            break
    return d


def box_to_canvas(m, box):
    """原图框 -> 画布坐标的 (cx, cy, w, h)。"""
    x0, y0, x1, y1 = box
    cx, cy = to_canvas(m, 0.5 * (x0 + x1), 0.5 * (y0 + y1))
    # 归一化后各向同性缩放：用面积比的平方根近似边长尺度
    sc = CANVAS / float(m["si"])
    return cx, cy, (x1 - x0) * sc, (y1 - y0) * sc


def report_rmse(pred_orig, true_orig, keys=None, verbose=True):
    """官方口径 RMSE：把 X/Y 拆成相邻两行后端到端计算。"""
    e = (pred_orig - true_orig)
    rmse = float(np.sqrt((e ** 2).mean()))
    eucl = np.sqrt((e ** 2).sum(1))
    if verbose:
        print("  RMSE=%.2f px | 欧氏距离 mean=%.2f med=%.2f p90=%.2f max=%.2f"
              % (rmse, eucl.mean(), np.median(eucl),
                 np.percentile(eucl, 90), eucl.max()))
        if keys is not None and len(keys) == len(eucl):
            order = np.argsort(-eucl)
            for j in order[:5]:
                print("     最差: %s  预测=(%.1f,%.1f) 真值=(%.1f,%.1f) 误差=%.1f"
                      % (keys[j], pred_orig[j, 0], pred_orig[j, 1],
                         true_orig[j, 0], true_orig[j, 1], eucl[j]))
    return rmse


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)