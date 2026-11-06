# =============================================================================
# Kaggle Notebook: 任务4 医学影像分割 (视网膜血管分割)
# 在 Kaggle Notebook 中运行：Settings -> Accelerator: GPU (T4/P100), Internet: ON
# 预计运行时间：约 2.5 小时（T4）
# =============================================================================

# %% [Cell 1] 环境准备
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "segmentation_models_pytorch", "albumentations"], check=True)
print("deps installed")

# %% [Cell 2] 数据路径自动探测
import os, glob

def _looks_like_image_dir(d):
    if not os.path.isdir(d):
        return False
    try:
        fs = os.listdir(d)
    except Exception:
        return False
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sum(1 for f in fs if f.lower().endswith(exts)) >= 5

def find_data_root(verbose=True):
    """在 /kaggle/input 下自动定位包含 train/image + test/image 的根目录。"""
    roots = []
    for c in ["/kaggle/input/2026task-4/segmentation",
              "/kaggle/input/2026task-4",
              "/kaggle/input/medical-image-segmentation"]:
        if _looks_like_image_dir(os.path.join(c, "train", "image")):
            roots.append(c)

    if not roots:
        base = "/kaggle/input"
        for dirpath, dirnames, filenames in os.walk(base):
            depth = dirpath[len(base):].count(os.sep)
            if depth > 5:
                dirnames[:] = []
                continue
            if os.path.basename(dirpath) == "image":
                parent = os.path.dirname(dirpath)
                root = os.path.dirname(parent)
                if (os.path.basename(parent) == "train"
                        and _looks_like_image_dir(dirpath)
                        and _looks_like_image_dir(os.path.join(root, "test", "image"))):
                    roots.append(root)

    if not roots and verbose:
        print("!! 未找到数据。当前 /kaggle/input 结构：")
        base = "/kaggle/input"
        if not os.path.isdir(base):
            print(f"   {base} 不存在 —— 请先在右侧 Add Input 添加竞赛数据集")
        else:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = dirpath[len(base):].count(os.sep)
                if depth > 3:
                    dirnames[:] = []
                    continue
                imgs = [f for f in filenames if f.lower().endswith((".jpg", ".png", ".tif"))]
                print(f"   {dirpath}  [{len(filenames)} files, {len(imgs)} images]")
        raise FileNotFoundError(
            "找不到 train/image 与 test/image。\n"
            "请在右侧 Add Input 添加竞赛数据 2026task-4。")

    roots = sorted(set(roots))
    root = roots[0]
    if verbose:
        print("DATA_ROOT =", root)
        if len(roots) > 1:
            print("  (其他候选:", roots[1:], ")")
    return root

DATA_ROOT = find_data_root()
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "image")
TRAIN_LBL = os.path.join(DATA_ROOT, "train", "label")
TEST_IMG  = os.path.join(DATA_ROOT, "test", "image")

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    names = sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)],
                   key=lambda s: int(os.path.splitext(s)[0]) if os.path.splitext(s)[0].isdigit() else s)
    return [(os.path.splitext(n)[0], os.path.join(folder, n)) for n in names]

_tr, _te = list_images(TRAIN_IMG), list_images(TEST_IMG)
print(f"train images: {len(_tr)}  test images: {len(_te)}")
print("  train ids:", [s for s, _ in _tr][:5], "...")
print("  label dir exists:", os.path.isdir(TRAIN_LBL),
      "| n labels:", len(os.listdir(TRAIN_LBL)) if os.path.isdir(TRAIN_LBL) else 0)
assert len(_tr) > 0 and len(_te) > 0, "数据为空，检查 Add Input"

# ---- 可选：外部公开数据集作为额外训练数据（CHASE_DB1 + HRF）----
# 需要先把 ext-data 作为 Dataset 添加到 Add Input，结构为：
#   ext-data/chase/input/*  ext-data/chase/label/*
#   ext-data/hrf/input/*    ext-data/hrf/label/*
USE_EXTERNAL = True

def find_external_dirs(verbose=True):
    found = {}
    for name in ("chase", "hrf"):
        for base in ("/kaggle/input/ext-data", "/kaggle/input",
                     "/kaggle/input/ext-data/ext_data"):
            cand = os.path.join(base, name)
            if os.path.isdir(os.path.join(cand, "input")):
                found[name] = cand
                break
        else:
            # 递归兜底
            for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
                if os.path.basename(dirpath) == name and os.path.isdir(os.path.join(dirpath, "input")):
                    found[name] = dirpath
                    break
    if verbose:
        print("external datasets:", {k: v for k, v in found.items()} or "无（将只用比赛数据）")
    return found

EXT_DIRS = find_external_dirs() if USE_EXTERNAL else {}

def list_external_pairs(verbose=True):
    """返回 [(stem|labelpath, image_path)]，标签用 '|' 编码在 stem 里。"""
    out = []
    for name, d in EXT_DIRS.items():
        idir, ldir = os.path.join(d, "input"), os.path.join(d, "label")
        if not (os.path.isdir(idir) and os.path.isdir(ldir)):
            continue
        labels = {}
        for f in os.listdir(ldir):
            stem = os.path.splitext(f)[0]
            for suf in ("_1stHO", "_2ndHO", "_manual1", "_manual"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            labels[stem] = os.path.join(ldir, f)
        n = 0
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue
            stem = os.path.splitext(f)[0]
            if stem in labels:
                out.append((f"{name}_{stem}|{labels[stem]}", os.path.join(idir, f)))
                n += 1
        if verbose:
            print(f"  external {name}: {n} labelled pairs")
    return out

EXT_PAIRS = list_external_pairs()
print(f"external training images: {len(EXT_PAIRS)}")

# %% [Cell 3] 配置 + 工具
import json, time, random
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
from PIL import Image

cv2.setNumThreads(0)

WORK = "/kaggle/working"
os.makedirs(WORK, exist_ok=True)

SIZE_BASE = 512     # 基础训练分辨率
SIZE_FT   = 768     # 微调分辨率（关键：提升细血管）
BATCH     = 4
N_SPLITS  = 4
SEED      = 42
# 多 seed 集成：每个 seed 重新划分 KFold，得到 4 个互补模型。
# SEEDS=[42] 即 v3 单 seed（4 模型集成）；[42,7] 为 8 模型集成。
# 每个 seed 多花约 2.8 小时，注意 Kaggle GPU 配额（30h/周）。
SEEDS     = [42]
# 集成架构列表：(arch, encoder, tag, 目标步数@512, 目标步数@768)
#
# 为什么用「步数」而不是「epoch」：训练集大小会变（15 张 vs 加了外部数据 53 张），
# 固定 epoch 会让计算量差 3.5 倍。按步数预算则总计算量恒定。
#   15 张 @bs4 = 4 步/epoch  -> 1200 步 = 300 epoch
#   53 张 @bs4 = 14 步/epoch -> 1200 步 = 86 epoch
#
# ⚠️ 实测记录：
#   v3 = 本表的 effb4 用 250ep/90ep 固定（15张时约 1000/720 步）-> 0.85143
#   v5 = 加了等权的 mit_b2 且把主模型降到 900/600 步     -> 0.85096（退步）
# 结论：主模型训练量不能降；弱架构等权混入会拉低总分。
MODEL_CFGS = [
    ("UnetPlusPlus", "efficientnet-b4", "upp_effb4",  1200, 850),   # 唯一启用，约 5.3h
    # 第二架构默认关闭。若要启用 Transformer 多样性，需先把下面的按 OOF 加权打开，
    # 否则等权平均会被较弱模型拖累。
    # ("Unet",        "mit_b2",          "unet_mitb2",  700, 450),
]
MIN_EPOCHS = 25          # 防止数据量大时 epoch 被压得太少
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

seed_everything(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# 注意：cv2.imread/imwrite 在含非 ASCII 的路径下会静默失败，统一用下面两个封装
def imread_u(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"imdecode failed: {path}")
    return img

def imwrite_u(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    buf.tofile(path)
    return os.path.exists(path)

def apply_clahe(rgb, clip=2.0, grid=8):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def fov_mask(rgb, thresh=10):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(gray.shape, np.uint8) * 255
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    return filled

def dice_score(pred, gt, eps=1e-7):
    p = pred.astype(np.float32).ravel(); g = gt.astype(np.float32).ravel()
    return float((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))

# %% [Cell 4] Dataset + 增强
def get_aug(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.20), translate_percent=(-0.06, 0.06),
                 rotate=(-45, 45), shear=(-8, 8),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.75),
        A.RandomResizedCrop(size=(size, size), scale=(0.60, 1.0), ratio=(0.85, 1.18), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

class VesselDataset(Dataset):
    def __init__(self, pairs, size, train=True, aug=None):
        self.pairs, self.size, self.train, self.aug = pairs, size, train, aug
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        stem, ipath = self.pairs[i]
        # 外部数据集把标签绝对路径编码在 stem 里，用 '|' 分隔
        name = stem.split("|", 1)[0] if "|" in stem else stem
        lpath = stem.split("|", 1)[1] if "|" in stem else os.path.join(TRAIN_LBL, stem + ".jpg")
        img = imread_u(ipath, cv2.IMREAD_COLOR)
        img = apply_clahe(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        msk = None
        if self.train:
            m = imread_u(lpath, cv2.IMREAD_GRAYSCALE)
            msk = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            msk = (msk > 127).astype(np.float32)
            if self.aug is not None:
                o = self.aug(image=img, mask=(msk * 255).astype(np.uint8))
                img, msk = o["image"], (o["mask"] > 127).astype(np.float32)
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        x = torch.from_numpy(x)
        if msk is None:
            return x, name
        return x, torch.from_numpy(msk)[None], name

# %% [Cell 5] 损失函数：BCE + Dice + soft-clDice（clDice 对细血管连通性很关键）
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth = smooth
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(2, 3)) + self.smooth
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        return 1 - (num / den).mean()

class SoftCLDice(nn.Module):
    def __init__(self, iters=10, smooth=1.0):
        super().__init__(); self.iters = iters; self.smooth = smooth
    def _skel(self, x):
        x = x.clamp(1e-6, 1 - 1e-6)
        for _ in range(self.iters):
            mx = F.max_pool2d(x, 3, 1, 1)
            mn = -F.max_pool2d(-x, 3, 1, 1)
            x = (mx + mn - mx * mn) * (1 - mx * mn)
            x = x.clamp(1e-6, 1 - 1e-6)
        return x
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        sp, st = self._skel(p), self._skel(target)
        tprec = (sp * target).sum(dim=(2, 3)) / (sp.sum(dim=(2, 3)) + self.smooth)
        tsens = (st * p).sum(dim=(2, 3)) / (st.sum(dim=(2, 3)) + self.smooth)
        return 1 - (2 * tprec * tsens / (tprec + tsens + 1e-6)).mean()

class CombinedLoss(nn.Module):
    """BCE + Dice + clDice + 细血管加权项。

    诊断结论（在真实标签上实测）：细血管(<3px)召回仅 0.599，粗血管 0.917，
    而细血管占 GT 的 25% —— 这是 Dice 卡在 0.85 的根本原因。
    因此这里做两件事：
      1. clDice 权重从 0.2 提到 0.4（它直接优化中心线连通性）；
      2. 加一个 thin-vessel 加权 BCE：用形态学开运算把 GT 拆成粗细两部分，
         对细血管像素额外加权，逼迫模型去拟合末梢。
    """
    def __init__(self, bce_w=0.30, dice_w=0.35, cl_w=0.35, pos_weight=2.0,
                 thin_w=0.15, thin_boost=3.0):
        super().__init__()
        self.bce_w, self.dice_w, self.cl_w = bce_w, dice_w, cl_w
        self.thin_w, self.thin_boost = thin_w, thin_boost
        self.dice, self.cldice = DiceLoss(), SoftCLDice()
        self.register_buffer("pw", torch.tensor([pos_weight]))

    @staticmethod
    def _thin_mask(target):
        """GT 中细血管的软掩膜：开运算去掉粗血管，保留末梢。"""
        # target: (B,1,H,W) in {0,1}
        thick = -F.max_pool2d(-F.max_pool2d(target, 3, 1, 1), 3, 1, 1)  # erosion
        thick = F.max_pool2d(-F.max_pool2d(-thick, 3, 1, 1), 3, 1, 1)   # dilation
        thin = (target - thick).clamp(0, 1)
        return thin

    def forward(self, logits, target):
        loss = self.bce_w * F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pw)
        loss = loss + self.dice_w * self.dice(logits, target)
        loss = loss + self.cl_w * self.cldice(logits, target)

        if self.thin_w > 0:
            thin = self._thin_mask(target)
            w = 1.0 + self.thin_boost * thin
            bce_thin = F.binary_cross_entropy_with_logits(logits, target, weight=w)
            loss = loss + self.thin_w * bce_thin
        return loss

# %% [Cell 6] 训练 / 验证
def build_model(arch="UnetPlusPlus", encoder="efficientnet-b4"):
    """smp 模型工厂。架构/编码器参数化，便于集成不同归纳偏置的模型。

    注意 UnetPlusPlus 不支持 mit_* (SegFormer) 编码器，那种组合要用 Unet。
    """
    cls = getattr(smp, arch)
    return cls(encoder_name=encoder, encoder_weights="imagenet",
               in_channels=3, classes=1)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ds = []
    for img, msk, _ in loader:
        img, msk = img.to(DEVICE), msk.to(DEVICE)
        logits = model(img)
        if logits.shape[-2:] != msk.shape[-2:]:
            logits = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
        pr = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[:, 0]
        gt = msk.cpu().numpy()[:, 0]
        ds += [dice_score(p, g) for p, g in zip(pr, gt)]
    return float(np.mean(ds))

@torch.no_grad()
def evaluate_detail(model, loader):
    """Dice + 精确率/召回率 + 细血管召回（诊断用）。

    细血管召回是本任务的核心瓶颈：上一版在真实标签上只有 0.60，
    而细血管占 GT 的 25%。这里把它显式打出来，方便判断改进是否有效。
    """
    model.eval()
    dices, precs, recs, thin_recs, thick_recs = [], [], [], [], []
    for img, msk, _ in loader:
        img, msk = img.to(DEVICE), msk.to(DEVICE)
        logits = model(img)
        if logits.shape[-2:] != msk.shape[-2:]:
            logits = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
        prs = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[:, 0]
        gts = msk.cpu().numpy()[:, 0]
        for pr, gt in zip(prs, gts):
            pr = pr.astype(np.uint8); gt = (gt > 0.5).astype(np.uint8)
            tp = int((pr & gt).sum()); fp = int((pr & ~gt.astype(bool)).sum())
            fn = int((~pr.astype(bool) & gt.astype(bool)).sum())
            dices.append(dice_score(pr, gt))
            precs.append(tp / max(tp + fp, 1))
            recs.append(tp / max(tp + fn, 1))
            # 细/粗血管拆分（与诊断脚本一致：3x3 开运算）
            thick = cv2.morphologyEx(gt, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            thin = gt - thick
            if thin.sum() > 0:
                thin_recs.append((pr & thin).sum() / thin.sum())
            if thick.sum() > 0:
                thick_recs.append((pr & thick).sum() / thick.sum())
    return dict(dice=float(np.mean(dices)), precision=float(np.mean(precs)),
                recall=float(np.mean(recs)),
                thin_recall=float(np.mean(thin_recs)) if thin_recs else 0.0,
                thick_recall=float(np.mean(thick_recs)) if thick_recs else 0.0)

def train_fold(fold, tr_pairs, va_pairs, size, epochs, lr, bs, tag, init_from=None,
               arch="UnetPlusPlus", encoder="efficientnet-b4", save=True):
    model = build_model(arch, encoder).to(DEVICE)
    if init_from and os.path.exists(init_from):
        model.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=False)["model"])
        print(f"  [fold{fold}] init from {os.path.basename(init_from)}")
    crit = CombinedLoss().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, int(np.ceil(len(tr_pairs) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps,
                                                pct_start=0.12, div_factor=10., final_div_factor=100.)
    tr = DataLoader(VesselDataset(tr_pairs, size, True, get_aug(size)), batch_size=bs,
                    shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    va = DataLoader(VesselDataset(va_pairs, size, True), batch_size=bs,
                    shuffle=False, num_workers=2, pin_memory=True)

    best, best_ep, out = 0.0, -1, f"{WORK}/{tag}_f{fold}.pth"
    best_detail = {}
    for ep in range(epochs):
        model.train(); t0 = time.time(); run = []
        for img, msk, _ in tr:
            img, msk = img.to(DEVICE, non_blocking=True), msk.to(DEVICE, non_blocking=True)
            logits = model(img)
            if logits.shape[-2:] != msk.shape[-2:]:
                logits = F.interpolate(logits, size=msk.shape[-2:], mode="bilinear", align_corners=False)
            loss = crit(logits, msk)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); run.append(loss.item())
        vd = evaluate(model, va)
        if (ep + 1) % 10 == 0 or ep == 0:
            d = evaluate_detail(model, va)
            print(f"  [{tag}_f{fold}] ep {ep+1:3d}/{epochs} loss {np.mean(run):.4f} "
                  f"dice {vd:.4f} P {d['precision']:.3f} R {d['recall']:.3f} "
                  f"thinR {d['thin_recall']:.3f} thickR {d['thick_recall']:.3f} "
                  f"{time.time()-t0:.1f}s", flush=True)
        if vd > best:
            best, best_ep = vd, ep
            best_detail = evaluate_detail(model, va)
            if save:
                torch.save({"model": model.state_dict(), "val_dice": vd, "size": size,
                            "arch": arch, "encoder": encoder}, out)
    print(f"  [{tag}_f{fold}] BEST dice {best:.4f} @ep{best_ep+1} | "
          f"P {best_detail.get('precision',0):.3f} R {best_detail.get('recall',0):.3f} "
          f"thinR {best_detail.get('thin_recall',0):.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    return best

# %% [Cell 7] 阶段一：512 训练（模型配置 × seed × 4 折）
# 遍历 MODEL_CFGS 中的所有架构，每个架构都做 4 折交叉验证。
# 这样既能得到每种架构的 CV 估计，又能得到可用于集成的 4 个模型。
from sklearn.model_selection import KFold

pairs = list_images(TRAIN_IMG)
print(f"train images: {len(pairs)}  | external: {len(EXT_PAIRS)}  | folds: {N_SPLITS}")
print(f"configs: {[(a, e) for a, e, _, _, _ in MODEL_CFGS]}")

SEED_SPLITS = {s: list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=s).split(pairs))
               for s in SEEDS}

def epochs_for(target_steps, n_train, bs, min_ep=MIN_EPOCHS):
    """把「目标优化步数」换算成 epoch 数，保证计算量与数据量无关。"""
    steps_per_ep = max(1, int(np.ceil(n_train / bs)))
    return max(min_ep, int(round(target_steps / steps_per_ep))), steps_per_ep

base_scores = {}
for (arch, enc, mtag, steps512, steps768) in MODEL_CFGS:
    for seed in SEEDS:
        # 每个 (架构, seed) 组合开始前重新设种，保证初始化不同
        seed_everything(seed)
        splits = SEED_SPLITS[seed]
        for fold in range(N_SPLITS):
            tr = [pairs[i] for i in splits[fold][0]] + EXT_PAIRS
            va = [pairs[i] for i in splits[fold][1]]
            ep, spe = epochs_for(steps512, len(tr), BATCH)
            if fold == 0:
                print(f"  [{mtag}] 512: {len(tr)} imgs, {spe} steps/ep -> {ep} epochs "
                      f"(~{ep*spe} steps)", flush=True)
            base_scores[f"{mtag}_s{seed}_f{fold}"] = train_fold(
                fold, tr, va, SIZE_BASE, epochs=ep, lr=3e-4, bs=BATCH,
                tag=f"{mtag}512_s{seed}", arch=arch, encoder=enc)
        print(f"{mtag} seed{seed} 512 done", flush=True)
    vals = [v for k, v in base_scores.items() if k.startswith(mtag)]
    print(f"{mtag} 512 CV mean = {float(np.mean(vals)):.4f}", flush=True)

# %% [Cell 8] 阶段二：768 微调（从对应 512 权重初始化）
# 诊断：512 训练时细血管只有 ~1px，下采样后基本消失，细血管召回仅 0.60。
# 768 让细血管保有 1.5~2px，是提升召回的关键一步。
ft_scores = {}
for (arch, enc, mtag, steps512, steps768) in MODEL_CFGS:
    for seed in SEEDS:
        splits = SEED_SPLITS[seed]
        for fold in range(N_SPLITS):
            tr = [pairs[i] for i in splits[fold][0]] + EXT_PAIRS
            va = [pairs[i] for i in splits[fold][1]]
            ep, spe = epochs_for(steps768, len(tr), 2)
            if fold == 0:
                print(f"  [{mtag}] 768: {len(tr)} imgs, {spe} steps/ep -> {ep} epochs "
                      f"(~{ep*spe} steps)", flush=True)
            ft_scores[f"{mtag}_s{seed}_f{fold}"] = train_fold(
                fold, tr, va, SIZE_FT, epochs=ep, lr=1.2e-4, bs=2,
                tag=f"{mtag}768_s{seed}",
                init_from=f"{WORK}/{mtag}512_s{seed}_f{fold}.pth",
                arch=arch, encoder=enc)
        print(f"{mtag} seed{seed} 768 done", flush=True)
    vals = [v for k, v in ft_scores.items() if k.startswith(mtag)]
    print(f"{mtag} 768 CV mean = {float(np.mean(vals)):.4f}", flush=True)

# %% [Cell 9] 推理：全模型集成 + 8 向 TTA
# TTA 集合：4 向翻转 + 180° 旋转。实测 4→8 向 +0.0015，16 向反而略降，
# 所以取 8 向。模型训练时用了 RandomRotate90 和 ±45° 仿射，旋转 TTA 是有效的。
TTA_TFS = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1)]

def _tta_apply(x, k, fh, fv):
    if k:  x = torch.rot90(x, k, dims=[2, 3])
    if fh: x = torch.flip(x, dims=[3])
    if fv: x = torch.flip(x, dims=[2])
    return x

def _tta_invert(x, k, fh, fv):
    if fv: x = torch.flip(x, dims=[2])
    if fh: x = torch.flip(x, dims=[3])
    if k:  x = torch.rot90(x, -k, dims=[2, 3])
    return x

def tta_forward(model, x):
    acc = None
    for tf in TTA_TFS:
        p = _tta_invert(torch.sigmoid(model(_tta_apply(x, *tf))), *tf)
        acc = p if acc is None else acc + p
    return acc / len(TTA_TFS)

@torch.no_grad()
def predict_all(ckpts, pairs, size, bs=2, weights=None):
    """返回原生分辨率(565x584)的概率图。

    关键：概率图必须保持原生分辨率，阈值化之后再缩放。
    若先缩放到 512 再阈值，BILINEAR 插值会稀释细血管的峰值，
    实测 Dice 会从 0.832 掉到 0.744（-0.09）。

    weights: 每个 checkpoint 的集成权重。默认等权；多架构时应传按 OOF 算出的权重，
             避免较弱模型等权拉低总分（v5 踩过这个坑）。
    """
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        # 从 checkpoint 里读架构，支持集成不同架构的模型
        m = build_model(ck.get("arch", "UnetPlusPlus"), ck.get("encoder", "efficientnet-b4"))
        m.load_state_dict(ck["model"])
        models.append(m.to(DEVICE).eval())
    if weights is None:
        weights = [1.0] * len(models)
    wsum = float(np.sum(weights))
    print(f"    loaded {len(models)} models, size={size}, weights="
          f"{[round(w / wsum, 3) for w in weights]}")
    ds = VesselDataset(pairs, size, train=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    # 每张图的原生尺寸
    native = {}
    for stem, path in pairs:
        h, w = imread_u(path, cv2.IMREAD_GRAYSCALE).shape[:2]
        native[stem] = (w, h)
    out = {}
    for img, stems in loader:
        img = img.to(DEVICE)
        acc = None
        for m, w in zip(models, weights):
            p = tta_forward(m, img) * (w / wsum)
            acc = p if acc is None else acc + p
        acc = acc.float().cpu().numpy()[:, 0]
        for st, pr in zip(stems, acc):
            out[st] = cv2.resize(pr, native[st], interpolation=cv2.INTER_LINEAR)
    return out

test_pairs = list_images(TEST_IMG)

def oof_eval(ckpt_list, split_seed):
    """用 split_seed 的划分，得到每张训练图的 OOF 概率（该图未被对应模型见过）。"""
    sp = SEED_SPLITS[split_seed]
    per_img = {}
    for fold in range(N_SPLITS):
        va = [pairs[i] for i in sp[fold][1]]
        pr = predict_all(ckpt_list, va, SIZE_FT)
        for stem, _ in va:
            per_img[stem] = pr[stem]
    return per_img

def _dice_official(per_img, stems=None):
    dices = []
    for stem, _ in (stems or pairs):
        gt_path = os.path.join(TRAIN_LBL, stem + ".jpg")
        g = imread_u(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (g > 127).astype(np.uint8)
        p = per_img[stem]
        if p.shape != gt.shape:
            p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pr = (p >= 0.5).astype(np.uint8)
        # 与官方口径一致：BILINEAR 到 512 再比
        pr512 = (np.asarray(Image.fromarray(pr * 255).resize((512, 512), Image.Resampling.BILINEAR)) > 0).astype(np.uint8)
        gt512 = (np.asarray(Image.fromarray(gt * 255).resize((512, 512), Image.Resampling.BILINEAR)) > 0).astype(np.uint8)
        inter = (pr512 & gt512).sum()
        dices.append(2 * inter / max(pr512.sum() + gt512.sum(), 1))
    return float(np.mean(dices))

# ---- 先按架构算 OOF 分数，据此给每个架构分配权重 ----
# 单架构时权重全为 1（等权，与 v3 一致）；多架构时按 "dice - 常数" 加权，
# 避免较弱架构把总分拖下来。
ARCH_WEIGHTS = {}
try:
    print("\n=== 逐架构 OOF（同一验证划分，均未见过各自验证图）===")
    per_arch = {}
    for (_, _, mtag, _, _) in MODEL_CFGS:
        lst = [f"{WORK}/{mtag}768_s{SEEDS[0]}_f{f}.pth" for f in range(N_SPLITS)]
        per_arch[mtag] = oof_eval(lst, SEEDS[0])
        d = _dice_official(per_arch[mtag])
        print(f"  {mtag:14s} OOF dice {d:.4f}")
        ARCH_WEIGHTS[mtag] = d

    if len(per_arch) > 1:
        base = max(ARCH_WEIGHTS.values())
        for mtag in ARCH_WEIGHTS:
            # 低于最佳架构 0.005 以上的，权重压到 0.3，否则 1.0
            ARCH_WEIGHTS[mtag] = 1.0 if ARCH_WEIGHTS[mtag] >= base - 0.005 else 0.3
        merged = {}
        for stem, _ in pairs:
            num = sum(ARCH_WEIGHTS[m] for m in per_arch)
            merged[stem] = sum(per_arch[m][stem] * ARCH_WEIGHTS[m] for m in per_arch) / num
        print(f"  {'WEIGHTED ENS':14s} {_dice_official(merged):.4f}")
    else:
        ARCH_WEIGHTS = {k: 1.0 for k in ARCH_WEIGHTS}
    print("架构权重:", {k: round(v, 2) for k, v in ARCH_WEIGHTS.items()}, flush=True)
except Exception as e:
    print("OOF 权重计算失败，回退到等权:", e)
    ARCH_WEIGHTS = {mtag: 1.0 for (_, _, mtag, _, _) in MODEL_CFGS}

# ---- 用上述权重做测试集预测 ----
ckpts, ws = [], []
for (_, _, mtag, _, _) in MODEL_CFGS:
    for s in SEEDS:
        for f in range(N_SPLITS):
            ckpts.append(f"{WORK}/{mtag}768_s{s}_f{f}.pth")
            ws.append(ARCH_WEIGHTS.get(mtag, 1.0))
for c in ckpts:
    assert os.path.exists(c), f"缺少权重 {c} —— 训练阶段未完成？"
print(f"\nensembling {len(ckpts)} checkpoints "
      f"({len(MODEL_CFGS)} 架构 x {len(SEEDS)} seed x {N_SPLITS} 折)")
probs = predict_all(ckpts, test_pairs, SIZE_FT, weights=ws)
print("predicted", len(probs), "test images")

# %% [Cell 10] 后处理 + 生成 submission.csv
# 阈值 0.50 + FOV 掩膜（在验证集上搜索得到的稳健值）
THR = 0.50
USE_FOV = True

def rle_encode(mask):
    """与官方 segmentation_to_csv.py 完全一致的 RLE（1-based, 行优先）"""
    flat = np.asarray(mask).ravel()
    dots = np.where(flat > 0)[0]
    if dots.size == 0:
        return ""
    runs, prev = [], -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = b
    return " ".join(str(r) for r in runs)

ids, preds = [], []
for stem, path in test_pairs:
    prob = probs[stem]                      # 原生分辨率概率
    m = (prob >= THR).astype(np.uint8)      # 先在原生分辨率二值化
    if USE_FOV:
        rgb = cv2.cvtColor(imread_u(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        f = fov_mask(rgb)
        if f.shape != m.shape:
            f = cv2.resize(f, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m & (f > 0).astype(np.uint8)
    # 官方 segmentation_to_csv.py 会把 PNG 用 PIL BILINEAR 缩放到 512 再取 >0 编码。
    # 这里必须复刻同一步，才能保证提交的下标与评测方一致。
    im512 = Image.fromarray((m * 255).astype(np.uint8)).convert("L")
    im512 = im512.resize((512, 512), Image.Resampling.BILINEAR)
    m512 = (np.asarray(im512) > 0).astype(np.uint8)
    ids.append(stem)
    preds.append(rle_encode(m512))
    print(f"  {stem}: vessel_frac={m.mean():.4f}")

df = pd.DataFrame({"Id": ids, "Predicted": preds})
df["_o"] = df.Id.astype(int)
df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
df.to_csv(f"{WORK}/submission.csv", index=False)
print("\nwrote submission.csv")
print(df.head(3).to_string()[:300])
