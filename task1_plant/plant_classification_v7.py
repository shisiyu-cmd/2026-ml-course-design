# %% [markdown]
# 任务一：植物分类 —— IN22K 双强模型 5 折 CV + D4-TTA 几何集成（v7 优化版）
# 架构：ConvNeXt-Base(IN22K->IN1K-384) + Swin-Base(IN22K->IN1K-384)，各 5 折
# 优化点：
#  1) 只在内存保留 CPU 权重，GPU 任意时刻只有 1 个模型，杜绝 10 模型常驻导致的 OOM
#  2) timm/HF 大权重下载：socket 超时 + 4 次重试，避免连接挂起卡死
#  3) 分类头热身 1 轮 + 全网络微调 11 轮（含 1 轮 warmup + cosine），合计 12 轮
#  4) RandomResizedCrop 尺度增强（应对不同生长阶段）+ D4 对称增强
#  5) 删除均衡数据上无效的类别权重，改用 label smoothing 0.03
#  6) 几何（对数概率）集成，权重 (Val F1)^3（降幂，避免单折验证噪声主导）
#  7) 提交前三重合法性校验
# 指标：macro-F1

# %%
# ================= 0. 依赖与下载健壮性 =================
import os, glob, random, time, copy, gc, socket
socket.setdefaulttimeout(60)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import timm

# ================= 1. 配置 =================
SEED = 42
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ON = DEVICE == "cuda"
print(f"设备: {DEVICE} | torch {torch.__version__} | timm {timm.__version__}", flush=True)

DATA_ROOT = None
for p in glob.glob("/kaggle/input/**/train", recursive=True):
    DATA_ROOT = os.path.dirname(p); break
assert DATA_ROOT is not None, "未找到 train 目录"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
TMPL_CSV  = glob.glob("/kaggle/input/**/*.csv", recursive=True)[0]

N_FOLDS   = 5
IMG_SIZE  = 384
BATCH     = 8      # 显存安全批次
GRAD_ACC  = 2      # 梯度累积，等效 batch=16
NWORK     = 2
HEAD_EP   = 1      # 分类头热身轮数
FT_EP     = 11     # 全网络微调轮数（合计 12 轮）
WARMUP_EP = 1      # 全网络阶段的线性 warmup 轮数
LABEL_SMOOTH = 0.03
WD = 1e-2
POWER_EXP = 3      # 集成权重 = (Val F1)^POWER_EXP * arch_mult

ARCH_CONFIGS = [
    {"name": "convnext_base.fb_in22k_ft_in1k_384",            "lr": 1e-4, "mult": 1.0, "brief": "ConvNeXt-B"},
    {"name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k","lr": 5e-5, "mult": 1.5, "brief": "Swin-B"},
]

probe = datasets.ImageFolder(TRAIN_DIR)
CLASSES = probe.classes
Y = np.array(probe.targets)
test_files = sorted(glob.glob(TEST_DIR + "/*.png") + glob.glob(TEST_DIR + "/*.jpg"))
tmpl_df = pd.read_csv(TMPL_CSV)
print(f"训练 {len(Y)} 张 | 测试 {len(test_files)} 张 | 类别 {CLASSES}", flush=True)

# %%
# ================= 2. 数据增强 =================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def make_transforms(img_size=IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

# %%
# ================= 3. 模型 / 工具 =================
def build_model(name, ncls, pretrained=True, retries=4):
    """timm 建模，预训练权重下载带超时重试"""
    last = None
    for k in range(retries):
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=ncls)
        except Exception as e:
            last = e
            print(f"[warn] {name} 准备失败({k+1}/{retries}): {type(e).__name__}，重试", flush=True)
            time.sleep(5)
    raise last

def get_head(model):
    return model.head if hasattr(model, "head") else model.get_classifier()

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head(model).parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.amp.autocast("cuda", enabled=AMP_ON):
            out.append(torch.softmax(model(x.to(DEVICE)), 1).cpu())
    return torch.cat(out, dim=0)

class TestDS(Dataset):
    def __init__(self, files, tf):
        self.files = files; self.tf = tf
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))

# %%
# ================= 4. 5 折 × 双架构训练（只保留 CPU 权重） =================
train_tf, eval_tf = make_transforms(IMG_SIZE)
base_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
base_eval_ds  = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(np.zeros(len(Y)), Y))
crit  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

trained = []  # (arch_name, brief, best_f1, mult, cpu_state_dict)
print("\n>>> 开始 5 折 × 双架构训练 ...", flush=True)
t_start = time.time()

for fold, (tr_idx, va_idx) in enumerate(folds):
    tl = DataLoader(Subset(base_train_ds, tr_idx), batch_size=BATCH, shuffle=True,
                    num_workers=NWORK, pin_memory=True)
    vl = DataLoader(Subset(base_eval_ds, va_idx),  batch_size=BATCH, shuffle=False,
                    num_workers=NWORK, pin_memory=True)

    for cfg in ARCH_CONFIGS:
        name, base_lr, mult, brief = cfg["name"], cfg["lr"], cfg["mult"], cfg["brief"]
        seed_everything(SEED)
        model = build_model(name, len(CLASSES)).to(DEVICE)
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ON)

        # ---- 4.1 分类头热身（冻骨干，只训 head）----
        freeze_backbone(model)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=1e-3, weight_decay=1e-4)
        for _ in range(HEAD_EP):
            model.train()
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=AMP_ON):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        # ---- 4.2 全网络微调（warmup + cosine，梯度累积）----
        unfreeze_all(model)
        opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=WD)
        def lr_lambda(ep):
            if ep < WARMUP_EP:                       # 线性 warmup
                return (ep + 1) / (WARMUP_EP + 1)
            prog = (ep - WARMUP_EP) / max(1, FT_EP - WARMUP_EP)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * prog))  # cosine，下限 1%
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        best_f1, best_state = 0.0, None
        for ep in range(FT_EP):
            model.train(); running = 0.0; opt.zero_grad()
            for step, (x, y) in enumerate(tl):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP_ON):
                    loss = crit(model(x), y) / GRAD_ACC
                scaler.scale(loss).backward()
                running += loss.item() * GRAD_ACC
                if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(tl):
                    scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()

            vp = predict_probs(model, vl)
            f1 = f1_score(Y[va_idx], vp.argmax(1).tolist(), average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Fold {fold+1}/{N_FOLDS}] {brief} ep{ep+1}/{FT_EP} "
                  f"loss={running/len(tl):.4f} valF1={f1:.4f} (best {best_f1:.4f})", flush=True)

        # 只保留 CPU 权重，活模型立即释放
        trained.append((name, brief, best_f1, mult, best_state))
        del model, opt, sched, scaler
        gc.collect(); torch.cuda.empty_cache()
        print(f"  >> {brief} Fold{fold+1} 完成 bestF1={best_f1:.4f}", flush=True)

print(f"\n训练总耗时 {(time.time()-t_start)/60:.1f} 分钟")
for cfg in ARCH_CONFIGS:
    vals = [t[2] for t in trained if t[1] == cfg["brief"]]
    print(f"{cfg['brief']} 5折 Val F1 均值 = {np.mean(vals):.4f}", flush=True)

# %%
# ================= 5. D4 8 形态 TTA + 几何对数概率集成 =================
def build_tta(size=IMG_SIZE):
    tfs = []
    for rot in [0, 90, 180, 270]:
        for flip in [False, True]:
            t = [transforms.Resize((size, size))]
            if rot > 0:
                t.append(transforms.Lambda(lambda img, r=rot: img.rotate(r)))
            if flip:
                t.append(transforms.RandomHorizontalFlip(p=1.0))
            t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            tfs.append(transforms.Compose(t))
    return tfs
TTA_TFS = build_tta()

raw_w   = [(f1 ** POWER_EXP) * mult for _, _, f1, mult, _ in trained]
total_w = sum(raw_w)
norm_w  = [w / total_w for w in raw_w]

all_logp = torch.zeros(len(test_files), len(CLASSES))
for (name, brief, f1, mult, state), w in zip(trained, norm_w):
    m = build_model(name, len(CLASSES), pretrained=False).to(DEVICE)
    m.load_state_dict(state); m.eval()
    m_logp = torch.zeros(len(test_files), len(CLASSES))
    with torch.no_grad():
        for tf in TTA_TFS:
            dl = DataLoader(TestDS(test_files, tf), batch_size=BATCH, shuffle=False,
                            num_workers=NWORK, pin_memory=True)
            parts = []
            for x in dl:
                with torch.amp.autocast("cuda", enabled=AMP_ON):
                    parts.append(torch.log(torch.softmax(m(x.to(DEVICE)), 1).cpu() + 1e-8))
            m_logp += torch.cat(parts, dim=0) / len(TTA_TFS)
    all_logp += m_logp * w
    print(f"  集成 {brief} valF1={f1:.4f} 权重={w:.4f}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

preds = all_logp.argmax(dim=1).tolist()

# %%
# ================= 6. 导出提交 + 合法性校验 =================
file_to_label = {os.path.basename(f): CLASSES[p] for f, p in zip(test_files, preds)}
out = tmpl_df.copy()
out["Category"] = out["ID"].map(
    lambda x: file_to_label.get(x, file_to_label.get(os.path.splitext(x)[0] + ".png")))

assert out["Category"].notna().all(), "存在未映射的 ID！"
assert set(out["Category"].unique()) <= set(CLASSES), "存在非法类别标签！"
assert len(out) == len(tmpl_df), "提交行数与模板不一致！"

out.to_csv("/kaggle/working/submission.csv", index=False)
print("\n>>> submission.csv 已生成，行数：", len(out))
print(out["Category"].value_counts())
print("\n=== DONE ===")
