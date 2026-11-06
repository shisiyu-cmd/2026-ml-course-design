"""
Step 2: Train ResNet34 with pseudo-labeled test images.
Pseudo labels: confidence >= 0.80, ~3008 images.
5-fold CV on real labels only; pseudo labels always in training set.
"""
import os, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
from PIL import Image
from sklearn.model_selection import StratifiedKFold

BASE = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
TRAIN_DIR = os.path.join(BASE, "train")
OUT_DIR  = r"C:\Users\古恩宇\Downloads\2026task2"
CKPT_DIR = os.path.join(OUT_DIR, "folds_pseudo")
os.makedirs(CKPT_DIR, exist_ok=True)
LABEL_MAP = {"Angry":0,"Fear":1,"Happy":2,"Sad":3,"Surprise":4,"Neutral":5}
SEED=42; BATCH=64; EPOCHS=30; LR=2e-4; N_FOLDS=5; IMG=112; PSEUDO_TH=0.80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def load_gray(p):
    with Image.open(p) as im: return np.asarray(im.convert("L"), dtype=np.uint8)

# Load real training data
print("loading real train images...", flush=True)
real_imgs, real_lbls = [], []
for folder, lbl in LABEL_MAP.items():
    d = os.path.join(TRAIN_DIR, folder)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg",".png")):
            real_imgs.append(load_gray(os.path.join(d,f))); real_lbls.append(lbl)
real_imgs = np.stack(real_imgs); real_lbls = np.array(real_lbls, dtype=np.int64)
print(f"  real: {real_imgs.shape}", flush=True)

# Load pseudo labels
test_arr = np.load(os.path.join(OUT_DIR, "test_arr.npy"))
test_conf = np.load(os.path.join(OUT_DIR, "test_conf.npy"))
test_pred = np.load(os.path.join(OUT_DIR, "test_pred.npy"))
mask = test_conf >= PSEUDO_TH
pseudo_imgs = test_arr[mask]
pseudo_lbls = test_pred[mask]
print(f"  pseudo: {pseudo_imgs.shape} (conf>={PSEUDO_TH}), class dist:", flush=True)
for c in range(6):
    print(f"    class{c}: {(pseudo_lbls==c).sum()}", flush=True)

# Augment: flip pseudo labels (horizontal flip preserves emotion)
print("augmenting pseudo labels with horizontal flip...", flush=True)
pseudo_flip = pseudo_imgs[:, :, ::-1].copy()
pseudo_imgs = np.concatenate([pseudo_imgs, pseudo_flip], axis=0)
pseudo_lbls = np.concatenate([pseudo_lbls, pseudo_lbls], axis=0)
print(f"  pseudo after flip augmentation: {pseudo_imgs.shape}", flush=True)

mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG,IMG)), transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(), transforms.Normalize(mean,std),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.15))])
eval_tf = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor(), transforms.Normalize(mean,std)])

class DS(Dataset):
    def __init__(s, im, lb, tf): s.im=im; s.lb=lb; s.tf=tf
    def __len__(s): return len(s.im)
    def __getitem__(s,i): return s.tf(Image.fromarray(s.im[i]).convert("RGB")), int(s.lb[i])

def build():
    m = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 6)
    return m.to(DEVICE)

def mixcut(x,y):
    r=random.random()
    if r<0.4: return x,y,y,1.0
    elif r<0.7:
        lam=np.random.beta(0.3,0.3); idx=torch.randperm(x.size(0),device=x.device)
        return lam*x+(1-lam)*x[idx], y, y[idx], lam
    else:
        lam=np.random.beta(0.3,0.3); idx=torch.randperm(x.size(0),device=x.device)
        H,W=x.size(2),x.size(3); cut=np.sqrt(1-lam); cw,ch=int(W*cut),int(H*cut)
        cx,cy=np.random.randint(0,W),np.random.randint(0,H)
        x1,x2=max(0,cx-cw//2),min(W,cx+cw//2); y1,y2=max(0,cy-ch//2),min(H,cy+ch//2)
        xm=x.clone(); xm[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
        return xm, y, y[idx], 1-(x2-x1)*(y2-y1)/(W*H)

crit = nn.CrossEntropyLoss(label_smoothing=0.1)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
accs=[]

for fold,(tri,vai) in enumerate(skf.split(real_imgs, real_lbls),1):
    print(f"\n--- PSEUDO FOLD {fold}/{N_FOLDS} ---", flush=True)
    # Real training subset + ALL pseudo labels
    tr_imgs = np.concatenate([real_imgs[tri], pseudo_imgs], axis=0)
    tr_lbls = np.concatenate([real_lbls[tri], pseudo_lbls], axis=0)
    va_imgs = real_imgs[vai]; va_lbls = real_lbls[vai]
    print(f"  train: {len(tr_imgs)} (real {len(tri)} + pseudo {len(pseudo_imgs)}), val: {len(va_imgs)}", flush=True)

    trl=DataLoader(DS(tr_imgs,tr_lbls,train_tf),batch_size=BATCH,shuffle=True,num_workers=0)
    val=DataLoader(DS(va_imgs,va_lbls,eval_tf),batch_size=BATCH,shuffle=False,num_workers=0)
    model=build(); opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best=0.0; ckpt=os.path.join(CKPT_DIR,f"fold{fold}.pt")
    for ep in range(1,EPOCHS+1):
        model.train(); tl=tn=0
        for x,y in trl:
            x=x.to(DEVICE); y=y.to(DEVICE)
            x,ya,yb,lam=mixcut(x,y)
            logits=model(x); loss=lam*crit(logits,ya)+(1-lam)*crit(logits,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()*x.size(0); tn+=x.size(0)
        sch.step()
        model.eval(); vc=vn=0
        with torch.no_grad():
            for x,y in val:
                x=x.to(DEVICE); y=y.to(DEVICE)
                vc+=(model(x).argmax(1)==y).sum().item(); vn+=x.size(0)
        acc=vc/vn
        if acc>best: best=acc; torch.save(model.state_dict(),ckpt)
        if ep%5==0 or ep==1:
            print(f"  ep{ep:02d} tr{tl/tn:.3f} va{acc:.4f} best{best:.4f}", flush=True)
    accs.append(best); print(f"  PSEUDO fold{fold} best={best:.4f}", flush=True)

print(f"\n===== Pseudo ResNet34: {[round(a,4) for a in accs]} mean={np.mean(accs):.4f} =====", flush=True)
