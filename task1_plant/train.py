import os, sys, random, time, argparse, json
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="resnet50", choices=["resnet50","efficientnet_b0","resnet18"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="model.pth")
parser.add_argument("--epochs_head", type=int, default=4)
parser.add_argument("--epochs_ft", type=int, default=12)
parser.add_argument("--img", type=int, default=224)
args = parser.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA = r"C:\Users\古恩宇\Doubao\chats\2026-09-21\new-chat\kaggle\dataset-for-task2\dataset-for-task2"
train_dir = os.path.join(DATA, "train")

mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(args.img, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.3),
])
val_tf = transforms.Compose([
    transforms.Resize((args.img,args.img)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

full = datasets.ImageFolder(train_dir)
n_val = int(len(full)*0.15)
n_train = len(full)-n_val
g = torch.Generator().manual_seed(args.seed)
tr, va = random_split(full, [n_train, n_val], generator=g)
tr.dataset.transform = train_tf
va.dataset.transform = val_tf
# Note: both splits share transform; set per-dataset
tr_ds = datasets.ImageFolder(train_dir, transform=train_tf)
va_ds = datasets.ImageFolder(train_dir, transform=val_tf)
# rebuild splits with proper transforms
tr_idx, va_idx = tr.indices, va.indices
from torch.utils.data import Subset
tr_ds = Subset(datasets.ImageFolder(train_dir, transform=train_tf), tr_idx)
va_ds = Subset(datasets.ImageFolder(train_dir, transform=val_tf), va_idx)

train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
classes = full.classes
print("classes:", classes, "train", len(tr_idx), "val", len(va_idx))

# model
if args.arch == "resnet50":
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    ft_params = [('layer3', None),('layer4', None),('fc', None)]
elif args.arch == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
else:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

def eval_f1():
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x,y in val_loader:
            out = model(x)
            preds += out.argmax(1).tolist()
            gts += y.tolist()
    from sklearn.metrics import f1_score
    try:
        m = f1_score(gts, preds, average="macro")
    except Exception:
        m = sum(p==g for p,g in zip(preds,gts))/len(gts)
    model.train()
    return m

def run_stage(params_to_train, epochs, lr, tag):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        for pat,_ in params_to_train:
            if name.startswith(pat):
                p.requires_grad = True
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0
    for ep in range(epochs):
        t0=time.time()
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        f1 = eval_f1()
        print(f"[{tag}] ep{ep+1}/{epochs} f1={f1:.4f} t={time.time()-t0:.0f}s", flush=True)
        if f1 > best:
            best = f1
            torch.save({"model":model.state_dict(),"classes":classes,"arch":args.arch,"img":args.img}, args.out)
    return best

# Stage1: head only
if args.arch.startswith("resnet"):
    b1 = run_stage([("fc",None)], args.epochs_head, 3e-3, "head")
    # Stage2: layer3,4 + fc
    b2 = run_stage([("layer3",None),("layer4",None),("fc",None)], args.epochs_ft, 1e-4, "finetune")
else:
    # efficientnet: features[-2:] + classifier
    b1 = run_stage([("classifier",None)], args.epochs_head, 3e-3, "head")
    b2 = run_stage([("features.7",None),("features.8",None),("classifier",None)], args.epochs_ft, 1e-4, "finetune")

print(f"DONE arch={args.arch} seed={args.seed} best_f1={max(b1,b2):.4f} saved={args.out}")
