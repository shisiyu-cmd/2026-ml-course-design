"""Verify that prior checkpoints align with our StratifiedKFold(42) split,
then generate OOF logits for every prior model."""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold

WORK = r"C:\Users\古恩宇\Downloads\2026task2"
CACHE = os.path.join(WORK, "cache", "data.npz")
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
DEV = torch.device("cuda")

d = np.load(CACHE, allow_pickle=True)
Xtr, ytr, Xte = d["Xtr"], d["ytr"], d["Xte"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(Xtr, ytr))


def build(arch):
    if arch == "r18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 6)
    elif arch == "r34":
        m = models.resnet34(weights=None); m.fc = nn.Linear(m.fc.in_features, 6)
    elif arch == "eff":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
    elif arch == "gnet":
        m = models.googlenet(weights=None, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, 6)
    else:
        raise ValueError(arch)
    return m


def tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def infer(model, arr, size, bs=256):
    model.eval()
    t = tf(size)
    out = []
    for i in range(0, len(arr), bs):
        chunk = arr[i:i + bs]
        x = torch.stack([t(Image.fromarray(a).convert("RGB")) for a in chunk]).to(DEV)
        out.append((0.5 * (model(x) + model(torch.flip(x, dims=[3])))).float().cpu())
    return torch.cat(out, 0).numpy()


def load_sd(model, path):
    sd = torch.load(path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("aux")}
    model.load_state_dict(sd, strict=False)


if __name__ == "__main__":
    # --- alignment check on r34 fold1 ---
    tri, vai = splits[0]
    m = build("r34"); load_sd(m, os.path.join(WORK, "folds_r34", "fold1.pt")); m = m.to(DEV)
    lg = infer(m, Xtr[vai], 112)
    acc = (lg.argmax(1) == ytr[vai]).mean()
    print(f"r34 fold1 on OUR fold1 val: acc={acc:.4f}  (prior log said 0.6937)")
    print("ALIGNED" if abs(acc - 0.6937) < 0.02 else "MISALIGNED -- investigate")
