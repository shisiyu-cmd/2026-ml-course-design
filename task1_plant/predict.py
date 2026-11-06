import os, glob, torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights

torch.set_num_threads(14)
DATA = r"C:\Users\古恩宇\Doubao\chats\2026-09-21\new-chat\kaggle\dataset-for-task2\dataset-for-task2"
test_dir = os.path.join(DATA, "test")
mean = [0.485,0.456,0.406]; std = [0.229,0.224,0.225]

# load model
def build(arch, ncls):
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, ncls)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, ncls)
    return m

ckpts = ["r50_s0.pth", "r50_s42.pth"]
weights = [0.6, 0.4]  # stronger model gets more weight
models_list = []
classes = None
for ck in ckpts:
    d = torch.load(ck, map_location="cpu", weights_only=False)
    classes = d["classes"]
    m = build(d["arch"], len(classes))
    m.load_state_dict(d["model"])
    m.eval()
    models_list.append(m)
    print("loaded", ck, d["arch"], "img", d["img"])

# TTA: multiple sizes + flip
sizes = [224, 256, 288]
test_files = sorted(glob.glob(os.path.join(test_dir, "*.png")) + glob.glob(os.path.join(test_dir, "*.jpg")))
print("test images:", len(test_files))

probs_sum = None
weight_sum = 0.0
n = len(classes)
for mi, m in enumerate(models_list):
    w = weights[mi]
    for sz in sizes:
        tf = transforms.Compose([transforms.Resize((sz,sz)), transforms.ToTensor(), transforms.Normalize(mean,std)])
        tf_flip = transforms.Compose([transforms.Resize((sz,sz)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean,std)])
        for fname in test_files:
            img = Image.open(fname).convert("RGB")
            with torch.no_grad():
                p1 = torch.softmax(m(tf(img).unsqueeze(0)), 1)
                p2 = torch.softmax(m(tf_flip(img).unsqueeze(0)), 1)
                p = (p1+p2)/2
            if probs_sum is None:
                probs_sum = torch.zeros(len(test_files), n)
            idx = test_files.index(fname)
            probs_sum[idx] += p[0] * w

# normalize by total weight
avg = probs_sum / (sum(weights) * len(sizes) * 2)
preds = avg.argmax(1).tolist()

# build submission (match template order)
import csv
tmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission-for-task2.csv")
rows = list(csv.reader(open(tmpl)))
header = rows[0]
# map test file order to template IDs
# template first col = id (image name)
out_rows = [header]
file_to_pred = {os.path.basename(f): classes[p] for f,p in zip(test_files,preds)}
miss=0
for r in rows[1:]:
    fid = r[0]
    if fid in file_to_pred:
        out_rows.append([fid, file_to_pred[fid]])
    else:
        # try without ext
        base = os.path.splitext(fid)[0]
        found=None
        for k,v in file_to_pred.items():
            if os.path.splitext(k)[0]==base:
                found=v; break
        if found: out_rows.append([fid, found])
        else: out_rows.append([fid, r[1] if len(r)>1 else classes[0]]); miss+=1

with open("submission_v4.csv","w",newline="",encoding="utf-8") as f:
    csv.writer(f).writerows(out_rows)
print("wrote submission_v4.csv rows", len(out_rows)-1, "miss", miss)
print("class distribution:")
from collections import Counter
print(Counter([r[1] for r in out_rows[1:]]))
