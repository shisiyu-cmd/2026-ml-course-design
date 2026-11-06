import os, sys, importlib

print("python:", sys.version)

for mod in ["numpy", "pandas", "PIL", "sklearn", "cv2", "scipy", "matplotlib", "torch", "torchvision"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  [NO] {mod}: {type(e).__name__}")

# Check GPU via torch if available
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  gpu:", torch.cuda.get_device_name(0))
except Exception:
    pass

# Image samples
base = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data"
from PIL import Image
for split in ["train", "test"]:
    p = os.path.join(base, split)
    if split == "train":
        folders = sorted(os.listdir(p))
        for f in folders:
            fp = os.path.join(p, f)
            files = os.listdir(fp)
            if files:
                im = Image.open(os.path.join(fp, files[0]))
                print(f"  {split}/{f}: n={len(files)} sample size={im.size} mode={im.mode}")
    else:
        files = os.listdir(p)
        if files:
            im = Image.open(os.path.join(p, files[0]))
            print(f"  {split}: n={len(files)} sample size={im.size} mode={im.mode}")
