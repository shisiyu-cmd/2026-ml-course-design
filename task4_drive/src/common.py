"""Shared utilities: paths, dataset, augmentation, metrics, RLE encoding.

Task: 2D fundus retinal vessel segmentation (DRIVE-style, 565x584 source).
Convention (matches provided training labels and segmentation_to_csv.py):
    vessel  -> non-zero  (255)
    background -> 0
segmentation_to_csv.py binarises with `image[image > 0] = 1`, i.e. any non-zero
pixel is treated as part of the predicted mask, so 255=vessel is the correct
convention for the submission pipeline.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMG = os.path.join(ROOT, "train", "image")
TRAIN_LBL = os.path.join(ROOT, "train", "label")
TEST_IMG = os.path.join(ROOT, "test", "image")
WORK = os.path.join(ROOT, "work")

# External public retinal-vessel datasets (used as EXTRA TRAINING DATA only).
# Layout expected: <dir>/input/*  and  <dir>/label/*.  We probe a few plausible
# locations so the same code works locally and on Kaggle.
_EXT_ROOTS = [
    os.path.join(os.path.dirname(ROOT), "ext_data"),   # <workspace>/ext_data
    os.path.join(ROOT, "ext_data"),
    os.path.join(os.path.dirname(os.path.dirname(ROOT)), "ext_data"),
    "/kaggle/input/ext-data",
]
EXT_DIRS = {}
for _n in ("chase", "hrf"):
    for _r in _EXT_ROOTS:
        _c = os.path.join(_r, _n)
        if os.path.isdir(os.path.join(_c, "input")):
            EXT_DIRS[_n] = _c
            break


def list_external_pairs(verbose=False):
    """Return [(stem|labelpath, image_path)] for every external dataset present.

    Pairs are encoded as "<name>|<absolute label path>" so VesselDataset can find
    the label without changing its signature. Competition pairs keep plain stems.
    """
    out = []
    for name, d in EXT_DIRS.items():
        idir, ldir = os.path.join(d, "input"), os.path.join(d, "label")
        if not (os.path.isdir(idir) and os.path.isdir(ldir)):
            continue
        # index labels by stem (strip trailing _1stHO / manual suffixes)
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
            if stem not in labels:
                continue
            out.append((f"{name}_{stem}|{labels[stem]}", os.path.join(idir, f)))
            n += 1
        if verbose:
            print(f"  external {name}: {n} labelled pairs")
    return out

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def list_images(folder):
    """Return sorted (stem, path) pairs of image files."""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    names = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    names.sort(key=lambda s: int(os.path.splitext(s)[0]) if os.path.splitext(s)[0].isdigit() else s)
    return [(os.path.splitext(n)[0], os.path.join(folder, n)) for n in names]


# ---------------------------------------------------------------- FOV
def fov_mask(rgb, thresh=10):
    """Circular field-of-view mask from a fundus image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (gray > thresh).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(gray.shape, np.uint8) * 255
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


# ---------------------------------------------------------------- CLAHE
def apply_clahe(rgb, clip=2.0, grid=8):
    """CLAHE on the L channel of LAB -> preserves colour, boosts local contrast."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread replacement that survives non-ASCII paths (Chinese usernames)."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"imdecode failed: {path}")
    return img


def imwrite_unicode(path, img):
    """cv2.imwrite replacement that survives non-ASCII paths; returns True on success."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return os.path.exists(path)


def load_image(path, size, use_clahe=True):
    img = imread_unicode(path, cv2.IMREAD_COLOR)      # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_clahe:
        img = apply_clahe(img)
    if size is not None and img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def load_label(path, size):
    m = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if size is not None and m.shape[:2] != (size, size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)


def to_tensor(img_uint8):
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


# ---------------------------------------------------------------- dataset
class VesselDataset(Dataset):
    """If `with_label` is True the label is loaded; `aug` is applied only when both
    `train` and `aug` are given. Returns (image, mask, stem) or (image, stem)."""

    def __init__(self, pairs, size=512, train=True, use_clahe=True, aug=None, with_label=None):
        self.pairs = pairs
        self.size = size
        self.train = train
        self.use_clahe = use_clahe
        self.aug = aug
        self.with_label = train if with_label is None else with_label

    def __len__(self):
        return len(self.pairs)

    def _label_path(self, stem, ipath):
        """Competition labels sit in TRAIN_LBL; external-dataset pairs carry an
        absolute label path in the stem field, signalled by a '|' separator."""
        if "|" in stem:
            return stem.split("|", 1)[1]
        return os.path.join(TRAIN_LBL, stem + ".jpg")

    def _stem_name(self, stem):
        return stem.split("|", 1)[0] if "|" in stem else stem

    def __getitem__(self, i):
        stem, ipath = self.pairs[i]
        img = load_image(ipath, self.size, self.use_clahe)
        name = self._stem_name(stem)
        if self.with_label:
            msk = load_label(self._label_path(stem, ipath), self.size)
            if self.aug is not None:
                out = self.aug(image=img, mask=(msk * 255).astype(np.uint8))
                img, msk = out["image"], out["mask"]
                msk = (msk > 127).astype(np.float32)
            return to_tensor(img), torch.from_numpy(msk)[None], name
        return to_tensor(img), name


# ---------------------------------------------------------------- metrics
def dice_score(pred, gt, eps=1e-7):
    pred = pred.astype(np.float32).ravel()
    gt = gt.astype(np.float32).ravel()
    inter = (pred * gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


# ---------------------------------------------------------------- RLE
def rle_encode(mask):
    """Row-major RLE identical to segmentation_to_csv.py (1-based start, length)."""
    flat = np.asarray(mask).ravel()
    dots = np.where(flat > 0)[0]
    if dots.size == 0:
        return ""
    runs = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            runs.extend((int(b) + 1, 0))
        runs[-1] += 1
        prev = b
    return " ".join(str(r) for r in runs)


def rle_decode(rle, shape):
    out = np.zeros(int(np.prod(shape)), dtype=np.uint8)
    if rle and str(rle).strip():
        vals = [int(v) for v in str(rle).split()]
        assert len(vals) % 2 == 0, "odd number of RLE values"
        for s, l in zip(vals[0::2], vals[1::2]):
            assert s >= 1 and l >= 1, "RLE start/length must be positive"
            out[s - 1:s - 1 + l] = 1
    return out.reshape(shape)


def check_rle(rle, n_pixels):
    """Validate sorted / positive / non-overlapping pairs (competition rules)."""
    if not rle:
        return True
    vals = [int(v) for v in rle.split()]
    if len(vals) % 2:
        return False
    prev_end = 0
    for s, l in zip(vals[0::2], vals[1::2]):
        if s < 1 or l < 1 or s <= prev_end or s + l - 1 > n_pixels:
            return False
        prev_end = s + l - 1
    return True
