"""Model factory + losses for retinal vessel segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


def build_model(arch="unetplusplus", encoder="efficientnet-b4", in_ch=3, classes=1):
    """Return a smp segmentation model with ImageNet-pretrained encoder."""
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_ch,
        classes=classes,
    )
    name = arch.lower()
    if name in ("unetplusplus", "unet++", "nestedunet"):
        return smp.UnetPlusPlus(**kwargs)
    if name == "unet":
        return smp.Unet(**kwargs)
    if name in ("deeplabv3plus", "deeplab"):
        return smp.DeepLabV3Plus(**kwargs)
    if name in ("manet", "mAnet".lower()):
        return smp.MAnet(**kwargs)
    if name == "fpn":
        return smp.FPN(**kwargs)
    if name == "linknet":
        return smp.LinkNet(**kwargs)
    if name == "segformer":
        return smp.Segformer(**kwargs)
    raise ValueError(f"unknown arch {arch}")


# ------------------------------------------------------------------ losses
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(2, 3)) + self.smooth
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        return 1 - (num / den).mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice, with a positive weight on BCE to fight the heavy class imbalance."""

    def __init__(self, bce_weight=0.5, pos_weight=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return self.bce_weight * bce + (1 - self.bce_weight) * self.dice(logits, target)


class SoftCLDiceLoss(nn.Module):
    """Soft centre-line Dice (Shit et al., CVPR 2021) - topology aware for thin vessels."""

    def __init__(self, iter_=10, smooth=1.0):
        super().__init__()
        self.iter = iter_
        self.smooth = smooth

    def _soft_skel(self, x):
        x = x.clamp(1e-6, 1 - 1e-6)
        for _ in range(self.iter):
            mx = F.max_pool2d(x, 3, 1, 1)
            mn = -F.max_pool2d(-x, 3, 1, 1)
            x = mx + mn - mx * mn          # soft dilation-ish (morphological closing)
            x = x * (1 - (mx * mn))        # soft erosion of the difference
            x = x.clamp(1e-6, 1 - 1e-6)
        return x

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        skel_pred = self._soft_skel(p)
        skel_true = self._soft_skel(target)
        tprec = (skel_pred * target).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + self.smooth)
        tsens = (skel_true * p).sum(dim=(2, 3)) / (skel_true.sum(dim=(2, 3)) + self.smooth)
        cldice = 2 * tprec * tsens / (tprec + tsens + 1e-6)
        return 1 - cldice.mean()


class CombinedLoss(nn.Module):
    """0.4*BCE + 0.4*Dice + 0.2*clDice."""

    def __init__(self, bce_w=0.4, dice_w=0.4, cldice_w=0.2, pos_weight=2.0, use_cldice=True):
        super().__init__()
        self.bce_w, self.dice_w, self.cldice_w = bce_w, dice_w, cldice_w
        self.dice = DiceLoss()
        self.cldice = SoftCLDiceLoss() if use_cldice else None
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits, target):
        loss = self.bce_w * F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight)
        loss = loss + self.dice_w * self.dice(logits, target)
        if self.cldice is not None and self.cldice_w > 0:
            loss = loss + self.cldice_w * self.cldice(logits, target)
        return loss


def build_loss(name="bcedice", **kw):
    n = name.lower()
    if n == "bcedice":
        return BCEDiceLoss(**kw)
    if n == "combined":
        return CombinedLoss(**kw)
    if n == "dice":
        return DiceLoss()
    raise ValueError(name)
