#!/usr/bin/env python3
"""Retrain petri dish 3-class boundary U-Net at 512x512 (was 256x256).
Run on the lab server (M2 Ultra, 64GB)."""
import os, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- UNet (same architecture as unet_model_class.py) ----
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_bottom = double_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up4 = double_conv(512 + 1024, 512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_bottom(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

# ---- paths (server) ----
DATA_ROOT = os.path.expanduser("~/Workspace/deepweed_512")
TRAIN_IMG = os.path.join(DATA_ROOT, "train_images")
TRAIN_MSK = os.path.join(DATA_ROOT, "train_masks")
VAL_IMG = os.path.join(DATA_ROOT, "val_images")
VAL_MSK = os.path.join(DATA_ROOT, "val_masks")
OUT_DIR = DATA_ROOT

# ---- hyperparams ----
IMAGE_SIZE = 512
N_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 300
LR = 1e-4
BOUNDARY_WIDTH = 2

def remap_mask(mask_np):
    out = np.zeros_like(mask_np, dtype=np.int64)
    out[mask_np == 85] = 1
    out[(mask_np == 170) | (mask_np == 255)] = 2
    return out


class DuckweedDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpeg', '.jpg', '.png')) and not f.startswith('.')])
        masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png') and not f.startswith('.')])
        mask_stems = {f.replace('_preview.png', ''): f for f in masks}
        self.pairs = []
        for img_f in imgs:
            stem = os.path.splitext(img_f)[0]
            if stem in mask_stems:
                self.pairs.append((img_f, mask_stems[stem]))
        print(f"  {img_dir}: {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_f, msk_f = self.pairs[idx]
        image = np.array(Image.open(os.path.join(self.img_dir, img_f)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, msk_f)).convert("L"))
        mask = remap_mask(mask)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].long()
        return image, mask


train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.CLAHE(clip_limit=2.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    ], p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=(3, 5)),
    ], p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])


class MultiClassDiceLoss(nn.Module):
    def __init__(self, n_classes=3, smooth=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.n_classes).permute(0, 3, 1, 2).float()
        loss = 0.0
        for c in range(self.n_classes):
            p = probs[:, c].reshape(-1)
            t = targets_oh[:, c].reshape(-1)
            inter = (p * t).sum()
            loss += 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return loss / self.n_classes


class CombinedLoss(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(n_classes)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def compute_metrics(logits, targets, n_classes=3):
    preds = logits.argmax(dim=1)
    metrics = {}
    for c in range(n_classes):
        p = (preds == c).float().reshape(-1)
        t = (targets == c).float().reshape(-1)
        inter = (p * t).sum().item()
        union = p.sum().item() + t.sum().item() - inter
        iou = inter / (union + 1e-7)
        dice = 2 * inter / (p.sum().item() + t.sum().item() + 1e-7)
        name = ["bg", "body", "boundary"][c]
        metrics[f"iou_{name}"] = iou
        metrics[f"dice_{name}"] = dice
    metrics["fg_mean_iou"] = (metrics["iou_body"] + metrics["iou_boundary"]) / 2
    metrics["fg_mean_dice"] = (metrics["dice_body"] + metrics["dice_boundary"]) / 2
    return metrics


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}, Batch size: {BATCH_SIZE}")

    train_ds = DuckweedDataset(TRAIN_IMG, TRAIN_MSK, train_transform)
    val_ds = DuckweedDataset(VAL_IMG, VAL_MSK, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet(n_channels=3, n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=30)
    criterion = CombinedLoss(N_CLASSES)

    best_fg_miou = 0.0
    save_path = os.path.join(OUT_DIR, "best_instance_unet_512.pt")

    print(f"\nTraining {len(train_ds)} images, validating {len(val_ds)} images")
    print(f"Saving best model to: {save_path}\n")

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_metrics_accum = {}
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                m = compute_metrics(logits, masks, N_CLASSES)
                for k, v in m.items():
                    val_metrics_accum[k] = val_metrics_accum.get(k, 0) + v
        for k in val_metrics_accum:
            val_metrics_accum[k] /= len(val_loader)

        fg_miou = val_metrics_accum["fg_mean_iou"]
        scheduler.step(fg_miou)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or fg_miou > best_fg_miou:
            print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={train_loss:.4f}  "
                  f"body_iou={val_metrics_accum['iou_body']:.4f}  "
                  f"bnd_iou={val_metrics_accum['iou_boundary']:.4f}  "
                  f"fg_miou={fg_miou:.4f}  lr={lr:.1e}  ({elapsed:.1f}s)",
                  flush=True)

        if fg_miou > best_fg_miou:
            best_fg_miou = fg_miou
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_fg_miou": best_fg_miou,
                "metrics": val_metrics_accum,
                "n_classes": N_CLASSES,
                "image_size": IMAGE_SIZE,
                "boundary_width": BOUNDARY_WIDTH,
            }, save_path)
            print(f"  >> Saved new best (fg_miou={best_fg_miou:.4f})", flush=True)

    print(f"\nTraining complete. Best fg_miou: {best_fg_miou:.4f}")
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    train()
