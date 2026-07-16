#!/usr/bin/env python3
"""Evaluate the petri-dish U-Net on its held-out val set under several metric
definitions, to (a) get defensible numbers and (b) identify which definition
reproduces the manuscript's reported Dice = 0.9420.
GT masks: 0=bg, 85=body, 170/255=boundary.  Model: 3-class argmax (1=body).
"""
import os, sys, glob
import numpy as np, cv2, torch
from skimage import measure

PA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PA)
import unet_model_class

import argparse
_ap = argparse.ArgumentParser(description="Segmentation metrics for the petri-dish U-Net")
_ap.add_argument("--images", required=True, help="dir of RGB frames (.jpeg)")
_ap.add_argument("--masks", required=True, help="dir of GT masks (*_preview.png; 0=bg,85=body,170/255=boundary)")
_a = _ap.parse_args()
IMGS, MSKS = _a.images, _a.masks
MODEL = os.path.join(PA, "data_model", "model", "best_instance_unet.pt")

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = unet_model_class.UNet(n_channels=3, n_classes=3).to(dev)
ck = torch.load(MODEL, map_location=dev, weights_only=False)
model.load_state_dict(ck["model_state_dict"]); model.eval()
print("checkpoint: epoch=%s best_fg_miou=%s" % (ck.get("epoch"), ck.get("best_fg_miou")))

def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s

def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u

rows = []
files = [f for f in sorted(glob.glob(os.path.join(MSKS, "*_preview.png")))
         if not os.path.basename(f).startswith("._")]
print("val masks:", len(files))
for mf in files:
    stem = os.path.basename(mf).replace("_preview.png", "")
    imf = os.path.join(IMGS, stem + ".jpeg")
    if not os.path.exists(imf):
        print("  MISSING image for", stem); continue
    gt = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(cv2.imread(imf), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    x = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        out = model(t)
    pred = out.argmax(dim=1).squeeze().cpu().numpy()
    pred_full = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    gt_body, gt_fg = gt == 85, gt > 0
    pr_body, pr_fg = pred_full == 1, pred_full > 0

    # per-class incl. background (the inflated definition)
    dices_allcls = []
    for cls_gt, cls_pr in ((gt == 0, pred_full == 0), (gt_body, pr_body), (gt > 85, pred_full == 2)):
        dices_allcls.append(dice(cls_gt, cls_pr))

    rows.append(dict(
        stem=stem,
        body_iou=iou(gt_body, pr_body), body_dice=dice(gt_body, pr_body),
        fg_iou=iou(gt_fg, pr_fg), fg_dice=dice(gt_fg, pr_fg),
        mean_dice_incl_bg=float(np.mean(dices_allcls)),
        bg_dice=dices_allcls[0],
        n_gt=measure.label(gt_body).max(), n_pr=measure.label(pr_body).max(),
    ))

def col(k):
    return np.array([r[k] for r in rows], float)

print("\n%-22s %6s %6s" % ("metric (n=%d)" % len(rows), "mean", "std"))
for k in ("body_iou", "body_dice", "fg_iou", "fg_dice", "bg_dice", "mean_dice_incl_bg"):
    print("  %-20s %.4f %.4f" % (k, col(k).mean(), col(k).std()))

print("\nper-image body IoU/Dice and frond counts:")
for r in rows:
    print("  %-22s IoU=%.3f Dice=%.3f  GT=%2d pred=%2d" %
          (r["stem"], r["body_iou"], r["body_dice"], r["n_gt"], r["n_pr"]))

print("\ncount error: GT total=%d pred total=%d" % (col("n_gt").sum(), col("n_pr").sum()))
print("\nmanuscript claims Dice=0.9420 -> closest definition:")
for k in ("body_dice", "fg_dice", "bg_dice", "mean_dice_incl_bg"):
    print("   %-20s %.4f   (diff %.4f)" % (k, col(k).mean(), abs(col(k).mean() - 0.9420)))
