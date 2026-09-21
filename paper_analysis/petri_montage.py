#!/usr/bin/env python3
"""Petri dish version of the training set size montage.

Reuses the layout from microfluidics_sweep.render_montage, but with the petri
conventions: ground truth is a 16 bit instance map scored as any labelled
pixel, the network is the 3 class boundary U-Net read at its body class, and
the input size is 512 x 512.

Numbers come from the existing sweep (petri_sweep_results.csv, sizes 10 to 70,
3 random draws each), which selected its best epoch on a validation split and
reported on a separate 11 image holdout, so nothing needs retraining.

Usage:
  python petri_montage.py [--csv PATH] [--out petri_montage.png]
"""
import argparse
import csv
import json
import os

import numpy as np
import cv2
import torch

import microfluidics_sweep as ms
import unet_model_class

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = "/Volumes/Extreme SSD/Deepweed/.scratch/petri_sweep_results.csv"
TEST_IMAGES = os.path.join(HERE, 'test_annotations', 'images')
TEST_MASKS = os.path.join(HERE, 'test_annotations', 'masks')
MODEL = os.path.join(HERE, 'data_model', 'model', 'best_instance_unet_512.pt')
INPUT_SIZE = 512
EPOCHS = 200


def load_gt(mask_path):
    """Any labelled pixel is frond, matching eval_petri in training_sweep.py."""
    g = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if g is None:
        return None
    if g.ndim == 3:
        g = g[:, :, 0]
    return g > 0


def predict(model, img_p, device):
    """Body class of the 3 class model, upscaled back to full resolution."""
    rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        pred = model(t).argmax(dim=1).squeeze().cpu().numpy()
    full = cv2.resize(pred.astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST)
    return full == 1


def pairs():
    """Image and mask pairs of the holdout set, by filename stem."""
    masks = {os.path.splitext(f)[0]: os.path.join(TEST_MASKS, f)
             for f in sorted(os.listdir(TEST_MASKS))
             if f.endswith('.png') and not f.startswith('._')}
    out = []
    for f in sorted(os.listdir(TEST_IMAGES)):
        if f.startswith('._'):
            continue
        stem = os.path.splitext(f)[0]
        if stem in masks:
            out.append((os.path.join(TEST_IMAGES, f), masks[stem]))
    return out


def results_from_csv(path):
    rows = list(csv.DictReader(open(path)))
    tp = pairs()
    return {
        'sizes': sorted({int(r['n_train']) for r in rows}),
        'seeds': len({r['seed'] for r in rows}),
        'epochs': EPOCHS,
        'n_test': int(rows[0]['n_images']),
        'test_images': [i for i, _ in tp],
        'test_masks': [m for _, m in tp],
        'runs': [{'size': int(r['n_train']), 'seed': int(r['seed']),
                  'dice': float(r['dice_mean']), 'iou': float(r['iou_mean'])}
                 for r in rows],
        'done': True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default=os.path.join(HERE, 'petri_montage.png'))
    ap.add_argument('--dpi', type=int, default=300)
    a = ap.parse_args()

    res = results_from_csv(a.csv)
    print(f"{'images':>7} {'Dice':>20} {'IoU':>20}")
    for r in ms.summarize(res):
        print(f"{r['size']:>7} {r['dice_mean']:>10.4f} +/- {r['dice_std']:.4f} "
              f"{r['iou_mean']:>10.4f} +/- {r['iou_std']:.4f}  ({r['n']} draws)")

    device = torch.device('cpu')
    model = unet_model_class.UNet(n_channels=3, n_classes=3).to(device)
    ck = torch.load(MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])

    ms.render_montage(res, a.out, model=model, device=device,
                      model_label='published model', gt_loader=load_gt,
                      predictor=predict, rotate=False, zoom=380,
                      title='Petri dish segmentation accuracy against '
                            'training set size', dpi=a.dpi,
                      reference=json.load(open(os.path.join(HERE,
                          'published_model_reference.json')))['petri'])


if __name__ == '__main__':
    main()
