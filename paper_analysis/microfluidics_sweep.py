#!/usr/bin/env python3
"""Segmentation accuracy against training set size, microfluidics dataset.

Trains the microfluidics U-Net from scratch on random subsets of the annotated
training pool (sizes 10, 20, 30, 40, 50), several random draws per size, and
scores every model on the held out validation frames. Dice and IoU are measured
at full image resolution, which is where the resolution loss of the network
input actually shows up.

Data and recipe follow the original microfluidics training notebook:
  ground truth  16 bit instance maps, one id per frond
  target        frond interior with inner boundaries removed
  input size    512 x 128, which keeps the 4:1 trench aspect ratio
  loss          Dice on sigmoid, plus binary cross entropy

Results are written to JSON as they are produced, so the sweep can be watched
while it runs. render_montage() turns that JSON into the summary figure.

Usage:
  python microfluidics_sweep.py                      # full sweep, 3 draws per size
  python microfluidics_sweep.py --sizes 10 --seeds 1 --epochs 2   # quick check
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unet_model_class

DATA_ROOT = ("/Volumes/Extreme SSD/04- Deepweed & Duckweed/duckweed_ish/"
             "linear_trench_mask_labelling")
INPUT_H, INPUT_W = 512, 128     # keeps the 1024 x 251 trench aspect ratio
NORM_MEAN, NORM_STD = 0.5, 0.5
DEFAULT_SIZES = [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_target(mask_path):
    """Frond interior as a boolean mask.

    Ground truth is a 16 bit instance map (0 background, 1..N one id per
    frond). Inner boundaries are dropped so that fronds touching inside a well
    stay separable, which is what the shipped microfluidics model was trained
    on. Scoring uses this same definition, so training and evaluation agree.
    """
    inst = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if inst is None:
        return None
    if inst.ndim == 3:
        inst = inst[:, :, 0]
    interior = inst > 0
    if interior.any():
        interior = interior & ~find_boundaries(inst, mode='inner')
    return interior


def find_pairs(images_dir, masks_dir):
    """Pair image files with mask files by filename stem."""
    def stem(p):
        return os.path.splitext(os.path.basename(p))[0]

    masks = {stem(p): p for p in sorted(glob.glob(os.path.join(masks_dir, '*.png')))
             if not os.path.basename(p).startswith('._')}
    pairs = []
    for p in sorted(glob.glob(os.path.join(images_dir, '*.png'))):
        if os.path.basename(p).startswith('._'):
            continue
        if stem(p) in masks:
            pairs.append((p, masks[stem(p)]))
    return pairs


class FrondDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, msk_p = self.pairs[i]
        rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        m = load_target(msk_p).astype(np.uint8)
        rgb = cv2.resize(rgb, (INPUT_W, INPUT_H))
        m = cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                rgb, m = rgb[:, ::-1].copy(), m[:, ::-1].copy()
            if random.random() < 0.5:
                rgb, m = rgb[::-1].copy(), m[::-1].copy()
            if random.random() < 0.5:          # lighting varies between runs
                a = 1.0 + random.uniform(-0.25, 0.25)
                b = random.uniform(-25, 25)
                rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        x = (rgb.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        return (torch.from_numpy(x.transpose(2, 0, 1)).float(),
                torch.from_numpy(m[None].astype(np.float32)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + 1.0) / (p.sum() + target.sum() + 1.0)
        return bce + dice


def train_model(pairs, device, epochs=None, batch_size=4, lr=1e-4, seed=0,
                steps=None, log=print):
    """Train for a fixed number of gradient steps when `steps` is given, so every
    training set size receives the same optimisation budget and only the amount of
    data differs. Falls back to a fixed epoch count otherwise."""
    torch.manual_seed(seed)
    model = unet_model_class.UNet(n_channels=3, n_classes=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine schedule rather than ReduceLROnPlateau: the frond target is only
    # about 0.3 percent positive, so a plateau scheduler decays the rate while
    # the model is still stuck predicting nothing and freezes it there.
    crit = DiceBCELoss()
    loader = DataLoader(FrondDataset(pairs, augment=True),
                        batch_size=min(batch_size, len(pairs)), shuffle=True)
    if steps is not None:
        epochs = max(1, -(-steps // len(loader)))      # ceil, equal step budget
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr * 0.01)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        if (ep + 1) % 20 == 0:
            log(f"      epoch {ep + 1}/{epochs} loss {total / len(loader):.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation, at full image resolution
# ---------------------------------------------------------------------------

def predict_full(model, img_p, device, threshold=0.5):
    rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    x = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    x = (x - NORM_MEAN) / NORM_STD
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR) > threshold


def _overlap(gt, pr):
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    s = gt.sum() + pr.sum()
    return (1.0 if s == 0 else 2.0 * inter / s,
            1.0 if union == 0 else inter / union)


def score(model, test_pairs, device, threshold=0.5):
    """Dice and IoU at both bases: full image resolution, and at the network
    input size where the ground truth is downsampled to match the prediction."""
    fd, fi, nd, ni = [], [], [], []
    model.eval()
    for img_p, msk_p in test_pairs:
        gt = load_target(msk_p)
        rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        x = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
        x = (x - NORM_MEAN) / NORM_STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
        d, i = _overlap(gt, cv2.resize(prob, (W, H),
                                       interpolation=cv2.INTER_LINEAR) > threshold)
        fd.append(d); fi.append(i)
        gt_small = cv2.resize(gt.astype(np.uint8), (INPUT_W, INPUT_H),
                              interpolation=cv2.INTER_NEAREST) > 0
        d, i = _overlap(gt_small, prob > threshold)
        nd.append(d); ni.append(i)
    return (float(np.mean(fd)), float(np.mean(fi)),
            float(np.mean(nd)), float(np.mean(ni)))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def get_device():
    return torch.device('mps' if torch.backends.mps.is_available()
                        else 'cuda' if torch.cuda.is_available() else 'cpu')


def run_sweep(data_root=DATA_ROOT, sizes=None, seeds=3, epochs=60,
              out_json=None, log=print, steps=None, save_all=False):
    device = get_device()
    pool = find_pairs(os.path.join(data_root, 'train_images'),
                      os.path.join(data_root, 'train_instance_map'))
    test_pairs = find_pairs(os.path.join(data_root, 'val_images'),
                            os.path.join(data_root, 'val_instance_masks'))
    log(f"training pool {len(pool)} images, held out test set {len(test_pairs)} "
        f"images, device {device}")
    if not pool or not test_pairs:
        raise SystemExit(f"no annotated pairs found under {data_root}")

    sizes = [s for s in (sizes or DEFAULT_SIZES) if s <= len(pool)]
    log(f"training set sizes {sizes}, {seeds} random draws each, {epochs} epochs")

    results = {'sizes': sizes, 'seeds': seeds, 'epochs': epochs, 'steps': steps,
               'n_test': len(test_pairs), 'n_pool': len(pool),
               'input_size': [INPUT_H, INPUT_W],
               'test_images': [p for p, _ in test_pairs],
               'test_masks': [m for _, m in test_pairs],
               'runs': [], 'done': False}

    for size in sizes:
        for seed in range(seeds):
            t0 = time.time()
            subset = random.Random(1000 * seed + size).sample(pool, size)
            log(f"  size {size:>3}, draw {seed + 1}/{seeds}")
            model = train_model(subset, device, epochs, seed=seed, steps=steps,
                                log=log)
            fdice, fiou, ndice, niou = score(model, test_pairs, device)
            log(f"      full-res dice {fdice:.4f} iou {fiou:.4f} | "
                f"network-res dice {ndice:.4f} iou {niou:.4f}  "
                f"({time.time() - t0:.0f}s)")
            results['runs'].append({'size': size, 'seed': seed,
                                    'dice': fdice, 'iou': fiou,
                                    'dice_net': ndice, 'iou_net': niou})
            if save_all and out_json:
                torch.save({'model_state_dict': model.state_dict(), 'size': size,
                            'seed': seed, 'dice': fdice, 'dice_net': ndice},
                           os.path.splitext(out_json)[0] + f'_n{size}_s{seed}.pt')
            # keep the largest-training-set model so the montage can show what
            # these scores look like on the actual images
            if out_json and size == sizes[-1] and seed == 0:
                ckpt = os.path.splitext(out_json)[0] + '_best_model.pt'
                torch.save({'model_state_dict': model.state_dict(),
                            'size': size, 'dice': fdice, 'iou': fiou}, ckpt)
                results['best_model'] = ckpt
            if out_json:
                json.dump(results, open(out_json, 'w'), indent=2)
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()

    results['done'] = True
    if out_json:
        json.dump(results, open(out_json, 'w'), indent=2)
    return results


def load_best_model(results, device=None):
    """Reload the saved largest-training-set model, for the montage overlay."""
    device = device or get_device()
    path = results.get('best_model')
    if not path or not os.path.exists(path):
        return None, device
    model = unet_model_class.UNet(n_channels=3, n_classes=1).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, device


def summarize(results, basis='full'):
    """Per size: mean and standard deviation of Dice and IoU across draws.
    basis 'full' scores at full image resolution, 'network' at the 512x128 input."""
    dk, ik = ('dice', 'iou') if basis == 'full' else ('dice_net', 'iou_net')
    out = []
    for size in results['sizes']:
        d = [r[dk] for r in results['runs'] if r['size'] == size and dk in r]
        i = [r[ik] for r in results['runs'] if r['size'] == size and ik in r]
        if not d:
            continue
        out.append({'size': size, 'n': len(d),
                    'dice_mean': float(np.mean(d)), 'dice_std': float(np.std(d)),
                    'iou_mean': float(np.mean(i)), 'iou_std': float(np.std(i))})
    return out


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

DICE_COLOR = '#2F6FBF'
IOU_COLOR = '#C0392B'


def _busiest_box(gt, bh, bw, stride=40):
    """Top left corner of the bh x bw window holding the most ground truth,
    for datasets where fronds are spread across the frame rather than in a
    vertical trench."""
    H, W = gt.shape
    bh, bw = min(bh, H), min(bw, W)
    ii = np.pad(np.cumsum(np.cumsum(gt.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    best, best_rc = -1, (0, 0)
    for r in range(0, H - bh + 1, stride):
        for c in range(0, W - bw + 1, stride):
            tot = (ii[r + bh, c + bw] - ii[r, c + bw]
                   - ii[r + bh, c] + ii[r, c])
            if tot > best:
                best, best_rc = tot, (r, c)
    return best_rc[0], best_rc[0] + bh, best_rc[1], best_rc[1] + bw


def _localize(path, subdir):
    """Results produced on the server carry server paths. Fall back to the same
    filename under the local DATA_ROOT so the montage can be rendered here."""
    if os.path.exists(path):
        return path
    local = os.path.join(DATA_ROOT, subdir, os.path.basename(path))
    return local if os.path.exists(local) else path


def _outline(ax, mask, color, lw=0.9):
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _busiest_window(gt, height=300):
    """Row range of the ground truth that holds the most fronds, so the zoom
    panel lands on wells that actually contain plants."""
    per_row = gt.sum(axis=1).astype(float)
    if per_row.sum() == 0 or len(per_row) <= height:
        return 0, min(height, len(per_row))
    csum = np.concatenate([[0], np.cumsum(per_row)])
    totals = csum[height:] - csum[:-height]
    r0 = int(np.argmax(totals))
    return r0, r0 + height


def render_montage(results, out_path, model=None, device=None, n_show=3,
                   threshold=0.5, model_label='model', gt_loader=None,
                   predictor=None, rotate=True, zoom=420, title=None,
                   image_subdir='val_images', mask_subdir='val_instance_masks',
                   dpi=300, reference=None):
    """Summary figure: held out test images on top, accuracy curves below.

    Pass `model` to outline its prediction in red beside the ground truth in
    green, so the reader can see what the scores correspond to.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    rows = summarize(results)
    if not rows:
        raise SystemExit('no finished runs in results yet')
    sizes = [r['size'] for r in rows]
    dice_m = np.array([r['dice_mean'] for r in rows])
    dice_s = np.array([r['dice_std'] for r in rows])
    iou_m = np.array([r['iou_mean'] for r in rows])
    iou_s = np.array([r['iou_std'] for r in rows])

    gt_loader = gt_loader or load_target
    if predictor is None:
        def predictor(mdl, path, dev):
            return predict_full(mdl, path, dev, threshold)
    orient = (lambda a: np.rot90(a)) if rotate else (lambda a: a)

    shown = [(_localize(i, image_subdir), _localize(m, mask_subdir))
             for i, m in zip(results['test_images'], results['test_masks'])][:n_show]

    fig = plt.figure(figsize=(15, 12.6))
    gs = gridspec.GridSpec(3, len(shown), figure=fig,
                           height_ratios=[1.25, 1.25, 2.5],
                           hspace=0.30, wspace=0.05)
    fig.suptitle(title or 'Microfluidics segmentation accuracy against '
                 'training set size', fontsize=22, fontweight='bold', y=0.965)

    top_axes, zoom_axes = [], []
    for k, (img_p, msk_p) in enumerate(shown):
        rgb = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        gt = gt_loader(msk_p)
        pr = predictor(model, img_p, device) if model is not None else None

        if rotate:
            r0, r1 = _busiest_window(gt, height=zoom)
            c0, c1 = 0, gt.shape[1]
        else:
            r0, r1, c0, c1 = _busiest_box(gt, zoom, int(zoom * 1.45))

        ax = fig.add_subplot(gs[0, k])          # same crop, ground truth only
        ax.imshow(orient(rgb[r0:r1, c0:c1]))
        _outline(ax, orient(gt[r0:r1, c0:c1]), '#00E36A', lw=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        top_axes.append(ax)
        az = fig.add_subplot(gs[1, k])
        az.imshow(orient(rgb[r0:r1, c0:c1]))
        _outline(az, orient(gt[r0:r1, c0:c1]), '#00E36A', lw=1.6)
        if pr is not None:
            _outline(az, orient(pr[r0:r1, c0:c1]), '#FF3B30', lw=1.4)
        az.set_xticks([])
        az.set_yticks([])
        for side in az.spines.values():
            side.set_color('#BBBBBB')
        zoom_axes.append(az)

    cap1 = f"{len(shown)} of the {results['n_test']} held out test images, "\
           f"ground truth in green"
    cap2 = 'zoom on the wells'
    if model is not None:
        cap2 += f': ground truth in green, {model_label} prediction in red'
    for axes_row, caption in ((top_axes, cap1), (zoom_axes, cap2)):
        y = min(a.get_position().y0 for a in axes_row) - 0.017
        fig.text(0.5, y, caption, ha='center', fontsize=13, color='#333333')

    ax = fig.add_subplot(gs[2, :])
    for m, s, c, name in ((dice_m, dice_s, DICE_COLOR, 'Dice coefficient'),
                          (iou_m, iou_s, IOU_COLOR, 'IoU')):
        ax.fill_between(sizes, m - s, m + s, color=c, alpha=0.16, linewidth=0)
        ax.plot(sizes, m, '-o', color=c, lw=2.2, ms=6.5, label=name,
                markerfacecolor='white', markeredgewidth=1.8)

    lo = max(0.0, min((dice_m - dice_s).min(), (iou_m - iou_s).min()) - 0.06)
    hi = min(1.0, max((dice_m + dice_s).max(), (iou_m + iou_s).max()) + 0.06)
    ax.set_ylim(lo, hi)
    ax.set_xticks(sizes)
    ax.set_xlabel('Number of annotated training images', fontsize=15)
    ax.set_ylabel('Score at full image resolution', fontsize=15)
    ax.grid(alpha=0.25, linestyle=':')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc='lower right')

    # Mark where the curve stops climbing, if it does so before the last point.
    # the deployed model, scored on the same held out frames
    if reference:
        for val, colour in ((reference.get('dice'), DICE_COLOR),
                            (reference.get('iou'), IOU_COLOR)):
            if val is not None:
                ax.axhline(val, color=colour, lw=1.1, linestyle=(0, (4, 3)), alpha=0.75)
        if reference.get('dice') is not None:
            ax.annotate('published model', xy=(sizes[0], reference['dice']),
                        xytext=(0, 4), textcoords='offset points',
                        fontsize=13, color=DICE_COLOR, va='bottom')

    # Where the curve stops climbing. If it never does, mark instead where it
    # first matches the deployed model, which is the honest equivalent.
    best = dice_m.max()
    plateau = next((s for s, m in zip(sizes, dice_m) if m >= best - 0.01), None)
    mark, label, va = None, None, 'top'
    if plateau is not None and sizes[0] < plateau < sizes[-1]:
        mark, label = plateau, f'flattens at {plateau} images'
    elif reference and reference.get('dice') is not None:
        hit = next((s for s, m in zip(sizes, dice_m)
                    if m >= reference['dice'] - 0.01), None)
        if hit is not None:
            mark, label, va = hit, f'matches published\nmodel at {hit} images', 'bottom'
    if mark is not None:
        ax.axvline(mark, color='#888888', lw=1, linestyle='--', alpha=0.8)
        y = hi if va == 'top' else lo
        ax.annotate(label, xy=(mark, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 6), textcoords='offset points', fontsize=12.5,
                    color='#555555', ha='center', va='bottom',
                    annotation_clip=False)

    ax.tick_params(labelsize=13)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"saved {out_path}")
    return fig


def render_plot_only(results, out_path, reference=None, dpi=300, xlabel=None):
    """Just the accuracy curve, styled exactly as in the montage, for slides."""
    import matplotlib.pyplot as plt

    rows = summarize(results)
    sizes = [r['size'] for r in rows]
    dm = np.array([r['dice_mean'] for r in rows]); ds = np.array([r['dice_std'] for r in rows])
    im = np.array([r['iou_mean'] for r in rows]);  isd = np.array([r['iou_std'] for r in rows])

    fig, ax = plt.subplots(figsize=(15.0, 5.0))
    for m, sd, c, name in ((dm, ds, DICE_COLOR, 'Dice coefficient'),
                           (im, isd, IOU_COLOR, 'IoU')):
        ax.fill_between(sizes, m - sd, m + sd, color=c, alpha=0.16, linewidth=0)
        ax.plot(sizes, m, '-o', color=c, lw=3.0, ms=9, label=name,
                markerfacecolor='white', markeredgewidth=2.4)

    lo = max(0.0, min((dm - ds).min(), (im - isd).min()) - 0.06)
    hi = min(1.0, max((dm + ds).max(), (im + isd).max()) + 0.06)
    if reference:
        for val, colour in ((reference.get('dice'), DICE_COLOR),
                            (reference.get('iou'), IOU_COLOR)):
            if val is not None:
                ax.axhline(val, color=colour, lw=1.6, linestyle=(0, (4, 3)), alpha=0.8)
        if reference.get('dice') is not None:
            ax.annotate('published model', xy=(sizes[0], reference['dice']),
                        xytext=(0, 6), textcoords='offset points',
                        fontsize=15, fontweight='bold', color=DICE_COLOR, va='bottom')

    best = dm.max()
    plateau = next((x for x, m in zip(sizes, dm) if m >= best - 0.01), None)
    mark, label = None, None
    if plateau is not None and sizes[0] < plateau < sizes[-1]:
        mark, label = plateau, f'flattens at {plateau} images'
    elif reference and reference.get('dice') is not None:
        hit = next((x for x, m in zip(sizes, dm) if m >= reference['dice'] - 0.01), None)
        if hit is not None:
            mark, label = hit, f'matches published model at {hit} images'
    if mark is not None:
        ax.axvline(mark, color='#888888', lw=1.4, linestyle='--', alpha=0.85)
        ax.annotate(label, xy=(mark, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, 7), textcoords='offset points', fontsize=15,
                    fontweight='bold', color='#555555', ha='center', va='bottom',
                    annotation_clip=False)

    ax.set_ylim(lo, hi)
    ax.set_xticks(sizes)
    ax.set_xlabel(xlabel or 'Number of annotated training images',
                  fontsize=19, fontweight='bold')
    ax.set_ylabel('Score at full image resolution', fontsize=19, fontweight='bold')
    ax.tick_params(labelsize=17)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    ax.grid(alpha=0.25, linestyle=':')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=18, loc='lower right')
    for t in leg.get_texts():
        t.set_fontweight('bold')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.25,
                facecolor='white')
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default=DATA_ROOT)
    p.add_argument('--sizes', default=None, help='comma separated, e.g. 10,20,30')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps', type=int, default=None,
                   help='equal gradient-step budget per model (overrides epochs)')
    p.add_argument('--save-all', action='store_true')
    p.add_argument('--out', default=os.path.join(HERE, 'microfluidics_sweep_results.json'))
    a = p.parse_args()
    sizes = [int(s) for s in a.sizes.split(',')] if a.sizes else None

    res = run_sweep(a.data_root, sizes, a.seeds, a.epochs, out_json=a.out,
                    log=lambda m: print(m, flush=True), steps=a.steps,
                    save_all=a.save_all)
    print('\nsummary', flush=True)
    for row in summarize(res):
        print(f"  {row['size']:>3} images: dice {row['dice_mean']:.4f} "
              f"+/- {row['dice_std']:.4f}   iou {row['iou_mean']:.4f} "
              f"+/- {row['iou_std']:.4f}", flush=True)
    print(f"saved {a.out}", flush=True)


if __name__ == '__main__':
    main()
