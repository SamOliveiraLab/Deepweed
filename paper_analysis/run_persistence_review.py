#!/usr/bin/env python
"""Frond-persistence y/n review, end to end (Reviewer 1 correction feature).

Segments the petri-dish series, tracks with btrack, then opens the
interactive popup where ambiguous track end->start pairs are confirmed or
rejected with y/n. Recomputes the ID-switch rate before vs. after the
human correction and saves both the decisions and the metrics.

Usage:
    python run_persistence_review.py              # 10-hour window (120 frames)
    python run_persistence_review.py --no-gui     # replay saved decisions only
    python run_persistence_review.py --frames 40  # shorter window

Detections are cached per frame count, so only the first run segments.
Decisions land in manual_review_decisions.json, metrics in
persistence_review_results.json.
"""
import argparse
import contextlib
import glob
import io
import json
import os

import numpy as np
import cv2
import torch
from skimage import measure
from skimage.measure import regionprops

import btrack
from btrack.utils import segmentation_to_objects

import unet_model_class
import persistence_review as pr

BASE = os.path.dirname(os.path.abspath(__file__))
PETRI_MODEL = os.path.join(BASE, 'data_model/model/best_instance_unet.pt')
BTRACK_CONFIG = os.path.join(BASE, 'cell_config.json')
_repo_images = os.path.join(BASE, 'data_model/data/petri_dish/')
_backup_images = ('/Volumes/Extreme SSD/04- Deepweed & Duckweed/'
                  'Amby_duckweed_analysis_backup/paper_analysis/data_model/'
                  'data/petri_dish/')
MINUTES_PER_FRAME = 5


def list_images(images_dir, n_frames):
    files = sorted(f for f in glob.glob(os.path.join(images_dir, '*.jpeg'))
                   if not os.path.basename(f).startswith('._'))
    return files[:n_frames] if n_frames else files


def segment_series(files, cache_path, min_area=30):
    """Per-frame frond centroids (t, x, y), cached to `cache_path`."""
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        print(f"loaded {len(z['t'])} cached detections ({os.path.basename(cache_path)})")
        return z['t'], z['x'], z['y']

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = unet_model_class.UNet(n_channels=3, n_classes=3).to(device)
    ckpt = torch.load(PETRI_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ts, xs, ys = [], [], []
    for idx, fp in enumerate(files):
        rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out = model(t)
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pf = cv2.resize(pred.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        inst = measure.label((pf == 1).astype(np.uint8))
        for r in regionprops(inst):
            if r.area < min_area:
                inst[inst == r.label] = 0
        inst = measure.label(inst > 0)
        l3 = np.zeros((1, *inst.shape), dtype=inst.dtype)
        l3[0] = inst
        with contextlib.redirect_stderr(io.StringIO()):
            objs = segmentation_to_objects(l3, properties=('centroid',))
        for o in objs:
            ts.append(idx)
            xs.append(float(o.x))
            ys.append(float(o.y))
        if (idx + 1) % 20 == 0:
            print(f"  segmented {idx + 1}/{len(files)}")

    ts, xs, ys = np.array(ts), np.array(xs), np.array(ys)
    np.savez(cache_path, t=ts, x=xs, y=ys)
    return ts, xs, ys


def run_tracking(ts, xs, ys):
    from btrack.btypes import PyTrackObject
    objects = [PyTrackObject.from_dict(
                   {'ID': i, 't': int(t), 'x': float(x), 'y': float(y), 'z': 0.0})
               for i, (t, x, y) in enumerate(zip(ts, xs, ys))]
    with btrack.BayesianTracker() as tk:
        tk.configure(btrack.config.load_config(BTRACK_CONFIG))
        tk.max_search_radius = 50
        tk.append(objects)
        tk.track(step_size=50)
        tk.optimize()
        return list(tk.tracks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', type=int, default=120,
                    help='number of frames (default 120 = 10 h; 0 = all)')
    ap.add_argument('--images', default=None, help='petri-dish image directory')
    ap.add_argument('--no-gui', action='store_true',
                    help='no popup; replay saved decisions and print metrics')
    ap.add_argument('--decisions', default='manual_review_decisions.json',
                    help='decision log; use a separate file per frame window')
    args = ap.parse_args()

    images_dir = args.images or (
        _repo_images if glob.glob(os.path.join(_repo_images, '*.jpeg'))
        else _backup_images)
    files = list_images(images_dir, args.frames)
    n_frames = len(files)
    decisions_path = os.path.join(BASE, args.decisions)
    results_path = os.path.join(
        BASE, os.path.splitext(args.decisions)[0].replace('_decisions', '')
        + '_results.json')
    cache_path = os.path.join(BASE, f'.detections_cache_{n_frames}.npz')

    ts, xs, ys = segment_series(files, cache_path)
    tracks = run_tracking(ts, xs, ys)
    print(f"{len(tracks)} raw tracks over {n_frames} frames")

    auto_links, review = pr.find_candidates(tracks)
    print(f"auto-linked: {len(auto_links)}   needs y/n review: {len(review)}")

    if args.no_gui:
        decisions = pr.load_decisions(decisions_path)
    else:
        decisions = pr.review_candidates_popup(review, files, decisions_path)

    # only decisions matching this run's candidates apply (track indices
    # from a different frame window would not line up)
    valid_keys = {pr.candidate_key(c) for c in review}
    decisions = {k: d for k, d in decisions.items() if k in valid_keys}
    pending = len(valid_keys) - len(decisions)
    if pending:
        print(f"WARNING: {pending} candidate(s) still unreviewed")

    canonical = pr.build_canonical_map(len(tracks), auto_links, decisions)
    before = pr.compute_id_switch_metrics(tracks, n_frames)
    after = pr.compute_id_switch_metrics(tracks, n_frames, canonical_map=canonical)

    hours = n_frames * MINUTES_PER_FRAME / 60
    print(f"\nPetri dish, {n_frames} frames ({hours:.1f} h)")
    print(f"{'':>28} {'before':>10} {'after':>10}")
    print(f"{'tracks (raw -> effective)':>28} {before['n_raw_tracks']:>10} "
          f"{after['n_effective_tracks']:>10}")
    print(f"{'ID switches':>28} {before['id_switches']:>10} "
          f"{after['id_switches']:>10}")
    print(f"{'ID switch rate':>28} {before['id_switch_rate']:>9.2%} "
          f"{after['id_switch_rate']:>9.2%}")
    print(f"\nmanual correction: {pr.summarize_decisions(decisions)}")

    results = {
        'dataset': 'petri_dish',
        'n_frames': n_frames,
        'hours': hours,
        'auto_links': len(auto_links),
        'review': pr.summarize_decisions(decisions),
        'pending': pending,
        'before': before,
        'after': after,
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"saved {results_path}")


if __name__ == '__main__':
    main()
