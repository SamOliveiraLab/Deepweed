#!/usr/bin/env python3
"""Detect sudden global image shifts across the petri time-lapse via phase
correlation on downsampled grayscale frames. Saves per-frame cumulative
offsets to petri_cache/offsets.json for drift-corrected stitching."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2

# usage: python3 detect_shifts.py <frames_dir> <output_dir>
FRAMES = sys.argv[1] if len(sys.argv) > 1 else "data_model/data/petri_dish"
CACHE = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(CACHE, exist_ok=True)

files = sorted(Path(FRAMES).glob("*.jpeg"))
files = [p for p in files if not p.name.startswith("._")]
print(f"{len(files)} frames")

SCALE = 0.25  # 1600x1200 -> 400x300, plenty for global shift
prev = None
shifts = []   # per-pair (dx, dy) at full-res scale
t0 = time.time()
for i, p in enumerate(files):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE).astype(np.float32)
    if prev is not None:
        (dx, dy), _ = cv2.phaseCorrelate(prev, small)
        shifts.append((dx / SCALE, dy / SCALE))
    prev = small
    if i % 400 == 0:
        print(f"  {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

mags = np.array([np.hypot(dx, dy) for dx, dy in shifts])
print(f"\nshift magnitude: p50={np.median(mags):.2f}px p95={np.percentile(mags,95):.2f}px max={mags.max():.2f}px")

# report all jumps > 8 px (frame i -> i+1)
print("\njumps > 8px:")
for i, m in enumerate(mags):
    if m > 8:
        dx, dy = shifts[i]
        print(f"  frame {i} -> {i+1}: {m:.1f}px  (dx={dx:+.1f}, dy={dy:+.1f})   [{files[i].name} -> {files[i+1].name}]")

# cumulative offset per frame (frame 0 anchored)
cum = [(0.0, 0.0)]
for dx, dy in shifts:
    cum.append((cum[-1][0] + dx, cum[-1][1] + dy))
json.dump({"cumulative": cum}, open(os.path.join(CACHE, "offsets.json"), "w"))
print(f"\ncumulative offsets saved -> {CACHE}/offsets.json")
print(f"net drift over video: dx={cum[-1][0]:+.1f}px dy={cum[-1][1]:+.1f}px")
