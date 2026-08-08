#!/usr/bin/env python3
"""Render Deepweed supplementary videos with POST-STITCHING track ID labels.

Fixes Reviewer 2 Comment 2: the original videos labelled fronds with per-frame
connected-component numbers, so labels changed between frames. This script runs
segmentation -> btrack -> fragment stitching, then renders the overlay video
labelling every frond with its stitched track ID (stable across the video).

Usage (petri dish):
  python3 render_stitched.py --dataset petri_dish \
      --frames render/frames --model best_instance_unet_512.pt \
      --input-size 512 --btrack-config cell_config.json \
      --out render/out/Oliveira2026_SI_Video-1_stitched.mp4

Usage (microfluidics):
  python3 render_stitched.py --dataset microfluidics \
      --video duckweed_25_0504_multiple.avi --model best_unet_microfluidics_boundary.pt \
      --btrack-config cell_config.json \
      --out render/out/Oliveira2026_SI_Video-2_stitched.mp4
"""
import argparse, os, sys, io, glob, logging, contextlib, time
from pathlib import Path

import numpy as np
import cv2
import torch
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize as sk_resize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import btrack
from btrack.utils import segmentation_to_objects

logging.getLogger("btrack").setLevel(logging.ERROR)

# ----------------------------------------------------------------- model ----
def load_unet(model_path, n_classes, device):
    """Load UNet trying unet_model_class first, then train_512's class."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    last_err = None
    for modname in ("unet_model_class", "train_512"):
        try:
            mod = __import__(modname)
            model = mod.UNet(n_channels=3, n_classes=n_classes).to(device)
            model.load_state_dict(state)
            model.eval()
            print(f"Model loaded via {modname}.UNet")
            return model
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load model: {last_err}")


def segment_petri(model, image_rgb, device, input_size=512, min_area=30):
    """3-class boundary model: bg/body/boundary -> connected components."""
    original_shape = image_rgb.shape[:2]
    img = cv2.resize(image_rgb, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(t)
    pred = output.argmax(dim=1).squeeze().cpu().numpy()
    pred_full = cv2.resize(pred.astype(np.uint8),
                           (original_shape[1], original_shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    body_mask = (pred_full == 1).astype(np.uint8)
    instance_mask = measure.label(body_mask)
    if min_area > 0:
        for r in regionprops(instance_mask):
            if r.area < min_area:
                instance_mask[instance_mask == r.label] = 0
        instance_mask = measure.label(instance_mask > 0)
    return instance_mask


def segment_micro(model, image_rgb, device, min_area=3):
    """1-class model: sigmoid -> threshold -> connected components."""
    original_shape = image_rgb.shape[:2]
    # equivalent of torchvision ToTensor + Resize((512, 128)) without torchvision
    img = cv2.resize(image_rgb, (128, 512), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(t)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_small = (pred_mask > 0.5).astype(np.uint8)
    binary_full = sk_resize(binary_small, original_shape, preserve_range=True)
    body_mask = (binary_full > 0.5).astype(np.uint8)
    instance_mask = measure.label(body_mask)
    if min_area > 0:
        for r in regionprops(instance_mask):
            if r.area < min_area:
                instance_mask[instance_mask == r.label] = 0
        instance_mask = measure.label(instance_mask > 0)
    return instance_mask


# ------------------------------------------------------------- stitching ----
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_stitch_map(tracks, max_gap=3, max_dist=20.0, offsets=None):
    """Union tracks whose end matches another track's start (same frond).
    If `offsets` (per-frame cumulative global drift) is given, endpoints are
    matched in drift-corrected coordinates so stage bumps do not break IDs."""
    def corrected(sign):
        starts, ends = {}, {}
        for i, tr in enumerate(tracks):
            t0i, t1i = int(tr.t[0]), int(tr.t[-1])
            if offsets is not None:
                ox0, oy0 = offsets[min(t0i, len(offsets)-1)]
                ox1, oy1 = offsets[min(t1i, len(offsets)-1)]
            else:
                ox0 = oy0 = ox1 = oy1 = 0.0
            starts.setdefault(t0i, []).append((i, tr.x[0] - sign*ox0, tr.y[0] - sign*oy0))
            ends.setdefault(t1i, []).append((i, tr.x[-1] - sign*ox1, tr.y[-1] - sign*oy1))
        return starts, ends

    def count_merges(starts, ends):
        n = 0
        for frame in sorted(ends.keys()):
            e = ends[frame]
            for gap in range(1, max_gap + 1):
                s = starts.get(frame + gap, [])
                if not e or not s:
                    continue
                D = cdist(np.array([(x, y) for _, x, y in e]),
                          np.array([(x, y) for _, x, y in s]))
                ri, ci = linear_sum_assignment(D)
                n += sum(1 for r, c in zip(ri, ci) if D[r, c] < max_dist)
        return n

    if offsets is not None:
        s_pos, e_pos = corrected(+1)
        s_neg, e_neg = corrected(-1)
        n_pos, n_neg = count_merges(s_pos, e_pos), count_merges(s_neg, e_neg)
        if n_neg > n_pos:
            track_starts, track_ends = s_neg, e_neg
            print(f"Drift correction: sign=-1 ({n_neg} vs {n_pos} merges)")
        else:
            track_starts, track_ends = s_pos, e_pos
            print(f"Drift correction: sign=+1 ({n_pos} vs {n_neg} merges)")
    else:
        track_starts, track_ends = corrected(0)

    uf = UnionFind(len(tracks))

    # each track end and each track start may participate in at most ONE
    # stitch, and merged groups must be disjoint in time (a frond is one
    # timeline; two coexisting tracks are never the same frond)
    used_end, used_start = set(), set()
    g_first = {i: int(tr.t[0]) for i, tr in enumerate(tracks)}
    g_last = {i: int(tr.t[-1]) for i, tr in enumerate(tracks)}

    def group_bounds(i):
        r = uf.find(i)
        return g_first[r], g_last[r]

    def merge(a, b):
        ra, rb = uf.find(a), uf.find(b)
        fa, la = g_first[ra], g_last[ra]
        fb, lb = g_first[rb], g_last[rb]
        uf.union(a, b)
        r = uf.find(a)
        g_first[r] = min(fa, fb)
        g_last[r] = max(la, lb)

    for gap in range(1, max_gap + 1):          # shortest gaps claim first
        for frame in sorted(track_ends.keys()):
            ends = [e for e in track_ends[frame] if e[0] not in used_end]
            starts = [s for s in track_starts.get(frame + gap, [])
                      if s[0] not in used_start]
            if not ends or not starts:
                continue
            end_pos = np.array([(x, y) for _, x, y in ends])
            start_pos = np.array([(x, y) for _, x, y in starts])
            dists = cdist(end_pos, start_pos)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] >= max_dist:
                    continue
                a, b = ends[r][0], starts[c][0]
                if uf.find(a) == uf.find(b):
                    continue
                # temporal disjointness of the merged groups
                fa, la = group_bounds(a)
                fb, lb = group_bounds(b)
                if not (la < fb or lb < fa):
                    continue
                merge(a, b)
                used_end.add(a)
                used_start.add(b)

    # sequential display IDs ordered by first appearance of each merged group
    groups = {}
    for i, tr in enumerate(tracks):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    order = sorted(groups.keys(), key=lambda r: min(int(tracks[i].t[0]) for i in groups[r]))
    display_id = {}
    for sid, root in enumerate(order, start=1):
        for i in groups[root]:
            display_id[i] = sid
    n_merged = len(tracks) - len(groups)
    print(f"Stitching: {len(tracks)} raw tracks -> {len(groups)} stitched tracks "
          f"({n_merged} fragments merged)")
    return display_id


# ------------------------------------------------------------- rendering ----
def sid_color(sid):
    hue = int((sid * 37) % 180)
    hsv = np.uint8([[[hue, 220, 230]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return tuple(int(c) for c in rgb)


def draw_overlay(frame_rgb, instance_mask, inst_to_sid, frame_idx,
                 minutes_per_frame, is_micro, alpha=0.55):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    uid_list = sorted([u for u in np.unique(instance_mask) if u > 0])
    num_fronds = len(uid_list)

    id_font_scale = 0.28 if is_micro else 0.7
    id_thickness = 1 if is_micro else 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for uid in uid_list:
        mask = instance_mask == uid
        sid = inst_to_sid.get(uid)
        c = sid_color(sid) if sid else (160, 160, 160)

        roi = canvas[mask].astype(np.float32)
        canvas[mask] = np.clip((1 - alpha) * roi + alpha * np.array(c, np.float32),
                               0, 255).astype(np.uint8)
        if sid:
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            label = str(sid)
            cv2.putText(canvas, label, (cx - 1, cy + 1), font, id_font_scale,
                        (0, 0, 0), id_thickness + 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (cx, cy), font, id_font_scale,
                        (0, 255, 255), id_thickness, cv2.LINE_AA)

    total_minutes = frame_idx * minutes_per_frame
    hours = total_minutes // 60
    mins = total_minutes % 60
    time_str = f"{int(hours):02d}:{int(mins):02d}" if hours > 0 else f"{int(mins):02d}:00"
    lines = [f"Frame: {frame_idx}", f"Time: {time_str} (min {int(total_minutes)})"]
    frond_line = f"Fronds: {num_fronds}"

    if is_micro:
        fs, th, line_h, box_w = 0.3, 1, 14, 150
    else:
        fs, th, line_h, box_w = 0.7, 2, 30, 340
    box_h = line_h * (len(lines) + 1) + 8
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (4, 12 + i * line_h), font, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(canvas, frond_line, (4, 12 + len(lines) * line_h), font, fs,
                (0, 255, 0), th, cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["petri_dish", "microfluidics"], required=True)
    ap.add_argument("--frames", help="directory of jpeg frames (petri)")
    ap.add_argument("--video", help="input video (microfluidics)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--btrack-config", default="cell_config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--minutes-per-frame", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--mask-cache", default="mask_cache")
    ap.add_argument("--frame-limit", type=int, default=None)
    ap.add_argument("--skip-seg", action="store_true",
                    help="reuse cached masks instead of running the model")
    ap.add_argument("--reuse-tracks", action="store_true",
                    help="load tracks_cache.json from mask-cache dir, skip btrack")
    ap.add_argument("--stitch-gap", type=int, default=3)
    ap.add_argument("--stitch-dist", type=float, default=20.0)
    ap.add_argument("--min-label-len", type=int, default=1,
                    help="only label tracks whose stitched span >= this many frames")
    args = ap.parse_args()

    is_micro = args.dataset == "microfluidics"
    minutes_per_frame = args.minutes_per_frame or (20 if is_micro else 5)
    min_area = args.min_area if args.min_area is not None else (3 if is_micro else 30)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = None
    if not args.skip_seg:
        model = load_unet(args.model, n_classes=1 if is_micro else 3, device=device)

    os.makedirs(args.mask_cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- frame source ----
    if is_micro:
        cap = cv2.VideoCapture(args.video)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        image_files = sorted(Path(args.frames).glob("*.jpeg")) + \
                      sorted(Path(args.frames).glob("*.jpg"))
        image_files = [p for p in image_files if not p.name.startswith("._")]
        n_frames = len(image_files)
    if args.frame_limit:
        n_frames = min(n_frames, args.frame_limit)
    print(f"Frames to process: {n_frames}")

    # ---- pass 1: segment + collect btrack objects ----
    _tcache = os.path.join(args.mask_cache, "tracks_cache.json")
    _skip_pass1 = args.reuse_tracks and os.path.exists(_tcache)
    all_objects = []
    t0 = time.time()
    for fidx in range(n_frames if not _skip_pass1 else 0):
        if is_micro:
            ret, frame_bgr = cap.read()
            if not ret:
                break
        else:
            frame_bgr = cv2.imread(str(image_files[fidx]))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cache_p = os.path.join(args.mask_cache, f"m{fidx:05d}.png")
        if args.skip_seg and os.path.exists(cache_p):
            inst = cv2.imread(cache_p, cv2.IMREAD_UNCHANGED).astype(np.int32)
        else:
            if is_micro:
                inst = segment_micro(model, frame_rgb, device, min_area=min_area)
            else:
                inst = segment_petri(model, frame_rgb, device,
                                     input_size=args.input_size, min_area=min_area)
            cv2.imwrite(cache_p, inst.astype(np.uint16))

        m3 = np.zeros((1, *inst.shape), dtype=inst.dtype)
        m3[0] = inst
        with contextlib.redirect_stderr(io.StringIO()):
            objs = segmentation_to_objects(m3, properties=("centroid",))
        for o in objs:
            o.t = fidx
        all_objects.extend(objs)

        if fidx % 100 == 0:
            el = time.time() - t0
            print(f"  seg {fidx}/{n_frames}  ({el:.0f}s elapsed)", flush=True)
    if is_micro:
        cap.release()
    print(f"Segmentation done: {len(all_objects)} detections "
          f"({time.time()-t0:.0f}s)")

    # ---- tracking (with cache) ----
    import json as _json

    class _Trk:
        __slots__ = ("t", "x", "y")
        def __init__(self, d):
            self.t, self.x, self.y = d["t"], d["x"], d["y"]

    tcache = os.path.join(args.mask_cache, "tracks_cache.json")
    if args.reuse_tracks and os.path.exists(tcache):
        tracks = [_Trk(d) for d in _json.load(open(tcache))]
        print(f"Tracks loaded from cache: {len(tracks)}")
    else:
        print("Running btrack...")
        with btrack.BayesianTracker() as tracker:
            if Path(args.btrack_config).exists():
                tracker.configure(btrack.config.load_config(args.btrack_config))
            else:
                tracker.configure(btrack.config.TrackerConfig())
            tracker.max_search_radius = 50
            tracker.append(all_objects)
            tracker.track(step_size=50)
            tracker.optimize()
            tracks = list(tracker.tracks)
        print(f"Tracking done: {len(tracks)} raw tracks")
        _json.dump([{"t": [int(v) for v in tr.t],
                     "x": [float(v) for v in tr.x],
                     "y": [float(v) for v in tr.y]} for tr in tracks],
                   open(tcache, "w"))
        print(f"Tracks cached -> {tcache}")

    # ---- stitching (with optional drift correction) ----
    offsets = None
    off_p = os.path.join(args.mask_cache, "offsets.json")
    if os.path.exists(off_p):
        import json as _j
        offsets = [tuple(v) for v in _j.load(open(off_p))["cumulative"]]
        print(f"Loaded per-frame drift offsets ({len(offsets)} frames)")
    display_id = build_stitch_map(tracks, max_gap=args.stitch_gap,
                                  max_dist=args.stitch_dist, offsets=offsets)

    # suppress labels for short-lived stitched groups (junk detections)
    if args.min_label_len > 1:
        span = {}
        for i, tr in enumerate(tracks):
            sid = display_id[i]
            lo, hi = int(tr.t[0]), int(tr.t[-1])
            if sid in span:
                span[sid] = (min(span[sid][0], lo), max(span[sid][1], hi))
            else:
                span[sid] = (lo, hi)
        short = {sid for sid, (lo, hi) in span.items()
                 if hi - lo + 1 < args.min_label_len}
        display_id = {i: (None if sid in short else sid)
                      for i, sid in display_id.items()}
        print(f"Label suppression: {len(short)} short-lived tracks unlabeled")

    # per-frame stitched positions: frame -> list[(sid, x, y)]
    frame_pos = {}
    for i, tr in enumerate(tracks):
        sid = display_id[i]
        if sid is None:
            continue
        for j, t in enumerate(tr.t):
            frame_pos.setdefault(int(t), []).append((sid, float(tr.x[j]), float(tr.y[j])))

    # ---- auto-detect coordinate convention (btrack x/y vs row/col) ----
    # sample a mid frame, compare both orientations
    probe_f = n_frames // 2
    probe_mask = cv2.imread(os.path.join(args.mask_cache, f"m{probe_f:05d}.png"),
                            cv2.IMREAD_UNCHANGED)
    props = regionprops(probe_mask.astype(np.int32))
    tps = frame_pos.get(probe_f, [])
    swap = False
    if props and tps:
        cent = np.array([[p.centroid[1], p.centroid[0]] for p in props])  # (x=col, y=row)
        tpos = np.array([[x, y] for _, x, y in tps])
        d_norm = cdist(cent, tpos).min(axis=1).mean()
        d_swap = cdist(cent, tpos[:, ::-1]).min(axis=1).mean()
        swap = d_swap < d_norm
        print(f"Coordinate check: normal={d_norm:.1f}px swapped={d_swap:.1f}px "
              f"-> {'SWAP' if swap else 'normal'}")

    # ---- pass 2: render ----
    if is_micro:
        cap = cv2.VideoCapture(args.video)
        ret, probe = cap.read()
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        probe = cv2.imread(str(image_files[0]))
        h, w = probe.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    t0 = time.time()

    for fidx in range(n_frames):
        if is_micro:
            ret, frame_bgr = cap.read()
            if not ret:
                break
        else:
            frame_bgr = cv2.imread(str(image_files[fidx]))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inst = cv2.imread(os.path.join(args.mask_cache, f"m{fidx:05d}.png"),
                          cv2.IMREAD_UNCHANGED).astype(np.int32)

        # assign each instance to nearest stitched track position this frame
        inst_to_sid = {}
        props = regionprops(inst)
        tps = frame_pos.get(fidx, [])
        if props and tps:
            cent = np.array([[p.centroid[1], p.centroid[0]] for p in props])
            tpos = np.array([[x, y] for _, x, y in tps])
            if swap:
                tpos = tpos[:, ::-1]
            dists = cdist(cent, tpos)
            row_ind, col_ind = linear_sum_assignment(dists)
            seen_sids = {}
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] < 30:
                    sid = tps[c][0]
                    if sid in seen_sids:      # never label two fronds alike
                        continue
                    seen_sids[sid] = True
                    inst_to_sid[props[r].label] = sid

        canvas = draw_overlay(frame_rgb, inst, inst_to_sid, fidx,
                              minutes_per_frame, is_micro)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        if fidx % 100 == 0:
            print(f"  render {fidx}/{n_frames}  ({time.time()-t0:.0f}s elapsed)",
                  flush=True)

    writer.release()
    if is_micro:
        cap.release()
    print(f"DONE -> {args.out}  ({time.time()-t0:.0f}s render)")


if __name__ == "__main__":
    main()
