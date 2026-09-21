#!/usr/bin/env python3
"""Side-by-side video: frond identities before and after manual correction.

Left panel labels each detection with its raw btrack id, so the number changes
whenever a track fragments. Right panel labels the same detections with the
identity assigned after confirmation, so the number stays put.
"""
import argparse, collections, os, subprocess
import numpy as np, cv2
import run_persistence_review as rp, persistence_review as pr

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PER_FRAME = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
CHANGE_HOLD = 12          # frames to keep a relabelled id highlighted


def palette(i):
    rng = np.random.RandomState(i * 7919 + 13)
    c = rng.randint(90, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def draw(frame, dets, labels, focus_mask, title, changes=0, flash=False):
    """Featured frond drawn boldly; the rest muted so the eye follows one plant."""
    out = frame.copy()
    if flash:
        cv2.rectangle(out, (2, 2), (out.shape[1] - 3, out.shape[0] - 3), (40, 40, 230), 6)
    for (x, y), lab, is_focus in zip(dets, labels, focus_mask):
        if lab is None:
            continue
        colour = (40, 40, 230) if is_focus else (150, 150, 150)
        r, th, fs = (15, 3, 0.72) if is_focus else (11, 1, 0.5)
        cv2.circle(out, (int(x), int(y)), r, colour, th)
        cv2.putText(out, str(lab), (int(x) + r + 3, int(y) - 8), FONT, fs, (255, 255, 255), 4)
        cv2.putText(out, str(lab), (int(x) + r + 3, int(y) - 8), FONT, fs, colour, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, title, (10, 24), FONT, 0.66, (25, 25, 25), 2)
    h = out.shape[0]
    cv2.rectangle(out, (0, h - 32), (out.shape[1], h), (255, 255, 255), -1)
    col = (40, 40, 230) if changes else (90, 140, 60)
    cv2.putText(out, f"identity changes: {changes}", (10, h - 10), FONT, 0.62, col, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='396-639', help='frame range to render')
    ap.add_argument('--pad', type=int, default=95, help='crop margin in px')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--single', action='store_true',
                    help='feature only the frond that fragments most')
    ap.add_argument('--out', default=os.path.join(HERE, 'persistence_before_after.mp4'))
    a = ap.parse_args()
    f0, f1 = (int(v) for v in a.frames.split('-'))

    files = rp.list_images(rp._backup_images, 0)
    z = np.load(os.path.join(HERE, '.detections_cache_1664.npz'))
    tracks = rp.run_tracking(z['t'], z['x'], z['y'])
    auto, review = pr.find_candidates(tracks)
    dec = pr.load_decisions(os.path.join(HERE, 'manual_review_decisions_full.json'))
    canon = pr.build_canonical_map(len(tracks), auto, dec)

    # detections per frame, tagged with raw id and corrected id
    per_frame = collections.defaultdict(list)
    for i, tr in enumerate(tracks):
        for j, t in enumerate(tr.t):
            per_frame[int(t)].append((float(tr.x[j]), float(tr.y[j]), i, canon[i]))

    # frame the region that the featured group occupies
    groups = collections.defaultdict(list)
    for i, c in canon.items():
        groups[c].append(i)
    # every frond that gets relabelled inside this window
    focus = set()
    for g in groups.values():
        if len(g) < 2:
            continue
        segs = sorted(g, key=lambda i: tracks[i].t[0])
        if any(f0 <= tracks[b].t[0] <= f1 for b in segs[1:]):
            focus |= set(g)
    if a.single or not focus:           # one frond only, keeps the crop tight
        cands = [g for g in groups.values() if len(g) > 1
                 and any(f0 <= tracks[b].t[0] <= f1 for b in sorted(g, key=lambda i: tracks[i].t[0])[1:])]
        focus = set(max(cands or [max((g for g in groups.values() if len(g) > 1), key=len)],
                        key=len))
    xs = [x for i in focus for x in tracks[i].x]
    ys = [y for i in focus for y in tracks[i].y]
    H, W = cv2.imread(files[f0]).shape[:2]
    x0, x1 = max(0, int(min(xs)) - a.pad), min(W, int(max(xs)) + a.pad)
    y0, y1 = max(0, int(min(ys)) - a.pad), min(H, int(max(ys)) + a.pad)
    print(f"featuring {len(focus)} fragments across "
          f"{len({canon[i] for i in focus})} fronds, crop {x1-x0}x{y1-y0}")

    writer = None
    focus_can = canon[sorted(focus)[0]]
    last_raw = {}
    n_raw_changes = n_fixed_changes = 0
    HOLD = 10                      # frames to linger on each relabelling
    for f in range(f0, f1 + 1):
        img = cv2.imread(files[f])[y0:y1, x0:x1]
        dets = [(x - x0, y - y0) for x, y, i, c in per_frame.get(f, [])]
        raw = [i for x, y, i, c in per_frame.get(f, [])]
        fixed = [c for x, y, i, c in per_frame.get(f, [])]

        # track the featured frond's label on each side
        cur = {canon[i]: i for i in raw if i in focus}
        changed = any(k in last_raw and last_raw[k] != v for k, v in cur.items())
        n_raw_changes += sum(1 for k, v in cur.items()
                             if k in last_raw and last_raw[k] != v)
        last_raw.update(cur)

        hrs = f * MIN_PER_FRAME / 60.0
        focus_mask = [i in focus for i in raw]
        left = draw(img, dets, raw, focus_mask, "before correction",
                    n_raw_changes, flash=changed)
        right = draw(img, dets, fixed, focus_mask, "after correction",
                     n_fixed_changes)
        gap = np.full((left.shape[0], 8, 3), 235, np.uint8)
        canvas = np.hstack([left, gap, right])
        cv2.putText(canvas, f"{hrs:.1f} h", (canvas.shape[1] - 78, 24),
                    FONT, 0.6, (110, 110, 110), 2)
        if canvas.shape[0] % 2:                    # H.264 needs even dimensions
            canvas = canvas[:-1]
        if canvas.shape[1] % 2:
            canvas = canvas[:, :-1]
        if writer is None:
            h, w = canvas.shape[:2]
            writer = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                 '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(a.fps), '-i', '-',
                 '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                 '-crf', '20', '-movflags', '+faststart', a.out],
                stdin=subprocess.PIPE)
        for _ in range(HOLD if changed else 1):     # linger when the label flips
            writer.stdin.write(canvas.tobytes())
    writer.stdin.close()
    writer.wait()
    print(f"saved {a.out}  ({f1 - f0 + 1} frames at {a.fps} fps, "
          f"{(f1 - f0 + 1) / a.fps:.0f} s)")


if __name__ == '__main__':
    main()
