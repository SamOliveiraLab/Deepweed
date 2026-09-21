#!/usr/bin/env python3
"""Two frames of one frond, labelled before and after manual correction."""
import collections, os
import numpy as np, cv2
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patheffects as pe
import run_persistence_review as rp, persistence_review as pr

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTE, HAIR = '#1A1A1A', '#6E6E6E', '#C9C9C9'
HOT, COOL = '#E63946', '#1A1A1A'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})

files = rp.list_images(rp._backup_images, 0)
z = np.load(os.path.join(HERE, '.detections_cache_1664.npz'))
tracks = rp.run_tracking(z['t'], z['x'], z['y'])
auto, review = pr.find_candidates(tracks)
dec = pr.load_decisions(os.path.join(HERE, 'manual_review_decisions_full.json'))
canon = pr.build_canonical_map(len(tracks), auto, dec)

groups = collections.defaultdict(list)
for i, c in canon.items():
    groups[c].append(i)
# the frond that fragments the most, so the relabelling is unmistakable
focus = sorted(max((g for g in groups.values() if len(g) > 1), key=len),
               key=lambda i: tracks[i].t[0])
first, last = focus[0], focus[-1]
f_a, f_b = int(tracks[first].t[0]) + 4, int(tracks[last].t[-1]) - 4

def at(track_i, f):
    t = list(tracks[track_i].t)
    j = t.index(f) if f in t else min(range(len(t)), key=lambda k: abs(t[k] - f))
    return float(tracks[track_i].x[j]), float(tracks[track_i].y[j])

x0, y0 = at(first, f_a)
x1, y1 = at(last, f_b)
PAD = 105
halo = [pe.withStroke(linewidth=2.2, foreground='white')]

fig = plt.figure(figsize=(7.4, 7.8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.06, wspace=0.05,
                       left=0.09, right=0.98, top=0.88, bottom=0.07)
rows = [(f_a, x0, y0, first, canon[first]), (f_b, x1, y1, last, canon[last])]
for r, (f, x, y, raw_id, fix_id) in enumerate(rows):
    img = cv2.cvtColor(cv2.imread(files[f]), cv2.COLOR_BGR2RGB)
    crop, (cx, cy) = pr._crop(img, x, y, PAD)
    for c, (lab, colour) in enumerate([(raw_id, HOT), (fix_id, COOL)]):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(crop)
        ax.plot(cx, cy, 'o', ms=26, mfc='none', mec=colour, mew=2.4)
        ax.annotate(str(lab), xy=(cx, cy), xytext=(20, -16), fontsize=15,
                    fontweight='bold', color=colour, textcoords='offset points',
                    path_effects=halo)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(HAIR); sp.set_linewidth(0.7)
        if r == 0:
            ax.set_title('before correction' if c == 0 else 'after correction',
                         fontsize=11.5, fontweight='bold', pad=8)
        if c == 0:
            ax.set_ylabel(f"{f * 5 / 60:.1f} h", fontsize=11, color=MUTE, labelpad=8)

gap_h = (f_b - f_a) * 5 / 60
fig.text(0.09, 0.975, f'The same frond, {gap_h:.0f} hours apart', fontsize=12.5,
         fontweight='bold', color=INK)
fig.text(0.09, 0.950, f'Its track fragments {len(focus) - 1} times, so btrack renames it '
         f'{first} to {last}. After confirmation it keeps one identity.',
         fontsize=9.5, color=MUTE)
fig.text(0.09, 0.022, f"Full series: 43 pairs reviewed, 42 confirmed. Identity switches "
         f"35 to 11 (0.11% to 0.03%); 367 raw tracks resolve to 319 fronds.",
         fontsize=9, color=MUTE)
out = os.path.join(HERE, 'persistence_before_after_frames.png')
fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
print("saved", out, "| frames", f_a, f_b, "| ids", rows[0][3], "->", rows[1][3],
      "| corrected", rows[0][4])
