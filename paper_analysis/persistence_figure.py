#!/usr/bin/env python3
"""Figure: interactive frond-persistence correction, full petri dish series."""
import os, collections
import numpy as np, cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import run_persistence_review as rp, persistence_review as pr

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = "/Volumes/Extreme SSD/Deepweed/.scratch/yn_review/frames"
MIN_PER_FRAME = 5

INK, MUTE, HAIR = '#1A1A1A', '#6E6E6E', '#C9C9C9'
END_C, START_C = '#E63946', '#3D7EA6'          # track end, candidate start
FRAG = ['#4C6E91', '#8AA6BF', '#B9C9D8']       # fragment segments, before
FRAG_LIGHT = '#AFC2D4'                         # raw tracks, muted
JOIN = '#1A1A1A'                               # merged identity, after
AMBER = '#C0392B'                              # the one skipped call

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK,
                     'axes.edgecolor': HAIR, 'axes.labelcolor': INK,
                     'xtick.color': MUTE, 'ytick.color': MUTE,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})


def build():
    files = rp.list_images(rp._backup_images, 0)
    z = np.load(os.path.join(HERE, ".detections_cache_1664.npz"))
    tracks = rp.run_tracking(z['t'], z['x'], z['y'])
    auto, review = pr.find_candidates(tracks)
    dec = pr.load_decisions(os.path.join(HERE, "manual_review_decisions_full.json"))
    canon = pr.build_canonical_map(len(tracks), auto, dec)
    n = len(files)
    return (files, tracks, auto, review, dec, canon,
            pr.compute_id_switch_metrics(tracks, n),
            pr.compute_id_switch_metrics(tracks, n, canonical_map=canon))


def frame(i, files):
    p = os.path.join(STAGE, f"frame_{i:05d}.jpeg")
    return cv2.cvtColor(cv2.imread(p if os.path.exists(p) else files[i]), cv2.COLOR_BGR2RGB)


def letter(fig, x, y, ch):
    fig.text(x, y, ch, fontsize=22, fontweight='bold', color=INK, va='top')


def candidate_row(fig, gs, r, cand, ans, files, note, first):
    e, s = frame(cand['end_frame'], files), frame(cand['start_frame'], files)
    views = [(e, cand['end_xy'], 70, f"track ends, frame {cand['end_frame']}"),
             (s, cand['start_xy'], 70, f"track starts, frame {cand['start_frame']}"),
             (e, cand['end_xy'], 240, f"zoomed out, frame {cand['end_frame']}")]
    halo = [pe.withStroke(linewidth=1.6, foreground='white')]

    def tag(ax, x, y, tid, colour, dx=9, dy=-9, size=10.5):
        ax.annotate(str(tid), xy=(x, y), xytext=(dx, dy),
                    textcoords='offset points', fontsize=size, fontweight='bold',
                    color=colour, path_effects=halo, ha='left', va='top')

    for k, (img, xy, half, title) in enumerate(views):
        ax = fig.add_subplot(gs[r, k])
        crop, (cx, cy) = pr._crop(img, xy[0], xy[1], half)
        ax.imshow(crop)
        if k == 0:
            ax.plot(cx, cy, 'o', ms=13, mfc='none', mec=END_C, mew=1.9)
            tag(ax, cx, cy, cand['end_track'], END_C)
        elif k == 1:
            ax.plot(cx, cy, 's', ms=13, mfc='none', mec=START_C, mew=1.9)
            tag(ax, cx, cy, cand['start_track'], START_C)
        else:
            dx, dy = (cand['start_xy'][0] - cand['end_xy'][0],
                      cand['start_xy'][1] - cand['end_xy'][1])
            ax.plot(cx, cy, 'o', ms=8, mfc='none', mec=END_C, mew=1.7)
            ax.plot(cx + dx, cy + dy, 's', ms=8, mfc='none', mec=START_C, mew=1.7)
            tag(ax, cx, cy, cand['end_track'], END_C, dx=-17, dy=-4, size=9.5)
            tag(ax, cx + dx, cy + dy, cand['start_track'], START_C, dx=8, dy=17, size=9.5)
        if first:
            ax.set_title(title, fontsize=11.5, color=MUTE, pad=6, fontweight='bold')
        else:
            ax.set_title(title, fontsize=11.5, color=MUTE, pad=6, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(HAIR); sp.set_linewidth(0.6)

    ax = fig.add_subplot(gs[r, 3]); ax.axis('off')
    label, colour = (("Confirmed  ·  same frond", INK) if ans == 'y'
                     else ("Skipped  ·  not called", AMBER))
    ax.plot([0.0, 0.34], [0.70, 0.70], color=colour, lw=2.2,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.0, 0.58, label, fontsize=13.5, fontweight='bold', color=colour,
            transform=ax.transAxes)
    ax.text(0.0, 0.42, f"gap {cand['gap']} frame{'s' if cand['gap'] > 1 else ''}"
                       f"   ·   {cand['dist']:.1f} px", fontsize=8.4, color=MUTE,
            transform=ax.transAxes)
    ax.text(0.0, 0.10, note, fontsize=11, color=MUTE, transform=ax.transAxes,
            linespacing=1.5, va='bottom')
    if first:
        T = ax.transAxes
        ax.plot(0.022, 1.30, 'o', ms=7, mfc='none', mec=END_C, mew=1.6,
                transform=T, clip_on=False)
        ax.text(0.075, 1.30, 'last position of the ending track', fontsize=11.5,
                color=MUTE, transform=T, va='center')
        ax.plot(0.022, 1.16, 's', ms=7, mfc='none', mec=START_C, mew=1.6,
                transform=T, clip_on=False)
        ax.text(0.075, 1.16, 'first position of the candidate track', fontsize=11.5,
                color=MUTE, transform=T, va='center')


def timeline(fig, gs, r, tracks, canon, n_show=5):
    """A few fronds: raw tracks with real gaps at the breaks, and the single
    identity obtained after confirmation."""
    ax = fig.add_subplot(gs[r, :])
    groups = collections.defaultdict(list)
    for i, c in canon.items():
        groups[c].append(i)
    merged = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)[:n_show]
    merged.sort(key=lambda g: min(tracks[i].t[0] for i in g))
    h = lambda f: f * MIN_PER_FRAME / 60.0
    span = max(h(tracks[i].t[-1]) for g in merged for i in g)
    pad = span * 0.006          # visible break, independent of track length

    labels = []
    for row, grp in enumerate(merged):
        segs = sorted(grp, key=lambda i: tracks[i].t[0])
        y_raw, y_fix = row + 0.22, row - 0.22
        for j, ti in enumerate(segs):
            t = tracks[ti].t
            a, b = h(t[0]), h(t[-1])
            if j:                       # pull the start back so the break shows
                a += pad
            if j < len(segs) - 1:
                b -= pad
            ax.plot([a, max(b, a + pad)], [y_raw] * 2, lw=6.5, color=FRAG_LIGHT,
                    solid_capstyle='butt', zorder=2)
        for a_i, b_i in zip(segs, segs[1:]):
            x = (h(tracks[a_i].t[-1]) + h(tracks[b_i].t[0])) / 2
            ax.plot([x, x], [y_raw - 0.17, y_raw + 0.17], lw=2.0, color=END_C, zorder=4)
        ax.plot([h(tracks[segs[0]].t[0]), h(tracks[segs[-1]].t[-1])],
                [y_fix] * 2, lw=6.5, color=JOIN, solid_capstyle='butt', zorder=2)
        labels.append((row, f"frond {row + 1}", f"{len(segs)} fragments"))

    for row, name, sub in labels:
        ax.text(-0.012, row + 0.22, name, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=13, color=INK, fontweight='bold')
        ax.text(-0.012, row - 0.22, sub, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=12, color=MUTE)

    ax.set_ylim(-0.75, len(merged) - 0.25)
    ax.set_xlim(0, 141)
    ax.set_xlabel('time (h)', fontsize=16, fontweight='bold')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.25, linestyle=':', linewidth=0.6)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(0.0, 1.22, f'{len(merged)} fronds whose tracks fragmented',
            transform=ax.transAxes, fontsize=14, color=INK, fontweight='bold')
    handles = [mlines.Line2D([], [], color=FRAG_LIGHT, lw=6.5, label='raw btrack tracks'),
               mlines.Line2D([], [], color=JOIN, lw=6.5, label='identity after confirmation'),
               mlines.Line2D([], [], color=END_C, lw=2.0, label='confirmed join')]
    ax.legend(handles=handles, frameon=False, fontsize=13, ncol=3,
              loc='lower left', bbox_to_anchor=(0.0, 1.005), handlelength=1.6,
              columnspacing=1.8, handletextpad=0.6, borderpad=0.0)


def summary(fig, gs, r, dec, review, auto, before, after):
    """The decisions and what they changed, as a single block."""
    g = collections.Counter((v['gap'], v['answer']) for v in dec.values())
    c = collections.Counter(v['answer'] for v in dec.values())
    by_gap = "   ·   ".join(f"gap {x}: {sum(g.get((x, a), 0) for a in 'yns')}"
                            for x in (1, 2, 3))

    ax = fig.add_subplot(gs[r, :2]); ax.axis('off')
    T = ax.transAxes
    ax.text(0.0, 0.94, "Manual correction", fontsize=9.5, fontweight='bold', transform=T)
    ax.plot([0.0, 0.95], [0.84, 0.84], color=HAIR, lw=0.8, transform=T, clip_on=False)
    ax.text(0.0, 0.62, f"{len(auto)} pairs linked automatically, "
            f"{len(review)} shown for confirmation.", fontsize=9, color=INK, transform=T)
    ax.text(0.0, 0.42, f"{c.get('y', 0)} confirmed   ·   {c.get('n', 0)} rejected   ·   "
            f"{c.get('s', 0)} skipped", fontsize=9, color=INK, transform=T)
    ax.text(0.0, 0.22, by_gap, fontsize=8.4, color=MUTE, transform=T)
    ax.text(0.0, 0.02, "roughly ten minutes of review for 138.7 h of imaging",
            fontsize=8.2, color=MUTE, transform=T)

    ax2 = fig.add_subplot(gs[r, 2:]); ax2.axis('off')
    T = ax2.transAxes
    ax2.text(0.0, 0.94, "Petri dish  ·  1,664 frames  ·  138.7 h", fontsize=9.5,
             fontweight='bold', transform=T)
    ax2.plot([0.0, 1.0], [0.84, 0.84], color=HAIR, lw=0.8, transform=T, clip_on=False)
    ax2.text(0.60, 0.72, "before", fontsize=8.6, color=MUTE, transform=T)
    ax2.text(0.85, 0.72, "after", fontsize=8.6, color=MUTE, transform=T)
    for i, (lbl, b, a) in enumerate([
            ("tracks", before['n_raw_tracks'], after['n_effective_tracks']),
            ("identity switches", before['id_switches'], after['id_switches']),
            ("switch rate", f"{before['id_switch_rate']:.2%}", f"{after['id_switch_rate']:.2%}")]):
        y = 0.54 - i * 0.155
        ax2.text(0.0, y, lbl, fontsize=9, color=INK, transform=T)
        ax2.text(0.60, y, str(b), fontsize=9, color=MUTE, transform=T)
        ax2.text(0.85, y, str(a), fontsize=9, fontweight='bold', color=INK, transform=T)
    ax2.plot([0.0, 1.0], [0.06, 0.06], color=HAIR, lw=0.8, transform=T, clip_on=False)
    ax2.text(0.0, -0.04, "the skipped pair is left unmerged, so it remains "
             "among the 11 residual switches", fontsize=8.2, color=MUTE, transform=T)


def main():
    files, tracks, auto, review, dec, canon, before, after = build()
    k = pr.candidate_key
    ex = [(next(c for c in review if dec.get(k(c), {}).get('answer') == 'y'
                and c['gap'] == 1 and c['dist'] > 15), 'y',
           "touching fronds segmented as one object,\nthen as two"),
          (next(c for c in review if dec.get(k(c), {}).get('answer') == 'y'
                and c['gap'] == 3 and c['dist'] < 1.0), 'y',
           "frond missed for three frames,\ncentroid unchanged"),
          (next(c for c in review if dec.get(k(c), {}).get('answer') == 's'), 's',
           "no frond beneath the marker,\nleft uncorrected")]
    c = collections.Counter(v['answer'] for v in dec.values())

    n = len(ex)
    figA = plt.figure(figsize=(11.2, 2.95 * n))
    gsA = gridspec.GridSpec(n, 4, figure=figA, hspace=0.42, wspace=0.11,
                            width_ratios=[1, 1, 1, 1.25],
                            left=0.04, right=0.985, top=0.88, bottom=0.03)
    for r, (cand, ans, note) in enumerate(ex):
        candidate_row(figA, gsA, r, cand, ans, files, note, first=(r == 0))
    letter(figA, 0.008, 0.985, 'A')
    outA = os.path.join(HERE, 'persistence_panel_A.png')
    figA.savefig(outA, dpi=300, bbox_inches='tight', pad_inches=0.22, facecolor='white')
    plt.close(figA)

    figB = plt.figure(figsize=(12.4, 6.4))
    gsB = gridspec.GridSpec(1, 4, figure=figB, left=0.15, right=0.98,
                            top=0.80, bottom=0.16)
    timeline(figB, gsB, 0, tracks, canon)
    letter(figB, 0.008, 0.985, 'B')
    figB.text(0.14, 0.045,
              f"1,664 frames, 138.7 h: {len(auto)} pairs linked automatically, "
              f"{len(review)} shown for confirmation "
              f"({c.get('y', 0)} confirmed, {c.get('n', 0)} rejected, {c.get('s', 0)} skipped). "
              f"Identity switches {before['id_switches']} to {after['id_switches']} "
              f"({before['id_switch_rate']:.2%} to {after['id_switch_rate']:.2%}).",
              fontsize=13, color=MUTE)
    outB = os.path.join(HERE, 'persistence_panel_B.png')
    figB.savefig(outB, dpi=300, bbox_inches='tight', pad_inches=0.22, facecolor='white')
    plt.close(figB)
    print("saved", outA)
    print("saved", outB)


if __name__ == '__main__':
    main()
