"""Human-in-the-loop y/n review of frond persistence across track gaps.

When a btrack track terminates and another begins nearby a few frames later,
the pair is either the same frond (a fragmented track that should be stitched)
or a genuine death/birth. Confident pairs are linked automatically; ambiguous
pairs are shown to the user as side-by-side crops for a y/n decision.

Decisions are logged to JSON keyed by (end_track, start_track, frame), so a
re-run replays them silently and the accepted/rejected counts can be reported
as the amount of manual correction.
"""

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------

def find_candidates(tracks, auto_dist=12.0, max_dist=30.0, max_gap=3):
    """Pair each track end with the nearest unmatched track start within
    `max_gap` frames and `max_dist` px.

    Returns (auto_links, review_candidates). Pairs at gap 1 and distance
    <= auto_dist are linked automatically; the rest go to the human.
    Each entry is a dict with end/start track indices, frames, positions,
    gap and distance.
    """
    ends_by_frame = {}
    starts_by_frame = {}
    for i, tr in enumerate(tracks):
        ends_by_frame.setdefault(int(tr.t[-1]), []).append(i)
        starts_by_frame.setdefault(int(tr.t[0]), []).append(i)

    used_starts = set()
    used_ends = set()
    auto_links, review = [], []

    for gap in range(1, max_gap + 1):  # tightest gaps claim matches first
        for f in sorted(ends_by_frame):
            ends = [i for i in ends_by_frame[f] if i not in used_ends]
            starts = [i for i in starts_by_frame.get(f + gap, [])
                      if i not in used_starts]
            if not ends or not starts:
                continue
            end_pos = np.array([(tracks[i].x[-1], tracks[i].y[-1]) for i in ends])
            start_pos = np.array([(tracks[i].x[0], tracks[i].y[0]) for i in starts])
            dists = cdist(end_pos, start_pos)
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] > max_dist:
                    continue
                cand = {
                    'end_track': ends[r],
                    'start_track': starts[c],
                    'end_frame': f,
                    'start_frame': f + gap,
                    'end_xy': (float(end_pos[r, 0]), float(end_pos[r, 1])),
                    'start_xy': (float(start_pos[c, 0]), float(start_pos[c, 1])),
                    'gap': gap,
                    'dist': float(dists[r, c]),
                }
                used_ends.add(ends[r])
                used_starts.add(starts[c])
                if gap == 1 and dists[r, c] <= auto_dist:
                    auto_links.append(cand)
                else:
                    review.append(cand)
    return auto_links, review


def candidate_key(cand):
    return f"{cand['end_track']}->{cand['start_track']}@{cand['end_frame']}"


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

def load_decisions(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_decisions(decisions, path):
    with open(path, 'w') as f:
        json.dump(decisions, f, indent=2)


def summarize_decisions(decisions):
    answers = [d['answer'] for d in decisions.values()]
    return {
        'reviewed': len(answers),
        'accepted': answers.count('y'),
        'rejected': answers.count('n'),
        'skipped': answers.count('s'),
    }


# ---------------------------------------------------------------------------
# Interactive review (notebook)
# ---------------------------------------------------------------------------

def _load_frame(image_files, idx):
    import cv2
    img = cv2.imread(image_files[idx])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _crop(img, x, y, half):
    h, w = img.shape[:2]
    r0, r1 = max(0, int(y) - half), min(h, int(y) + half)
    c0, c1 = max(0, int(x) - half), min(w, int(x) + half)
    return img[r0:r1, c0:c1], (int(x) - c0, int(y) - r0)


def _draw_candidate(axes, cand, image_files, crop_half=64, context_half=200):
    """Draw one candidate onto three axes: track-end crop, track-start crop,
    zoomed-out context (red = end, cyan = start)."""
    end_img = _load_frame(image_files, cand['end_frame'])
    start_img = _load_frame(image_files, cand['start_frame'])

    for ax in axes:
        ax.clear()
    for ax, img, (x, y), title in [
        (axes[0], end_img, cand['end_xy'],
         f"track {cand['end_track']} ends (frame {cand['end_frame']})"),
        (axes[1], start_img, cand['start_xy'],
         f"track {cand['start_track']} starts (frame {cand['start_frame']})"),
    ]:
        crop, (cx, cy) = _crop(img, x, y, crop_half)
        ax.imshow(crop)
        ax.plot(cx, cy, 'o', ms=18, mfc='none', mec='red', mew=2)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    ctx, (ex, ey) = _crop(end_img, *cand['end_xy'], context_half)
    sx = ex + (cand['start_xy'][0] - cand['end_xy'][0])
    sy = ey + (cand['start_xy'][1] - cand['end_xy'][1])
    axes[2].imshow(ctx)
    axes[2].plot(ex, ey, 'o', ms=10, mfc='none', mec='red', mew=2)
    axes[2].plot(sx, sy, 's', ms=10, mfc='none', mec='cyan', mew=2)
    axes[2].set_title('context (red=end, cyan=start)', fontsize=10)
    axes[2].axis('off')


def review_candidates_popup(candidates, image_files, decisions_path,
                            crop_half=64, context_half=200):
    """Interactive y/n popup review — the human-in-the-loop feature.

    Opens one window and steps through every unreviewed candidate. Keys:

    - ``y``  same frond, merge the two track IDs
    - ``n``  different fronds, keep them separate
    - ``s``  skip (asked again next session)
    - ``u``  undo the previous answer of this session
    - ``q``  quit early

    Every answer is written to ``decisions_path`` immediately, so quitting
    and rerunning resumes where the session left off. Returns the decisions.
    """
    import matplotlib.pyplot as plt

    decisions = load_decisions(decisions_path)
    pending = [c for c in candidates if candidate_key(c) not in decisions]
    print(f"{len(candidates)} candidates, {len(pending)} pending review")
    if not pending:
        print(f"nothing to review: {summarize_decisions(decisions)}")
        return decisions

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    # matplotlib's own shortcuts (s=save, q=quit, ...) would swallow our keys
    manager = fig.canvas.manager
    if manager is not None and getattr(manager, 'key_press_handler_id', None):
        fig.canvas.mpl_disconnect(manager.key_press_handler_id)

    state = {'i': 0, 'answered': []}

    def _show():
        cand = pending[state['i']]
        _draw_candidate(axes, cand, image_files, crop_half, context_half)
        fig.suptitle(
            f"[{state['i'] + 1}/{len(pending)}]  gap={cand['gap']} frame(s), "
            f"dist={cand['dist']:.1f} px  —  same frond?   "
            f"[y]es  [n]o  [s]kip  [u]ndo  [q]uit", fontsize=11)
        fig.canvas.draw_idle()

    def _on_key(event):
        k = (event.key or '').lower()
        if k in ('y', 'n', 's'):
            cand = pending[state['i']]
            decisions[candidate_key(cand)] = {**cand, 'answer': k}
            save_decisions(decisions, decisions_path)
            state['answered'].append(candidate_key(cand))
            if state['i'] + 1 >= len(pending):
                plt.close(fig)
            else:
                state['i'] += 1
                _show()
        elif k == 'u' and state['answered']:
            decisions.pop(state['answered'].pop(), None)
            save_decisions(decisions, decisions_path)
            state['i'] = max(0, state['i'] - 1)
            _show()
        elif k == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)
    _show()
    plt.show(block=True)
    print(f"Review session done: {summarize_decisions(decisions)}")
    return decisions


def review_candidates_widget(candidates, image_files, decisions_path,
                             crop_half=64, context_half=200, on_complete=None):
    """Inline y/n review for notebooks, driven by buttons instead of a popup.

    Shows one candidate at a time (track-end crop, track-start crop, context)
    with Yes / No / Skip / Undo buttons. Each answer is written to
    `decisions_path` immediately. When the queue empties, `on_complete` is
    called with the decisions dict so the caller can print metrics.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    decisions = load_decisions(decisions_path)
    pending = [c for c in candidates if candidate_key(c) not in decisions]

    status = widgets.HTML()
    out = widgets.Output()
    b_yes = widgets.Button(description='Yes, same frond',
                           button_style='success', icon='check')
    b_no = widgets.Button(description='No, different fronds',
                          button_style='danger', icon='times')
    b_skip = widgets.Button(description='Skip', button_style='warning')
    b_undo = widgets.Button(description='Undo')
    buttons = widgets.HBox([b_yes, b_no, b_skip, b_undo])
    state = {'i': 0, 'answered': []}

    def finish():
        buttons.layout.display = 'none'
        summary = summarize_decisions(decisions)
        status.value = (f"<b>Review complete.</b> reviewed {summary['reviewed']}, "
                        f"accepted {summary['accepted']}, rejected {summary['rejected']}, "
                        f"skipped {summary['skipped']}")
        with out:
            clear_output(wait=True)
        if on_complete is not None:
            with out:
                on_complete(decisions)

    def render():
        if state['i'] >= len(pending):
            finish()
            return
        cand = pending[state['i']]
        status.value = (f"<b>[{state['i'] + 1}/{len(pending)}]</b> "
                        f"gap {cand['gap']} frame(s), "
                        f"distance {cand['dist']:.1f} px &nbsp;&nbsp; same frond?")
        with out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
            _draw_candidate(axes, cand, image_files, crop_half, context_half)
            plt.tight_layout()
            plt.show()

    def answer(a):
        cand = pending[state['i']]
        decisions[candidate_key(cand)] = {**cand, 'answer': a}
        save_decisions(decisions, decisions_path)
        state['answered'].append(candidate_key(cand))
        state['i'] += 1
        render()

    def undo(_):
        if state['answered']:
            decisions.pop(state['answered'].pop(), None)
            save_decisions(decisions, decisions_path)
            state['i'] = max(0, state['i'] - 1)
            render()

    b_yes.on_click(lambda _: answer('y'))
    b_no.on_click(lambda _: answer('n'))
    b_skip.on_click(lambda _: answer('s'))
    b_undo.on_click(undo)

    display(widgets.VBox([status, out, buttons]))
    if not pending:
        finish()
    else:
        render()
    return decisions


def review_candidates(candidates, image_files, decisions_path,
                      crop_half=64, context_half=200):
    """Notebook (inline-backend) fallback of the y/n review: shows each
    candidate as a static figure and asks via input(). Prefer
    review_candidates_popup outside `%matplotlib inline`.
    """
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    decisions = load_decisions(decisions_path)
    pending = [c for c in candidates if candidate_key(c) not in decisions]
    print(f"{len(candidates)} candidates, {len(pending)} pending review")

    for n, cand in enumerate(pending):
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
        _draw_candidate(axes, cand, image_files, crop_half, context_half)
        fig.suptitle(
            f"[{n + 1}/{len(pending)}]  gap={cand['gap']} frame(s), "
            f"dist={cand['dist']:.1f} px  —  same frond?", fontsize=11)
        plt.tight_layout()
        plt.show()

        while True:
            ans = input("same frond? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
            if ans in ('y', 'n', 's', 'q'):
                break
        if ans == 'q':
            break
        decisions[candidate_key(cand)] = {**cand, 'answer': ans}
        save_decisions(decisions, decisions_path)

    clear_output(wait=True)
    print(f"Review session done: {summarize_decisions(decisions)}")
    return decisions


# ---------------------------------------------------------------------------
# Applying decisions
# ---------------------------------------------------------------------------

def build_canonical_map(n_tracks, auto_links, decisions):
    """Union-find over auto links plus human-accepted links.

    Returns {track_index: canonical_index} covering all n_tracks.
    """
    parent = list(range(n_tracks))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(b)] = find(a)

    for cand in auto_links:
        union(cand['end_track'], cand['start_track'])
    for d in decisions.values():
        if d['answer'] == 'y':
            union(d['end_track'], d['start_track'])
    return {i: find(i) for i in range(n_tracks)}


# ---------------------------------------------------------------------------
# Metrics (same definitions as evaluate_tracking.ipynb, with optional
# canonical remapping so corrected links no longer count as errors)
# ---------------------------------------------------------------------------

def compute_id_switch_metrics(tracks, n_frames, canonical_map=None,
                              assoc_dist=30.0):
    """Frame-to-frame ID switch rate via Hungarian assignment, identical to
    evaluate_tracking.ipynb. With canonical_map, stitched tracks share one ID
    so corrected fragmentations stop counting as switches.
    """
    ids = canonical_map or {i: i for i in range(len(tracks))}
    frame_tracks = {}
    for i, tr in enumerate(tracks):
        for j, t in enumerate(tr.t):
            frame_tracks.setdefault(int(t), []).append((ids[i], tr.x[j], tr.y[j]))

    id_switches = 0
    total_associations = 0
    for f in range(1, n_frames):
        prev = frame_tracks.get(f - 1, [])
        curr = frame_tracks.get(f, [])
        if not prev or not curr:
            continue
        prev_pos = np.array([(x, y) for _, x, y in prev])
        curr_pos = np.array([(x, y) for _, x, y in curr])
        dists = cdist(prev_pos, curr_pos)
        row_ind, col_ind = linear_sum_assignment(dists)
        for r, c in zip(row_ind, col_ind):
            if dists[r, c] < assoc_dist:
                total_associations += 1
                if prev[r][0] != curr[c][0]:
                    id_switches += 1

    n_effective = len(set(ids.values()))
    return {
        'n_raw_tracks': len(tracks),
        'n_effective_tracks': n_effective,
        'id_switches': id_switches,
        'total_associations': total_associations,
        'id_switch_rate': id_switches / total_associations if total_associations else 0,
    }
