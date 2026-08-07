#!/usr/bin/env python3
"""Correct the model's starting masks in napari to build held-out ground truth.

Loads each holdout frame with the model prediction pre-loaded as a labels layer.
You fix it rather than draw from scratch. Corrected masks go to masks/.

Keys:
    S  save current mask
    N  save and go to next image
    L  next label id (start a new frond)

What to look for, in priority order:
    1. MISSED fronds        paint them in with a new label
    2. MERGED fronds        one label covering two touching fronds; split them
    3. FALSE positives      bubbles or rim reflections; erase them
    4. Boundary accuracy    least important, do not over-invest here

Progress is resumable: frames already present in masks/ are skipped.
"""
import os, sys, glob
import numpy as np
import napari
from skimage import io

HERE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(HERE, "images")

# --blank: start from an empty canvas instead of the model's prediction.
# Seeding from predictions makes annotation fast, but the resulting ground truth
# inherits whatever the model missed, so detection recall measured against it is
# circular. A few blank-annotated frames give an unbiased reference.
BLANK = "--blank" in sys.argv
PRED = None if BLANK else os.path.join(HERE, "pred_masks")
OUT = os.path.join(HERE, "masks_blank" if BLANK else "masks")
os.makedirs(OUT, exist_ok=True)

files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(IMGS, "*.jpeg"))
               if not os.path.basename(f).startswith("._"))


def spread_order(items):
    """Bisection order, so stopping early still leaves the window evenly covered."""
    out, ranges = [], [(0, len(items) - 1)]
    out.append(items[0])
    if len(items) > 1:
        out.append(items[-1])
    while ranges:
        lo, hi = ranges.pop(0)
        mid = (lo + hi) // 2
        if mid != lo and mid != hi and items[mid] not in out:
            out.append(items[mid])
            ranges.append((lo, mid))
            ranges.append((mid, hi))
    return out + [f for f in items if f not in out]


files = spread_order(files)
todo = [f for f in files if not os.path.exists(os.path.join(OUT, os.path.splitext(f)[0] + ".png"))]

if not todo:
    print("All %d frames already annotated in %s" % (len(files), OUT))
    sys.exit(0)

print("=" * 62)
print("HOLDOUT ANNOTATION  (%d of %d remaining)" % (len(todo), len(files)))
print("  S = save    N = save + next    L = new label")
print("  Fix: missed fronds, merged fronds, false positives")
print("=" * 62)

state = {"i": 0}
viewer = napari.Viewer(title="Deepweed holdout annotation")


def load(idx):
    name = todo[idx]
    stem = os.path.splitext(name)[0]
    img = io.imread(os.path.join(IMGS, name))
    pred_p = os.path.join(PRED, stem + ".png") if PRED else None
    lab = io.imread(pred_p).astype(np.int32) if pred_p and os.path.exists(pred_p) \
        else np.zeros(img.shape[:2], np.int32)
    viewer.layers.clear()
    viewer.add_image(img, name=stem)
    ll = viewer.add_labels(lab, name="fronds")
    ll.mode = "paint"
    ll.brush_size = 12
    print("\n[%d/%d] %s  (%d fronds predicted)" % (idx + 1, len(todo), stem, int(lab.max())))


def save(idx):
    stem = os.path.splitext(todo[idx])[0]
    lab = viewer.layers["fronds"].data.astype(np.uint16)
    io.imsave(os.path.join(OUT, stem + ".png"), lab, check_contrast=False)
    print("  saved %s  -> %d fronds" % (stem, int(lab.max())))


@viewer.bind_key("s")
def _s(v):
    save(state["i"])


@viewer.bind_key("n")
def _n(v):
    save(state["i"])
    if state["i"] + 1 < len(todo):
        state["i"] += 1
        load(state["i"])
    else:
        print("\nDone. All frames annotated.")


@viewer.bind_key("l")
def _l(v):
    ll = viewer.layers["fronds"]
    ll.selected_label = int(ll.data.max()) + 1
    print("  new label: %d" % ll.selected_label)


load(0)
napari.run()
