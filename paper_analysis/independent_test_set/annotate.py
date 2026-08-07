#!/usr/bin/env python3
"""Annotate the Deepweed independent test set.

Draw every duckweed frond, one label id per frond, on a blank canvas.

    python3 annotate.py

Keys
    L   start a new frond (next label id)
    S   save
    N   save and go to the next image

Frames already saved in masks/ are skipped, so you can stop and resume.
Paths are relative to this script, so run it from wherever the folder lives.

Why blank and not pre-filled: these masks measure how many fronds the model
misses. If the model's own output were loaded as a starting point, anything it
missed would be missing from the ground truth too, and the measurement would be
meaningless. So every frond has to be drawn by eye.

What matters, in order
    1. Every frond gets a label. Completeness is the whole point.
    2. Touching fronds get SEPARATE ids. A mother and its budding daughter are
       two fronds, not one.
    3. Cover the full frond out to its edge.
    4. Do not label bubbles, reflections or the plate rim.

Useful napari settings
    contour: 2          outlines only, so you can see the frond underneath
    preserve labels: on stops you painting over a neighbouring frond
    opacity: 0.4        while checking for fronds you have not labelled yet

Requires: napari, scikit-image, numpy   (pip install "napari[all]" scikit-image)
"""
import os
import sys
import glob

import numpy as np
import napari
from skimage import io

HERE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "masks")
os.makedirs(OUT, exist_ok=True)

files = sorted(
    os.path.basename(f)
    for f in glob.glob(os.path.join(IMGS, "*.jpeg"))
    if not os.path.basename(f).startswith("._")
)
todo = [f for f in files
        if not os.path.exists(os.path.join(OUT, os.path.splitext(f)[0] + ".png"))]

print("=" * 64)
print("DEEPWEED INDEPENDENT TEST SET")
print("  %d frames total, %d already done, %d to go" %
      (len(files), len(files) - len(todo), len(todo)))
print("  L = new frond    S = save    N = save and next")
print("  Label EVERY frond. Touching fronds get separate ids.")
print("=" * 64)

if not todo:
    print("\nAll frames are annotated. Nothing to do.")
    sys.exit(0)

state = {"i": 0}
viewer = napari.Viewer(title="Deepweed independent test set")


def load(idx):
    name = todo[idx]
    stem = os.path.splitext(name)[0]
    img = io.imread(os.path.join(IMGS, name))
    viewer.layers.clear()
    viewer.add_image(img, name=stem)
    layer = viewer.add_labels(np.zeros(img.shape[:2], np.int32), name="fronds")
    layer.mode = "paint"
    layer.brush_size = 12
    layer.selected_label = 1
    print("\n[%d/%d] %s" % (idx + 1, len(todo), stem))


def save(idx):
    stem = os.path.splitext(todo[idx])[0]
    lab = viewer.layers["fronds"].data.astype(np.uint16)
    n = len([v for v in np.unique(lab) if v != 0])
    if n == 0:
        print("  nothing drawn, not saving")
        return False
    io.imsave(os.path.join(OUT, stem + ".png"), lab, check_contrast=False)
    print("  saved %s with %d fronds" % (stem, n))
    return True


@viewer.bind_key("s")
def _s(v):
    save(state["i"])


@viewer.bind_key("n")
def _n(v):
    if not save(state["i"]):
        return
    if state["i"] + 1 < len(todo):
        state["i"] += 1
        load(state["i"])
    else:
        print("\nAll done. Send the masks/ folder back.")


@viewer.bind_key("l")
def _l(v):
    layer = viewer.layers["fronds"]
    layer.selected_label = int(layer.data.max()) + 1
    print("  frond %d" % layer.selected_label)


load(0)
napari.run()
