# Deepweed independent test set

Ground truth for Reviewer 1, major comment 1: *"quantify performance on an
independent dataset."*

## Run

    python3 annotate.py

Keys: `L` new frond, `S` save, `N` save and next.
Already-saved frames are skipped, so you can stop and resume any time.

Needs `napari`, `scikit-image`, `numpy`:

    pip install "napari[all]" scikit-image

## Status

10 frames. **3 are done, 7 to go.**

## What to do

Draw every duckweed frond on a blank canvas, one label id per frond.

1. **Label every frond.** Completeness is the entire point of this set.
2. **Touching fronds get separate ids.** A mother and its budding daughter are
   two fronds, not one.
3. **Cover the full frond**, out to its edge.
4. **Do not label** bubbles, reflections or the plate rim.

Useful napari settings:

- `contour: 2` outlines only, so you can see the frond underneath
- `preserve labels: on` stops you painting over a neighbour
- `opacity: 0.4` while checking for fronds you have not labelled yet

## Why it is blank and not pre-filled

We first tried seeding these masks with the model's own predictions, to save
time. That silently broke the measurement: any frond the model missed was also
missing from the ground truth, so the model scored 95.8% recall against labels
derived from itself. Only 3 fronds out of 288 had been added by hand.

Annotated blind on the same frames, real recall came out at 84.1%. Every frond
therefore has to be drawn by eye, with no starting mask.

## Why these frames

The original validation set cannot answer the reviewer. All 10 of its frames sit
inside the training window, a median of 15 minutes from the nearest training
frame, two only 5 minutes away. Duckweed doubles every 24 to 72 hours, so at
that spacing those frames show the same plants in the same positions.

These 10 frames are all at least **24 hours after the last training frame**
(`2024_02_23_21_50`), spanning +24 h to +59 h, sampled evenly. None were used in
training or in model selection.

## When finished

Send the `masks/` folder back. Expect roughly 23 to 30 fronds per frame; the
count should not fall as time goes on, since fronds do not un-divide.
