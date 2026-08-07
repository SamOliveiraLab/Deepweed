# Held-out test set: 24 to 26 February

Ground truth for answering Reviewer 1, major comment 1 ("quantify performance
on an independent dataset").

## Why these frames

The current validation set cannot answer that comment. All 10 of its frames sit
inside the training window, a median of 15 minutes from the nearest training
frame and two of them only 5 minutes away. Duckweed doubles every 24 to 72
hours, so at that spacing the "held-out" frames are effectively duplicates of
training images.

These 15 frames are all at least **24 hours after the last training frame**
(`2024_02_23_21_50`), 24 h being the lower bound of the doubling time, so the
population has genuinely changed. They were sampled evenly from the 419
eligible frames, spanning +24.0 h to +58.9 h.

## Layout

    images/       15 frames, untouched
    pred_masks/   model predictions, the STARTING POINT (not ground truth)
    masks/        corrected ground truth, what you produce

## How to annotate

    python3 annotate_holdout.py

Keys: `S` save, `N` save and next, `L` new label id.
Already-saved frames are skipped, so you can stop and resume.

Frames are ordered by bisection rather than chronologically, so stopping early
still leaves the time window evenly covered. **10 frames is enough**; the
existing validation set is only 10.

Useful napari settings:

- `contour: 2` outlines only, makes missed fronds obvious
- `preserve labels: on` stops you painting over a neighbouring frond
- `opacity: 0.4` while hunting for misses, `0.7` while painting

## What to fix, in order

1. **Expand masks to the true frond edge.** This is the most common error and it
   affects nearly every frond. The model emits a 256x256 mask that is upscaled
   6.25x, so every prediction sits inside the real frond.
2. **Merge fragments.** About 12% of predictions are a small blob within 12 px
   of a much larger one, usually a broken-off tip.
3. **Add missed fronds.** The model misses roughly 18% of fronds, concentrated
   where they touch.
4. **Remove false positives.** Rare here, mostly not worth hunting.

### The judgement that matters most

A small mask beside a large one is either a **fragment** (merge it) or a
**daughter frond budding off the mother** (keep it separate). Only the eye can
tell. This call directly determines the paper's budding and lineage results, so
it is worth slowing down for.

## Notes

Boundary accuracy matters here, unlike in earlier annotation rounds. Frond areas
feed the growth curves, and the model currently underestimates them.

The existing training annotations were checked and are fine: they cover 91.8% of
frond pixels and miss only 2% of fronds. They do not need redoing. This holdout
set is the measuring stick, so it should be *better* than the training data.
