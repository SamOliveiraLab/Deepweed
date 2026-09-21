#!/usr/bin/env python3
"""Stack the petri dish and microfluidics montages into one figure.

Pads the narrower panel with white so neither is rescaled, then verifies both
halves are pixel identical to their sources. Use --side-by-side for a landscape
layout instead of the default portrait stack.

Usage:
  python combine_montages.py [--dpi 600] [--side-by-side]
"""
import argparse
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
Image.MAX_IMAGE_PIXELS = None


def flatten(im):
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert('RGB')


def combine(petri_path, micro_path, out_path, dpi=600, side_by_side=False):
    petri = flatten(Image.open(petri_path))
    micro = flatten(Image.open(micro_path))

    if side_by_side:
        W, H = petri.width + micro.width, max(petri.height, micro.height)
        pos = [(0, (H - petri.height) // 2), (petri.width, (H - micro.height) // 2)]
    else:
        W, H = max(petri.width, micro.width), petri.height + micro.height
        pos = [((W - petri.width) // 2, 0), ((W - micro.width) // 2, petri.height)]

    out = Image.new('RGB', (W, H), 'white')
    out.paste(petri, pos[0])
    out.paste(micro, pos[1])
    out.save(out_path, dpi=(dpi, dpi))

    c = np.array(out)
    for name, im, (x, y) in (('petri', petri, pos[0]), ('microfluidics', micro, pos[1])):
        a = np.array(im)
        ok = np.array_equal(c[y:y + a.shape[0], x:x + a.shape[1]], a)
        print(f"  {name} half pixel identical: {ok}")
    print(f"saved {out_path}  {W}x{H} px = {W / dpi:.1f} x {H / dpi:.1f} inches at {dpi} dpi")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--petri', default=os.path.join(HERE, 'petri_montage.png'))
    ap.add_argument('--micro', default=os.path.join(HERE, 'microfluidics_montage.png'))
    ap.add_argument('--out', default=os.path.join(HERE, 'segmentation_accuracy_montage.png'))
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--side-by-side', action='store_true')
    a = ap.parse_args()
    combine(a.petri, a.micro, a.out, a.dpi, a.side_by_side)


if __name__ == '__main__':
    main()
