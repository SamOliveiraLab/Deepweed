#!/usr/bin/env python3
"""Seed annotation with model predictions.

Runs the petri-dish U-Net over the holdout frames and writes an instance mask
per frame. These are a STARTING POINT for hand correction, not ground truth:
the model is known to under-segment touching fronds, which is exactly the
regime these late frames are in.

Output: pred_masks/<stem>.png, uint16, 0 = background, 1..N = frond instances.
"""
import os, sys, glob
import numpy as np, cv2, torch
from skimage import measure
from skimage.measure import regionprops

HERE = os.path.dirname(os.path.abspath(__file__))
PA = os.path.dirname(HERE)
sys.path.insert(0, PA)
import unet_model_class

MODEL = os.path.join(PA, "data_model", "model", "best_instance_unet.pt")
IMGS = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "pred_masks")
MIN_AREA = 30

os.makedirs(OUT, exist_ok=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = unet_model_class.UNet(n_channels=3, n_classes=3).to(dev)
ck = torch.load(MODEL, map_location=dev, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print("U-Net epoch %s on %s" % (ck.get("epoch"), dev))

files = sorted(f for f in glob.glob(os.path.join(IMGS, "*.jpeg"))
               if not os.path.basename(f).startswith("._"))
for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    x = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        pred = model(t).argmax(dim=1).squeeze().cpu().numpy()

    full = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    inst = measure.label(full == 1)
    for r in regionprops(inst):
        if r.area < MIN_AREA:
            inst[inst == r.label] = 0
    inst = measure.label(inst > 0).astype(np.uint16)

    cv2.imwrite(os.path.join(OUT, stem + ".png"), inst)
    print("  %s -> %2d fronds" % (stem, int(inst.max())))

print("\nwrote %d starting masks to %s" % (len(files), OUT))
