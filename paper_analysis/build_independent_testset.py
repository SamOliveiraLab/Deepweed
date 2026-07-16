#!/usr/bin/env python3
"""Build the independent test set: Wolffia-only images from Senay's Roboflow
COCO export -> RGB frame + binary GT mask (Wolffia instances)."""
import json, os, glob, shutil
import numpy as np, cv2

ROOT = "/Volumes/Extreme SSD/04- Deepweed & Duckweed/Deepweed_senay_lab/duckweed.noresize.6.25.26.coco"
OUT = "/tmp/indep"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT + "/images", exist_ok=True)
os.makedirs(OUT + "/masks", exist_ok=True)
os.makedirs(OUT + "/inst", exist_ok=True)

kept = 0
meta = []
for split in ("train", "valid", "test"):
    p = os.path.join(ROOT, split, "_annotations.coco.json")
    d = json.load(open(p))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    per = {}
    for a in d["annotations"]:
        per.setdefault(a["image_id"], []).append(a)
    for iid, anns in per.items():
        names = {cats[a["category_id"]] for a in anns}
        if names != {"Wolffia"}:
            continue
        info = imgs[iid]
        src = os.path.join(ROOT, split, info["file_name"])
        if not os.path.exists(src):
            continue
        H, W = info["height"], info["width"]
        binm = np.zeros((H, W), np.uint8)
        inst = np.zeros((H, W), np.int32)
        n = 0
        for a in anns:
            seg = a.get("segmentation")
            if not seg or not isinstance(seg, list):
                continue
            n += 1
            for poly in seg:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(binm, [pts], 255)
                cv2.fillPoly(inst, [pts], n)
        if n == 0:
            continue
        stem = "%s_%03d" % (split, iid)
        shutil.copy(src, os.path.join(OUT, "images", stem + ".jpg"))
        cv2.imwrite(os.path.join(OUT, "masks", stem + ".png"), binm)
        cv2.imwrite(os.path.join(OUT, "inst", stem + ".png"), inst.astype(np.uint16))
        cov = 100.0 * (binm > 0).sum() / binm.size
        meta.append((stem, W, H, n, cov))
        kept += 1

print("independent Wolffia-only images: %d" % kept)
print("%-14s %10s %6s %8s" % ("stem", "WxH", "fronds", "cov%"))
for stem, W, H, n, cov in meta:
    print("%-14s %5dx%-4d %6d %7.2f%%" % (stem, W, H, n, cov))
tot = sum(m[3] for m in meta)
print("\ntotal annotated Wolffia fronds: %d" % tot)
print("median coverage: %.2f%%" % np.median([m[4] for m in meta]))
