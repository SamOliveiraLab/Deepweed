#!/usr/bin/env python3
"""Segmentation metrics for the petri-dish U-Net.

Scores the model against ground-truth masks and reports IoU, Dice and
instance-level frond recall. Handles both ground-truth conventions:

  * 3-class preview masks  (0 = background, 85 = body, 170/255 = boundary)
  * instance label masks   (0 = background, 1..N = one id per frond)

Metrics are reported at full image resolution. This matters: the network runs
at 256x256 and its output is upscaled, so scoring at the network input size
flatters the model and hides the resolution loss.

Usage:
  python3 evaluate_segmentation.py --images DIR --masks DIR [--model PATH]
                                   [--input-size 256] [--pattern '*.png']
"""
import os, sys, glob, argparse
import numpy as np, cv2, torch
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unet_model_class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="directory of RGB frames")
    p.add_argument("--masks", required=True, help="directory of ground-truth masks")
    p.add_argument("--model", default=os.path.join(HERE, "data_model", "model", "best_instance_unet.pt"))
    p.add_argument("--input-size", type=int, default=256, help="network input size")
    p.add_argument("--pattern", default="*.png", help="glob for mask files")
    p.add_argument("--min-area", type=int, default=30, help="drop predicted fronds below this area")
    p.add_argument("--iou-match", type=float, default=0.5, help="IoU for counting a frond as detected")
    return p.parse_args()


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


def load_gt(path):
    """Return (body_mask, instance_label_map) for either GT convention."""
    g = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if g is None:
        return None, None
    if g.ndim == 3:
        g = g[:, :, 0]
    vals = np.unique(g)
    if set(vals.tolist()) <= {0, 85, 170, 255}:      # 3-class preview mask
        body = g == 85
        inst = measure.label(body)
    else:                                             # instance label mask
        body = g > 0
        inst = g.astype(np.int32)
    return body, inst


def find_image(images_dir, stem):
    for ext in (".jpeg", ".jpg", ".png"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def instance_recall(gt_inst, pred_inst, thr):
    """Greedy one-to-one matching of predicted fronds to ground-truth fronds."""
    gt_ids = [v for v in np.unique(gt_inst) if v != 0]
    pr_ids = [v for v in np.unique(pred_inst) if v != 0]
    used, matched = set(), 0
    for g in gt_ids:
        gs = gt_inst == g
        best, best_iou = None, 0.0
        for p in pr_ids:
            if p in used:
                continue
            ps = pred_inst == p
            if not (gs & ps).any():
                continue
            v = iou(gs, ps)
            if v > best_iou:
                best, best_iou = p, v
        if best is not None and best_iou >= thr:
            used.add(best)
            matched += 1
    return len(gt_ids), len(pr_ids), matched


def main():
    a = parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model = unet_model_class.UNet(n_channels=3, n_classes=3).to(dev)
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print("model      : %s" % os.path.basename(a.model))
    print("checkpoint : epoch %s, best_fg_miou %s" % (ck.get("epoch"), ck.get("best_fg_miou")))
    print("input size : %dx%d   device: %s" % (a.input_size, a.input_size, dev))
    print("ground truth: %s\n" % a.masks)

    rows = []
    for mf in sorted(glob.glob(os.path.join(a.masks, a.pattern))):
        if os.path.basename(mf).startswith("._"):
            continue
        stem = os.path.splitext(os.path.basename(mf))[0].replace("_preview", "")
        imf = find_image(a.images, stem)
        if imf is None:
            print("  no image for %s, skipped" % stem)
            continue
        gt_body, gt_inst = load_gt(mf)
        rgb = cv2.cvtColor(cv2.imread(imf), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        x = cv2.resize(rgb, (a.input_size, a.input_size)).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            pred = model(t).argmax(dim=1).squeeze().cpu().numpy()
        full = cv2.resize(pred.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

        pr_body = full == 1
        pr_inst = measure.label(pr_body)
        if a.min_area > 0:
            for r in measure.regionprops(pr_inst):
                if r.area < a.min_area:
                    pr_inst[pr_inst == r.label] = 0
            pr_inst = measure.label(pr_inst > 0)
            pr_body = pr_inst > 0

        n_gt, n_pr, matched = instance_recall(gt_inst, pr_inst, a.iou_match)
        rows.append(dict(stem=stem,
                         body_iou=iou(gt_body, pr_body), body_dice=dice(gt_body, pr_body),
                         fg_iou=iou(gt_body, full > 0), fg_dice=dice(gt_body, full > 0),
                         n_gt=n_gt, n_pr=n_pr, matched=matched,
                         gt_area=100.0 * gt_body.mean(), pr_area=100.0 * pr_body.mean()))

    if not rows:
        print("no frames evaluated")
        return

    def col(k):
        return np.array([r[k] for r in rows], float)

    print("%-22s %7s %7s %5s %5s %6s %8s %8s" %
          ("frame", "bIoU", "bDice", "GT", "pred", "match", "GTarea%", "PRarea%"))
    for r in rows:
        print("%-22s %7.3f %7.3f %5d %5d %6d %7.2f%% %7.2f%%" %
              (r["stem"][:22], r["body_iou"], r["body_dice"], r["n_gt"], r["n_pr"],
               r["matched"], r["gt_area"], r["pr_area"]))

    n = len(rows)
    tg, tp, tm = int(col("n_gt").sum()), int(col("n_pr").sum()), int(col("matched").sum())
    print("\n--- %d frames ---" % n)
    for k, lbl in (("body_iou", "body IoU"), ("body_dice", "body Dice"),
                   ("fg_iou", "foreground IoU"), ("fg_dice", "foreground Dice")):
        print("  %-16s %.4f +/- %.4f" % (lbl, col(k).mean(), col(k).std()))
    print("  %-16s %d ground truth, %d predicted" % ("fronds", tg, tp))
    print("  %-16s %d / %d = %.1f%%  (IoU >= %.2f)" % ("detection recall", tm, tg, 100.0 * tm / max(tg, 1), a.iou_match))
    print("  %-16s %d / %d = %.1f%%" % ("precision", tm, tp, 100.0 * tm / max(tp, 1)))
    print("  %-16s GT %.2f%%  predicted %.2f%%  (ratio %.3f)" %
          ("area coverage", col("gt_area").mean(), col("pr_area").mean(),
           col("pr_area").mean() / max(col("gt_area").mean(), 1e-9)))


if __name__ == "__main__":
    main()
