"""
Microfluidics segmentation: correct preprocessing to match segment_boundary_model.

The model was trained with:
  - Input size: 256x256 (not 512x128)
  - Normalization: (img/255 - 0.5) / 0.5  =>  range [-1, 1]
  - Output: raw logits; use threshold THRESHOLD (e.g. 0.2), not sigmoid then 0.5

Using 512x128 and no normalization causes severe under-detection (e.g. only 3 fronds).

Usage from notebook (after loading boundary_model and setting DATASET_TYPE='microfluidics'):
  from microfluidics_segmentation_fixed import analyze_single_image, run_segmentation_on_video
  frame_rgb = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
  region_info = analyze_single_image(frame_rgb, boundary_model, device, threshold=THRESHOLD, min_area=MIN_AREA)
  # For video:
  run_segmentation_on_video(VIDEO_PATH, output_path, boundary_model, device, threshold=THRESHOLD, min_area=MIN_AREA)
"""

import numpy as np
import cv2
import torch
from skimage import measure
from skimage.measure import regionprops, regionprops_table
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib import gridspec


def analyze_single_image(frame, model, device='cpu', threshold=0.2, min_area=20):
    """
    Run microfluidics segmentation with correct preprocessing (matches segment_boundary_model).

    - Resize to 256x256, normalize to [-1, 1]
    - Raw output > threshold (no sigmoid)
    - Returns regionprops_table and displays Original | Predicted mask with bboxes.
    """
    model.eval()
    original_shape = frame.shape[:2]

    # Match segment_boundary_model: 256x256, normalize to [-1, 1]
    img = cv2.resize(frame, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(input_tensor)
    pred = output.squeeze().cpu().numpy()

    # Resize prediction back to original size (same as segment_boundary_model)
    pred_full = cv2.resize(pred, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    binary_mask = (pred_full > threshold).astype(np.uint8)

    labeled_mask = measure.label(binary_mask)

    # Remove small objects
    if min_area > 0:
        for region in measure.regionprops(labeled_mask):
            if region.area < min_area:
                labeled_mask[labeled_mask == region.label] = 0
        labeled_mask = measure.label(labeled_mask > 0)

    props = regionprops_table(
        labeled_mask,
        properties=['label', 'area', 'perimeter', 'major_axis_length', 'bbox', 'centroid']
    )

    # Figure: Original | Predicted mask with bboxes
    fig = plt.figure(figsize=(frame.shape[1] * 2 / 100, frame.shape[0] / 100), dpi=100)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0, hspace=0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.imshow(frame)
    ax1.set_title("Original", fontsize=10)
    ax1.axis('off')
    overlay = np.zeros_like(frame)
    overlay[binary_mask.astype(bool)] = frame[binary_mask.astype(bool)]
    ax2.imshow(overlay)
    n_fronds = labeled_mask.max()
    ax2.set_title(f"Predicted Mask ({n_fronds} fronds detected)", fontsize=10)
    ax2.axis('off')
    for region in measure.regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        cy, cx = region.centroid
        ax2.plot([minc, maxc, maxc, minc, minc], [minr, minr, maxr, maxr, minr], color='red', linewidth=1.2)
        ax2.text(cx, cy, str(region.label), color='yellow', fontsize=4, ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.4, lw=0, pad=1))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.show()
    return props


def run_segmentation_on_video(video_path, output_path, model, device='cpu', threshold=0.2, min_area=20):
    """
    Run segmentation on each frame with correct preprocessing (256x256, normalize [-1,1], raw > threshold).
    Writes side-by-side: original | label2rgb overlay.
    """
    model.eval()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = w * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_shape = frame_rgb.shape[:2]

        img = cv2.resize(frame_rgb, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)
        pred = output.squeeze().cpu().numpy()
        pred_full = cv2.resize(pred, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        binary_mask = (pred_full > threshold).astype(np.uint8)
        labeled_mask = measure.label(binary_mask)

        if min_area > 0:
            for region in measure.regionprops(labeled_mask):
                if region.area < min_area:
                    labeled_mask[labeled_mask == region.label] = 0
            labeled_mask = measure.label(labeled_mask > 0)

        overlay_rgb = label2rgb(labeled_mask, image=frame_rgb, bg_label=0)
        overlay_rgb = (overlay_rgb * 255).astype(np.uint8)
        combined = np.hstack([frame_rgb, overlay_rgb])
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Done. Output saved to {output_path}")
