# Deepweed

Automated duckweed segmentation, tracking, and growth analysis pipeline using deep learning. This tool processes time-lapse microscopy data from **petri dish** and **microfluidics** experiments to segment individual duckweed fronds, track them across frames, infer lineage (budding events), and generate publication-ready figures.

Developed at [Oliveira Lab](https://www.oliveiralab.me)

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Conda (Recommended)](#option-a-conda-recommended)
  - [Option B: pip with venv](#option-b-pip-with-venv)
  - [GPU Support (Optional)](#gpu-support-optional)
- [Data Setup](#data-setup)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Configuration](#configuration)
  - [Running the Pipeline](#running-the-pipeline)
- [Pipeline Overview](#pipeline-overview)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

## Overview

Deepweed provides an end-to-end computer vision pipeline:

1. **Segmentation** - A U-Net deep learning model identifies individual duckweed fronds in each frame.
2. **Tracking** - Bayesian tracking (btrack) assigns persistent IDs across frames.
3. **Track Stitching** — Fragmented tracks from the same plant are merged.
4. **Lineage Inference** - Parent-child (budding) relationships are detected.
5. **Cluster Analysis** - Fronds are grouped into spatial clusters (e.g., individual petri dishes or microfluidic wells).
6. **Visualization** - Publication-ready figures: population dynamics, growth curves, lineage timelines, trajectory plots, and animated GIFs.

Two dataset modes are supported:

| Feature | Petri Dish | Microfluidics |
|---|---|---|
| Input | JPEG/PNG image sequence | AVI video |
| Model | 3-class boundary U-Net (bg/body/boundary) | 1-class binary U-Net |
| Clustering | KMeans (6 known dishes) | Distance-based hierarchical |
| Time resolution | 5 min/frame | 20 min/frame |

## Repository Structure

```
Deepweed/
├── README.md
├── paper_analysis/
│   ├── Duckweed_Paper_Analysis.ipynb   # Main analysis notebook (run this)
│   ├── unet_model_class.py            # U-Net model architecture (PyTorch)
│   ├── microfluidics_segmentation_fixed.py  # Standalone segmentation module
│   ├── cell_config.json               # btrack Kalman filter & hypothesis config
│   ├── _apply_pipeline.py             # Dev utility (notebook patch script)
│   └── data_model/
│       └── model/
│           ├── best_instance_unet.pt                # Petri dish model weights (Git LFS)
│           └── best_unet_microfluidics_boundary.pt  # Microfluidics model weights (Git LFS)
```

## Requirements

- **Python** 3.10 – 3.12 (tested on 3.12.7)
- **Git LFS** (for downloading model weights)
- ~2 GB disk space for model weights
- NVIDIA GPU recommended but not required (CPU inference is supported)

### Python Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework and U-Net inference |
| `opencv-python` (`cv2`) | Image/video I/O and preprocessing |
| `numpy` | Array operations |
| `pandas` | Tabular data handling |
| `scikit-image` (`skimage`) | Connected component labeling, region properties |
| `scikit-learn` | KMeans clustering |
| `matplotlib` | Plotting and figure generation |
| `btrack` | Bayesian multi-object tracking (>= 0.7) |
| `albumentations` | Image augmentation / preprocessing transforms |
| `scipy` | Hierarchical clustering, signal filtering |
| `networkx` | Lineage graph construction |
| `imageio` | GIF generation |
| `tqdm` | Progress bars |
| `Pillow` (`PIL`) | Image utilities |
| `ipykernel` | Jupyter notebook kernel |

## Installation

### Prerequisites

1. **Install Git LFS** (required to download model weights):

   ```bash
   # macOS
   brew install git-lfs

   # Ubuntu/Debian
   sudo apt-get install git-lfs

   # Windows (with Git for Windows — LFS is included)
   # Or download from https://git-lfs.com
   ```

   Then initialize Git LFS:
   ```bash
   git lfs install
   ```

2. **Install Conda** (recommended) from [Miniforge](https://github.com/conda-forge/miniforge) or [Anaconda](https://www.anaconda.com/download).

### Option A: Conda (Recommended)

```bash
# 1. Clone the repository (includes model weights via Git LFS)
git clone https://github.com/SamOliveiraLab/Deepweed.git
cd Deepweed

# 2. Create a conda environment
conda create -n deepweed python=3.12 -y
conda activate deepweed

# 3. Install PyTorch (CPU version — see GPU section below for CUDA)
conda install pytorch torchvision cpuonly -c pytorch -y

# 4. Install remaining dependencies
pip install opencv-python numpy pandas scikit-image scikit-learn \
            matplotlib btrack albumentations scipy networkx imageio \
            tqdm Pillow ipykernel

# 5. Register the Jupyter kernel
python -m ipykernel install --user --name deepweed --display-name "Deepweed"

# 6. Verify model weights were downloaded (should be ~MB files, not LFS pointers)
ls -lh paper_analysis/data_model/model/
# Expected: two .pt files, each several MB to hundreds of MB
```

### Option B: pip with venv

```bash
# 1. Clone the repository
git clone https://github.com/SamOliveiraLab/Deepweed.git
cd Deepweed

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install PyTorch (CPU version — see GPU section below for CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install opencv-python numpy pandas scikit-image scikit-learn \
            matplotlib btrack albumentations scipy networkx imageio \
            tqdm Pillow ipykernel

# 5. Register the Jupyter kernel
python -m ipykernel install --user --name deepweed --display-name "Deepweed"
```

### GPU Support (Optional)

If you have an NVIDIA GPU with CUDA, replace the PyTorch installation step:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Or with conda:
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Check the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the latest CUDA options.

## Data Setup

### Download the Dataset

The imaging data used in this study is publicly available on Zenodo:

**[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18843918.svg)](https://doi.org/10.5281/zenodo.18843918)**

Download the dataset and extract it to a location of your choice (referred to as `BASE_PATH` in the notebook configuration).

### Expected Directory Layout

The pipeline expects your imaging data organized as follows:

```
<BASE_PATH>/
├── data_model/
│   ├── data/
│   │   ├── petri_dish/          # JPEG/PNG image sequence (petri dish mode)
│   │   │   ├── frame_0000.jpeg
│   │   │   ├── frame_0001.jpeg
│   │   │   └── ...
│   │   └── microfluidics/       # AVI video file (microfluidics mode)
│   │       └── duckweed_25_0504_multiple.avi
│   └── model/                   # Pretrained weights (included in repo via Git LFS)
│       ├── best_instance_unet.pt
│       └── best_unet_microfluidics_boundary.pt
├── cell_config.json             # btrack config (included in repo)
└── paper_analysis_output/       # Created automatically
    ├── petri_dish/
    └── microfluidics/
```

**Note:** The model weights are included in this repository via Git LFS. The imaging data (images and video) must be downloaded separately from Zenodo using the link above.

## Usage

### Quick Start

```bash
cd Deepweed

# Activate your environment
conda activate deepweed   # or: source .venv/bin/activate

# Launch Jupyter
jupyter notebook paper_analysis/Duckweed_Paper_Analysis.ipynb
```

### Configuration

All parameters are set in **Cell 4** (Section 1.2) of the notebook. Edit these before running:

```python
# Choose your dataset type
DATASET_TYPE = "petri_dish"    # or "microfluidics"

# Set the base path to your data
BASE_PATH = "/path/to/your/data"

# Optionally limit frames for testing
FRAME_LIMIT = 50   # Set to None for all frames
```

**Key parameters by dataset:**

| Parameter | Petri Dish | Microfluidics | Description |
|---|---|---|---|
| `THRESHOLD` | 0.2 | 0.2 | Segmentation confidence threshold |
| `MIN_AREA` | 30 | 3 | Minimum frond area (pixels) |
| `CLUSTER_DISTANCE` | 150 | 40 | Spatial clustering distance (pixels) |
| `N_CLUSTERS` | 6 | `None` (auto) | Number of clusters |
| `MINUTES_PER_FRAME` | 5 | 20 | Time between frames |

### Running the Pipeline

Run the notebook cells **in order**. The pipeline follows this sequence:

1. **Part 1** - Setup: imports, configuration, model loading
2. **Part 2** - Segmentation demo on a single image (sanity check)
3. **Part 3** - Full pipeline: segmentation + tracking + lineage inference on all frames
4. **Part 4** - Cluster analysis: group fronds, select regions of interest
5. **Part 5** - Generate visualizations (population plots, growth curves, timelines)
6. **Part 6** - Lineage and generation analysis

**Checkpointing:** After Part 3, you can save a checkpoint (Cell 24) and reload it later (Cell 8) to skip reprocessing.

## Pipeline Overview

```
Input (images/video)
        │
        ▼
┌─────────────────┐
│  U-Net Model    │  Semantic segmentation
│  (PyTorch)      │  → binary or 3-class mask
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Connected      │  Instance segmentation
│  Components     │  → individual frond masks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  btrack         │  Bayesian tracking with
│  (Kalman + HMM) │  Kalman filter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Track Stitching│  Merge fragmented tracks
│  + Lineage      │  Detect budding events
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cluster        │  Group by spatial location
│  Analysis       │  (KMeans / hierarchical)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │  Figures, plots, GIFs,
│  & Statistics   │  CSV exports
└─────────────────┘
```

## Outputs

The pipeline generates the following in `OUTPUT_DIR`:

| Output | Description |
|---|---|
| `unified_tracking_data.csv` | All detections with track IDs, areas, centroids, and time |
| `lineage_data.csv` | Parent-child relationships from budding events |
| `cluster_*_population_smoothed.csv` | Filtered population counts per cluster |
| `segmentation_overlay.mp4` | Video with colored instance masks and bounding boxes |
| `cluster_timeline_*.gif` | Animated GIF of cluster evolution |
| `population_plot.png` | Smoothed population count over time |
| `area_over_time.png` | Individual growth trajectories |
| `trajectories.png` | Movement paths of tracked fronds |
| `doubling_time.png` | Doubling time analysis |
| `generation_distribution.png` | Size distribution by generation |

## Troubleshooting

### Git LFS: Model weights are text pointers instead of binary files

If `.pt` files are tiny (~130 bytes) and contain text like `version https://git-lfs.github.com/spec/v1`, Git LFS did not download them:

```bash
git lfs pull
```

### `ModuleNotFoundError: No module named 'unet_model_class'`

The notebook expects to be run from the `paper_analysis/` directory. Either:
- Open the notebook directly from `paper_analysis/`
- Or add the path in the notebook before imports:
  ```python
  import sys
  sys.path.insert(0, '/path/to/Deepweed/paper_analysis')
  ```

### `ModuleNotFoundError: No module named 'btrack'`

Make sure you installed btrack >= 0.7:
```bash
pip install btrack
```

### CUDA / GPU not detected

Verify your PyTorch sees the GPU:
```python
import torch
print(torch.cuda.is_available())   # Should print True
print(torch.cuda.get_device_name(0))
```

If `False`, reinstall PyTorch with the correct CUDA version (see [GPU Support](#gpu-support-optional)).
The pipeline works on CPU — it will just be slower.

### `cv2.error` when reading video

Ensure your OpenCV build supports the video codec:
```bash
pip install opencv-python-headless   # or: pip install opencv-python
```

For AVI files with specific codecs, you may need system-level codec libraries (e.g., FFmpeg):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Out of memory

- Set `FRAME_LIMIT` to a smaller number in the configuration cell to process fewer frames.
- Use CPU instead of GPU if GPU memory is limited (the notebook auto-detects and falls back to CPU).

## License

See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite the associated paper. (Citation details to be added upon publication.)
