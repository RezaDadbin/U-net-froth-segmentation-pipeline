# 🫧 UNet Froth Segmentation Pipeline

A modular **PyTorch** implementation of **UNet** for industrial froth image segmentation. This repository follows the **same workflow style** as the SBS‑Net froth pipeline: central configuration file, and four top‑level scripts for **train**, **evaluate**, **predict**, and **post‑process**.

**Authors:** [Reza Dadbin](https://github.com/RezaDadbin) · [Sina Lotfi](https://github.com/cinaLotfi)

---

## Table of contents

- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Prepare your dataset](#prepare-your-dataset)
- [Configure the pipeline](#configure-the-pipeline)
- [Run the training & evaluation workflow](#run-the-training--evaluation-workflow)
  - [1. Train](#1-train)
  - [2. Evaluate](#2-evaluate)
  - [3. Predict soft masks](#3-predict-soft-masks)
  - [4. Watershed post‑processing](#4-watershed-post-processing)
- [Outputs](#outputs)
- [Reproducibility tips](#reproducibility-tips)
- [Extending the project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Authors & citation](#authors--citation)

---

## Key features

- **UNet** backbone implemented in `unet_froth/models/unet.py`.
- **scripts/** for the full workflow: training, evaluation, prediction of probability maps, and **watershed** post‑processing.
- **Polygon JSON** labels (LabelMe/simple/COCO‑like) automatically rasterized into binary masks.
- **Single `config.py`** holds paths & hyperparameters (mirrors the SBS‑Net style).
- Inference is **dtype‑safe** and resizes probability maps back to the **original image size** for saving.
- Designed to be **clear, reproducible, and production‑friendly**.

> This pipeline’s **structure** is inspired by the SBS‑Net froth segmentation repo; we adapt the same ideas to a UNet implementation.

---

## Repository structure

```
UNet-froth-segmentation-pipeline/
├── README.md
├── config.py                 # global configuration used by all scripts
├── unet_froth/
│   ├── models/               # UNet architecture
│   └── utils/                # metrics, common helpers, visualize, postprocess
├── scripts/
│   ├── train.py              # training loop (saves best checkpoint)
│   ├── eval.py               # compute IoU/Dice on eval split
│   ├── predict.py            # export soft probability maps as PNGs
│   └── postprocess.py        # watershed instance refinement -> binary masks
├── data/
│   ├── train/                # paired images + same‑stem JSON polygons
│   ├── val/
│   └── eval/
├── checkpoints/
└── outputs/
    ├── pred_masks/
    └── postprocessed_masks/
```

---

## Getting started

> Python **3.11+** recommended. On macOS, if OpenCV complains, run `xcode-select --install`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Prepare your dataset

Place your images and annotations in `data/`:

```
data/
├── train/
│   ├── img_0001.tiff
│   ├── img_0001.json
│   └── ...
├── val/
│   ├── img_0101.png
│   ├── img_0101.json
│   └── ...
└── eval/
    ├── img_0201.jpg
    ├── img_0201.json      # optional; present if you want metrics on eval
    └── ...
```

- Supported image formats: `.tiff/.tif/.png/.jpg` (RGB).  
- **JSON must share the same stem** as the image (`img_0001.tiff` ↔ `img_0001.json`).  
- Supported JSON schemas:
  - LabelMe: `{"shapes":[{"points":[[x,y],...]},...]}`
  - Simple polygons: `{"polygons":[[[x,y],...],...],"height":H,"width":W}`
  - COCO‑like: `{"annotations":[{"segmentation":[[x1,y1,x2,y2,...]]},...],"height":H,"width":W}`
- Any image **without** a JSON is **skipped** (warning printed), so training won’t crash.

---

## Configure the pipeline

Edit **`config.py`** to adjust everything in one place:

- `data_root`, `train_dir`, `val_dir`, `eval_dir`
- `checkpoint_dir` (`checkpoints/`) and `outputs_root` (`outputs/`)
- `image_size` `(W,H)` used during training & prediction
- `batch_size`, `epochs`, `lr`, `weight_decay`, `save_every`
- `pixel_threshold` for binarization
- `device`: `"auto"`, `"cuda"`, or `"cpu"`
- `resume_checkpoint`: name of a `.pth` file under `checkpoints/`

---

## Run the training & evaluation workflow

> Run all commands from the repository root with your virtualenv activated.

### 1. Train
```bash
python scripts/train.py
```
- Tracks **train/val loss, IoU, and Dice**.  
- Saves `checkpoints/model_best.pth` (by validation IoU).  
- To resume/fine‑tune, set `resume_checkpoint` in `config.py`.

### 2. Evaluate
```bash
python scripts/eval.py
```
- Loads `model_best.pth` (or `resume_checkpoint`) and prints **IoU/Dice** on `data/eval` (skips samples without JSON).

### 3. Predict soft masks
```bash
python scripts/predict.py
```
- Runs on `data/eval/` and writes **probability maps** (`*_prob.png`) to `outputs/pred_masks/`.  
- Each map is resized back to the original image resolution.

### 4. Watershed post‑processing
```bash
python scripts/postprocess.py
```
- Converts probability maps to **final binary masks** using a watershed refinement that splits merged regions and removes small artifacts.  
- Saves results to `outputs/postprocessed_masks/`.

---

## Outputs

```
checkpoints/
└── model_best.pth

outputs/
├── pred_masks/
│   └── img_0201_prob.png
└── postprocessed_masks/
    └── img_0201_mask.png
```

---

## Reproducibility tips

- Fix `seed` in `config.py` for deterministic runs.  
- Keep `image_size` and augmentations unchanged between train/val/eval.  
- Track dataset snapshots and Git commits for each experiment.

---

## Extending the project

- Swap the backbone: add new architectures under `unet_froth/models/`.  
- Add stronger augmentations in the dataset pipeline.  
- Integrate TensorBoard or Weights & Biases in `scripts/train.py`.  
- Add a sliding‑window predictor for very large images.

---

## Troubleshooting

- **CUDA not available** → set `device="cpu"` in `config.py`.  
- **Missing JSON** → the dataset warns and skips the image.  
- **Masks too thin/thick** → tune `pixel_threshold` (0.4–0.6 typical).  
- **Noisy instances** → rely on `postprocess.py` (watershed), or increase `min_area` inside `unet_froth/utils/postprocess.py`.

---

## Authors & citation

- **Reza Dadbin** — <https://github.com/RezaDadbin>  
- **Sina Lotfi** — <https://github.com/cinaLotfi>

If you use this pipeline, please acknowledge:

> Reza Dadbin & Sina Lotfi — *UNet‑based Froth Segmentation Pipeline*

---

**Note:** The overall structure and README flow are inspired by the SBS‑Net froth segmentation pipeline (mirrored design: single config, scripts for train/eval/predict/postprocess).
