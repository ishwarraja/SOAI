
# CIFAR-10: Dilated + Depthwise-Separable CNN (Session 7)

## Overview / Goal
This repository contains a compact convolutional neural network and training script for CIFAR-10 that meets the assignment requirements:

- Works on CIFAR-10.
- Architecture form: **C1 C2 C3 C4** (no MaxPooling; the last block includes a convolution with `stride=2` for spatial downsampling).
- Total effective **receptive field (RF) > 44**.
- At least **one Depthwise Separable Convolution** is present.
- At least **one Dilated Convolution** is present (we use multiple dilated convs).
- Uses **Global Average Pooling (GAP)** followed by an FC head.
- Uses **albumentations** with the requested augmentations:
  - `HorizontalFlip`
  - `ShiftScaleRotate` (implemented via `Affine` — recommended alias)
  - `CoarseDropout` (with `max_holes=1`, `height/width = 16`, and `fill_value` set to CIFAR-10 mean)
- Aim for **>= 85% accuracy** (requires sufficient epochs + GPU). Model kept **< 200k parameters**.

## Files
- `S7.ipynb / cifar10_dilated_ds.py` — main model + training script (ready-to-run).
- `README.md` — this file.

## How the architecture maps to C1 C2 C3 C4
- **C1**: `conv1` (standard 3x3 conv)
- **C2**: `ds` DepthwiseSeparableConv (3x3 depthwise + 1x1 pointwise)
- **C3**: a stack of dilated convolutions (`dilated1` .. `dilated4`) — these expand receptive field without pooling
- **C4**: `down` — final 3x3 conv with `stride=2` for spatial downsampling (no MaxPool)

After C4 we have `post1` and `post2` (extra capacity) followed by GAP and final FC.

## Receptive Field
An estimated receptive field is printed at start. The chosen dilation stack ensures RF > 44 for the default configuration.

## Augmentations (albumentations)
We use:
- `A.HorizontalFlip(p=0.5)`
- `A.Affine` (replacement for ShiftScaleRotate; same effects)
- `A.CoarseDropout(max_holes=1, max_height=16, max_width=16, ... fill_value=<CIFAR mean in 0..255>)`
- `A.Normalize` and `ToTensorV2()`

**Note:** Different versions of albumentations accept slightly different parameter names; the script uses a compatible call (works with most 1.x versions). If you get a warning, update albumentations or reduce CoarseDropout args to the minimal set.

## Model size
By default `base_ch=24`. This keeps total params under 200k (printed at script start). If you want more capacity increase `base_ch`, but keep params < 200k to meet the constraint.

## Training tips to reach 85%+
- Use a GPU. On CPU the run will be extremely slow.
- Train for at least **150–300 epochs** (experiment).
- Consider adding MixUp/CutMix and a tiny label smoothing to push accuracy up.
- Increase batch size (if GPU memory allows) and tune learning rate accordingly.
- Use `--base_ch` to trade param count vs capacity (watch `Model params` printed at start).

## Usage
Directly execute on google collab

OR

```bash
pip install torch torchvision albumentations tqdm
python cifar10_dilated_ds.py --epochs 180 --batch 128 --base_ch 24 --lr 0.1 --workers 4 --save best.pth

