# SOF-Net

This repository provides the PyTorch implementation of SOF-Net, a SAR-prior-guided optical fusion network for offshore oil and gas infrastructure mapping and component-level interpretation.

## Scope

The repository contains four reproducible stages:

1. Multitemporal GF-3 and Sentinel-1 prior construction
2. Aligned optical, SAR, prior, quality, detection and component-mask patch preparation
3. Joint platform detection and five-class component segmentation training
4. Detection, segmentation and facility-level priority evaluation

The five component classes are platform body, bridge, crane, flare-related structure and helipad.

## Method correspondence

The implementation follows the manuscript configuration:

- ConvNeXt-Tiny optical encoder
- ResNet-18-FPN SAR encoder
- Four pyramid levels from P2 to P5
- Prior-guided deformable cross-attention with 8 heads and 4 sampling points
- Maximum displacement radii of 8, 4, 2 and 1 pixels
- Quality-aware gated fusion
- Anchor-free classification, centerness and distance regression
- Scale-aware RoI component decoder
- Focal, centerness, generalized-IoU, component, boundary and prior-consistency losses
- SGD optimization for 300 epochs with linear warm-up and cosine decay

## Repository structure

```text
SOF-Net/
  config.yaml
  build_sar_prior.py
  prepare_dataset.py
  train.py
  test.py
  manifests/
    raw_manifest.csv
    sar_prior_manifest.csv
  sofnet/
    __init__.py
    config.py
    data.py
    engine.py
    losses.py
    metrics.py
    model.py
    sar_prior.py
    utils.py
```

## Input products

All raster products for one sample must be co-registered on the optical grid before patch preparation.

| Product | Expected content |
|---|---|
| Optical raster | Pan-sharpened multispectral bands, panchromatic band and NDWI |
| SAR stack | Calibrated backscatter, standardized anomalies, CFAR confidence and temporal persistence |
| SAR prior | One continuous channel in the range 0 to 1 |
| Quality map | One optical-validity channel in the range 0 to 1 |
| Component mask | Integer values 0 to 5, with 0 as background |
| Bounding boxes | Pixel-coordinate platform boxes in JSON format |

The box JSON file must contain either a list of `[xmin, ymin, xmax, ymax]` boxes or an object with a `boxes` field.

The manuscript partition contains 700 training patches, 100 validation patches and 200 test patches from 200 unique physical platforms. Scene identities and physical-platform identities must remain mutually exclusive across the three subsets. Augmentation is applied only to the training subset.

## Manifest format

`manifests/sar_prior_manifest.csv` contains:

```text
sample_id,gf3,sentinel1,sentinel1_quality,output_prior,output_stack
```

Multiple scenes are separated with semicolons.

`manifests/raw_manifest.csv` contains:

```text
sample_id,split,optical,sar,prior,quality,mask,boxes,selection,x,y
```

The optional `x` and `y` fields select one 1000 by 1000 window. The `selection` field accepts `grid` for regular tiling or `prior` for SAR-thresholded ROIs with the configured optical-pixel buffer.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python build_sar_prior.py --config "Real Filepath"
python prepare_dataset.py --config "Real Filepath"
python train.py --config "Real Filepath"
python test.py --config "Real Filepath"
```

