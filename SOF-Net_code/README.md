# SOF-Net revision implementation

This local revision implements the specified mechanisms and evaluation interfaces in the 22 November 2026 manuscript and supplementary material. It is based on the existing SOF-Net source, not on recovered training logs. No trained weights or original evaluation rasters were supplied. Consequently, this package does **not** establish reproduction of the paper's numerical results. It never fills missing experimental scores from the manuscript tables.

See [修订说明.md](修订说明.md) for the source-to-code mapping, unresolved specification choices and known manuscript conflicts. The original code and Word files are unchanged. Nothing has been uploaded to GitHub.

## Main changes

- CFAR confidence uses `clip((I-T)/(8T-T),0,1)`, valid-date averaging and per-pixel available-sensor weight normalization.
- Six segmentation channels: background, platform body, bridge, crane, flare-related structure, helipad. Ignore value 255 and the independent annotation-validity raster are excluded from focal, Dice, boundary and confusion counts.
- Detection scores use objectness × centerness. Ranked AP candidates and the 0.30 operating threshold are separated.
- Original prior loss remains the default benchmark; revised positive-label, strong-prior, one-sided loss is a separate retraining option.
- Matched controls for prior logits, gate inputs, fixed axial sampling, loss and radius settings are configurable.
- Training checkpoint selection uses validation AP50, not validation loss.
- Test exports integer TP/FP/FN, canonical scene/platform IDs, raw predictions, class confusion counts, quality-group counts and checkpoint SHA-256.
- `G=1` means greater **SAR** contribution. P2–P5 raw gates and their equally weighted, bilinearly upsampled mean are exported for Fig. S3-1.
- Scene-level NMS removes repeated patch detections. Common-grid evaluation uses complete reference rasters, including areas without candidate patches.

## Installation and verification

Use a dedicated Python environment with matching PyTorch/torchvision builds and the requirements in `requirements.txt`. The verification environment used Python 3.10, PyTorch 2.3.0+cu118 and torchvision 0.18.0+cu118 on CPU. No GPU performance benchmark was run.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic software fixtures in temporary directories, including a random-weight smoke-test checkpoint. They are not scientific experiments, trained-model gate maps or evidence for the reported accuracy. The source package contains no generated experimental-result tables.

## Data contracts

All SAR inputs to prior construction must be calibrated **dB** rasters on one aligned projected grid. Each file holds the available polarizations for one acquisition. Use one non-duplicated file/mosaic per acquisition date. Raster no-data masks must be valid; zero is a legitimate dB value and must not be used to hide missing observations. Grid differences are rejected, not silently interpreted as registration.

`manifests/sar_prior_manifest.csv` is a header-only template. Semicolons separate scene paths. Leave `sentinel1_quality` empty for equal acquisition weights. Explicit weights must match acquisition count and be finite/nonnegative. Acquisitions are retained across infrastructure changes, following the current manuscript; no temporal-state filter is silently introduced.

The eight SAR stack channels are retained from the original implementation: GF-3 mean backscatter, maximum anomaly, confidence; S1 mean backscatter, maximum anomaly, continuous persistence, thresholded persistence; fused prior. Only backscatter/anomaly channels 0,1,3,4 receive patch percentile scaling. Confidence and persistence channels retain their original scales.

The optical input has six channels (four pan-sharpened multispectral bands, PAN, water index). Supply the actual band order and preprocessing used for training. RGB display indices must be explicitly specified to the gate renderer. The network input uses the inherited per-channel 2nd–98th percentile stretch; this implementation choice is documented, not inferred from a checkpoint.

`manifests/raw_manifest.csv` fields:

```text
sample_id,scene_id,region,split,optical,sar,prior,quality,quality_valid,mask,valid,boxes,selection,x,y
```

- `scene_id` and globally stable `platform_ids` must not cross train/validation/test splits.
- Box files are JSON: `{"boxes":[[xmin,ymin,xmax,ymax]],"platform_ids":["actual-platform-id"]}`. Coordinates are full-scene optical pixels. Do not substitute invented IDs for unavailable records.
- `quality_valid` flags acquisition-quality no-data; `valid` flags annotation reliability. They are separate and optional only when the input rasters already encode correct no-data/255 labels.
- `selection=platform` creates reference-platform-centred crops. Explicit archived `x,y` can specify the original crop selection. The development test has no deliberately empty-platform patches. Neither the exact 200 crops nor their 67 identities is reconstructed from aggregate counts.
- `selection=prior` creates deployment candidates from the continuous prior at 0.25 with a 128-optical-pixel buffer. The persistence-channel threshold does not select these ROIs.
- Equipment-first label precedence is an annotation requirement. The code does not manufacture equipment masks from deck labels.
- RPC orthorectification, SAR calibration/terrain correction, tie-point registration and independent annotation review are upstream tasks, not implemented by simple reprojection in this package.

## Preparation and training

Replace the paths in `config.yaml` with actual data paths. Pretrained backbones are enabled for fresh training; inference loads checkpoint weights without downloading backbone weights.

```bash
python build_sar_prior.py --config config.yaml
python build_quality.py --config config.yaml --indicators five_indicators.tif --output quality.tif
python prepare_dataset.py --config config.yaml
python train.py --config config.yaml
python test.py --config config.yaml
```

`build_quality.py` consumes five externally computed indicator rasters. It implements the exact bounds and weights in S3.3, but **not** an invented HOT/cloud/striping extraction formula. An already computed quality raster can be used directly instead. Missing-quality pixels do not enter the patch quality mean.

The default 300 epochs, SGD settings, batch size 16 and 1000-pixel patches are manuscript settings, not a claim that training fits every GPU. A full GPU run requires actual data and sufficient memory. Existing checkpoints may load structurally but cannot be assumed scientifically equivalent after CFAR, score, loss or preprocessing changes. Retrain affected controls. Old checkpoints produce an explicit warning.

## Detection, parsing and common-grid mapping

`test.py` defaults to reference-box parsing. It exports patch segmentation separately from scene-deduplicated detection. `evaluation.protocol=predicted_patch` gives a patch-level predicted-box diagnostic; it is **not** labelled regional/end-to-end accuracy.

`metrics.json` reports rates on a 0–1 scale, raw integer detection counts, the original patch denominator for FP/image, and pooled confusion matrices. Quality-stratified AP uses patch instances and is explicitly separated from the overall scene-deduplicated AP. An absent segmentation class has undefined IoU; the fixed-five-class mean is null rather than silently averaging fewer classes.

For full end-to-end scoring, supply a JSON list of independent full scene references:

```json
[
  {"scene_id":"actual-scene-id","mask":"full_reference.tif","valid":"full_valid.tif","boxes":"full_boxes.json"}
]
```

These references must include platforms outside supplied candidates. Use the same extent, validity, CRS and reference manifest for both protocols. Valid scene domains on the same grid must be disjoint; overlapping domains are rejected until ownership masks are supplied, preventing repeated geographic pixel counts. Every reference scene is scored even when no patch prediction exists. A physical platform repeated across source scenes needs one explicit canonical evaluation scene; ambiguous repeated ownership is rejected.

```bash
python evaluate_scene_grid.py --config config.yaml --predictions evaluation/predictions.json --scenes full_scenes.json --protocol predicted_boxes --output end_to_end.json
python evaluate_scene_grid.py --config config.yaml --predictions evaluation/predictions.json --scenes full_scenes.json --protocol reference_boxes --output conditional_grid.json
```

Conditional common-grid scoring requires an untruncated decoder result for every full-reference box, potentially a separate reference-crop inference run. It stops if one is missing. Predicted-box scoring includes all valid scene pixels and counts foreground omission/commission. Miss attribution uses intersection of reference boxes with supplied patch coverage. Unmatched boxes remain detection false positives even with an empty mask. Full scenes are processed one at a time; very large rasters may require additional tiled/memory-mapped evaluation engineering.

## Real gate panels and uncertainty

`test.py` exports `gates/<actual-patch-id>.npz` plus provenance JSON. The renderer uses a fixed 0–1 scale, real patch IDs and the SAR-direction gate. It does not ingest or generate conceptual heatmaps.

```bash
python render_gate_panels.py --inputs evaluation/gates/patch1.npz evaluation/gates/patch2.npz --rgb-bands 2 1 0 --output Fig_S3-1.png
python bootstrap_comparison.py --method-a methodA/scene_detections.json --method-b methodB/scene_detections.json --replicates 2000 --seed 20261119 --output paired_AP50.json
```

Use RGB indices appropriate for the real input. Gate means use equal level and equal patch weights as specified in S3.4. The bootstrap recomputes pooled AP independently for both methods on the same resampled scenes; it does not bootstrap a prefilled 1.9-point difference.

## Supplementary controls

```bash
python make_experiment_configs.py --config config.yaml --output experiments --seeds 42 43 44
```

The displayed seeds are an example; use the actual registered three seeds. Each training control needs its own trained checkpoint. The no-gate implementation uses a transparent 0.5/0.5 mean; this specific replacement is not stated in the manuscript and must be confirmed before attributing results to its existing Table 8. It retains the prior loss weight.

S2 shifts translate all SAR channels and prior together, not optical data. Shift configs share a 48-pixel horizontal margin and use each model's fixed checkpoint. Detection is restricted by box centres inside the common horizontal interior; masks use a common pixel-validity margin. This precise boundary convention must accompany reported results. Reconstruct the dense-cluster wrong-association analysis only after specifying and exporting the attention-centroid reduction rule; it is not replaced with box-centre proximity.

Sensor-weight/ROI/persistence configurations are provided, but use separate output manifests and actual independent screening references for each rerun. Persistence-only sweeps must retain the archived candidate list. No screening hit counts, independent audit decisions or cross-domain performance are supplied by this package.

```bash
python audit_statistics.py --mode audit --input audit_strata.csv --output audit_summary.json
python audit_statistics.py --mode registration --input checkpoint_residuals.csv --output registration_summary.json
```

Audit CSV fields: `stratum,population,sampled,confirmed`. Registration CSV fields: `dx_pixels,dy_pixels`. These utilities calculate statistics from real review/checkpoint records. They cannot establish missing manual observations.

## Inventory attributes

Per-box attributes are four-connected mask-component counts, maximum flare probability and mask-derived deck area. They are explicitly marked raw until cross-box identity review. The retained complexity rule is the original equal mean of saturated deck area and total visible-component count; flare-area fraction uses total predicted foreground area. The manuscript does not fully disambiguate these denominators and count terms. Shared deck-pair identities must be supplied/reviewed before producing deduplicated connected-platform totals. No learned attribute head is invented.
