import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
from scipy.ndimage import find_objects, label

from sofnet.config import load_config
from sofnet.utils import save_json, set_seed


def read_raster(path):
    with rasterio.open(path) as source:
        return source.read(), source.profile.copy()


def read_aligned(path, reference, resampling):
    with rasterio.open(path) as source:
        destination = np.zeros(
            (source.count, reference["height"], reference["width"]), dtype=np.float32
        )
        for band in range(source.count):
            reproject(
                source=source.read(band + 1),
                destination=destination[band],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=reference["transform"],
                dst_crs=reference["crs"],
                resampling=resampling,
            )
    return destination


def read_boxes(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    boxes = data["boxes"] if isinstance(data, dict) else data
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def patch_array(array, x, y, size, fill=0):
    result = np.full((array.shape[0], size, size), fill, dtype=array.dtype)
    height = min(size, array.shape[1] - y)
    width = min(size, array.shape[2] - x)
    if height > 0 and width > 0:
        result[:, :height, :width] = array[:, y : y + height, x : x + width]
    return result


def patch_boxes(boxes, x, y, size):
    if boxes.shape[0] == 0:
        return boxes
    intersection_x1 = np.maximum(boxes[:, 0], x)
    intersection_y1 = np.maximum(boxes[:, 1], y)
    intersection_x2 = np.minimum(boxes[:, 2], x + size)
    intersection_y2 = np.minimum(boxes[:, 3], y + size)
    valid = (intersection_x2 > intersection_x1) & (intersection_y2 > intersection_y1)
    clipped = np.stack(
        [intersection_x1[valid] - x, intersection_y1[valid] - y, intersection_x2[valid] - x, intersection_y2[valid] - y],
        axis=1,
    )
    return clipped.astype(np.float32)


def grid_positions(width, height, size, stride, x_min=0, y_min=0, x_max=None, y_max=None):
    x_max = width if x_max is None else x_max
    y_max = height if y_max is None else y_max
    x_values = list(range(x_min, max(x_max - size + 1, x_min + 1), stride))
    y_values = list(range(y_min, max(y_max - size + 1, y_min + 1), stride))
    final_x = min(max(x_max - size, x_min), max(width - size, 0))
    final_y = min(max(y_max - size, y_min), max(height - size, 0))
    if not x_values or x_values[-1] != final_x:
        x_values.append(final_x)
    if not y_values or y_values[-1] != final_y:
        y_values.append(final_y)
    return [(max(x, 0), max(y, 0)) for y in y_values for x in x_values]


def positions(row, prior, width, height, size, stride, prior_config):
    if row.get("x", "").strip() and row.get("y", "").strip():
        return [(int(row["x"]), int(row["y"]))]
    if row.get("selection", "grid").strip().lower() != "prior":
        return grid_positions(width, height, size, stride)
    components, _ = label(prior[0] >= prior_config["roi_threshold"])
    windows = []
    buffer_size = prior_config["roi_buffer"]
    for region in find_objects(components):
        if region is None:
            continue
        y_slice, x_slice = region
        x_min = max(x_slice.start - buffer_size, 0)
        y_min = max(y_slice.start - buffer_size, 0)
        x_max = min(x_slice.stop + buffer_size, width)
        y_max = min(y_slice.stop + buffer_size, height)
        if x_max - x_min <= size and y_max - y_min <= size:
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            windows.append(
                (
                    min(max(center_x - size // 2, 0), max(width - size, 0)),
                    min(max(center_y - size // 2, 0), max(height - size, 0)),
                )
            )
        else:
            windows.extend(grid_positions(width, height, size, stride, x_min, y_min, x_max, y_max))
    return sorted(set(windows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    set_seed(config["seed"])
    data_config = config["data"]
    output_root = Path(data_config["processed_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    with open(data_config["raw_manifest"], "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        optical, profile = read_raster(row["optical"])
        sar = read_aligned(row["sar"], profile, Resampling.bilinear)
        prior = read_aligned(row["prior"], profile, Resampling.bilinear)
        quality = read_aligned(row["quality"], profile, Resampling.bilinear)
        mask = read_aligned(row["mask"], profile, Resampling.nearest).astype(np.uint8)
        boxes = read_boxes(row["boxes"])
        if optical.shape[0] != data_config["optical_channels"]:
            raise ValueError(f"{row['sample_id']} optical channels do not match config")
        if sar.shape[0] != data_config["sar_channels"]:
            raise ValueError(f"{row['sample_id']} SAR channels do not match config")
        for x, y in positions(
            row,
            prior,
            profile["width"],
            profile["height"],
            data_config["patch_size"],
            data_config["stride"],
            config["sar_prior"],
        ):
            current_boxes = patch_boxes(boxes, x, y, data_config["patch_size"])
            if current_boxes.shape[0] == 0 and random.random() > data_config["negative_keep_probability"]:
                continue
            split = row["split"]
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{row['sample_id']}_{x}_{y}.npz"
            relative = Path(split) / file_name
            affine = window_transform(
                Window(x, y, data_config["patch_size"], data_config["patch_size"]), profile["transform"]
            )
            np.savez_compressed(
                output_root / relative,
                optical=patch_array(optical, x, y, data_config["patch_size"]).astype(np.float32),
                sar=patch_array(sar, x, y, data_config["patch_size"]).astype(np.float32),
                prior=np.clip(
                    patch_array(prior, x, y, data_config["patch_size"])[0:1].astype(np.float32), 0.0, 1.0
                ),
                quality=np.clip(
                    patch_array(quality, x, y, data_config["patch_size"])[0:1].astype(np.float32), 0.0, 1.0
                ),
                mask=patch_array(mask, x, y, data_config["patch_size"])[0].astype(np.uint8),
                boxes=current_boxes,
                labels=np.ones((current_boxes.shape[0],), dtype=np.int64),
                transform=np.asarray(tuple(affine), dtype=np.float64),
                crs=np.asarray(profile["crs"].to_string()),
            )
            records.append(
                {
                    "sample_id": f"{row['sample_id']}_{x}_{y}",
                    "split": split,
                    "file": relative.as_posix(),
                }
            )
    save_json(records, output_root / "manifest.json")
    summary = {}
    for split in ("train", "validation", "test"):
        summary[split] = sum(record["split"] == split for record in records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
