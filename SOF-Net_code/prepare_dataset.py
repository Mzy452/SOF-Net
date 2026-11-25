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
        return source.read(masked=True).astype('float32').filled(np.nan), source.profile.copy()


def read_aligned(path, reference, resampling):
    with rasterio.open(path) as source:
        destination = np.full(
            (source.count, reference["height"], reference["width"]), dtype=np.float32
            , fill_value=np.nan
        )
        for band in range(source.count):
            reproject(
                source=source.read(band + 1, masked=True).astype('float32').filled(np.nan),
                destination=destination[band],
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=reference["transform"],
                dst_crs=reference["crs"],
                resampling=resampling,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
    return destination


def read_boxes(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    boxes = data["boxes"] if isinstance(data, dict) else data
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def read_identity_boxes(path):
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or 'platform_ids' not in data:
        raise ValueError('Box JSON requires boxes and globally stable platform_ids')
    boxes = np.asarray(data['boxes'], dtype=np.float32).reshape(-1, 4)
    ids = list(map(str, data['platform_ids']))
    if len(boxes) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid platform_ids')
    return boxes, ids


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


def positions(row, prior, width, height, size, stride, prior_config, boxes=None):
    if row.get("x", "").strip() and row.get("y", "").strip():
        return [(int(row["x"]), int(row["y"]))]
    if row.get('selection') == 'platform':
        if boxes is None:
            raise ValueError('Platform-centred crops need reference boxes')
        return sorted(set((min(max(int((b[0]+b[2])/2)-size//2, 0), max(width-size,0)),
                           min(max(int((b[1]+b[3])/2)-size//2, 0), max(height-size,0))) for b in boxes))
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
    if not rows:
        raise ValueError('Raw manifest is empty; provide actual scene/patch records')
    records = []
    seen_scenes, seen_platforms, seen_patches = {}, {}, set()
    for row in rows:
        if not row.get('scene_id') or not row.get('region'):
            raise ValueError('Manifest requires scene_id and region')
        if row['split'] not in ('train', 'validation', 'test', 'application'):
            raise ValueError('Unknown split')
        if row['split'] == 'test' and row.get('selection') != 'platform' and not row.get('x'):
            raise ValueError('Development test requires platform-centred or explicit archived crops')
        optical, profile = read_raster(row["optical"])
        sar = read_aligned(row["sar"], profile, Resampling.cubic)
        prior = read_aligned(row["prior"], profile, Resampling.bilinear)
        quality = read_aligned(row["quality"], profile, Resampling.bilinear)
        raw_mask = read_aligned(row["mask"], profile, Resampling.nearest)
        valid = np.isfinite(raw_mask) & (raw_mask >= 0) & (raw_mask <= 5)
        valid &= np.isfinite(optical).all(axis=0)[None]
        if row.get('valid'):
            v = read_aligned(row['valid'], profile, Resampling.nearest)
            valid &= np.isfinite(v) & (v > 0)
        mask = np.where(valid, raw_mask, 255).astype(np.uint8)
        quality_valid = np.isfinite(quality)
        if row.get('quality_valid'):
            qv = read_aligned(row['quality_valid'], profile, Resampling.nearest)
            quality_valid &= np.isfinite(qv) & (qv > 0)
        boxes, platform_ids = read_identity_boxes(row['boxes'])
        for mapping, keys in ((seen_scenes, [row['scene_id']]), (seen_platforms, platform_ids)):
            for key in keys:
                if key in mapping and mapping[key] != row['split']:
                    raise ValueError(f'Split leakage: {key}')
                mapping[key] = row['split']
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
            boxes,
        ):
            current_boxes = patch_boxes(boxes, x, y, data_config["patch_size"])
            hit = (boxes[:, 2] > x) & (boxes[:, 0] < x+data_config['patch_size']) & (boxes[:, 3] > y) & (boxes[:, 1] < y+data_config['patch_size'])
            current_ids = [v for v, keep in zip(platform_ids, hit) if keep]
            if row['split'] == 'test' and not len(current_boxes):
                raise ValueError('Development test contains an empty-platform patch')
            if current_boxes.shape[0] == 0 and random.random() > data_config["negative_keep_probability"]:
                continue
            split = row["split"]
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{row['sample_id']}_{x}_{y}.npz"
            if file_name in seen_patches:
                raise ValueError(f'Duplicate output patch {file_name}')
            seen_patches.add(file_name)
            relative = Path(split) / file_name
            affine = window_transform(
                Window(x, y, data_config["patch_size"], data_config["patch_size"]), profile["transform"]
            )
            np.savez_compressed(
                output_root / relative,
                optical=patch_array(optical, x, y, data_config["patch_size"], fill=np.nan).astype(np.float32),
                sar=patch_array(sar, x, y, data_config["patch_size"], fill=np.nan).astype(np.float32),
                prior=np.clip(
                    patch_array(prior, x, y, data_config["patch_size"])[0:1].astype(np.float32), 0.0, 1.0
                ),
                quality=np.clip(
                    patch_array(quality, x, y, data_config["patch_size"])[0:1].astype(np.float32), 0.0, 1.0
                ),
                mask=patch_array(mask, x, y, data_config["patch_size"], fill=255)[0].astype(np.uint8),
                valid=patch_array(valid, x, y, data_config['patch_size'])[0],
                quality_valid=patch_array(quality_valid, x, y, data_config['patch_size'])[0],
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
                    'scene_id': row['scene_id'],
                    'region': row['region'],
                    'platform_ids': current_ids,
                    'reference_boxes_scene': boxes[hit].tolist(),
                    'window_xy': [x, y],
                    'scene_shape': [profile['height'], profile['width']],
                    'scene_transform': list(profile['transform']),
                    'selection': row.get('selection', 'explicit'),
                }
            )
    save_json(records, output_root / "manifest.json")
    summary = {}
    for split in ("train", "validation", "test"):
        summary[split] = sum(record["split"] == split for record in records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
