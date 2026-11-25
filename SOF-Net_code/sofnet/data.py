import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import collate_fn


def normalize_channels(array, indices=None):
    result = array.astype(np.float32).copy()
    for index in range(result.shape[0]):
        if indices is not None and index not in indices:
            result[index] = np.nan_to_num(result[index])
            continue
        channel = result[index]
        valid = np.isfinite(channel)
        if not valid.any():
            result[index] = 0.0
            continue
        low, high = np.percentile(channel[valid], [2.0, 98.0])
        scale = max(float(high - low), 1e-6)
        result[index] = np.clip((np.nan_to_num(channel, nan=low) - low) / scale, 0.0, 1.0)
    return result


def rotate_boxes(boxes, height, width):
    if boxes.shape[0] == 0:
        return boxes, width, height
    corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )
    x = corners[..., 0].copy()
    y = corners[..., 1].copy()
    corners[..., 0] = y
    corners[..., 1] = width - x
    minimum = corners.min(axis=1)
    maximum = corners.max(axis=1)
    return np.concatenate([minimum, maximum], axis=1), width, height


def augment_sample(images, target):
    height, width = images["optical"].shape[-2:]
    boxes = target["boxes"].copy()
    mask = target["mask"].copy()
    if random.random() < 0.5:
        images = {key: value[..., :, ::-1].copy() for key, value in images.items()}
        mask = mask[:, ::-1].copy()
        if boxes.shape[0] > 0:
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
    if random.random() < 0.5:
        images = {key: value[..., ::-1, :].copy() for key, value in images.items()}
        mask = mask[::-1, :].copy()
        if boxes.shape[0] > 0:
            y1 = boxes[:, 1].copy()
            y2 = boxes[:, 3].copy()
            boxes[:, 1] = height - y2
            boxes[:, 3] = height - y1
    rotations = random.randint(0, 3)
    for _ in range(rotations):
        images = {key: np.rot90(value, 1, axes=(-2, -1)).copy() for key, value in images.items()}
        mask = np.rot90(mask, 1).copy()
        boxes, height, width = rotate_boxes(boxes, height, width)
    target["boxes"] = boxes
    target["mask"] = mask
    return images, target


class SOFDataset(Dataset):
    def __init__(self, root, split, augment=False, config=None):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.config = config or {}
        if augment and split != 'train':
            raise ValueError('Augmentation is training-only')
        manifest_path = self.root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f'No records for split {split}')
        scene_splits, platform_splits = {}, {}
        for record in records:
            for mapping, ids in ((scene_splits, [record.get('scene_id')]), (platform_splits, record.get('platform_ids', []))):
                for identity in ids:
                    if identity is None:
                        raise ValueError('Rebuild processed data with scene and platform identities')
                    if identity in mapping and mapping[identity] != record['split']:
                        raise ValueError(f'Split leakage: {identity}')
                    mapping[identity] = record['split']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        file_path = self.root / record["file"]
        with np.load(file_path, allow_pickle=False) as sample:
            images = {
                "optical": normalize_channels(sample["optical"]),
                "sar": normalize_channels(sample["sar"], self.config.get('sar_normalize_channels', [0,1,3,4])),
                "prior": np.clip(np.nan_to_num(sample["prior"].astype(np.float32)), 0, 1),
                "quality": np.clip(np.nan_to_num(sample["quality"].astype(np.float32)), 0, 1),
                "quality_valid": sample['quality_valid'][None].astype(np.float32),
            }
            target = {
                "boxes": sample["boxes"].astype(np.float32),
                "labels": sample["labels"].astype(np.int64),
                "mask": sample["mask"].astype(np.int64),
                "image_id": record["sample_id"],
                "transform": sample["transform"].astype(np.float64),
                "crs": str(sample["crs"].item()),
            }
            target['mask'][~sample['valid'].astype(bool)] = 255
        if images["prior"].ndim == 2:
            images["prior"] = images["prior"][None]
        if images["quality"].ndim == 2:
            images["quality"] = images["quality"][None]
        if self.augment:
            images, target = augment_sample(images, target)
        tensor_images = {key: torch.from_numpy(value.copy()).float() for key, value in images.items()}
        tensor_target = {
            "boxes": torch.from_numpy(target["boxes"]).float(),
            "labels": torch.from_numpy(target["labels"]).long(),
            "mask": torch.from_numpy(target["mask"]).long(),
            "image_id": target["image_id"],
            "transform": torch.from_numpy(target["transform"]).double(),
            "crs": target["crs"],
            'valid': torch.from_numpy(((target['mask'] >= 0) & (target['mask'] <= 5)).copy()),
            'scene_id': record['scene_id'],
            'region': record['region'],
            'platform_ids': record['platform_ids'],
            'reference_boxes_scene': torch.tensor(record['reference_boxes_scene'], dtype=torch.float32).reshape(-1,4),
            'window_xy': record['window_xy'],
            'scene_shape': record['scene_shape'],
            'scene_transform': record['scene_transform'],
        }
        return tensor_images, tensor_target


def build_dataloader(config, split, shuffle=False, augment=False):
    dataset = SOFDataset(config["data"]["processed_root"], split, augment=augment, config=config['data'])
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=shuffle,
    )
