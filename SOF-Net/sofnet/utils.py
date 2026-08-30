import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images = {
        key: torch.stack([item[0][key] for item in batch])
        for key in batch[0][0]
    }
    targets = [item[1] for item in batch]
    return images, targets


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def save_json(data, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def cosine_warmup_lambda(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += float(value) * count
        self.count += count

    @property
    def average(self):
        return self.total / max(self.count, 1)

