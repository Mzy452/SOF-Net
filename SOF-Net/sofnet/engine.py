from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .utils import AverageMeter, move_targets


def train_one_epoch(model, criterion, loader, optimizer, scheduler, scaler, device, gradient_clip):
    model.train()
    meters = {}
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(
                images["optical"], images["sar"], images["prior"], images["quality"], targets=targets
            )
            losses = criterion(outputs, targets)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for name, value in losses.items():
            meters.setdefault(name, AverageMeter()).update(value.detach().item(), images["optical"].shape[0])
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def validate_loss(model, criterion, loader, device):
    model.eval()
    meters = {}
    for images, targets in tqdm(loader, desc="validation", leave=False):
        images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
        targets = move_targets(targets, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(
                images["optical"], images["sar"], images["prior"], images["quality"], targets=targets
            )
            losses = criterion(outputs, targets)
        for name, value in losses.items():
            meters.setdefault(name, AverageMeter()).update(value.detach().item(), images["optical"].shape[0])
    return {name: meter.average for name, meter in meters.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, config, best_loss):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_loss": best_loss,
        },
        destination,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
