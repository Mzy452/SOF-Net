import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import label
from tqdm import tqdm

from sofnet.config import load_config
from sofnet.data import build_dataloader
from sofnet.engine import load_checkpoint
from sofnet.metrics import summarize_metrics
from sofnet.model import SOFNet
from sofnet.utils import move_targets


def platform_records(detection, target, config):
    priority = config["priority"]
    logits = detection["roi_mask_logits"]
    transform = target["transform"].cpu().numpy()
    pixel_area = abs(transform[0] * transform[4] - transform[1] * transform[3])
    probabilities = torch.softmax(logits, dim=1)
    masks = logits.argmax(dim=1).cpu().numpy()
    records = []
    for index, (box, confidence, mask) in enumerate(zip(detection["boxes"], detection["scores"], masks)):
        flare_confidence = float(probabilities[index, 4].amax())
        platform_pixels = max(int((mask > 0).sum()), 1)
        flare_ratio = min(
            float((mask == 4).sum()) / platform_pixels / priority["flare_area_saturation"], 1.0
        )
        box_width = max(float(box[2] - box[0]), 0.0)
        box_height = max(float(box[3] - box[1]), 0.0)
        platform_area = box_width * box_height * pixel_area
        component_counts = {}
        component_count = 0
        for class_index, class_name in enumerate(config["model"]["component_names"], start=1):
            count = int(label(mask == class_index)[1])
            component_counts[class_name] = count
            component_count += count
        complexity = 0.5 * min(platform_area / priority["platform_area_saturation"], 1.0)
        complexity += 0.5 * min(component_count / priority["component_count_saturation"], 1.0)
        weights = priority["weights"]
        score = weights[0] * flare_confidence + weights[1] * flare_ratio + weights[2] * complexity
        records.append(
            {
                "box": box.detach().cpu().tolist(),
                "confidence": float(confidence),
                "platform_type": "connected" if component_counts.get("bridge", 0) > 0 else "isolated",
                "component_counts": component_counts,
                "flare_confidence": flare_confidence,
                "flare_area_ratio_score": flare_ratio,
                "complexity_score": complexity,
                "priority_score": min(max(score, 0.0), 1.0),
                "platform_area": platform_area,
                "visible_component_count": component_count,
            }
        )
    return records


def write_mask(path, mask, target):
    transform = rasterio.Affine(*target["transform"].cpu().numpy()[:6])
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": target["crs"],
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(mask.astype(np.uint8), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(config, "test", shuffle=False, augment=False)
    model = SOFNet(config).to(device)
    load_checkpoint(config["evaluation"]["checkpoint"], model, device=device)
    model.eval()
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    targets_all = []
    records = []
    inference_seconds = 0.0
    inference_images = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="test"):
            images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
            device_targets = move_targets(targets, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                detections = model(images["optical"], images["sar"], images["prior"], images["quality"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start_time
            inference_images += len(targets)
            for batch_index, (detection, target, device_target) in enumerate(
                zip(detections, targets, device_targets)
            ):
                cpu_detection = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in detection.items()
                }
                predictions.append(cpu_detection)
                target["quality_score"] = float(images["quality"][batch_index].mean().detach().cpu())
                targets_all.append(target)
                image_id = target["image_id"]
                mask_path = output_dir / f"{image_id}_components.tif"
                write_mask(mask_path, cpu_detection["mask"].numpy(), target)
                platforms = platform_records(detection, device_target, config)
                records.append(
                    {
                        "image_id": image_id,
                        "component_mask": mask_path.name,
                        "platforms": platforms,
                    }
                )
    metrics = summarize_metrics(predictions, targets_all, config)
    metrics["inference_ms_per_image"] = 1000.0 * inference_seconds / max(inference_images, 1)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
