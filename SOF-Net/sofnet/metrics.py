import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou


def compute_ap(recalls, precisions):
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = precisions[recalls >= recall_level]
        values.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(values))


def average_precision(predictions, targets, iou_threshold, area_limit=None):
    records = []
    total_ground_truth = 0
    ground_truth = {}
    ignored_ground_truth = {}
    for image_index, target in enumerate(targets):
        boxes = target["boxes"]
        if area_limit is not None:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ignored_ground_truth[image_index] = boxes[area >= area_limit]
            boxes = boxes[area < area_limit]
        else:
            ignored_ground_truth[image_index] = boxes.new_zeros((0, 4))
        ground_truth[image_index] = boxes
        total_ground_truth += boxes.shape[0]
    for image_index, prediction in enumerate(predictions):
        for box, score in zip(prediction["boxes"], prediction["scores"]):
            records.append((float(score), image_index, box))
    records.sort(key=lambda item: item[0], reverse=True)
    matched = {index: set() for index in ground_truth}
    true_positive = []
    false_positive = []
    for _, image_index, box in records:
        candidates = ground_truth[image_index]
        if candidates.shape[0] == 0:
            ignored = ignored_ground_truth[image_index]
            if ignored.shape[0] > 0 and float(box_iou(box[None], ignored).max()) >= iou_threshold:
                continue
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue
        overlaps = box_iou(box[None], candidates)[0]
        value, index = overlaps.max(dim=0)
        index_value = int(index)
        if float(value) >= iou_threshold and index_value not in matched[image_index]:
            matched[image_index].add(index_value)
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            ignored = ignored_ground_truth[image_index]
            if ignored.shape[0] > 0 and float(box_iou(box[None], ignored).max()) >= iou_threshold:
                continue
            true_positive.append(0.0)
            false_positive.append(1.0)
    if total_ground_truth == 0 or not records:
        return 0.0
    true_positive = np.cumsum(np.asarray(true_positive))
    false_positive = np.cumsum(np.asarray(false_positive))
    recalls = true_positive / total_ground_truth
    precisions = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    return compute_ap(recalls, precisions)


def operating_point(predictions, targets, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for prediction, target in zip(predictions, targets):
        predicted_boxes = prediction["boxes"]
        target_boxes = target["boxes"]
        matched = set()
        order = torch.argsort(prediction["scores"], descending=True)
        for index in order:
            if target_boxes.shape[0] == 0:
                false_positive += 1
                continue
            overlaps = box_iou(predicted_boxes[index][None], target_boxes)[0]
            value, target_index = overlaps.max(dim=0)
            target_index = int(target_index)
            if float(value) >= iou_threshold and target_index not in matched:
                matched.add(target_index)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += target_boxes.shape[0] - len(matched)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, false_positive


def segmentation_confusion(predictions, targets, classes):
    confusion = torch.zeros((classes + 1, classes + 1), dtype=torch.float64)
    for prediction, target in zip(predictions, targets):
        predicted = prediction["mask"].reshape(-1).long().cpu()
        reference = target["mask"].reshape(-1).long().cpu()
        valid = (reference >= 0) & (reference <= classes)
        index = reference[valid] * (classes + 1) + predicted[valid].clamp(0, classes)
        confusion += torch.bincount(index, minlength=(classes + 1) ** 2).reshape(classes + 1, classes + 1)
    return confusion


def boundary_map(mask, classes):
    one_hot = F.one_hot(mask.long(), num_classes=classes + 1).permute(2, 0, 1).float()[None]
    maximum = F.max_pool2d(one_hot, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-one_hot, 3, stride=1, padding=1)
    return (maximum - minimum).amax(dim=1, keepdim=True) > 0


def boundary_f1(predictions, targets, classes, tolerance):
    matched_prediction = 0
    matched_reference = 0
    predicted_total = 0
    reference_total = 0
    kernel = tolerance * 2 + 1
    for prediction, target in zip(predictions, targets):
        predicted = boundary_map(prediction["mask"].cpu(), classes)
        reference = boundary_map(target["mask"].cpu(), classes)
        reference_dilated = F.max_pool2d(reference.float(), kernel, stride=1, padding=tolerance) > 0
        predicted_dilated = F.max_pool2d(predicted.float(), kernel, stride=1, padding=tolerance) > 0
        matched_prediction += int((predicted & reference_dilated).sum())
        matched_reference += int((reference & predicted_dilated).sum())
        predicted_total += int(predicted.sum())
        reference_total += int(reference.sum())
    precision = matched_prediction / max(predicted_total, 1)
    recall = matched_reference / max(reference_total, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def summarize_metrics(predictions, targets, config):
    ap50 = average_precision(predictions, targets, 0.5)
    thresholds = np.arange(0.5, 0.96, 0.05)
    ap5095 = float(np.mean([average_precision(predictions, targets, float(value)) for value in thresholds]))
    apsmall = average_precision(
        predictions, targets, 0.5, area_limit=config["evaluation"]["small_object_area"]
    )
    precision, recall, f1, false_positive = operating_point(predictions, targets)
    confusion = segmentation_confusion(
        predictions, targets, config["model"]["component_classes"]
    )
    intersection = confusion.diag()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    iou = intersection / union.clamp(min=1.0)
    foreground_iou = iou[1:]
    boundary = boundary_f1(
        predictions,
        targets,
        config["model"]["component_classes"],
        config["evaluation"]["boundary_tolerance"],
    )
    metrics = {
        "AP50": ap50,
        "AP50_95": ap5095,
        "APsmall": apsmall,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "FP_per_image": false_positive / max(len(targets), 1),
        "class_IoU": [float(value) for value in foreground_iou],
        "mIoU_FG": float(foreground_iou.mean()),
        "boundary_F1": boundary,
    }
    quality_groups = {
        "high": [
            index
            for index, target in enumerate(targets)
            if target.get("quality_score", 0.0) >= config["evaluation"]["high_quality_threshold"]
        ],
        "medium": [
            index
            for index, target in enumerate(targets)
            if config["evaluation"]["medium_quality_threshold"]
            <= target.get("quality_score", 0.0)
            < config["evaluation"]["high_quality_threshold"]
        ],
        "low": [
            index
            for index, target in enumerate(targets)
            if target.get("quality_score", 0.0) < config["evaluation"]["medium_quality_threshold"]
        ],
    }
    metrics["AP50_by_quality"] = {}
    for name, indices in quality_groups.items():
        metrics["AP50_by_quality"][name] = (
            average_precision([predictions[index] for index in indices], [targets[index] for index in indices], 0.5)
            if indices
            else None
        )
    return metrics
