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
        if matched[image_index]:
            overlaps[list(matched[image_index])] = -1
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
            if matched:
                overlaps[list(matched)] = -1
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
        if 'valid' in target:
            valid &= target['valid'].reshape(-1).cpu().bool()
        if ((predicted[valid] < 0) | (predicted[valid] > classes)).any():
            raise ValueError('Prediction contains an unknown class')
        index = reference[valid] * (classes + 1) + predicted[valid].clamp(0, classes)
        confusion += torch.bincount(index, minlength=(classes + 1) ** 2).reshape(classes + 1, classes + 1)
    return confusion


def boundary_map(mask, classes):
    one_hot = F.one_hot(mask.long(), num_classes=classes + 1).permute(2, 0, 1).float()[None]
    maximum = F.max_pool2d(one_hot, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-one_hot, 3, stride=1, padding=1)
    return (maximum - minimum).amax(dim=1, keepdim=True) > 0


def boundary_counts(predictions, targets, classes, tolerance):
    matched_prediction = 0
    matched_reference = 0
    predicted_total = 0
    reference_total = 0
    kernel = tolerance * 2 + 1
    for prediction, target in zip(predictions, targets):
        ref = target['mask'].cpu()
        valid = (ref >= 0) & (ref <= classes)
        if 'valid' in target:
            valid &= target['valid'].cpu().bool()
        safe = F.max_pool2d((~valid)[None,None].float(), kernel, 1, tolerance) == 0
        predicted = boundary_map(prediction["mask"].cpu().masked_fill(~valid,0), classes) & safe
        reference = boundary_map(ref.masked_fill(~valid,0), classes) & safe
        reference_dilated = F.max_pool2d(reference.float(), kernel, stride=1, padding=tolerance) > 0
        predicted_dilated = F.max_pool2d(predicted.float(), kernel, stride=1, padding=tolerance) > 0
        matched_prediction += int((predicted & reference_dilated).sum())
        matched_reference += int((reference & predicted_dilated).sum())
        predicted_total += int(predicted.sum())
        reference_total += int(reference.sum())
    return np.array([matched_prediction, matched_reference, predicted_total, reference_total], dtype=np.int64)


def boundary_f1(predictions, targets, classes, tolerance):
    mp,mr,npred,nref = boundary_counts(predictions,targets,classes,tolerance)
    precision = mp / max(npred,1)
    recall = mr / max(nref,1)
    return float(2*precision*recall/max(precision+recall,1e-12))


def segmentation_summary(predictions, targets, config):
    confusion = segmentation_confusion(predictions, targets, config['model']['component_classes'])
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = torch.where(union > 0, intersection / union.clamp(min=1), torch.nan)
    f1den = confusion.sum(0) + confusion.sum(1)
    f1 = torch.where(f1den > 0, 2*intersection / f1den.clamp(min=1), torch.nan)
    clean = lambda a: [float(v) if torch.isfinite(v) else None for v in a]
    scalar = lambda a: float(a) if torch.isfinite(a) else None
    return {'confusion_matrix': confusion.long().tolist(), 'class_IoU_all': clean(iou),
            'class_IoU': clean(iou[1:]), 'class_F1_all': clean(f1),
            'mIoU_FG': scalar(iou[1:].mean()), 'mIoU_all': scalar(iou.mean()),
            'boundary_F1': boundary_f1(predictions, targets, config['model']['component_classes'], config['evaluation']['boundary_tolerance']),
            'valid_pixels': int(confusion.sum()), 'absent_class_policy': 'undefined; fixed-class mean is null if a class has zero union'}


def detection_summary(predictions, targets, config, patch_count=None):
    if len(predictions) != len(targets):
        raise ValueError('Prediction/reference length mismatch')
    threshold = config['model']['detection_threshold']
    operating = [{'boxes': p['boxes'][p['scores'] >= threshold], 'scores':p['scores'][p['scores'] >= threshold]} for p in predictions]
    precision, recall, f1, fp = operating_point(operating, targets)
    nref = sum(len(t['boxes']) for t in targets)
    tp = sum(len(p['boxes']) for p in operating) - fp
    n = len(targets) if patch_count is None else patch_count
    return {'AP50':average_precision(predictions,targets,.5),
            'AP50_95': float(np.mean([average_precision(predictions,targets,float(t)) for t in np.arange(.5,.96,.05)])),
            'APsmall':average_precision(predictions,targets,.5,config['evaluation']['small_object_area']),
            'precision':precision,'recall':recall,'F1':f1,'TP':tp,'FP':fp,'FN':nref-tp,
            'reference_count':nref,'patch_count':n,'FP_per_image':fp/n if n else None,
            'operating_score_threshold':threshold, 'AP_candidate_floor':config['model'].get('candidate_score_floor',.001)}


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
    metrics.update(detection_summary(predictions, targets, config))
    metrics.update(segmentation_summary(predictions, targets, config))
    quality_groups = {
        "high": [
            index
            for index, target in enumerate(targets)
            if target.get('quality_score') is not None and target['quality_score'] >= config["evaluation"]["high_quality_threshold"]
        ],
        "medium": [
            index
            for index, target in enumerate(targets)
            if target.get('quality_score') is not None and config["evaluation"]["medium_quality_threshold"]
            <= target.get("quality_score", 0.0)
            < config["evaluation"]["high_quality_threshold"]
        ],
        "low": [
            index
            for index, target in enumerate(targets)
            if target.get('quality_score') is not None and target['quality_score'] < config["evaluation"]["medium_quality_threshold"]
        ],
    }
    metrics["AP50_by_quality"] = {}
    metrics['quality_groups'] = {}
    for name, indices in quality_groups.items():
        metrics["AP50_by_quality"][name] = (
            average_precision([predictions[index] for index in indices], [targets[index] for index in indices], 0.5)
            if indices
            else None
        )
        pp, tt = [predictions[i] for i in indices], [targets[i] for i in indices]
        metrics['quality_groups'][name] = {'patch_count':len(indices),'patch_ids':[t['image_id'] for t in tt],
                                          'segmentation':segmentation_summary(pp,tt,config) if tt else None,
                                          'mean_SAR_gate':float(np.mean([t['gate_mean'] for t in tt])) if tt and all('gate_mean' in t for t in tt) else None}
    return metrics
