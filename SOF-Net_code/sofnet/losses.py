import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou_loss


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulating = torch.where(targets > 0, 1.0 - probability, probability).pow(gamma)
    weighting = torch.where(targets > 0, alpha, 1.0 - alpha)
    loss = weighting * modulating * cross_entropy
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    return loss.sum()


def boxes_from_points(points, distances):
    return torch.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        dim=1,
    )


def assign_points(points, stride, boxes, lower, upper):
    point_count = points.shape[0]
    if boxes.shape[0] == 0:
        return (
            points.new_zeros((point_count,)),
            points.new_zeros((point_count,)),
            points.new_zeros((point_count, 4)),
        )
    x = points[:, 0, None]
    y = points[:, 1, None]
    left = x - boxes[None, :, 0]
    top = y - boxes[None, :, 1]
    right = boxes[None, :, 2] - x
    bottom = boxes[None, :, 3] - y
    distances = torch.stack([left, top, right, bottom], dim=2)
    inside = distances.min(dim=2).values > 0
    maximum = distances.max(dim=2).values
    scale_valid = (maximum >= lower) & (maximum < upper)
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    radius = stride * 1.5
    center_valid = (torch.abs(x - center_x) <= radius) & (torch.abs(y - center_y) <= radius)
    valid = inside & scale_valid & center_valid
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area = area[None].repeat(point_count, 1)
    area[~valid] = torch.inf
    selected_area, selected_index = area.min(dim=1)
    positive = torch.isfinite(selected_area)
    target_distances = distances[torch.arange(point_count, device=points.device), selected_index]
    target_distances[~positive] = 0
    left_right = target_distances[:, [0, 2]]
    top_bottom = target_distances[:, [1, 3]]
    centerness = torch.sqrt(
        torch.clamp(
            left_right.min(dim=1).values / left_right.max(dim=1).values.clamp(min=1e-6)
            * top_bottom.min(dim=1).values
            / top_bottom.max(dim=1).values.clamp(min=1e-6),
            min=0.0,
        )
    )
    centerness[~positive] = 0
    return positive.float(), centerness, target_distances


def crop_mask_targets(targets, mask_size):
    crops = []
    for target in targets:
        mask = target["mask"].clone()
        if "valid" in target:
            mask[~target["valid"].bool()] = 255
        height, width = mask.shape
        for box in target["boxes"]:
            coordinates = box.round().long()
            x1 = int(coordinates[0].clamp(0, width - 1))
            y1 = int(coordinates[1].clamp(0, height - 1))
            x2 = int(coordinates[2].clamp(x1 + 1, width))
            y2 = int(coordinates[3].clamp(y1 + 1, height))
            crop = mask[y1:y2, x1:x2][None, None].float()
            crops.append(F.interpolate(crop, size=(mask_size, mask_size), mode="nearest")[0, 0].long())
    if not crops:
        return targets[0]["mask"].new_zeros((0, mask_size, mask_size))
    return torch.stack(crops)


def component_loss(logits, targets, classes, class_balanced=True):
    if logits.shape[0] == 0:
        return logits.sum(), logits.sum()
    valid = (targets >= 0) & (targets <= classes)
    if not valid.any():
        return logits.sum() * 0, logits.sum() * 0
    safe_targets = targets.masked_fill(~valid, 0)
    counts = torch.bincount(targets[valid], minlength=classes + 1).float()
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.mean()
    if not class_balanced:
        weights = torch.ones_like(weights)
    cross_entropy = F.cross_entropy(logits, safe_targets, weight=weights, reduction="none")
    probability = torch.softmax(logits, dim=1)
    selected_probability = probability.gather(1, safe_targets[:, None]).squeeze(1)
    focal = ((((1.0 - selected_probability) ** 2) * cross_entropy) * valid).sum() / valid.sum()
    one_hot = F.one_hot(safe_targets, num_classes=classes + 1).permute(0, 3, 1, 2).float()
    v = valid[:, None].to(probability.dtype)
    intersection = (v * probability[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3))
    denominator = (v * probability[:, 1:]).sum(dim=(0, 2, 3)) + (v * one_hot[:, 1:]).sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    target_max = F.max_pool2d(one_hot, 3, stride=1, padding=1)
    target_min = -F.max_pool2d(-one_hot, 3, stride=1, padding=1)
    target_boundary = (target_max - target_min).amax(dim=1, keepdim=True).clamp(0.0, 1.0)
    probability_max = F.max_pool2d(probability, 3, stride=1, padding=1)
    probability_min = -F.max_pool2d(-probability, 3, stride=1, padding=1)
    probability_boundary = (probability_max - probability_min).amax(dim=1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)

    boundary_valid = F.max_pool2d((~valid)[:, None].float(), 5, 1, 2) == 0
    boundary_values = F.binary_cross_entropy(probability_boundary, target_boundary, reduction="none")
    boundary = (boundary_values * boundary_valid).sum() / boundary_valid.sum().clamp(min=1)
    return focal + dice, boundary


def prior_consistency_loss(responses, priors, targets, mode="original", threshold=0.35):

    if mode not in ("original", "revised_lower_bound", "none"):
        raise ValueError("Unknown prior loss mode")
    total = responses[0].sum() * 0
    if mode == "none":
        return total
    for response, prior in zip(responses, priors):
        if mode == "original":
            weights = prior
            penalty = torch.abs(response.sigmoid() - prior)
        else:

            positive = torch.stack([(t['mask'] > 0) & (t['mask'] <= 5) &
                                    t.get('valid', torch.ones_like(t['mask'], dtype=torch.bool)).bool()
                                    for t in targets])[:, None].float()
            positive = F.interpolate(positive, size=prior.shape[-2:], mode='nearest') > 0
            weights = prior * positive * (prior >= threshold)
            penalty = F.relu(prior - response.sigmoid())
        total = total + (weights * penalty).sum() / weights.sum().clamp(min=1e-6)
    return total / len(responses)


class SOFLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weights = config["loss"]
        self.classes = config["model"]["component_classes"]
        self.mask_size = config["model"]["mask_size"]
        self.prior_mode = config["loss"].get("prior_mode", "original")
        self.prior_threshold = config["loss"].get("prior_positive_threshold", 0.35)
        self.class_balanced = config["loss"].get("class_balanced", True)
        self.ranges = [(0, 64), (64, 128), (128, 256), (256, float("inf"))]

    def forward(self, outputs, targets):
        classification_loss = outputs["classifications"][0].new_tensor(0.0)
        centerness_loss = classification_loss.clone()
        box_loss = classification_loss.clone()
        positive_count = 0
        for batch_index, target in enumerate(targets):
            for level, (classification, center, regression, points, stride, scale_range) in enumerate(
                zip(
                    outputs["classifications"],
                    outputs["centerness"],
                    outputs["regressions"],
                    outputs["points"],
                    outputs["strides"],
                    self.ranges,
                )
            ):
                positive, center_target, distance_target = assign_points(
                    points, stride, target["boxes"], scale_range[0], scale_range[1]
                )
                class_logits = classification[batch_index].reshape(-1)
                center_logits = center[batch_index].reshape(-1)
                distance_prediction = regression[batch_index].permute(1, 2, 0).reshape(-1, 4)
                classification_loss = classification_loss + sigmoid_focal_loss(class_logits, positive)
                selected = positive > 0
                current_positive = int(selected.sum())
                positive_count += current_positive
                if current_positive > 0:
                    centerness_loss = centerness_loss + F.binary_cross_entropy_with_logits(
                        center_logits[selected], center_target[selected], reduction="sum"
                    )
                    predicted_boxes = boxes_from_points(points[selected], distance_prediction[selected])
                    target_boxes = boxes_from_points(points[selected], distance_target[selected])
                    box_loss = box_loss + generalized_box_iou_loss(
                        predicted_boxes, target_boxes, reduction="sum"
                    )
        normalizer = max(positive_count, 1)
        classification_loss = classification_loss / normalizer
        centerness_loss = centerness_loss / normalizer
        box_loss = box_loss / normalizer
        mask_targets = crop_mask_targets(targets, self.mask_size)
        segmentation_loss, boundary_loss = component_loss(
            outputs["mask_logits"], mask_targets, self.classes, self.class_balanced
        )
        prior_loss = prior_consistency_loss(outputs["responses"], outputs["prior_levels"], targets,
                                            self.prior_mode, self.prior_threshold)
        total = (
            self.weights["cls"] * classification_loss
            + self.weights["centerness"] * centerness_loss
            + self.weights["box"] * box_loss
            + self.weights["segmentation"] * segmentation_loss
            + self.weights["boundary"] * boundary_loss
            + self.weights["prior"] * prior_loss
        )
        return {
            "total": total,
            "classification": classification_loss,
            "centerness": centerness_loss,
            "box": box_loss,
            "segmentation": segmentation_loss,
            "boundary": boundary_loss,
            "prior": prior_loss,
        }
