import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, convnext_tiny, resnet18
from torchvision.ops import nms, roi_align


def replace_conv_input(conv, in_channels):
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        if conv.weight.shape[1] == 3:
            mean_weight = conv.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


class ConvNeXtPyramid(nn.Module):
    def __init__(self, in_channels, pretrained):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        network = convnext_tiny(weights=weights)
        network.features[0][0] = replace_conv_input(network.features[0][0], in_channels)
        self.features = network.features
        self.channels = [96, 192, 384, 768]

    def forward(self, x):
        outputs = []
        x = self.features[0](x)
        x = self.features[1](x)
        outputs.append(x)
        x = self.features[2](x)
        x = self.features[3](x)
        outputs.append(x)
        x = self.features[4](x)
        x = self.features[5](x)
        outputs.append(x)
        x = self.features[6](x)
        x = self.features[7](x)
        outputs.append(x)
        return outputs


class ResNetPyramid(nn.Module):
    def __init__(self, in_channels, pretrained):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        network.conv1 = replace_conv_input(network.conv1, in_channels)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layers = nn.ModuleList([network.layer1, network.layer2, network.layer3, network.layer4])
        self.channels = [64, 128, 256, 512]

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class FeaturePyramid(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(channel, output_channels, 1) for channel in input_channels])
        self.smooth = nn.ModuleList(
            [nn.Conv2d(output_channels, output_channels, 3, padding=1) for _ in input_channels]
        )

    def forward(self, features):
        lateral = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(lateral) - 2, -1, -1):
            lateral[index] = lateral[index] + F.interpolate(
                lateral[index + 1], size=lateral[index].shape[-2:], mode="nearest"
            )
        return [layer(feature) for layer, feature in zip(self.smooth, lateral)]


class PriorGuidedDeformableAttention(nn.Module):
    def __init__(self, channels, heads, points, radius, use_prior=True, sampling="deformable"):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("feature_channels must be divisible by attention_heads")
        self.channels = channels
        self.heads = heads
        self.points = points
        self.radius = radius
        self.use_prior = use_prior
        self.sampling = sampling
        if sampling not in ("deformable", "fixed_axial"):
            raise ValueError("Unknown sampling control")
        if sampling == "fixed_axial" and points != 4:
            raise ValueError("The fixed axial control requires exactly four points")
        self.head_channels = channels // heads
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.offset = nn.Conv2d(channels, heads * points * 2, 3, padding=1)
        self.output = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(heads))
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

        with torch.no_grad():
            angles = torch.arange(points, dtype=torch.float32) * (2 * math.pi / points)
            initial = torch.stack([angles.cos(), angles.sin()], dim=-1) * .25
            self.offset.bias.copy_(torch.atanh(initial).repeat(heads, 1).reshape(-1))

    def forward(self, optical, sar, prior):
        batch, channels, height, width = optical.shape
        query = self.query(optical).view(batch, self.heads, self.head_channels, height, width)
        key = self.key(sar).view(batch, self.heads, self.head_channels, height, width)
        value = self.value(sar).view(batch, self.heads, self.head_channels, height, width)
        offsets = self.offset(self.query(optical)).view(batch, self.heads, self.points, 2, height, width)
        offsets = torch.tanh(offsets) * self.radius
        if self.sampling == "fixed_axial":
            offsets = optical.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])
            offsets = offsets.view(1, 1, 4, 2, 1, 1).expand(batch, self.heads, 4, 2, height, width) * self.radius
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=optical.device, dtype=optical.dtype),
            torch.linspace(-1.0, 1.0, width, device=optical.device, dtype=optical.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([x, y], dim=-1).view(1, 1, 1, height, width, 2)
        offset_grid = offsets.permute(0, 1, 2, 4, 5, 3)
        offset_grid_x = offset_grid[..., 0] * (2.0 / max(width - 1, 1))
        offset_grid_y = offset_grid[..., 1] * (2.0 / max(height - 1, 1))
        grid = base_grid + torch.stack([offset_grid_x, offset_grid_y], dim=-1)
        prior_heads = prior[:, None].expand(batch, self.heads, 1, height, width)
        prior_heads = prior_heads.reshape(batch * self.heads, 1, height, width)
        grids = []
        logits = []
        for point_index in range(self.points):
            point_grid = grid[:, :, point_index].reshape(batch * self.heads, height, width, 2)
            grids.append(point_grid)
            sampled_key = self._sample_heads(key, point_grid)
            sampled_prior = F.grid_sample(
                prior_heads, point_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(batch, self.heads, height, width)
            point_logit = (query * sampled_key).sum(dim=2) / math.sqrt(self.head_channels)
            if self.use_prior:
                point_logit = point_logit + self.gamma.view(1, self.heads, 1, 1) * sampled_prior
            logits.append(point_logit)
        attention = torch.softmax(torch.stack(logits, dim=2), dim=2)
        attended = torch.zeros_like(query)
        for point_index, point_grid in enumerate(grids):
            sampled_value = self._sample_heads(value, point_grid)
            attended = attended + attention[:, :, point_index, None] * sampled_value
        attended = attended.reshape(batch, channels, height, width)
        return self.output(attended)

    def _sample_heads(self, feature, grid):
        batch, heads, channels, height, width = feature.shape
        flattened = feature.reshape(batch * heads, channels, height, width)
        sampled = F.grid_sample(flattened, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled.view(batch, heads, channels, height, width)


class QualityAwareGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, optical, attended, prior, quality):
        gate = self.gate(torch.cat([optical, attended, prior, quality], dim=1))
        return gate * attended + (1.0 - gate) * optical, gate


class DetectionHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        tower = []
        for _ in range(4):
            tower.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels), nn.ReLU()])
        self.tower = nn.Sequential(*tower)
        self.classification = nn.Conv2d(channels, 1, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)
        self.regression = nn.Conv2d(channels, 4, 3, padding=1)
        prior_probability = 0.01
        nn.init.constant_(self.classification.bias, -math.log((1.0 - prior_probability) / prior_probability))

    def forward(self, feature, stride):
        hidden = self.tower(feature)
        classification = self.classification(hidden)
        centerness = self.centerness(hidden)
        regression = F.softplus(self.regression(hidden)) * stride
        return classification, centerness, regression


class ScaleAwareMaskHead(nn.Module):
    def __init__(self, channels, classes, mask_size, strides):
        super().__init__()
        self.classes = classes
        self.mask_size = mask_size
        self.strides = strides
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, classes + 1, 1),
        )

    def forward(self, features, boxes):
        rois = []
        counts = []
        for batch_index, image_boxes in enumerate(boxes):
            counts.append(image_boxes.shape[0])
            if image_boxes.shape[0] > 0:
                indices = torch.full(
                    (image_boxes.shape[0], 1), batch_index, dtype=image_boxes.dtype, device=image_boxes.device
                )
                rois.append(torch.cat([indices, image_boxes], dim=1))
        if not rois:
            return features[0].new_zeros((0, self.classes + 1, self.mask_size, self.mask_size)), counts
        rois = torch.cat(rois, dim=0)
        scales = torch.sqrt(torch.clamp((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]), min=1.0))
        levels = torch.floor(4.0 + torch.log2(scales / 224.0 + 1e-6)).long().clamp(2, 5)
        pooled = features[0].new_zeros((rois.shape[0], features[0].shape[1], self.mask_size // 2, self.mask_size // 2))
        for level in range(2, 6):
            selected = torch.where(levels == level)[0]
            if selected.numel() == 0:
                continue
            pooled[selected] = roi_align(
                features[level - 2],
                rois[selected],
                output_size=self.mask_size // 2,
                spatial_scale=1.0 / self.strides[level - 2],
                sampling_ratio=2,
                aligned=True,
            )
        return self.encoder(pooled), counts


class SOFNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        data_config = config["data"]
        model_config = config["model"]
        channels = model_config["feature_channels"]
        pretrained = model_config["pretrained_backbones"]
        self.strides = [4, 8, 16, 32]
        self.optical_backbone = ConvNeXtPyramid(data_config["optical_channels"], pretrained)
        self.sar_backbone = ResNetPyramid(data_config["sar_channels"], pretrained)
        self.optical_fpn = FeaturePyramid(self.optical_backbone.channels, channels)
        self.sar_fpn = FeaturePyramid(self.sar_backbone.channels, channels)
        self.attention = nn.ModuleList(
            [
                PriorGuidedDeformableAttention(
                    channels,
                    model_config["attention_heads"],
                    model_config["attention_points"],
                    radius,
                    use_prior=model_config.get("attention_prior", True),
                    sampling=model_config.get("sampling", "deformable"),
                )
                for radius in model_config["displacement_radii"]
            ]
        )
        self.gates = nn.ModuleList([QualityAwareGate(channels) for _ in self.strides])
        self.prior_response = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in self.strides])
        self.detection_head = DetectionHead(channels)
        self.mask_head = ScaleAwareMaskHead(
            channels, model_config["component_classes"], model_config["mask_size"], self.strides
        )
        self.detection_threshold = model_config["detection_threshold"]
        self.nms_iou = model_config["nms_iou"]
        self.max_detections = model_config["max_detections"]
        self.score_floor = model_config.get("candidate_score_floor", 0.001)
        self.score_mode = model_config.get("score_mode", "product")
        self.gate_mode = model_config.get("gate_mode", "full")
        self.neutral_quality = model_config.get("neutral_quality", 0.75)
        if self.gate_mode not in ("full", "neutral_quality", "no_prior", "feature_only", "mean"):
            raise ValueError("Unknown gate control")

    def forward(self, optical, sar, prior, quality, targets=None, return_aux=False, reference_boxes=None):
        optical_features = self.optical_fpn(self.optical_backbone(optical))
        sar_features = self.sar_fpn(self.sar_backbone(sar))
        fused_features = []
        gates = []
        prior_levels = []
        quality_levels = []
        for optical_level, sar_level, attention, gate in zip(
            optical_features, sar_features, self.attention, self.gates
        ):

            sar_level = F.interpolate(sar_level, size=optical_level.shape[-2:], mode="bilinear", align_corners=False)
            level_prior = F.interpolate(prior, size=optical_level.shape[-2:], mode="bilinear", align_corners=False)
            level_quality = F.interpolate(quality, size=optical_level.shape[-2:], mode="bilinear", align_corners=False)
            attended = attention(optical_level, sar_level, level_prior)
            gate_prior, gate_quality = level_prior, level_quality
            if self.gate_mode in ("no_prior", "feature_only"):
                gate_prior = torch.zeros_like(level_prior)
            if self.gate_mode == "neutral_quality":
                gate_quality = torch.full_like(level_quality, self.neutral_quality)
            if self.gate_mode == "feature_only":
                gate_quality = torch.zeros_like(level_quality)
            if self.gate_mode == "mean":
                gate_map = torch.full_like(level_prior, 0.5)
                fused = 0.5 * (optical_level + attended)
            else:
                fused, gate_map = gate(optical_level, attended, gate_prior, gate_quality)
            fused_features.append(fused)
            gates.append(gate_map)
            prior_levels.append(level_prior)
            quality_levels.append(level_quality)
        classifications = []
        centerness = []
        regressions = []
        points = []
        responses = []
        for feature, stride, response_layer in zip(fused_features, self.strides, self.prior_response):
            classification, center, regression = self.detection_head(feature, stride)
            classifications.append(classification)
            centerness.append(center)
            regressions.append(regression)
            points.append(self._points(feature.shape[-2], feature.shape[-1], stride, feature.device, feature.dtype))
            responses.append(response_layer(feature))
        output = {
            "features": fused_features,
            "gates": gates,
            "prior_levels": prior_levels,
            "quality_levels": quality_levels,
            "classifications": classifications,
            "centerness": centerness,
            "regressions": regressions,
            "points": points,
            "responses": responses,
            "strides": self.strides,
            "image_size": optical.shape[-2:],
        }
        if targets is not None:
            boxes = [target["boxes"] for target in targets]
            output["mask_logits"], output["mask_counts"] = self.mask_head(fused_features, boxes)
            return output
        detections = self._decode(output)
        mask_logits, mask_counts = self.mask_head(fused_features, [item["boxes"] for item in detections])
        start = 0
        for detection, count in zip(detections, mask_counts):
            current = mask_logits[start : start + count]
            detection["roi_mask_logits"] = current
            detection["mask"] = self._paste_masks(
                current, detection["boxes"], detection["scores"], optical.shape[-2:]
            )
            start += count
        if reference_boxes is not None:
            ref_logits, ref_counts = self.mask_head(fused_features, reference_boxes)
            cursor = 0
            for det, boxes, count in zip(detections, reference_boxes, ref_counts):
                current = ref_logits[cursor:cursor + count]
                det["reference_roi_logits"] = current
                det["reference_mask"] = self._paste_masks(current, boxes, boxes.new_ones(len(boxes)), optical.shape[-2:])
                cursor += count
        return (detections, output) if return_aux else detections

    def _points(self, height, width, stride, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([(x.reshape(-1) + 0.5) * stride, (y.reshape(-1) + 0.5) * stride], dim=1)

    def _decode(self, output):
        batch = output["classifications"][0].shape[0]
        height, width = output["image_size"]
        detections = []
        for batch_index in range(batch):
            boxes_all = []
            scores_all = []
            for classification, center, regression, points in zip(
                output["classifications"], output["centerness"], output["regressions"], output["points"]
            ):
                score = classification[batch_index].sigmoid().reshape(-1) * center[batch_index].sigmoid().reshape(-1)
                if self.score_mode == "legacy_sqrt":
                    score = torch.sqrt(score)
                distances = regression[batch_index].permute(1, 2, 0).reshape(-1, 4)
                keep = torch.where(score >= self.score_floor)[0]
                if keep.numel() == 0:
                    continue
                if keep.numel() > 1000:
                    keep = keep[score[keep].topk(1000).indices]
                selected_points = points[keep]
                selected_distances = distances[keep]
                boxes = torch.stack(
                    [
                        selected_points[:, 0] - selected_distances[:, 0],
                        selected_points[:, 1] - selected_distances[:, 1],
                        selected_points[:, 0] + selected_distances[:, 2],
                        selected_points[:, 1] + selected_distances[:, 3],
                    ],
                    dim=1,
                )
                boxes[:, 0::2].clamp_(0, width)
                boxes[:, 1::2].clamp_(0, height)
                boxes_all.append(boxes)
                scores_all.append(score[keep])
            if not boxes_all:
                device = output["classifications"][0].device
                detections.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    }
                )
                continue
            boxes = torch.cat(boxes_all)
            scores = torch.cat(scores_all)
            keep = nms(boxes, scores, self.nms_iou)[: self.max_detections]
            detections.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": torch.ones((keep.numel(),), dtype=torch.long, device=boxes.device),
                }
            )
        return detections

    def _paste_masks(self, logits, boxes, scores, image_size):
        height, width = image_size
        canvas = logits.new_zeros((height, width), dtype=torch.long)
        confidence = logits.new_zeros((height, width))
        if logits.shape[0] == 0:
            return canvas
        labels = logits.argmax(dim=1)
        for index in torch.argsort(scores):
            box = boxes[index].round().long()
            x1 = int(box[0].clamp(0, width - 1))
            y1 = int(box[1].clamp(0, height - 1))
            x2 = int(box[2].clamp(x1 + 1, width))
            y2 = int(box[3].clamp(y1 + 1, height))
            resized = F.interpolate(
                labels[index][None, None].float(), size=(y2 - y1, x2 - x1), mode="nearest"
            )[0, 0].long()
            region_score = confidence[y1:y2, x1:x2]
            update = (resized > 0) & (scores[index] >= region_score)
            canvas_region = canvas[y1:y2, x1:x2]
            canvas_region[update] = resized[update]
            region_score[update] = scores[index]
        return canvas
