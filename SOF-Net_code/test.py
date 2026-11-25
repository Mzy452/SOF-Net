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
from sofnet.metrics import detection_summary
from sofnet.evaluation import deduplicate_scenes, paste_roi_masks, shift_sar_inputs
from sofnet.gates import aggregate_gates, export_gate_patch, checkpoint_sha256
from sofnet.quality import patch_quality
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
        out_h=max(int(round(float(box[3]-box[1]))),1)
        out_w=max(int(round(float(box[2]-box[0]))),1)
        mask=torch.nn.functional.interpolate(torch.from_numpy(mask)[None,None].float(),size=(out_h,out_w),mode='nearest')[0,0].numpy().astype('uint8')
        platform_pixels = max(int((mask > 0).sum()), 1)
        flare_ratio = min(
            float((mask == 4).sum()) / platform_pixels / priority["flare_area_saturation"], 1.0
        )
        box_width = max(float(box[2] - box[0]), 0.0)
        box_height = max(float(box[3] - box[1]), 0.0)

        cell_area = box_width * box_height * pixel_area / max(mask.size, 1)
        platform_area = int((mask == 1).sum()) * cell_area
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
                'attribute_status': 'raw mask-derived; cross-box deck-pair verification required',
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
    config['model']['pretrained_backbones'] = False
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(config, "test", shuffle=False, augment=False)
    model = SOFNet(config).to(device)
    load_checkpoint(config["evaluation"]["checkpoint"], model, device=device)
    model.eval()
    checkpoint_hash = checkpoint_sha256(config['evaluation']['checkpoint'])
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
            dx = int(config['evaluation'].get('shift_pixels',0))
            margin = int(config['evaluation'].get('shift_common_margin',0))
            if abs(dx)>margin:
                raise ValueError('S2 shifts require a shared margin >= largest absolute shift')
            images['sar'], images['prior'] = shift_sar_inputs(images['sar'], images['prior'],dx)
            if margin:
                for t,dt in zip(targets,device_targets):
                    for item in (t,dt):
                        item['valid'][...,:margin] = False
                        item['valid'][...,-margin:] = False
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                detections,aux = model(images["optical"], images["sar"], images["prior"], images["quality"],
                                       return_aux=True, reference_boxes=[t['boxes'] for t in device_targets])
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
                predicted_mask=paste_roi_masks(cpu_detection['roi_mask_logits'],cpu_detection['boxes'],cpu_detection['scores'],
                                              target['mask'].shape,config['model']['detection_threshold'])
                protocol=config['evaluation'].get('protocol','reference_boxes')
                if protocol not in ('reference_boxes','predicted_patch'):
                    raise ValueError('Use evaluate_scene_grid.py for full end-to-end mapping')
                cpu_detection['mask']=cpu_detection['reference_mask'] if protocol=='reference_boxes' else predicted_mask
                predictions.append(cpu_detection)
                target["quality_score"] = patch_quality(images['quality'][batch_index,0].cpu().numpy(),
                                                        images['quality_valid'][batch_index,0].cpu().numpy())
                targets_all.append(target)
                if margin:

                    h,w=target['mask'].shape
                    def inside(b):
                        cx=(b[:,0]+b[:,2])/2;cy=(b[:,1]+b[:,3])/2
                        return (cx>=margin)&(cx<w-margin)&(cy>=0)&(cy<h)
                    kp=inside(cpu_detection['boxes'])
                    for key in ('boxes','scores','labels','roi_mask_logits'):
                        cpu_detection[key]=cpu_detection[key][kp]
                    kr=inside(target['boxes'])
                    cpu_detection['reference_roi_logits']=cpu_detection['reference_roi_logits'][kr]
                    target['boxes']=target['boxes'][kr]
                    target['reference_boxes_scene']=target['reference_boxes_scene'][kr]
                    target['platform_ids']=[v for v,k in zip(target['platform_ids'],kr) if k]
                image_id = target["image_id"]
                if Path(image_id).name != image_id:
                    raise ValueError('Patch IDs must be filename-safe')
                levels=[g[batch_index].detach().float().cpu() for g in aux['gates']]
                target['gate_mean']=float(aggregate_gates([g[None] for g in levels],target['mask'].shape).mean())
                if config['evaluation'].get('export_gates',True):
                    export_gate_patch(output_dir/'gates',image_id,levels,images['optical'][batch_index].cpu(),
                                      images['prior'][batch_index].cpu(),images['quality'][batch_index].cpu(),
                                      target['mask'],predicted_mask,target['valid'],checkpoint_hash)
                archive=output_dir/(image_id+'_raw.npz')
                np.savez_compressed(archive,boxes=cpu_detection['boxes'].numpy(),scores=cpu_detection['scores'].numpy(),
                                    roi_logits=cpu_detection['roi_mask_logits'].numpy(),reference_roi_logits=cpu_detection['reference_roi_logits'].numpy(),
                                    reference_boxes=target['boxes'].numpy())
                mask_path = output_dir / f"{image_id}_components.tif"
                write_mask(mask_path, cpu_detection["mask"].numpy(), target)
                keep=detection['scores']>=config['model']['detection_threshold']
                retained={k:(v[keep] if k in ('boxes','scores','labels','roi_mask_logits') else v) for k,v in detection.items()}
                platforms = platform_records(retained, device_target, config)
                records.append(
                    {
                        "image_id": image_id,
                        "component_mask": mask_path.name,
                        "platforms": platforms,
                        'scene_id':target['scene_id'], 'window_xy':target['window_xy'],
                        'scene_shape':target['scene_shape'], 'scene_transform':target['scene_transform'],
                        'platform_ids':target['platform_ids'], 'reference_boxes_scene':target['reference_boxes_scene'].tolist(),
                        'crs':target['crs'], 'raw_archive':archive.name, 'quality_score':target['quality_score'],
                    }
                )
    metrics = summarize_metrics(predictions, targets_all, config)
    scene_predictions,scene_targets=deduplicate_scenes(predictions,targets_all,config['model']['nms_iou'])
    metrics['patch_detection_diagnostic']={k:metrics[k] for k in ('AP50','AP50_95','APsmall','precision','recall','F1','TP','FP','FN','FP_per_image','reference_count')}
    metrics.update(detection_summary(scene_predictions,scene_targets,config,patch_count=len(targets_all)))
    metrics['detection_unit']='scene-level NMS and unique reference IDs; FP divided by evaluated patch count'
    metrics['segmentation_protocol']=config['evaluation'].get('protocol','reference_boxes')
    metrics['segmentation_scope']='patch-level; overlapping patches are not a common-grid mosaic'
    metrics['checkpoint_sha256']=checkpoint_hash
    metrics['quality_detection_unit']='patch instances, not unique physical platforms across quality groups'
    metrics['shift_domain']='box centres in shared horizontal interior; segmentation excludes shared edge margin' if margin else 'unshifted patch domain'
    metrics["inference_ms_per_image"] = 1000.0 * inference_seconds / max(inference_images, 1)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    scene_records=[]
    for p,t in zip(scene_predictions,scene_targets):
        scene_records.append({'scene_id':t['scene_id'],'boxes':p['boxes'].tolist(),'scores':p['scores'].tolist(),
                              'reference_boxes':t['boxes'].tolist(),'platform_ids':t['platform_ids']})
    with (output_dir/'scene_detections.json').open('w',encoding='utf-8') as stream:
        json.dump(scene_records,stream,indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
