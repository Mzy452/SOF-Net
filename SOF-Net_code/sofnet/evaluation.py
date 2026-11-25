import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms
from .metrics import average_precision, detection_summary


def deduplicate_scenes(predictions, targets, nms_iou=.5):
    if len(predictions) != len(targets):
        raise ValueError('Mismatched predictions and targets')
    groups, ownership = {}, {}
    for pi, (p, t) in enumerate(zip(predictions, targets)):
        sid = t['scene_id']
        g = groups.setdefault(sid, {'boxes':[], 'scores':[], 'refs':{}, 'origins':[]})
        x, y = t['window_xy']
        offset = torch.tensor([x,y,x,y], dtype=torch.float32)
        g['boxes'].append(p['boxes'].cpu() + offset)
        g['scores'].append(p['scores'].cpu())
        g['origins'].extend((pi,j) for j in range(len(p['boxes'])))
        for identity, box in zip(t['platform_ids'],t['reference_boxes_scene']):
            if identity in ownership and ownership[identity] != sid:
                raise ValueError(f'{identity} appears in multiple scenes: specify one canonical evaluation scene per physical platform')
            ownership[identity] = sid
            box = torch.as_tensor(box).cpu().float()
            if identity in g['refs'] and not torch.allclose(g['refs'][identity],box,atol=.01):
                raise ValueError(f'Conflicting full reference box for {identity}')
            g['refs'][identity] = box
    pp, tt = [], []
    for sid, g in sorted(groups.items()):
        boxes = torch.cat(g['boxes'])
        scores = torch.cat(g['scores'])
        keep = nms(boxes,scores,nms_iou)
        pp.append({'boxes':boxes[keep], 'scores':scores[keep], 'scene_id':sid,
                   'origins':[g['origins'][i] for i in keep.tolist()]})
        tt.append({'boxes':torch.stack(list(g['refs'].values())) if g['refs'] else torch.empty((0,4)),
                   'platform_ids':list(g['refs']), 'scene_id':sid})
    return pp, tt


def paste_roi_masks(logits, boxes, scores, shape, score_threshold=.3):

    canvas = torch.zeros(tuple(shape),dtype=torch.uint8)
    claimed = torch.zeros(tuple(shape),dtype=torch.bool)
    h,w = shape
    for j in torch.argsort(scores,descending=True).tolist():
        if float(scores[j]) < score_threshold:
            continue
        b = boxes[j].round().long().tolist()
        x1,y1,x2,y2 = b
        if x2 <= x1 or y2 <= y1:
            continue
        label = logits[j].argmax(0)[None,None].float()
        label = F.interpolate(label,size=(y2-y1,x2-x1),mode='nearest')[0,0].byte()
        xa,ya,xb,yb = max(x1,0),max(y1,0),min(x2,w),min(y2,h)
        if xa >= xb or ya >= yb:
            continue
        crop = label[ya-y1:yb-y1,xa-x1:xb-x1]
        update = (crop>0) & ~claimed[ya:yb,xa:xb]
        canvas[ya:yb,xa:xb][update] = crop[update]
        claimed[ya:yb,xa:xb] |= update
    return canvas


def shift_sar_inputs(sar, prior, dx):

    if abs(dx) >= sar.shape[-1]:
        raise ValueError('Shift exceeds patch width')
    def shift(a):
        out = torch.zeros_like(a)
        if dx == 0:
            return a
        if dx > 0:
            out[...,dx:] = a[...,:-dx]
        else:
            out[...,:dx] = a[...,-dx:]
        return out
    return shift(sar),shift(prior)


def paired_scene_bootstrap(pred_a, pred_b, targets, replicates=2000, seed=20261119):
    if not targets or len(pred_a)!=len(targets) or len(pred_b)!=len(targets):
        raise ValueError('Paired predictions and references must have the same nonzero scene count')
    for pa,pb,t in zip(pred_a,pred_b,targets):
        if pa['scene_id'] != t['scene_id'] or pb['scene_id'] != t['scene_id']:
            raise ValueError('Scene ordering differs')
    rng = np.random.default_rng(seed)
    differences=[]
    for _ in range(replicates):
        idx=rng.integers(0,len(targets),len(targets))
        ta=[targets[i] for i in idx]
        differences.append(100*(average_precision([pred_a[i] for i in idx],ta,.5)-average_precision([pred_b[i] for i in idx],ta,.5)))
    return {'replicates':replicates,'seed':seed,'scene_count':len(targets),
            'difference_percentage_points':100*(average_precision(pred_a,targets,.5)-average_precision(pred_b,targets,.5)),
            'percentile_95_interval':np.percentile(differences,[2.5,97.5]).tolist()}
