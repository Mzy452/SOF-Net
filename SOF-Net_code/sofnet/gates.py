from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F


def checkpoint_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def aggregate_gates(levels, shape):
    return torch.stack([F.interpolate(g,size=shape,mode='bilinear',align_corners=False) for g in levels]).mean(0)


def export_gate_patch(directory, patch_id, levels, optical, prior, quality, reference, prediction, valid, checkpoint_hash):
    path=Path(directory)
    path.mkdir(parents=True,exist_ok=True)
    pooled=aggregate_gates([g[None] for g in levels],quality.shape[-2:])[0,0]
    arrays={f'G_P{i+2}':g.detach().float().cpu().numpy() for i,g in enumerate(levels)}
    arrays.update(G_mean=pooled.detach().cpu().numpy(), optical=optical.cpu().numpy(),
                  prior=prior.cpu().numpy(), quality=quality.cpu().numpy(),
                  reference=reference.cpu().numpy(), prediction=prediction.cpu().numpy(), valid=valid.cpu().numpy())
    np.savez_compressed(path/(patch_id+'.npz'),**arrays)
    metadata={'patch_id':patch_id,'checkpoint_sha256':checkpoint_hash,'gate_direction':'larger = more SAR weight',
              'levels':['P2','P3','P4','P5'],'aggregation':'equal level weights after bilinear upsampling; align_corners=False',
              'color_limits':[0,1],'source':'actual forward-pass tensor'}
    (path/(patch_id+'.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return float(pooled.mean())
