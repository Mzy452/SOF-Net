import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from torchvision.ops import nms, box_iou
from sofnet.config import load_config
from sofnet.evaluation import paste_roi_masks
from sofnet.metrics import segmentation_confusion, boundary_counts, detection_summary
from sofnet.utils import save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--predictions',required=True,help='test.py predictions.json')
    p.add_argument('--scenes',required=True,help='Independent full-extent scene reference manifest JSON')
    p.add_argument('--protocol',choices=['reference_boxes','predicted_boxes'],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(); config=load_config(a.config)
    records=json.loads(Path(a.predictions).read_text(encoding='utf-8'))
    scenes=json.loads(Path(a.scenes).read_text(encoding='utf-8'))
    basedir=Path(a.predictions).parent
    confusion=torch.zeros((6,6),dtype=torch.float64); bc=np.zeros(4,dtype=np.int64)
    pp,tt=[],[]; screening_misses=detection_misses=0; area=0.; used_ids=set(); reports=[]
    prior_extents=[]
    if len({s['scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('Duplicate scene IDs in full reference manifest')
    unknown={r['scene_id'] for r in records}-{s['scene_id'] for s in scenes}
    if unknown:
        raise ValueError(f'Prediction scenes lack full references: {unknown}')
    for s in scenes:
        with rasterio.open(s['mask']) as ds:
            reference=ds.read(1); transform=ds.transform; crs=ds.crs
        with rasterio.open(s['valid']) as ds:
            if ds.shape!=reference.shape or ds.transform!=transform or ds.crs!=crs:
                raise ValueError('Reference/validity grids differ')
            valid=(ds.read(1)>0)&(ds.read_masks(1)>0)&(reference<=5)
        if crs is None or not crs.is_projected or crs.linear_units != 'metre':
            raise ValueError('Common-grid area requires projected metre units')
        if transform.b or transform.d or transform.a<=0 or transform.e>=0:
            raise ValueError('Common-grid scoring requires a north-up projected grid')


        for previous,pt,pc,pshape in prior_extents:
            if pc!=crs:
                continue
            if not np.isclose(pt.a,transform.a) or not np.isclose(pt.e,transform.e):
                raise ValueError('Scene grid spacings differ')
            dx=(transform.c-pt.c)/pt.a;dy=(transform.f-pt.f)/pt.e
            if not np.isclose(dx,round(dx),atol=1e-5) or not np.isclose(dy,round(dy),atol=1e-5):
                raise ValueError('Scene grids have noninteger pixel offsets')
            dx,dy=int(round(dx)),int(round(dy))
            x0,y0=max(0,-dx),max(0,-dy)
            x1,y1=min(reference.shape[1],pshape[1]-dx),min(reference.shape[0],pshape[0]-dy)
            if x1>x0 and y1>y0:
                window=rasterio.windows.Window(x0+dx,y0+dy,x1-x0,y1-y0)
                with rasterio.open(previous['valid']) as vds, rasterio.open(previous['mask']) as mds:
                    prev_valid=(vds.read(1,window=window)>0)&(vds.read_masks(1,window=window)>0)&(mds.read(1,window=window)<=5)
                if (valid[y0:y1,x0:x1]&prev_valid).any():
                    raise ValueError('Valid scene domains overlap: provide disjoint ownership/validity masks so each geographic pixel is counted once')
        prior_extents.append((s,transform,crs,reference.shape))
        refs=json.loads(Path(s['boxes']).read_text(encoding='utf-8'))
        if len(refs['boxes']) != len(refs['platform_ids']):
            raise ValueError('Reference box/identity length mismatch')
        if used_ids & set(refs['platform_ids']):
            raise ValueError('Physical platform occurs on more than one evaluation scene; assign canonical scene')
        used_ids.update(refs['platform_ids'])
        boxes=[]; scores=[]; logits=[]; coverage=np.zeros(reference.shape,dtype=bool); ref_choices={}
        current=[r for r in records if r['scene_id']==s['scene_id']]
        for r in current:
            if not np.allclose(r['scene_transform'][:6],list(transform)[:6]) or rasterio.crs.CRS.from_string(r['crs'])!=crs or tuple(r['scene_shape'])!=reference.shape:
                raise ValueError('Prediction grid differs from full reference grid')
            x,y=r['window_xy']
            size=config['data']['patch_size']
            coverage[y:min(y+size,reference.shape[0]),x:min(x+size,reference.shape[1])]=True
            with np.load(basedir/r['raw_archive'],allow_pickle=False) as d:
                if a.protocol=='predicted_boxes':
                    boxes.extend(d['boxes']+np.array([x,y,x,y])); scores.extend(d['scores']); logits.extend(d['roi_logits'])
                else:

                    for k,identity in enumerate(r['platform_ids']):
                        b=d['reference_boxes'][k]+np.array([x,y,x,y]); area_box=(b[2]-b[0])*(b[3]-b[1])
                        if identity not in ref_choices or area_box>ref_choices[identity][0]:
                            ref_choices[identity]=(area_box,b,d['reference_roi_logits'][k].copy())
        if a.protocol=='reference_boxes':
            for identity,b in zip(refs['platform_ids'],refs['boxes']):
                if identity not in ref_choices or not np.allclose(ref_choices[identity][1],b,atol=.01):
                    raise ValueError(f'Conditional evaluation needs an untruncated reference ROI for {identity}')
                boxes.append(b); scores.append(1.); logits.append(ref_choices[identity][2])
        boxes=torch.tensor(np.array(boxes),dtype=torch.float32).reshape(-1,4)
        scores=torch.tensor(np.array(scores),dtype=torch.float32)
        logits=torch.tensor(np.array(logits),dtype=torch.float32) if logits else torch.empty((0,6,28,28))
        keep=nms(boxes,scores,config['model']['nms_iou']) if a.protocol=='predicted_boxes' else torch.arange(len(boxes))
        boxes,scores,logits=boxes[keep],scores[keep],logits[keep]
        mask=paste_roi_masks(logits,boxes,scores,reference.shape,config['model']['detection_threshold'])
        target={'mask':torch.tensor(reference.astype('int64')),'valid':torch.tensor(valid),'boxes':torch.tensor(refs['boxes'],dtype=torch.float32).reshape(-1,4)}
        pred={'mask':mask,'boxes':boxes,'scores':scores}
        confusion+=segmentation_confusion([pred],[target],5)
        bc+=boundary_counts([pred],[target],5,config['evaluation']['boundary_tolerance'])
        pixel_area=abs(transform.a*transform.e-transform.b*transform.d)
        area+=valid.sum()*pixel_area/1e6
        matched=set()
        for j in torch.argsort(scores,descending=True):
            if scores[j]<config['model']['detection_threshold'] or not len(target['boxes']):
                continue
            overlaps=box_iou(boxes[j][None],target['boxes'])[0]
            if matched: overlaps[list(matched)]=-1
            v,k=overlaps.max(0)
            if v>=.5: matched.add(int(k))
        for k,b in enumerate(target['boxes'].round().long().tolist()):
            if k in matched: continue
            x1,y1,x2,y2=b
            hit=coverage[max(0,y1):min(y2,coverage.shape[0]),max(0,x1):min(x2,coverage.shape[1])].any()
            detection_misses+=int(hit); screening_misses+=int(not hit)
        pp.append({'boxes':boxes,'scores':scores}); tt.append({'boxes':target['boxes']})
        reports.append({'scene_id':s['scene_id'],'reference_count':len(target['boxes']),'matched':len(matched),'valid_pixels':int(valid.sum())})
    union=confusion.sum(0)+confusion.sum(1)-confusion.diag()
    iou=np.divide(confusion.diag().numpy(),union.numpy(),out=np.full(6,np.nan),where=union.numpy()>0)
    mp,mr,pn,rn=bc; precision=mp/max(pn,1); recall=mr/max(rn,1)
    result={'protocol':a.protocol,'scene_count':len(scenes),'valid_area_km2':float(area),'confusion_matrix':confusion.long().tolist(),
            'class_IoU_all':[float(v) if np.isfinite(v) else None for v in iou],
            'mIoU_FG':float(iou[1:].mean()) if np.isfinite(iou[1:]).all() else None,
            'boundary_counts':bc.tolist(),'boundary_F1':float(2*precision*recall/max(precision+recall,1e-12)),
            'screening_misses':screening_misses,'detection_stage_misses':detection_misses,'scenes':reports}
    if a.protocol=='predicted_boxes':result['detection']=detection_summary(pp,tt,config,patch_count=len(records))
    save_json(result,a.output)


if __name__=='__main__':
    main()
