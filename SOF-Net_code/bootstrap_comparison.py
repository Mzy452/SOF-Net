import argparse
import json
from pathlib import Path
import torch
from sofnet.evaluation import paired_scene_bootstrap
from sofnet.utils import save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method-a',required=True); p.add_argument('--method-b',required=True)
    p.add_argument('--output',required=True); p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261119)
    a=p.parse_args()
    ra=sorted(json.loads(Path(a.method_a).read_text()),key=lambda x:x['scene_id'])
    rb=sorted(json.loads(Path(a.method_b).read_text()),key=lambda x:x['scene_id'])
    if len(ra)!=len(rb):raise ValueError('Different scene counts')
    pp,qq,tt=[],[],[]
    for x,y in zip(ra,rb):
        if x['scene_id']!=y['scene_id'] or x['platform_ids']!=y['platform_ids'] or x['reference_boxes']!=y['reference_boxes']:
            raise ValueError('Reference identities or boxes differ between detectors')
        for source,dest in ((x,pp),(y,qq)):
            dest.append({'scene_id':source['scene_id'],'boxes':torch.tensor(source['boxes'],dtype=torch.float32).reshape(-1,4),'scores':torch.tensor(source['scores'])})
        tt.append({'scene_id':x['scene_id'],'boxes':torch.tensor(x['reference_boxes'],dtype=torch.float32).reshape(-1,4)})
    save_json(paired_scene_bootstrap(pp,qq,tt,a.replicates,a.seed),a.output)


if __name__=='__main__':main()
