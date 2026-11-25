import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import yaml
from sofnet.config import load_config
from sofnet.model import SOFNet

ROOT=Path(__file__).resolve().parents[1]


class PipelineSmokeTest(unittest.TestCase):
    def test_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);transform=from_origin(500000,3000000,.65,.65)
            def write(name,array):
                file=d/(name+'.tif')
                if array.ndim==2:array=array[None]
                with rasterio.open(file,'w',driver='GTiff',height=64,width=64,count=len(array),dtype=str(array.dtype),crs='EPSG:32631',transform=transform) as ds:ds.write(array)
                return str(file)
            rng=np.random.default_rng(9)
            mask=np.zeros((64,64),np.uint8);mask[12:44,12:44]=1;mask[20:25,20:25]=5
            valid=np.ones((64,64),np.uint8);valid[0,0]=0
            paths={'optical':write('optical',rng.random((6,64,64)).astype('float32')),
                   'sar':write('sar',rng.random((8,64,64)).astype('float32')),
                   'prior':write('prior',rng.random((1,64,64)).astype('float32')),
                   'quality':write('quality',np.full((1,64,64),.85,np.float32)),
                   'quality_valid':write('quality_valid',valid),
                   'mask':write('mask',mask),'valid':write('valid',valid)}
            b=d/'boxes.json';b.write_text(json.dumps({'boxes':[[12,12,44,44]],'platform_ids':['SYNTHETIC-P1']}))
            row={'sample_id':'SYNTHETIC','scene_id':'SYNTHETIC-S1','region':'fixture','split':'test',**paths,'boxes':str(b),'selection':'platform','x':'','y':''}
            manifest=d/'raw.csv'
            with manifest.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
            cfg=load_config(ROOT/'config.yaml');cfg['device']='cpu'
            cfg['model'].update(pretrained_backbones=False,feature_channels=32,max_detections=2)
            cfg['data'].update(raw_manifest=str(manifest),processed_root=str(d/'processed'),patch_size=64,stride=64,num_workers=0)
            cfg['training']['batch_size']=1
            cfg['evaluation'].update(checkpoint=str(d/'random_fixture.pt'),output_dir=str(d/'evaluation'))
            torch.save({'model':SOFNet(cfg).state_dict(),'config':cfg,'implementation_version':'20261123','fixture_only':True},cfg['evaluation']['checkpoint'])
            config=d/'config.yaml';config.write_text(yaml.safe_dump(cfg))
            def run(script,*args):
                result=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,msg=result.stdout+result.stderr)
            run('prepare_dataset.py','--config',config)
            run('test.py','--config',config)
            result=json.loads((d/'evaluation/metrics.json').read_text())
            self.assertEqual(result['reference_count'],1)
            self.assertEqual(result['quality_groups']['high']['patch_count'],1)
            run('render_gate_panels.py','--inputs',d/'evaluation/gates/SYNTHETIC_0_0.npz','--rgb-bands','2','1','0','--output',d/'gates.png')
            self.assertTrue((d/'gates.png').is_file())
            scenes=d/'scenes.json';scenes.write_text(json.dumps([{'scene_id':'SYNTHETIC-S1','mask':paths['mask'],'valid':paths['valid'],'boxes':str(b)}]))
            for protocol in ['reference_boxes','predicted_boxes']:
                run('evaluate_scene_grid.py','--config',config,'--predictions',d/'evaluation/predictions.json','--scenes',scenes,'--protocol',protocol,'--output',d/(protocol+'.json'))
                rr=json.loads((d/(protocol+'.json')).read_text());self.assertEqual(sum(map(sum,rr['confusion_matrix'])),4095)
            run('bootstrap_comparison.py','--method-a',d/'evaluation/scene_detections.json','--method-b',d/'evaluation/scene_detections.json','--replicates','10','--output',d/'bootstrap.json')
            self.assertEqual(json.loads((d/'bootstrap.json').read_text())['percentile_95_interval'],[0.,0.])


if __name__=='__main__':unittest.main(verbosity=2)
