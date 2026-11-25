import argparse
import copy
from pathlib import Path
import yaml
from sofnet.config import load_config


def variants(base):
    result={}
    def add(name,changes):
        c=copy.deepcopy(base)
        for section,key,value in changes:c[section][key]=value
        result[name]=c
    add('full_original',[])
    add('no_prior_logit',[('model','attention_prior',False)])
    add('neutral_quality',[('model','gate_mode','neutral_quality')])
    add('gate_no_prior',[('model','gate_mode','no_prior')])
    add('gate_feature_only',[('model','gate_mode','feature_only')])
    add('no_gate_mean',[('model','gate_mode','mean')])
    add('fixed_axial',[('model','sampling','fixed_axial')])
    add('loss_revised',[('loss','prior_mode','revised_lower_bound')])
    add('loss_none',[('loss','prior_mode','none'),('loss','prior',0.)])
    add('unweighted_focal',[('loss','class_balanced',False)])
    for w in [0.,.1,.2,.4]:add(f'prior_weight_{w}',[('loss','prior',w)])
    for scale in [.5,1.,1.5]:add(f'radius_{scale}',[('model','displacement_radii',[r*scale for r in base['model']['displacement_radii']])])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs=3,type=int,required=True,help='Provide three actual matched seeds; historical seeds were not supplied')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base=load_config(a.config)
    for name,cfg in variants(base).items():
        for seed in a.seeds:
            c=copy.deepcopy(cfg);c['seed']=seed;c['training']['resume']=None
            c['training']['output_dir']=str(out/'runs'/f'{name}_{seed}')
            c['evaluation']['checkpoint']=str(Path(c['training']['output_dir'])/'best.pt')
            c['evaluation']['output_dir']=str(Path(c['training']['output_dir'])/'evaluation')
            (out/f'{name}_{seed}.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')

    for sampling in ['deformable','fixed_axial']:
        for dx in [0,4,8,16,24,32,40,48]:
            c=copy.deepcopy(base);c['model']['sampling']=sampling
            c['evaluation'].update(shift_pixels=dx,shift_common_margin=48)
            c['evaluation']['checkpoint']=str(out/'runs'/f'{"full_original" if sampling=="deformable" else "fixed_axial"}_{a.seeds[0]}'/'best.pt')
            c['evaluation']['output_dir']=str(out/'shift_results'/f'{sampling}_{dx}')
            (out/f'shift_{sampling}_{dx}.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
    for key,values in [('gf3_weight',[.4,.5,.6,.7]),('roi_threshold',[.15,.2,.25,.3,.35]),('persistence_threshold',[.4,.5,.6,.7,.8])]:
        for value in values:
            c=copy.deepcopy(base);c['sar_prior'][key]=value
            if key=='gf3_weight':c['sar_prior']['sentinel1_weight']=1-value
            c['data']['processed_root']=str(out/'sensitivity_inputs'/f'{key}_{value}')
            c['data']['sar_prior_manifest']=str(out/'manifests'/f'{key}_{value}_prior.csv')
            c['data']['raw_manifest']=str(out/'manifests'/f'{key}_{value}_raw.csv')
            (out/'manifests').mkdir(exist_ok=True)
            Path(c['data']['sar_prior_manifest']).write_text('sample_id,gf3,sentinel1,sentinel1_quality,output_prior,output_stack\n',encoding='utf-8')
            Path(c['data']['raw_manifest']).write_text('sample_id,scene_id,region,split,optical,sar,prior,quality,quality_valid,mask,valid,boxes,selection,x,y\n',encoding='utf-8')
            c['evaluation']['output_dir']=str(out/'sensitivity_results'/f'{key}_{value}')
            (out/f'sweep_{key}_{value}.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')


if __name__=='__main__':main()
