import argparse
import csv
import math
import numpy as np
from sofnet.utils import save_json


def stratified_confirmation(rows):
    populations=[int(r['population']) for r in rows]
    total=sum(populations)
    if total<=0:raise ValueError('Empty population')
    estimate=variance=0.
    details=[]
    for r,N in zip(rows,populations):
        n,c=int(r['sampled']),int(r['confirmed'])
        if not 0<=c<=n<=N or n==0:raise ValueError('Invalid stratum counts')
        if n==1 and n<N:raise ValueError('Stratum variance needs n>=2 unless census')
        w=N/total;p=c/n
        estimate+=w*p
        if n<N:variance+=w*w*(1-n/N)*p*(1-p)/(n-1)
        details.append({'stratum':r['stratum'],'population':N,'sampled':n,'confirmed':c,'sampling_fraction':n/N})
    error=1.96*math.sqrt(variance)
    return {'weighted_confirmation':estimate,'variance':variance,'normal_95_interval':[max(0.,estimate-error),min(1.,estimate+error)],
            'population':total,'strata':details,'interpretation':'confirmation in sampled frame, not regional completeness'}


def registration_residuals(rows,spacing=.65):
    if not rows:raise ValueError('No independent checkpoints')
    xy=np.array([[float(r['dx_pixels']),float(r['dy_pixels'])] for r in rows])
    if not np.isfinite(xy).all():raise ValueError('Nonfinite checkpoint error')
    distance=np.linalg.norm(xy,axis=1)
    return {'checkpoints':len(xy),'bias_xy_pixels':xy.mean(0).tolist(),'median_pixels':float(np.median(distance)),
            'p95_pixels':float(np.percentile(distance,95)),'planar_RMSE_pixels':float(np.sqrt((xy*xy).sum(1).mean())),
            'planar_RMSE_metres':float(np.sqrt((xy*xy).sum(1).mean())*spacing)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['audit','registration'],required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    with open(a.input,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result=stratified_confirmation(rows) if a.mode=='audit' else registration_residuals(rows)
    save_json(result,a.output)


if __name__=='__main__':main()
