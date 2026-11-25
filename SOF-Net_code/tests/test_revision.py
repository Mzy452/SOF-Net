import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from sofnet.config import load_config
from sofnet.losses import component_loss, prior_consistency_loss, SOFLoss
from sofnet.metrics import operating_point, average_precision, segmentation_summary, detection_summary
from sofnet.evaluation import deduplicate_scenes, paste_roi_masks, shift_sar_inputs, paired_scene_bootstrap
from sofnet.sar_prior import cfar_confidence, build_prior, _mean
from sofnet.quality import combine_quality, patch_quality
from sofnet.gates import aggregate_gates
from sofnet.model import SOFNet, PriorGuidedDeformableAttention, QualityAwareGate
from sofnet.data import SOFDataset
from make_experiment_configs import variants
from audit_statistics import stratified_confirmation, registration_residuals

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config(ROOT/'config.yaml')

    def test_invalid_loss_and_gradients(self):
        logits=torch.randn(1,6,9,9,requires_grad=True)
        labels=torch.ones(1,9,9,dtype=torch.long); labels[:,4,4]=255
        l,b=component_loss(logits,labels,5); (l+b).backward()
        self.assertTrue(torch.isfinite(l+b)); self.assertEqual(float(logits.grad[0,:,4,4].abs().sum()),0.)
        l,b=component_loss(logits,torch.full_like(labels,255),5)
        self.assertEqual(float(l+b),0.)

    def test_revised_prior_guard(self):
        response=torch.tensor([[[[3.,-3.,-3.,-3.]]]],requires_grad=True)
        prior=torch.tensor([[[[.8,.1,.9,.8]]]])
        target={'mask':torch.tensor([[1,1,0,1]]),'valid':torch.ones(1,4,dtype=torch.bool)}
        loss=prior_consistency_loss([response],[prior],[target],'revised_lower_bound')
        loss.backward()
        self.assertEqual(float(response.grad[0,0,0,:3].abs().sum()),0.)
        self.assertLess(float(response.grad[0,0,0,3]),0.)

    def test_highest_unmatched_iou(self):
        ref={'boxes':torch.tensor([[0.,0,10,10],[2.,0,12,10]])}
        pred={'boxes':torch.tensor([[0.,0,10,10],[0.,0,10,10]]),'scores':torch.tensor([.9,.8])}
        precision,recall,_,fp=operating_point([pred],[ref])
        self.assertEqual((precision,recall,fp),(1.,1.,0))
        self.assertEqual(average_precision([pred],[ref],.5),1.)

    def test_mask_valid_metrics(self):
        ref=torch.zeros(7,7,dtype=torch.long); ref[3,3]=255
        p=ref.clone();p[3,3]=4
        result=segmentation_summary([{'mask':p}],[{'mask':ref}],self.config)
        self.assertEqual(result['valid_pixels'],48)
        self.assertEqual(result['boundary_F1'],0.)

    def test_cfar_upper_8T(self):
        n=31**2-7**2; threshold=n*np.expm1(-np.log(1e-6)/n)
        a=np.zeros((41,41)); a[20,20]=10*np.log10(4.5*threshold)
        out=cfar_confidence(a,31,7,1e-6,1e-6)
        self.assertAlmostEqual(float(out[20,20]),.5,places=5)
        a[:]=np.nan
        self.assertTrue(np.isnan(cfar_confidence(a,31,7,1e-6,1e-6)).all())

    def test_missing_dates(self):
        a=np.array([[[1.,np.nan]],[[np.nan,.4]]])
        np.testing.assert_allclose(_mean(a),[[1.,.4]])
        np.testing.assert_allclose(_mean(a,[2.,1.]),[[1.,.4]])

    def test_missing_sensor_and_persistence_independence(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'s1.tif';a=np.zeros((1,41,41),np.float32);a[0,20,20]=30
            with rasterio.open(f,'w',driver='GTiff',height=41,width=41,count=1,dtype='float32',crs='EPSG:32631',transform=from_origin(0,41,1,1),nodata=np.nan) as ds:ds.write(a)
            cfg=self.config['sar_prior'];p,s,_=build_prior([], [str(f)], [],cfg)
            np.testing.assert_allclose(p,s[5])
            q,t,_=build_prior([], [str(f)], [],{**cfg,'persistence_threshold':.9})
            np.testing.assert_array_equal(p,q)

    def test_quality_valid_denominator(self):
        cfg=self.config['quality']
        a=np.broadcast_to(np.array(cfg['lower'])[:,None,None],(5,2,2)).copy();a[:,0,0]=np.nan
        q,v=combine_quality(a,cfg)
        self.assertEqual(patch_quality(q,v),1.)
        self.assertFalse(v[0,0])

    def test_gate_aggregation(self):
        a=[torch.full((1,1,2,2),v) for v in [.1,.3,.5,.7]]
        self.assertTrue(torch.allclose(aggregate_gates(a,(8,8)),torch.full((1,1,8,8),.4)))

    def test_scene_dedup(self):
        p={'boxes':torch.tensor([[2.,2.,8.,8.]]),'scores':torch.tensor([.9])}
        t={'scene_id':'s','window_xy':[0,0],'platform_ids':['p'],'reference_boxes_scene':torch.tensor([[2.,2.,8.,8.]])}
        pp,tt=deduplicate_scenes([p,p],[t,t])
        m=detection_summary(pp,tt,self.config,patch_count=2)
        self.assertEqual((m['TP'],m['reference_count'],m['patch_count']),(1,1,2))

    def test_overlap_foreground_priority(self):
        logits=torch.zeros(2,6,2,2);logits[0,1]=5;logits[1,4]=5
        b=torch.tensor([[0.,0.,4.,4.],[0.,0.,4.,4.]])
        c=paste_roi_masks(logits,b,torch.tensor([.8,.9]),(5,5))
        self.assertEqual(int(c[1,1]),4)

    def test_no_shift_wrap(self):
        a=torch.arange(8.).reshape(1,1,1,8)
        b,p=shift_sar_inputs(a,a,2)
        np.testing.assert_array_equal(b.numpy().ravel(),[0,0,0,1,2,3,4,5])

    def test_paired_bootstrap_identity(self):
        p={'scene_id':'s','boxes':torch.tensor([[0.,0.,2.,2.]]),'scores':torch.tensor([.9])}
        t={'scene_id':'s','boxes':p['boxes']}
        r=paired_scene_bootstrap([p],[p],[t],20)
        self.assertEqual(r['percentile_95_interval'],[0.,0.])

    def test_attention_control(self):
        torch.manual_seed(1)
        m=PriorGuidedDeformableAttention(32,8,4,2,use_prior=False)
        o=torch.randn(1,32,4,4);s=torch.randn_like(o)
        torch.testing.assert_close(m(o,s,torch.zeros(1,1,4,4)),m(o,s,torch.ones(1,1,4,4)))
        fixed=PriorGuidedDeformableAttention(32,8,4,2,sampling='fixed_axial')
        self.assertEqual(fixed(o,s,torch.zeros(1,1,4,4)).shape,o.shape)

    def test_full_forward_backward(self):
        cfg=copy.deepcopy(self.config);cfg['model']['pretrained_backbones']=False
        cfg['model']['feature_channels']=32;cfg['model']['max_detections']=3
        model=SOFNet(cfg).eval()

        x=torch.randn(1,6,68,68);sar=torch.randn(1,8,68,68);p=torch.rand(1,1,68,68)
        target={'boxes':torch.tensor([[8.,8.,45.,45.]]),'mask':torch.zeros(68,68,dtype=torch.long),'valid':torch.ones(68,68,dtype=torch.bool)}
        target['mask'][8:45,8:45]=1
        output=model(x,sar,p,p,targets=[target])
        losses=SOFLoss(cfg)(output,[target]);losses['total'].backward()
        self.assertTrue(torch.isfinite(losses['total']))
        with torch.no_grad():dets,aux=model(x,sar,p,p,return_aux=True,reference_boxes=[target['boxes']])
        self.assertEqual(dets[0]['reference_mask'].shape,(68,68))
        self.assertEqual(len(aux['gates']),4)

    def test_matched_control_configs(self):
        v=variants(self.config)
        self.assertEqual(v['no_gate_mean']['loss']['prior'],.2)
        self.assertEqual(v['no_prior_logit']['model']['sampling'],'deformable')
        self.assertEqual(v['loss_revised']['loss']['prior_mode'],'revised_lower_bound')

    def test_audit_statistics(self):
        r=stratified_confirmation([{'stratum':'a','population':10,'sampled':10,'confirmed':9}])
        self.assertAlmostEqual(r['weighted_confirmation'],.9)
        self.assertEqual(r['variance'],0.)
        r=registration_residuals([{'dx_pixels':3,'dy_pixels':4}])
        self.assertEqual(r['planar_RMSE_pixels'],5.)


if __name__=='__main__':unittest.main(verbosity=2)
