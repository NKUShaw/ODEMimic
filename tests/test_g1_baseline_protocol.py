"""Keep the MLP comparison on its original StateSpec loss/checkpoint schema."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from odemimic.g1.world_model.world_model import WorldModel,WorldModelCfg

class Heads:
    @staticmethod
    def termination_features(pos,quat,ref):return pos[...,:2]*0

class G1BaselineProtocolTest(unittest.TestCase):
    def test_mlp_keeps_70d_error_and_no_ode_buffers(self):
        torch.set_num_threads(2)
        wm=WorldModel(WorldModelCfg(kind='mlp'),Heads(),'cpu')
        s=torch.zeros(2,71);s[:,3]=1.;target=s.clone();target[:,0]=.1
        action=torch.zeros(2,29);xi=torch.zeros(2,33)
        pred,squared,_,_=wm._loss_step(s,action,xi,target,torch.zeros(2),s[:,:3],s[:,3:7])
        self.assertEqual(squared.shape,(2,70))
        torch.testing.assert_close(squared,wm.spec.error_vector(pred,target).square())
        self.assertNotIn('state_delta_std',wm.state_dict())
        self.assertNotIn('input_mean',wm.state_dict())

    def test_euler_keeps_frozen_processing_buffers(self):
        wm=WorldModel(WorldModelCfg(kind='euler'),Heads(),'cpu')
        self.assertEqual(tuple(wm.input_mean.shape),(281,))
        self.assertEqual(tuple(wm.state_delta_std.shape),(71,))
        self.assertFalse(bool(wm.normalization_fitted))

if __name__=='__main__':unittest.main()
