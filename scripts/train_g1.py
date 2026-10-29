"""Train the current G1 MLP/Euler protocol, from scratch or a matching checkpoint."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['baseline','euler'],default='euler');p.add_argument('--data',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--gpu',default='0');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--action-output',choices=['linear','tanh'],default='linear');p.add_argument('--saturation-weight',type=float,default=0.)
    p.add_argument('--microbatch',type=int,default=4096);p.add_argument('--resume',type=Path)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from odemimic.g1.config import TrainConfig
    from odemimic.g1.training.loop import Trainer
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    cfg=TrainConfig(run_name=a.out.name,log_root=a.out.resolve().parent,seed=a.seed)
    cfg.data.npz_dir=a.data;cfg.world_model.kind='mlp' if a.method=='baseline' else 'euler'
    cfg.policy.action_output=a.action_output;cfg.policy_trainer.saturation_weight=a.saturation_weight
    cfg.policy_trainer.microbatch=a.microbatch;cfg.schedule.target_updates=a.updates
    cfg.schedule.checkpoint_updates=500;cfg.schedule.total_iterations=320000;cfg.schedule.log_interval=10;cfg.resume=a.resume
    trainer=Trainer(cfg)
    try:trainer.run()
    finally:trainer.env.close()

if __name__=='__main__':main()
