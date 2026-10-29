"""Evaluate a trusted PyTorch H1 learner checkpoint on the native fixed starts."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',default='0')
    p.add_argument('--upstream',type=Path,default=ROOT/'external/loco-mujoco');p.add_argument('--variables')
    a=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                                      NUMPY_MADVISE_HUGEPAGE='0',LOCO_MOTION_MANIFEST=str(ROOT/'configs/lafan1_40.json'))
    if a.variables:os.environ['LOCO_VARIABLES_FILE']=str(Path(a.variables).resolve())
    import torch
    import jax
    sys.path.insert(0,str(ROOT/'supermimic'))
    from adapter import Adapter,make_config,make_env
    from evaluate import evaluate
    from odemimic.h1.networks import H1
    torch.set_num_threads(4);model=H1(a.assets).cuda()
    ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);model.eval()
    ad=Adapter(make_env(make_config(a.upstream,160)))
    import json
    ad.motion_names=json.loads((ROOT/'configs/lafan1_40.json').read_text())
    def act(h,g,obs,x):
        with torch.no_grad():command=model.bounded(model.policy(torch.from_dlpack(h),torch.from_dlpack(g)))
        return jax.dlpack.from_dlpack(command.contiguous())
    evaluate(ad,act,a.out,f"iter_{ck['iteration']}",n=160,horizon=1000)

if __name__=='__main__':main()
