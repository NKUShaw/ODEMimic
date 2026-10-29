"""Convert a trusted H1 JAX checkpoint, preserving all Adam moments and counters."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--upstream',type=Path,default=ROOT/'external/loco-mujoco')
    p.add_argument('--variables');p.add_argument('--motion-manifest',type=Path,default=ROOT/'configs/lafan1_40.json')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ck=pickle.loads(a.checkpoint.read_bytes())
    assert ck['action_transform']=='tanh_native_bounds','This H1 learner uses Tanh actions.'
    arrays={}
    def flatten(tree,prefix):
        for key,value in tree.items():
            if isinstance(value,dict):flatten(value,prefix+'/'+key)
            else:arrays[prefix+'/'+key]=np.asarray(value)
    for key in ['wp','pp','vp','norm']:flatten(ck[key],key)
    def find_adam(state):
        if hasattr(state,'mu') and hasattr(state,'nu') and hasattr(state,'count'):return state
        if isinstance(state,(list,tuple)):
            for child in state:
                found=find_adam(child)
                if found is not None:return found
        return None
    counts={}
    for key in ['ow','op','ov']:
        state=find_adam(ck[key]);assert state is not None,key
        counts[key]=int(np.asarray(state.count));flatten(state.mu,'opt/'+key+'/m');flatten(state.nu,'opt/'+key+'/v')
    np.savez(a.out/'weights.npz',**arrays)
    from odemimic.h1.assets import native_adapter,write_model_assets
    meta=write_model_assets(native_adapter(a.upstream,a.variables,a.motion_manifest),a.out)
    meta.update(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                optimizer_steps=counts,training_args=ck['args'],iteration=ck['iteration'],controls=ck['controls'])
    (a.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    state={key:ck[key] for key in ['key','numpy_rng','curriculum','iteration','controls','policy_updates','args','action_transform','objective_version']}
    (a.out/'resume_state.pkl').write_bytes(pickle.dumps(state))
    print(json.dumps(dict(iteration=ck['iteration'],optimizer_steps=counts,out=str(a.out))),flush=True)

if __name__=='__main__':main()
