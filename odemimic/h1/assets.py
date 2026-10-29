"""Portable model metadata and native-reference cache preparation."""
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def native_adapter(upstream,variables=None,manifest=None):
    sys.path.insert(0,str(ROOT/'supermimic'))
    os.environ['LOCO_MOTION_MANIFEST']=str(Path(manifest or ROOT/'configs/lafan1_40.json').resolve())
    if variables:os.environ['LOCO_VARIABLES_FILE']=str(Path(variables).resolve())
    from adapter import Adapter,make_config,make_env
    cfg=make_config(Path(upstream),1);cfg.experiment.env_params.use_mjwarp=False
    return Adapter(make_env(cfg))

def write_model_assets(ad,out):
    import mujoco
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model=ad.env._model
    fields=['body_parentid','body_pos','body_quat','body_jntadr','body_jntnum','jnt_type',
            'jnt_pos','jnt_axis','jnt_qposadr','jnt_dofadr','site_bodyid','site_pos','site_quat','qpos0']
    np.savez(out/'model.npz',**{k:np.asarray(getattr(model,k)) for k in fields})
    mujoco.mj_saveModel(model,str(out/'model.mjb'),None)
    return dict(dt=float(ad.env.dt),ids=np.asarray(ad.ids).tolist(),reward_ids=np.asarray(ad.rf._rel_site_ids).tolist(),
                obs_indexes=np.asarray(ad.obs_state_indexes).tolist(),lengths=np.asarray(ad.lengths).tolist(),
                nq=ad.nq,nv=ad.nv,sd=ad.sd,od=ad.od,observation_state_dim=ad.observation_state_dim,
                low=np.asarray(ad.env.info.action_space.low).tolist(),high=np.asarray(ad.env.info.action_space.high).tolist(),
                reward={k:float(getattr(ad.rf,k)) for k in ['_qpos_w_sum','_qpos_w_exp','_qvel_w_sum','_qvel_w_exp',
                        '_rpos_w_sum','_rpos_w_exp','_rquat_w_sum','_rquat_w_exp','_rvel_w_sum','_rvel_w_exp','_action_out_of_bounds_coeff']})

def build_reference_cache(ad,dest):
    import jax.numpy as jnp
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    lengths=np.asarray(ad.lengths);tids=np.repeat(np.arange(len(lengths)),lengths)
    frames=np.concatenate([np.arange(n) for n in lengths]);temp=dest.with_name(dest.stem+'.incomplete.npy')
    table=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(tids),ad.sd))
    for start in range(0,len(tids),8192):
        n=min(8192,len(tids)-start)
        ii=np.pad(tids[start:start+n],(0,8192-n),mode='edge');ff=np.pad(frames[start:start+n],(0,8192-n),mode='edge')
        table[start:start+n]=np.asarray(ad.reference(jnp.asarray(ii),jnp.asarray(ff)))[:n]
        if start%65536==0:print(f'Native reference cache: {start}/{len(tids)}',flush=True)
    table.flush();del table;temp.replace(dest)
