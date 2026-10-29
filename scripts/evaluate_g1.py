"""40 clips x 4 fixed starts x 10 seconds; identical manifest across methods."""
import argparse
import sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import numpy as np
import torch
from odemimic.g1.envs import SuperMimicEnvCfgArgs, make_env
from odemimic.g1.envs.mdp.motion_library import discover_npz
from odemimic.g1.eval.benchmark import load_policy_from_checkpoint

p=argparse.ArgumentParser()
p.add_argument('--ckpt',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--data-dir',type=Path,required=True)
p.add_argument('--action-diagnostics',action='store_true')
a=p.parse_args()
torch.set_num_threads(4)
root=Path(__file__).resolve().parent
policy,cfg=load_policy_from_checkpoint(a.ckpt,'cuda:0')
files=[str(f) for f in discover_npz(a.data_dir)]
assert len(files)==40
env=make_env(SuperMimicEnvCfgArgs(npz_files=files,num_envs=160,
    future_frames=cfg['env']['future_frames'],episode_length_s=10.,
    action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
    sampling_mode='start',record_transitions=False,seed=20261027,
    anchor_obs=cfg['env'].get('anchor_obs',False)),device='cuda:0')
try:
    cmd=env.command_manager.get_term('motion');lib=cmd.library
    rng=np.random.default_rng(20261027)
    manifest=[];starts=[]
    for i,clip in enumerate(lib.clips):
        assert clip.length>501
        for repeat,offset in enumerate(rng.integers(0,clip.length-500,size=4)):
            starts.append(int(lib.clip_start[i])+int(offset))
            manifest.append(dict(motion=clip.name,repeat=repeat,start_frame=int(offset),horizon=500))
    cmd.forced_frame[:]=torch.tensor(starts,device=env.device)
    obs,_=env.reset()
    # Reset places the robot at start; the command advances to its next target frame.
    start_tensor=torch.tensor(starts,device=env.device)
    torch.testing.assert_close(cmd.frame,start_tensor+1)
    torch.testing.assert_close(cmd.robot.data.root_link_pos_w-env.scene.env_origins,
        lib.ref(start_tensor).root_pos_w,atol=1e-5,rtol=1e-5)
    active=torch.ones(160,dtype=torch.bool,device=env.device)
    ret=torch.zeros(160,device=env.device);error=torch.zeros_like(ret)
    steps=torch.zeros(160,dtype=torch.long,device=env.device)
    fail=torch.zeros_like(active);numerical=torch.zeros_like(active)
    diag=[]
    raw_holder={}
    if a.action_diagnostics:
        hook=policy.decoder.mlp.register_forward_hook(lambda module,args,output: raw_holder.update(raw=output.detach()))
    with torch.no_grad():
        for t in range(500):
            err=torch.norm(cmd.body_pos_relative_w-cmd.robot_body_pos_w,dim=-1).mean(-1)
            act=policy(obs['policy'],obs['goal'])
            if a.action_diagnostics:
                deriv=1-act.square() if cfg['policy'].get('action_output','linear')=='tanh' else torch.ones_like(act)
                diag.append(torch.stack([active.float(),(act.abs()>.99).float().mean(-1),(deriv<.01).float().mean(-1),(deriv==0).float().mean(-1),deriv.mean(-1),raw_holder['raw'].abs().amax(-1),(deriv<.01).all(-1).float()],-1).cpu())
            obs,rew,term,tout,_=env.step(act)
            bad=~torch.isfinite(rew)|~torch.isfinite(err)|~torch.isfinite(act).all(-1)
            valid=active & ~bad
            ret+=torch.where(valid,rew,0.)
            error+=torch.where(valid,err,0.)
            steps+=active.long()
            fail|=active & term
            numerical|=active & bad
            active&=~(term|tout|bad)
            if not bool(active.any()):break
    rows=[]
    for i,m in enumerate(manifest):
        rows.append(dict(**m,steps=int(steps[i]),raw_return=float(ret[i]),
            norm_return=float(ret[i])/50.,mean_body_error_m=float(error[i])/max(int(steps[i]),1),
            failed=bool(fail[i]),numerical_failure=bool(numerical[i])))
    summary=dict(episodes=160,motions=40,horizon_seconds=10,horizon_steps=500,
        mean_return=float(ret.mean()),mean_norm_return=float(ret.mean())/50.,
        mean_body_error_m=sum(x['mean_body_error_m'] for x in rows)/160,
        failed=int((fail|numerical).sum()),absorbing_failures=int(fail.sum()),
        numerical_failures=int(numerical.sum()),
        completed_horizon=int(((steps==500)&~fail&~numerical).sum()))
    result=dict(protocol='G1_40x4_fixed_random_starts_10s_v1',reset_seed=20261027,
        checkpoint=str(a.ckpt),normalization_denominator=50.,summary=summary,episodes=rows)
    if a.action_diagnostics:
        data=torch.stack(diag)
        masks={'all':torch.ones(160,dtype=torch.bool),'failed':(fail|numerical).cpu(),'completed':(~fail&~numerical).cpu()}
        stats={}
        for label,mask in masks.items():
            subset=data[:,mask];valid=subset[...,0]>0
            vals=subset[...,1:][valid]
            stats[label]={'episodes':int(mask.sum()),'action_abs_gt_099':float(vals[:,0].mean()),'derivative_below_001':float(vals[:,1].mean()),'derivative_exact_zero':float(vals[:,2].mean()),'mean_derivative':float(vals[:,3].mean()),'raw_absmax':float(vals[:,4].max()),'all_joints_derivative_below_001_step_fraction':float(vals[:,5].mean())} if len(vals) else {'episodes':0}
        result['action_diagnostics']=stats
        for i,row in enumerate(rows):
            vals=data[:,i,1:][data[:,i,0]>0]
            row['saturation_099']=float(vals[:,0].mean())
            row['mean_output_derivative']=float(vals[:,3].mean())
            row['last50_derivative_below_001']=float(vals[-50:,1].mean())
        hook.remove()
    assert summary['failed']+summary['completed_horizon']==160
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(summary),flush=True)
finally:
    env.close()
