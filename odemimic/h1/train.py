"""H1 PyTorch learner; unchanged LocoMuJoCo/MJX simulator and evaluation.

Dynamics, policy, critic, BPTT and Adam run in PyTorch. JAX remains only
in the unchanged LocoMuJoCo/MJX simulator and data adapter.
"""
import argparse
import importlib.util
import json
import os
import pickle
import signal
import sys
import time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',dest='fixture',required=True)
    p.add_argument('--upstream',required=True);p.add_argument('--reference-cache',required=True)
    p.add_argument('--variables');p.add_argument('--motion-manifest',default=str(Path(__file__).resolve().parents[2]/'configs/lafan1_40.json'))
    p.add_argument('--out',required=True);p.add_argument('--microbatch',type=int,default=4096)
    p.add_argument('--iterations',type=int,default=10000);p.add_argument('--smoke-updates',type=int,default=0)
    p.add_argument('--compile',action='store_true');p.add_argument('--resume-torch');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    fixture,out=Path(a.fixture),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=root/'supermimic'
    os.environ.update(MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',NUMPY_MADVISE_HUGEPAGE='0',
                      LOCO_MOTION_MANIFEST=str(Path(a.motion_manifest).resolve()))
    if a.variables:os.environ['LOCO_VARIABLES_FILE']=str(Path(a.variables).resolve())
    import jax
    import jax.numpy as jnp
    import torch
    from odemimic.h1.networks import H1
    from odemimic.h1.optimizers import make_optimizers,step
    sys.path.insert(1,str(source))
    from adapter import Adapter,make_config,make_env
    from curriculum import ReturnCurriculum
    from evaluate import evaluate
    from odemimic.h1.replay import Replay
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    # World fitting, frozen-world BPTT, collection and evaluation specialize
    # the shared MLP forward in more than eight legitimate ways. Keep those
    # compiled variants instead of silently falling back to eager execution.
    if hasattr(torch._dynamo.config,'recompile_limit'):torch._dynamo.config.recompile_limit=32
    else:torch._dynamo.config.cache_size_limit=32
    model=H1(fixture).cuda();optim,counts=make_optimizers(model,fixture)
    restored=pickle.loads((fixture/'resume_state.pkl').read_bytes());cfg_args=restored['args'];start_iteration=restored['iteration']
    rng=np.random.default_rng();rng.bit_generator.state=restored['numpy_rng'];key=jnp.asarray(restored['key'])
    if a.resume_torch:
        ck=torch.load(a.resume_torch,map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
        for k in optim:optim[k].load_state_dict(ck['optimizers'][k])
        counts=ck['optimizer_counts'];start_iteration=ck['iteration'];restored.update(ck['resume_state'])
        rng.bit_generator.state=restored['numpy_rng'];key=jnp.asarray(restored['key'])
    target=start_iteration+a.smoke_updates if a.smoke_updates else a.iterations
    controls=restored['controls'];stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    def status(stage,**kw):
        p=out/'status.tmp';p.write_text(json.dumps(dict(stage=stage,iteration=kw.pop('iteration',start_iteration),target=target,**kw),indent=2));p.replace(out/'status.json')
    status('initializing',microbatch=a.microbatch,framework='pytorch',world='euler' if model.is_ode else 'mlp')
    ad=Adapter(make_env(make_config(Path(a.upstream),4096)));ad.motion_names=json.loads(Path(a.motion_manifest).read_text())
    Replay.velocity_indexes=np.flatnonzero((np.asarray(ad.obs_state_indexes)>=ad.nq)&(np.asarray(ad.obs_state_indexes)<ad.nq+ad.nv))
    Replay.diagnostic_dir=out
    # One cached table of the native reference features, not ODE predictions.
    cache=Path(a.reference_cache)
    if not cache.exists():
        raise FileNotFoundError('Build the shared native reference table before launching learners: '+str(cache))
    table=torch.from_numpy(np.load(cache).copy()).cuda()
    lengths=torch.as_tensor(np.array(ad.lengths,copy=True),device='cuda',dtype=torch.long)
    offsets=torch.cat((lengths.new_zeros(1),lengths.cumsum(0)[:-1]))
    def refs(tid,frame,n=42):
        tid=tid.long();frames=frame.long()[:,None]+torch.arange(n,device='cuda')
        frames=torch.minimum(frames.clamp_min(0),lengths[tid,None]-1)
        return table[offsets[tid,None]+frames]
    def tt(x):
        if isinstance(x,np.ndarray):return torch.from_numpy(x).cuda()
        return torch.from_dlpack(x)
    def jj(x):return jax.dlpack.from_dlpack(x.detach().contiguous())
    def nextkey():
        nonlocal key
        key,k=jax.random.split(key);return k
    curriculum=ReturnCurriculum(ad,4096)
    for name in ['returns','visits','history','probs']:setattr(curriculum,name,np.array(restored['curriculum'][name],copy=True))
    if a.compile:
        model.features=torch.compile(model.features)
        for name in ['world','encoder','decoder','critic']:setattr(model,name,torch.compile(getattr(model,name)))
    def policy(h,tid,frame):
        with torch.no_grad():return model.bounded(model.policy(h,refs(tt(tid),tt(frame),10)[...,2:219].flatten(1)))
    def eval_act(h,g,obs,x):
        with torch.no_grad():return jj(model.bounded(model.policy(tt(h),tt(g))))
    state=ad.train_reset(jax.random.split(nextkey(),4096),jnp.asarray(curriculum.probs));x,tid,f=ad.extract(state)
    h=model.obs(tt(x))[:,None].repeat(1,10,1);replay=Replay(1000,4096)
    lo,hi=jnp.asarray(model.meta['low']),jnp.asarray(model.meta['high'])
    def collect():
        nonlocal state,x,tid,f,h,controls
        action=jnp.clip(jj(policy(h,tid,f))+.1*jax.random.normal(nextkey(),(4096,19)),lo,hi)
        bc=state.bad_count;state,tr=ad.train_step(state,action,jnp.asarray(curriculum.probs));xn,tn,fn,rr,ab,dd,valid=tr
        replay.add(x=x,x1=xn,a=action,tid=tid,f=f,r=rr,absorbing=ab,done=dd,h=jj(h),valid=valid,bad_count=bc)
        controls+=4096;x,tid,f=ad.extract(state);ob=model.obs(tt(x));done=tt(dd).bool()
        h=torch.where(done[:,None,None],ob[:,None].expand(-1,10,-1),torch.cat((h[:,1:],ob[:,None]),1))
        return rr,dd,valid
    status('refilling_replay')
    for _ in range(64):collect()
    curriculum.begin(np.asarray(tid),np.asarray(f))
    def save(iteration,stage):
        model_state=model.state_dict()
        # Remove torch.compile wrapper prefixes so eager/compiled loads agree.
        model_state={k.replace('._orig_mod.','.'):v for k,v in model_state.items()}
        resume=dict(restored,numpy_rng=rng.bit_generator.state,key=np.asarray(key),controls=controls,iteration=iteration,policy_updates=iteration,curriculum=curriculum.snapshot())
        ck=dict(model=model_state,optimizers={k:v.state_dict() for k,v in optim.items()},optimizer_counts=counts.copy(),resume_state=resume,
                iteration=iteration,controls=controls,framework='pytorch',metadata=model.meta,args=vars(a),parent_checkpoint=model.meta['checkpoint'])
        temp=out/'latest.tmp';torch.save(ck,temp);temp.replace(out/'latest.pt')
        if iteration%500==0:
            dest=out/'checkpoints';dest.mkdir(exist_ok=True);p=dest/f'iter_{iteration}.pt'
            if not p.exists():os.link(out/'latest.pt',p)
        status(stage,iteration=iteration)
    save(start_iteration,'ready')
    def sample(n):return {k:tt(v) for k,v in replay.sample(rng,n,8).items()}
    begin=time.perf_counter()
    for it in range(start_iteration,target):
        t=time.perf_counter();collect_t=t
        for _ in range(3 if it%2==0 else 2):
            rr,dd,valid=collect();curriculum.observe(np.asarray(rr),np.asarray(dd),np.asarray(valid),np.asarray(tid),np.asarray(f))
        curriculum.refresh();torch.cuda.synchronize();collection_seconds=time.perf_counter()-collect_t
        model.world.requires_grad_(True);wm_t=time.perf_counter()
        for _ in range(10):
            optim['ow'].zero_grad(set_to_none=True);wl=model.wm_loss(sample(2048));assert torch.isfinite(wl).item()
            wl.backward();wgn,lr=step(optim['ow'],counts['ow'],'ow');counts['ow']+=1
        optim['ow'].zero_grad(set_to_none=True);model.world.requires_grad_(False);model.critic.requires_grad_(False)
        torch.cuda.synchronize();world_seconds=time.perf_counter()-wm_t;actor_t=time.perf_counter()
        batch=sample(4096);reference=refs(batch['tid'][:,0],batch['f'][:,0]);optim['op'].zero_grad(set_to_none=True)
        auxes=[];pl=0.
        for start in range(0,4096,a.microbatch):
            micro={k:v[start:start+a.microbatch] for k,v in batch.items()}
            ll,aux=model.actor_loss(micro,reference[start:start+a.microbatch],return_aux=True,
                                    saturation_weight=cfg_args.get('saturation_weight',0.),saturation_threshold=cfg_args.get('saturation_threshold',1.))
            (ll*(a.microbatch/4096)).backward();pl+=float(ll.detach())*(a.microbatch/4096);auxes.append(aux)
        pgn,lr=step(optim['op'],counts['op'],'op');counts['op']+=1
        traces=tuple(torch.cat([v[0][i] for v in auxes],1) for i in range(6));boot=torch.cat([v[1] for v in auxes])
        torch.cuda.synchronize();actor_seconds=time.perf_counter()-actor_t;critic_t=time.perf_counter()
        xs,gs,rs,cs,vs,mask=traces;nextvalues=torch.cat((vs[1:],boot[None]),0);ret=boot;targets=[]
        for index in range(31,-1,-1):
            ret=rs[index]+.99*cs[index]*(.05*nextvalues[index]+.95*ret);targets.append(ret)
        yy=torch.stack(targets[::-1]).flatten();cx=((xs-model.mean)/model.std).reshape(-1,267);gg=model.normg(gs.reshape(-1,2170));weight=mask.flatten().float()
        model.critic.requires_grad_(True)
        for epoch in range(5):
            for index in rng.permutation(32*4096).reshape(4,-1):
                ix=torch.as_tensor(index,device='cuda');optim['ov'].zero_grad(set_to_none=True)
                vv=model.critic(torch.cat((cx[ix],gg[ix]),-1))[...,0]
                diff=torch.where(weight[ix]>0,vv-yy[ix],0.);cl=(diff.square()*weight[ix]).sum()/weight[ix].sum().clamp_min(1)
                assert torch.isfinite(cl).item();cl.backward();cgn,lr=step(optim['ov'],counts['ov'],'ov');counts['ov']+=1
        optim['ov'].zero_grad(set_to_none=True);torch.cuda.synchronize();critic_seconds=time.perf_counter()-critic_t
        row=dict(iteration=it+1,policy_updates=it+1,controls=controls,framework='pytorch',microbatch=a.microbatch,wm_loss=float(wl.detach()),policy_loss=pl,critic_loss=float(cl.detach()),
                 collection_seconds=collection_seconds,world_seconds=world_seconds,actor_seconds=actor_seconds,critic_seconds=critic_seconds,iteration_seconds=time.perf_counter()-t,
                 parameters_and_optimizer_finite=True,elapsed_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,learning_rate=lr)
        row.update(wm_gradient_norm=wgn,policy_gradient_norm=pgn,critic_gradient_norm=cgn,
                   wm_optimizer_steps=counts['ow'],actor_optimizer_steps=counts['op'],critic_optimizer_steps=counts['ov'],
                   saturation_weight=cfg_args.get('saturation_weight',0.),rejected_real_transitions=Replay.rejected_transitions,
                   objective_version=restored['objective_version'],action_transform=restored['action_transform'])
        with torch.no_grad():
            survival=torch.cat((torch.ones_like(cs[:1]),cs.cumprod(0)[:-1]),0)
            discounted=.99**torch.arange(32,device='cuda')[:,None]
            row.update(imagined_reward=float((discounted*survival*rs).sum(0).mean()),
                       critic_bootstrap_tail=float((.99**32*cs.prod(0)*boot).mean()),
                       bootstrap_value_mean=float(boot.mean()),bootstrap_value_absmax=float(boot.abs().max()),
                       imagined_qvel_absmax=float(xs[...,26:51].abs().max()),
                       imagined_normalized_feature_absmax=float(((xs-model.mean)/model.std).abs().max()))
        if (it+1)%10==0:
            with torch.no_grad():
                raw=model.policy(batch['h'],reference[:,:10,2:219].flatten(1));unit=torch.tanh(raw);derivative=1-unit.square()
                row.update(raw_action_absmax=float(raw.abs().max()),action_output_derivative_mean=float(derivative.mean()),
                           action_output_derivative_below_001=float((derivative<.01).float().mean()),
                           action_saturation_099=float((unit.abs()>.99).float().mean()))
        with (out/'metrics.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        del batch,reference,traces,auxes,xs,gs,rs,cs,vs,mask,yy,cx,gg,targets,ret,boot,nextvalues
        if (it+1)%100==0 or stop_requested or it+1==target:save(it+1,'paused' if stop_requested else 'training')
        if not a.smoke_updates and (it+1)%500==0:
            evaluate(ad,eval_act,out/'evaluation',f'iter_{it+1}',n=160,horizon=1000)
        if stop_requested:return
    save(target,'smoke_complete' if a.smoke_updates else 'complete')

if __name__=='__main__':main()
