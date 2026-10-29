"""Host replay, preserved from the H1 reference implementation."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from supermimic.numerical_safety import state_valid,history_valid

class Replay:
    diagnostic_dir=None
    rejected_transitions=0
    sample_rejections=0
    velocity_indexes=np.arange(24,49)
    def __init__(self,capacity,n):
        self.capacity,self.n=capacity,n;self.ptr=0;self.filled=0;self.arr={}
    def add(self,**items):
        host={k:np.array(jax.device_get(v),copy=True) for k,v in items.items()}
        valid=(np.asarray(host.get('valid',np.ones(self.n,dtype=bool)))
               & state_valid(host['x']) & state_valid(host['x1'])
               & history_valid(host['h'],Replay.velocity_indexes))
        valid &= np.isfinite(host['a']).all(axis=-1) & np.isfinite(host['r'])
        rejected=~valid
        if rejected.any():
            Replay.rejected_transitions+=int(rejected.sum())
            path=Path(self.diagnostic_dir)/f'rejected_simulator_transition_{Replay.rejected_transitions}.npz'
            np.savez(path,**{k:v[rejected] for k,v in host.items()})
            with (Path(self.diagnostic_dir)/'numerical_events.jsonl').open('a') as fp:
                fp.write(json.dumps({'rejected_total':Replay.rejected_transitions,
                    'motion_ids':host['tid'][rejected].tolist(),'frames':host['f'][rejected].tolist(),
                    'raw_transition_file':path.name})+'\n')
            # Raw values above remain recoverable. Safe placeholders never become
            # sampled starts or regression targets, even for a finite outlier.
            host['x'][rejected]=0.;host['x'][rejected,3]=1.
            host['x1'][rejected]=host['x'][rejected]
            host['h'][rejected]=0.;host['a'][rejected]=0.;host['r'][rejected]=0.
            host['done'][rejected]=True;host['absorbing'][rejected]=True
        host['valid']=valid
        bad=np.zeros(self.n,dtype=bool)
        for v in host.values():
            if np.issubdtype(v.dtype,np.floating):
                bad|=~np.isfinite(v).reshape(self.n,-1).all(axis=1)
        if bad.any():
            path=Path(self.diagnostic_dir)/'nonfinite_simulator_transition.npz'
            np.savez(path,**{k:v[bad] for k,v in host.items()})
            raise FloatingPointError(f'Nonfinite real transition in {int(bad.sum())} environments; {path}')
        for k,v in host.items():
            if k not in self.arr:self.arr[k]=np.empty((self.capacity,*v.shape),dtype=v.dtype)
            self.arr[k][self.ptr]=v
        self.ptr=(self.ptr+1)%self.capacity;self.filled=min(self.filled+1,self.capacity)
    def sample(self,rng,b,window=8):
        oldest=0 if self.filled<self.capacity else self.ptr
        t=rng.integers(0,self.filled-window+1,size=b)
        e=rng.integers(0,self.n,size=b)
        for attempt in range(100):
            ok=self.arr['valid'][(oldest+t)%self.capacity,e]
            if ok.all():break
            bad=~ok;Replay.sample_rejections+=int(bad.sum())
            t[bad]=rng.integers(0,self.filled-window+1,size=int(bad.sum()))
            e[bad]=rng.integers(0,self.n,size=int(bad.sum()))
        else:raise RuntimeError('Could not sample a valid real-state start; no invalid batch was used')
        idx=(oldest+t[:,None]+np.arange(window)[None])%self.capacity
        result={k:jnp.asarray(v[idx,e[:,None]]) for k,v in self.arr.items() if k!='h'}
        result['h']=jnp.asarray(self.arr['h'][(oldest+t)%self.capacity,e])
        d=np.asarray(result['done'])|~np.asarray(result['valid'])
        result['mask']=jnp.asarray(np.concatenate([np.ones((b,1)),np.cumprod(1-d[:,:-1],axis=1)],axis=1))*result['valid']
        assert np.asarray(result['mask'][:,0]).all()
        assert state_valid(np.asarray(result['x'][:,0])).all()
        assert history_valid(np.asarray(result['h']),Replay.velocity_indexes).all()
        return result
