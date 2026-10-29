"""Restore Flax/Optax Adam moments without resetting optimizer history."""
import math
import numpy as np
import torch

def mappings(model):
    result={}
    for role,modules in [('ow',[model.world]),('op',[model.encoder,model.decoder]),('ov',[model.critic])]:
        rows=[]
        for i,module in enumerate(modules):
            for j,layer in enumerate(module.layers):
                prefix=f'params/MLP_{i}/Dense_{j}'
                rows += [(layer.weight,prefix+'/kernel',True),(layer.bias,prefix+'/bias',False)]
            for j,layer in enumerate(module.norms):
                prefix=f'params/MLP_{i}/LayerNorm_{j}'
                rows += [(layer.weight,prefix+'/scale',False),(layer.bias,prefix+'/bias',False)]
        result[role]=rows
    return result

def make_optimizers(model,fixture):
    weights=dict(np.load(str(fixture)+'/weights.npz'));optimizers={};counts=model.meta['optimizer_steps'].copy()
    for role,rows in mappings(model).items():
        optimizer=torch.optim.Adam([p for p,_,_ in rows],lr=2e-4,betas=(.9,.95 if role=='op' else .999),eps=1e-8,foreach=False)
        for param,name,transpose in rows:
            first=weights[f'opt/{role}/m/{name}'];second=weights[f'opt/{role}/v/{name}']
            if transpose:first=first.T;second=second.T
            optimizer.state[param]=dict(step=torch.tensor(float(counts[role])),exp_avg=torch.tensor(first.copy(),device=param.device),exp_avg_sq=torch.tensor(second.copy(),device=param.device))
        optimizers[role]=optimizer
    return optimizers,counts

def step(optimizer,count,role):
    per_iteration={'ow':10,'op':1,'ov':20}[role]
    lr=2e-4*(.03+.97*.5*(1+math.cos(math.pi*min(count//per_iteration,250000)/250000)))
    params=[p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
    assert params,'no gradients before update'
    # clip_grad_norm_ rejects NaN/Inf global norms before Adam, avoiding a
    # separate device synchronization for every individual gradient tensor.
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if role=='op':torch.nn.utils.clip_grad_value_(params,.1)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step()
    assert torch.stack([torch.isfinite(p).all() for p in params]).all().item(),'nonfinite candidate parameters'
    return float(norm),lr
