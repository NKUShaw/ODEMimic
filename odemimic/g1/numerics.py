import torch
import os
from pathlib import Path

def checked_clip(parameters, max_norm, label):
    params=[p for p in parameters if p.grad is not None]
    norms=torch.stack([torch.linalg.vector_norm(p.grad.detach(),dtype=torch.float64) for p in params])
    total=torch.linalg.vector_norm(norms)
    if not bool(torch.isfinite(total)):
        if os.environ.get('NUMERIC_FAILURE_DIR'):
            dest=Path(os.environ['NUMERIC_FAILURE_DIR']);dest.mkdir(parents=True,exist_ok=True)
            torch.save({'norm':total.detach().cpu(),'gradients':[p.grad.detach().cpu() for p in params]},dest/(label+'_gradients.pt'))
        raise FloatingPointError(f"{label}: nonfinite gradient before optimizer step")
    scale=(max_norm/(total+1e-6)).clamp(max=1.)
    for p in params:p.grad.mul_(scale.to(p.grad.dtype))
    return total

def require_finite(x,label):
    if not bool(torch.isfinite(x).all()):
        if os.environ.get('NUMERIC_FAILURE_DIR'):
            dest=Path(os.environ['NUMERIC_FAILURE_DIR']);dest.mkdir(parents=True,exist_ok=True)
            torch.save(x.detach().cpu(),dest/(label.replace(' ','_')+'_tensor.pt'))
        raise FloatingPointError(f"{label}: nonfinite tensor before backward/optimizer step")
