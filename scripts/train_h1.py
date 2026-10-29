"""Continue H1 training with the validated PyTorch MLP/Euler learner."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True)
    p.add_argument('--out',required=True);p.add_argument('--gpu',default='0');p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--microbatch',type=int,default=4096);p.add_argument('--upstream',default=str(ROOT/'external/loco-mujoco'))
    p.add_argument('--reference-cache',default=str(ROOT/'cache/h1_references.npy'));p.add_argument('--variables')
    p.add_argument('--resume-torch');p.add_argument('--smoke-updates',type=int,default=0);p.add_argument('--no-compile',action='store_true')
    a=p.parse_args();env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                            NUMPY_MADVISE_HUGEPAGE='0',JAX_DEFAULT_MATMUL_PRECISION='highest',JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS='true',
                            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4')
    cmd=[sys.executable,'-u','-m','odemimic.h1.train','--assets',a.assets,'--out',a.out,'--upstream',a.upstream,
         '--reference-cache',a.reference_cache,'--iterations',str(a.iterations),'--microbatch',str(a.microbatch),'--smoke-updates',str(a.smoke_updates)]
    if a.variables:cmd+=['--variables',a.variables]
    if a.resume_torch:cmd+=['--resume-torch',a.resume_torch]
    if not a.no_compile:cmd+=['--compile']
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
