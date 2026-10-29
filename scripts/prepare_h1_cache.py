"""Build the shared native H1 reference table used by the PyTorch learner."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',type=Path,default=ROOT/'external/loco-mujoco')
    p.add_argument('--variables');p.add_argument('--out',type=Path,default=ROOT/'cache/h1_references.npy')
    a=p.parse_args()
    from odemimic.h1.assets import native_adapter,build_reference_cache
    build_reference_cache(native_adapter(a.upstream,a.variables),a.out)

if __name__=='__main__':main()
