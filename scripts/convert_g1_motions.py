"""Convert 29-DOF G1 retargeted motion CSVs to the mjlab NPZ format."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--input-fps',type=float,required=True)
    a=p.parse_args()
    from odemimic.g1.data.motion_convert import convert_csv_to_npz,load_g1_model
    model=load_g1_model()
    for path in sorted(a.input.glob('*.csv')):
        print(convert_csv_to_npz(path,a.out/(path.stem+'.npz'),a.input_fps,50.,model),flush=True)

if __name__=='__main__':main()
