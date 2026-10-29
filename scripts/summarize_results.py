"""Print three-seed statistics from completed per-seed evaluation records."""
import argparse
import csv
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results/per_seed.csv');a=p.parse_args()
    rows=list(csv.DictReader(a.input.open()))
    print('| Robot | Method | Step | Return | Error (m) | Failures |')
    print('|---|---|---:|---:|---:|---:|')
    for key in sorted({(r['robot'],r['method'],int(r['step'])) for r in rows}):
        group=[r for r in rows if (r['robot'],r['method'],int(r['step']))==key]
        if sorted(int(r['seed']) for r in group)!=[42,43,44]:continue
        cells=[]
        for field,digits in [('return_display',2),('error_m',4),('failures',2)]:
            values=[float(r[field]) for r in group];cells.append(f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}')
        print('| '+' | '.join([key[0],key[1],str(key[2]),*cells])+' |')

if __name__=='__main__':main()
