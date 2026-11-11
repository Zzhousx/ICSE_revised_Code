"""Generate the full STRaP-OS rankings and expose them in the paper's ranking format."""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.io import savemat
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--dataset',type=int,choices=(1,2,3),required=True);a=p.parse_args()
subprocess.run([sys.executable,str(root/'strap_os'/'launch_fullrank.py'),'--config','main','--dataset',str(a.dataset),'--kept-interval','half_open'],cwd=root,check=True)
source=root/'outputs'/'fullrank'/'main'/f'D{a.dataset}';target=root/'outputs'/'rankings';target.mkdir(parents=True,exist_ok=True)
for seed in range(30):
 rank=np.loadtxt(source/f'ranking_seed{seed:02d}.csv',dtype=np.int32,delimiter=',').reshape(-1,1)
 savemat(target/f'Dataset_{a.dataset}_STRaP-OS_seed{seed:02d}.mat',{'rank':rank,'seed':seed,'dataset':a.dataset})
print(f'Dataset_{a.dataset} STRaP-OS rankings ready.')
