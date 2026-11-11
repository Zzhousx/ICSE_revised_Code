from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
for dataset in (1,2,3):
    command=f"addpath('scripts');run_method({dataset},'VoD-Sampler',0);run_benchmarks({dataset})"
    subprocess.run(['matlab','-batch',command],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/rank_full.py'),'--dataset',str(dataset)],cwd=ROOT,check=True)
