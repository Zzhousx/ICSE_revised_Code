from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
subprocess.run(['matlab','-batch',"addpath('scripts');table_IV"],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'strap_os/time_fullrank.py')],cwd=ROOT,check=True)
subprocess.run(['matlab','-batch',"addpath('scripts');table_IV_summary"],cwd=ROOT,check=True)
