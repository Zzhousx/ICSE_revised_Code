from pathlib import Path
import subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
subprocess.run(['matlab','-batch',"addpath('scripts');section_V_C"],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/section_V_C.py')],cwd=ROOT,check=True)
