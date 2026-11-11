"""Time the full-ranking configuration from preloaded input logs to a final ranking."""
from pathlib import Path
import csv,io,sys,tempfile,time
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
from scipy.io import loadmat
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'strap_os'))
import data,grouping,semantics
sys.path.insert(0,str(root/'strap_os'/'fullrank'))
import strapos_fullrank as fr
import run_strapos_fullrank as driver
seg,prio=fr.import_official(str(root/'outputs'/'strap'))
class MemoryArchive:
 def __init__(self,items):self.items=items
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self,name):return self.items[name]
 def namelist(self):return list(self.items)
output=root/'outputs'/'strapos_runtime.csv'
with output.open('w',newline='') as f:csv.writer(f).writerow(['dataset','seed','seconds','matches_saved_ranking'])
for d in (1,2,3):
 table_file=root/'outputs'/f'Table_IV_STRaP-OS_Dataset_{d}.csv'
 with table_file.open('w',newline='') as f:csv.writer(f).writerow(['Dataset','Method','Seed','Seconds','Language'])
 metadata,pieces=grouping.recordings(d)
 with data.archive(d) as z:records={r['log_id']:z.read(r['log_id']) for piece in pieces for r in piece}
 for seed in range(30):
  with tempfile.TemporaryDirectory(dir=root/'outputs') as wd:
   with patch.object(data,'metadata',lambda dataset:metadata),patch.object(data,'archive',lambda dataset:MemoryArchive(records)),patch.object(grouping,'recordings',lambda dataset:(metadata,pieces)),redirect_stdout(io.StringIO()):
    start=time.perf_counter();recs=driver.load_recordings(d,'log','default');[fr.segment_recording(seg,rec,kept_length=3) for rec in recs];order=np.random.default_rng(seed).permutation(len(recs)).tolist()
    runs,seginfo=fr.run_configuration_repeated(recs,[order],seg,prio,wd,kept_length=3,criterion='rsc',kept_interval='half_open')
    ranking,info,seg_ranking=runs[0];fr.diagnostics_repeated(recs,seginfo,ranking,info,seg_ranking,driver.n_budget_5(d));rows=driver.logs_to_ranking_rows(d,ranking);elapsed=time.perf_counter()-start
   expected=loadmat(root/'outputs'/'rankings'/f'Dataset_{d}_STRaP-OS_seed{seed:02d}.mat')['rank'].ravel()
   matches=np.array_equal(rows,expected)
   if not matches:raise RuntimeError(f'Ranking mismatch dataset {d} seed {seed}')
   with output.open('a',newline='') as f:csv.writer(f).writerow([d,seed,format(elapsed,'.9f'),int(matches)])
   with table_file.open('a',newline='') as f:csv.writer(f).writerow([f'Dataset_{d}','STRaP-OS',seed,format(elapsed,'.9f'),'Python'])
   print(f'Dataset_{d} seed {seed}: {elapsed:.3f} s, ranking matches',flush=True)
