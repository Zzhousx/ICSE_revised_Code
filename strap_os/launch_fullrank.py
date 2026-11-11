"""Initialize the input adapter and run the full-ranking configurations."""
from pathlib import Path
import argparse
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'strap_os'))
import semantics
sys.path.insert(0,str(root/'strap_os'/'fullrank'))
import strapos_fullrank as fr
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config')
options, _ = parser.parse_known_args()
if options.config == 'single_pass':
    import run_strapos_fullrank as driver
    driver.CONFIGS['single_pass'] = dict(driver.CONFIGS['per_log'])
else:
    import run_strapos_repeated as driver
pinned=root/'outputs'/'strap'
fr.verify_official_files(pinned)
sys.argv=['run_strapos_fullrank.py','--strap-dir',str(pinned),'--out',str(root/'outputs'/'fullrank')]+sys.argv[1:]
driver.main()
