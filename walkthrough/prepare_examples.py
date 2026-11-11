from pathlib import Path
import csv, io, os, zipfile

ROOT=Path(__file__).resolve().parents[1]
IDS=[
 'random-strategy/reward-dto/road1-rain_day-scenarios/2_scenario_7_pedestrian.deepscenario',
 'greedy-strategy/reward-jerk/road3-sunny_night-scenarios/0_scenario_9_npc_vehicle.deepscenario',
 'rl_based-strategy/reward-dto/road4-sunny_night-scenarios/15_scenario_4_npc_vehicle.deepscenario']

def local_path(path):
    value=str(path.resolve())
    return Path('\\\\?\\'+value) if os.name=='nt' and not value.startswith('\\\\?\\') else Path(value)

def main():
    for dataset, log in enumerate(IDS,1):
        parent=log.rsplit('/',1)[0];scenario=log.rsplit('/',1)[1][:-len('.deepscenario')]
        dest=ROOT/'outputs'/'examples'/f'Dataset_{dataset}'/'deepscenario-dataset'
        with zipfile.ZipFile(ROOT/'data'/f'Dataset_{dataset}_raw.zip') as z:
            candidates=[n for n in z.namelist() if n.rsplit('/',1)[0]==parent and n.endswith('-attributes.csv')]
            found=[]
            for name in candidates:
                reader=csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig')))
                rows=[r for r in reader if r['ScenarioID']==scenario]
                if rows:found.append((name,reader.fieldnames,rows))
            if len(found)!=1 or len(found[0][2])!=1:raise ValueError('Expected one attribute row for the selected log.')
            name,fields,rows=found[0];csvpath=local_path(dest/name);csvpath.parent.mkdir(parents=True,exist_ok=True)
            with csvpath.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            target=local_path(dest/log);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(log))
        print(f'Dataset_{dataset}: one recorded log and its attribute row prepared.')

if __name__=='__main__':main()
