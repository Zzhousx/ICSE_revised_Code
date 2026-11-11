from semantics import *
import csv, io, zipfile

def archive(dataset):
    return zipfile.ZipFile(ROOT / 'data' / f'Dataset_{dataset}_raw.zip')

def metadata(dataset):
    rows = []
    with archive(dataset) as z:
        for name in sorted(n for n in z.namelist() if n.endswith('-attributes.csv')):
            for row in csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))):
                row['log_id'] = name.rsplit('/', 1)[0] + '/' + row['ScenarioID'] + '.deepscenario'
                rows.append(row)
    return rows

class MapInputs:
    def __init__(self):
        self.full = NS(lanes={}, crosswalk={})
