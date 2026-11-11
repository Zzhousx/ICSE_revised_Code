from collections import Counter, defaultdict
import math, re
from data import *

def recordings(dataset):
    meta = metadata(dataset)
    lookup = defaultdict(list)
    for i, row in enumerate(meta):
        lookup[row['log_id']].append(i)
    groups = defaultdict(list)
    with archive(dataset) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.deepscenario'))
        if set(names) != set(lookup):
            raise ValueError('Recorded logs do not match the feature rows.')
        for name in names:
            folder, filename = name.rsplit('/', 1)
            match = re.fullmatch(r'(\d+)_scenario_(\d+)(?:_[^.]+)?\.deepscenario', filename)
            if not match:
                raise ValueError('A log identifier does not match the recording naming scheme.')
            root = ET.fromstring(z.read(name))
            objects = root.findall('.//Entities/ScenarioObject')
            ego = next(e.get('name') for e in objects if e.find('ObjectParameter').get('ObjectType', '').lower() == 'ego')
            action = next(e for e in root.findall('.//ObjectAction') if e.get('name') == 'Act_' + ego)
            wp = action.findall('WayPoint'); dt = float(root.get('timestep', '.5'))
            if dt != .5:
                raise ValueError('The input sampling interval is not 0.5 seconds.')
            def point(w):
                return [float(w.find('.//ObjectPosition').get('position' + a)) for a in 'XYZ']
            stamps = [w.get('timeStamp') for w in wp]
            numeric = [float(x) for x in stamps] if all(x is not None for x in stamps) else None
            if numeric == list(range(1, len(wp) + 1)):
                times = None
            elif numeric is not None and len(numeric) > 1 and np.allclose(np.diff(numeric), dt):
                times = numeric
            elif numeric is not None and len(numeric) > 1 and np.allclose(np.diff(numeric), 1):
                times = [v*dt for v in numeric]
            else:
                times = None
            speed = wp[-1].find('.//Speed'); velocity = wp[-1].find('.//Velocity')
            if speed is not None:
                speed = float(speed.get('speed'))
            elif velocity is not None:
                speed = math.sqrt(sum(float(velocity.get('velocity' + a))**2 for a in 'XYZ'))
            else:
                speed = None
            record = dict(log_id=name, log_index=lookup[name][0], log_indices=lookup[name],
                          folder=folder, execution=int(match[1]), k=int(match[2]),
                          first=point(wp[0]), last=point(wp[-1]), last_speed=speed, times=times)
            groups[(folder, record['execution'])].append(record)
    pieces = []
    for key, seq in sorted(groups.items()):
        seq.sort(key=lambda r: (r['k'], r['log_id']))
        counts = Counter(r['k'] for r in seq); current = []
        for row in seq:
            continuous = False
            if current:
                previous = current[-1]
                gap = float(np.linalg.norm(np.asarray(row['first']) - np.asarray(previous['last'])))
                expected = previous['last_speed']*.5 if previous['last_speed'] is not None else None
                residual = abs(gap - expected) if expected is not None else None
                comparable = previous['times'] is not None and row['times'] is not None
                delta = row['times'][0] - previous['times'][-1] if comparable else None
                continuous = (row['k'] == previous['k'] + 1 and residual is not None and residual <= 1.0
                              and (not comparable or .25 <= delta <= .75)
                              and counts[previous['k']] == 1 and counts[row['k']] == 1)
                if not continuous:
                    pieces.append(current); current = []
            current.append(row)
        if current:
            pieces.append(current)
    return meta, pieces
