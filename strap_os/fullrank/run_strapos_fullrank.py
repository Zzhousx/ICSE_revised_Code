"""
run_strapos_fullrank.py

Usage:
  python3 run_strapos_fullrank.py --strap-dir /path/to/official/STRAP --out results/strapos_full \
      --config check_earlier_recording --dataset 1
  configs: check_earlier_recording, check_earlier_per_log, main, per_log, semantic_coverage,
           time_equivalent, distance_30, distance_50, distance_100, pedestrian_proxy,
           cross_recording_reduction
"""

import argparse
import csv
import json
import os
import tempfile

import strapos_fullrank as fr

# unit:      'recording' = the execution-level recordings, 'log' = each log is one recording
# kept_length: clipping length passed to the official record2segments
CONFIGS = {
    # checks: must reproduce the saved rankings of the earlier runs exactly
    "check_earlier_recording":   dict(unit="recording", adapter="default", kept_length=3,
                                      criterion="rsc", pooled_reduction=True, tail="input"),
    "check_earlier_per_log":     dict(unit="log", adapter="default", kept_length=3,
                                      criterion="rsc", pooled_reduction=True, tail="input"),
    # the new main configuration
    "main":                      dict(unit="recording", adapter="default", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    # sensitivity variants: each changes one thing relative to "main"
    "per_log":                   dict(unit="log", adapter="default", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "semantic_coverage":         dict(unit="recording", adapter="default", kept_length=3,
                                      criterion="sc", pooled_reduction=False, tail="score"),
    "time_equivalent":           dict(unit="recording", adapter="time_equivalent", kept_length=None,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "distance_30":               dict(unit="recording", adapter="distance_30", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "distance_50":               dict(unit="recording", adapter="distance_50", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "distance_100":              dict(unit="recording", adapter="distance_100", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "pedestrian_proxy":          dict(unit="recording", adapter="pedestrian_proxy", kept_length=3,
                                      criterion="rsc", pooled_reduction=False, tail="score"),
    "cross_recording_reduction": dict(unit="recording", adapter="default", kept_length=3,
                                      criterion="rsc", pooled_reduction=True, tail="score"),
}


def load_recordings(dataset, unit, adapter):
    """Return a list of fr.Recording for the dataset.

    unit == 'recording': the 960 / 962 / 976 execution-level recordings of the earlier run.
    unit == 'log':       one Recording per log, as in the earlier per-log run.
    adapter:             'default', 'time_equivalent', 'distance_30/50/100', 'pedestrian_proxy'.
                         Use exactly the adapter options of the earlier runs of these settings.

    Each Recording needs record_data (with smoothed 'actor_vec', 'time', 'localization_msgs'),
    frame_log (log id of every RAW frame) and logs (log ids in order of k).
    Reuse cached record_data pickles of the earlier runs if they exist.
    """
    import sys
    from pathlib import Path
    from collections import defaultdict
    from types import SimpleNamespace
    import numpy as np
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'strap_os'))
    import data as earlier_data
    import grouping as earlier_grouping
    import semantics as earlier_semantics
    meta = earlier_data.metadata(dataset)
    row_map = defaultdict(list)
    for i, row in enumerate(meta): row_map[row['log_id']].append(i + 1)
    globals().setdefault('_row_maps', {})[dataset] = dict(row_map)
    if unit == 'recording':
        _, pieces = earlier_grouping.recordings(dataset)
    elif unit == 'log':
        pieces = [[{'log_id':row['log_id']}] for row in meta]
    else: raise ValueError(unit)
    records=[]; by_name={}; map_inputs=earlier_data.MapInputs()
    with earlier_data.archive(dataset) as archive:
        for index,piece in enumerate(pieces):
            row_id='row:'+str(index+1) if unit=='log' else None
            if unit=='log' and piece[0]['log_id'] in by_name:
                rd=by_name[piece[0]['log_id']]
                records.append(fr.Recording(row_id,rd,[row_id]*len(rd['time']),[row_id]))
                continue
            combined={'loc':[],'obs':[]};owners=[]
            for item in piece:
                parsed=earlier_semantics.parse(archive.read(item['log_id']).decode('utf-8-sig'))
                for message in parsed['loc']:
                    message['header']['timestamp_sec']=1000000000+len(combined['loc'])*.5
                    combined['loc'].append(message)
                combined['obs']+=parsed['obs']
                owners.extend([row_id if unit=='log' else item['log_id']]*len(parsed['loc']))
            if adapter=='pedestrian_proxy': raw=earlier_semantics.encode_proxy(combined)
            else:
                radius=int(adapter.split('_')[1]) if adapter.startswith('distance_') else None
                raw=earlier_semantics.encode(combined,map_inputs.full,radius)
            smooth=earlier_semantics.official_utils.vec_denoise_v3(raw,1 if adapter=='time_equivalent' else 3)
            rd={'actor_vec':smooth,'time':[i*.5 for i in range(len(raw))],
                'localization_msgs':[SimpleNamespace(header=SimpleNamespace(timestamp_sec=1000000000+i*.5)) for i in range(len(raw))]}
            if unit=='log': by_name[piece[0]['log_id']]=rd
            ids=[row_id] if unit=='log' else [x['log_id'] for x in piece]
            records.append(fr.Recording(str(index),rd,owners,ids))
    expected={1:13565,2:13262,3:6703}[dataset] if unit=='log' else {1:960,2:962,3:976}[dataset]
    assert len(records)==expected
    distinct=len({tuple(v) for r in records for v in r.record_data['actor_vec']})
    if adapter=='default': assert distinct=={1:12,2:11,3:12}[dataset]
    print(f'Dataset_{dataset} {unit}: {len(records)} records, {distinct} smoothed vectors',flush=True)
    return records


def load_input_orders(dataset, unit, recs):
    """Generate the earlier input-order protocol using NumPy's seeded PCG64."""
    import numpy as np
    return [np.random.default_rng(seed).permutation(len(recs)).tolist() for seed in range(30)]


def logs_to_ranking_rows(dataset, ranking):
    rows=[];row_map=globals()['_row_maps'][dataset]
    for log in ranking: rows.extend([int(log[4:])] if log.startswith('row:') else row_map[log])
    assert sorted(rows)==list(range(1,{1:13565,2:13262,3:6703}[dataset]+1))
    return rows


def kept_length_for(adapter):
    return 3


def n_budget_5(dataset):
    return int({1:13565,2:13262,3:6703}[dataset]*.05+.5)



# ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strap-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--dataset", required=True, type=int, choices=[1, 2, 3])
    ap.add_argument("--kept-interval", default="closed", choices=["closed", "half_open"])
    ap.add_argument("--stubs", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    seg_mod, prio_mod = fr.import_official(args.strap_dir, use_stubs=args.stubs)

    recs = load_recordings(args.dataset, cfg["unit"], cfg["adapter"])
    orders = load_input_orders(args.dataset, cfg["unit"], recs)
    if len(orders) != 30:
        raise RuntimeError("expected 30 input orders")
    kept_length = cfg["kept_length"] if cfg["kept_length"] is not None else kept_length_for(cfg["adapter"])

    out_dir = os.path.join(args.out, args.config, f"D{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as wd:
        runs, seginfo = fr.run_configuration(
            recs, orders, seg_mod, prio_mod, wd, kept_length=kept_length,
            criterion=cfg["criterion"], pooled_reduction=cfg["pooled_reduction"],
            tail=cfg["tail"], kept_interval=args.kept_interval)

    diag_rows = []
    for seed, (ranking, info) in enumerate(runs):
        rows = logs_to_ranking_rows(args.dataset, ranking)
        with open(os.path.join(out_dir, f"ranking_seed{seed:02d}.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows([[r] for r in rows])
        with open(os.path.join(out_dir, f"tiers_seed{seed:02d}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["position", "log", "tier", "score"])
            for pos, lg in enumerate(ranking):
                t, s = info[lg]
                w.writerow([pos, lg, t, "" if s is None else repr(float(s))])
        d = fr.diagnostics(recs, seginfo, ranking, info, n_budget_5(args.dataset))
        d["seed"] = seed
        diag_rows.append(d)

    with open(os.path.join(out_dir, "diagnostics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_rows[0]))
        w.writeheader()
        w.writerows(diag_rows)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(dict(cfg, kept_length=kept_length, kept_interval=args.kept_interval,
                       official_commit=fr.PINNED_COMMIT), fh, indent=2)
    print(f"done: {args.config} D{args.dataset} -> {out_dir}")


if __name__ == "__main__":
    main()
