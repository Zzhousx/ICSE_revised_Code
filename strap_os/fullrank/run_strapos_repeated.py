"""
run_strapos_repeated.py

Driver for repeated STRaP: STRaP's duplicate reduction and prioritization are applied again
to the segments not yet ranked, until every log is ranked (strapos_fullrank.py, Section 7).

Usage:
  python3 run_strapos_repeated.py --strap-dir /path/to/official/STRAP \
      --out results/strapos_repeated --config main --dataset 1
  configs: main, semantic_coverage, time_equivalent, distance_30, distance_50, distance_100,
           pedestrian_proxy, joined_recordings
"""

import argparse
import csv
import json
import os
import tempfile

import strapos_fullrank as fr
from run_strapos_fullrank import (kept_length_for, load_input_orders, load_recordings,
                                  logs_to_ranking_rows, n_budget_5)

# unit:        'log' = each log is one recording (main), 'recording' = the joined drives
# kept_length: clipping length passed to the official record2segments (None = kept_length_for)
CONFIGS = {
    # the main configuration
    "main":              dict(unit="log", adapter="default", kept_length=3, criterion="rsc"),
    # sensitivity variants: each changes one thing relative to "main"
    "semantic_coverage": dict(unit="log", adapter="default", kept_length=3, criterion="sc"),
    "time_equivalent":   dict(unit="log", adapter="time_equivalent", kept_length=None,
                              criterion="rsc"),
    "distance_30":       dict(unit="log", adapter="distance_30", kept_length=3, criterion="rsc"),
    "distance_50":       dict(unit="log", adapter="distance_50", kept_length=3, criterion="rsc"),
    "distance_100":      dict(unit="log", adapter="distance_100", kept_length=3, criterion="rsc"),
    "pedestrian_proxy":  dict(unit="log", adapter="pedestrian_proxy", kept_length=3,
                              criterion="rsc"),
    "joined_recordings": dict(unit="recording", adapter="default", kept_length=3,
                              criterion="rsc"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strap-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--dataset", required=True, type=int, choices=[1, 2, 3])
    ap.add_argument("--kept-interval", default="half_open", choices=["closed", "half_open"])
    ap.add_argument("--stubs", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    seg_mod, prio_mod = fr.import_official(args.strap_dir, use_stubs=args.stubs)

    recs = load_recordings(args.dataset, cfg["unit"], cfg["adapter"])
    orders = load_input_orders(args.dataset, cfg["unit"], recs)
    if len(orders) != 30:
        raise RuntimeError("expected 30 input orders")
    kept_length = cfg["kept_length"] if cfg["kept_length"] is not None else kept_length_for(cfg["adapter"])
    budget = n_budget_5(args.dataset)

    out_dir = os.path.join(args.out, args.config, f"D{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as wd:
        runs, seginfo = fr.run_configuration_repeated(
            recs, orders, seg_mod, prio_mod, wd, kept_length=kept_length,
            criterion=cfg["criterion"], kept_interval=args.kept_interval)

    diag_rows = []
    for seed, (ranking, info, seg_ranking) in enumerate(runs):
        rows = logs_to_ranking_rows(args.dataset, ranking)
        with open(os.path.join(out_dir, f"ranking_seed{seed:02d}.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows([[r] for r in rows])
        with open(os.path.join(out_dir, f"rounds_seed{seed:02d}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["position", "log", "status", "round", "vector", "score"])
            for pos, lg in enumerate(ranking):
                t, rnd, vec, sc = info[lg]
                w.writerow([pos, lg, t, "" if rnd is None else rnd,
                            "" if vec is None else "".join(map(str, vec)),
                            "" if sc is None else repr(float(sc))])
        d = fr.diagnostics_repeated(recs, seginfo, ranking, info, seg_ranking, budget)
        d["seed"] = seed
        diag_rows.append(d)

    with open(os.path.join(out_dir, "diagnostics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_rows[0]))
        w.writeheader()
        w.writerows(diag_rows)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(dict(cfg, mode="repeated", kept_length=kept_length,
                       kept_interval=args.kept_interval, official_commit=fr.PINNED_COMMIT),
                  fh, indent=2)
    print(f"done: {args.config} D{args.dataset} -> {out_dir}")


if __name__ == "__main__":
    main()
