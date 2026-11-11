# VoD-Sampler

Code and DeepScenario inputs for log prioritization. Run commands from the
repository root. Rankings, tables, figures and diagnostics are generated in
`outputs/` when the corresponding commands are run.

## Setup

Requirements: Windows, MATLAB R2021b with Statistics and Machine Learning
Toolbox, Python 3.10 and Git. Keep the package versions in `requirements.txt`.

```text
python -m pip install -r requirements.txt
python strap_os/setup.py
```

The setup command obtains the official STRaP code at commit
`1ba86ff54e77ff61548f458ea7559578564bce70` and the Apollo protocol schemas.
The STRaP checkout and its GPL-3.0 license are placed in `outputs/strap/`.

## Inputs and parameters

`data/Dataset_1.mat`, `Dataset_2.mat` and `Dataset_3.mat` contain feature
sequences, collision labels and category IDs. The corresponding raw ZIP files
contain recorded scenarios and their attribute tables, including TTC and DTO.

| Dataset | Generation strategy | Logs | Window length τ | Quality weight α |
|---|---|---:|---:|---:|
| Dataset_1 | Random-based | 13,565 | 5 | 0.5 |
| Dataset_2 | Greedy-based | 13,262 | 5 | 0.5 |
| Dataset_3 | RL-based | 6,703 | 6 | 0.1 |

Each frame has 18 features. The augmented state has 36 dimensions and the
log-Euclidean tangent vector has 666 dimensions. Windows use stride 2 and
max-norm selection. Covariance regularization is ε = 1e-5 and the basis
selection threshold is 1e-10. The quality score uses min-max normalized
covariance trace and mean intensity. The 68 collision categories combine
17 behaviors with four weather/time conditions. ΔGS uses a 1e-10 ridge and
retains eigenvalues above 1e-12.

## Generate rankings

Generate rankings for all three datasets and all methods:

```text
python scripts/rank_all.py
```

The methods are VoD-Sampler, SO-SDC, MO-SDC, STRaP-IR, STRaP-OS,
BehAVExplor and Random. Stochastic methods use seeds 0–29. STRaP-OS uses
30 randomized input orders.

For Dataset_1 only:

```text
matlab -batch "addpath('scripts');run_method(1,'VoD-Sampler',0)"
matlab -batch "addpath('scripts');run_benchmarks(1)"
python scripts/rank_full.py --dataset 1
```

Replace `1` with `2` or `3` for the other datasets. `run_method` accepts one
method and seed; `run_benchmarks` generates the baseline rankings.

The main STRaP-OS configuration uses one recording per feature row, a
smoothing window of 3 frames, a clipping length of 3 seconds, rarity-based
prioritization and half-open segment intervals. Repeated selection continues
until every log with an accepted segment is ranked.

Generate the STRaP-OS variants and evaluate their generated rankings:

```text
python scripts/rank_strapos_variants.py
matlab -batch "addpath('scripts');export_strapos_variants"
```

To run an individual repeated-selection configuration:

```text
python strap_os/launch_fullrank.py --config semantic_coverage --dataset 1 --kept-interval half_open
```

Other configuration names are `main`, `time_equivalent`, `distance_30`,
`distance_50`, `distance_100`, `pedestrian_proxy` and `joined_recordings`.

## Reproduce paper elements

Generate the required rankings before running the table, curve and coverage
commands. Table IV measures runtime by running the methods.

| Paper element | Command |
|---|---|
| Feature preprocessing, Dataset_1 | `matlab -batch "addpath('scripts');preprocess(1)"` |
| Figure 3 | `matlab -batch "addpath('scripts');figure3"` |
| Figure 4 | `matlab -batch "addpath('scripts');figure4"` |
| Figure 5 | `matlab -batch "addpath('scripts');figure5"` |
| Table I | `matlab -batch "addpath('scripts');table_I"` |
| Table II | `matlab -batch "addpath('scripts');table_II"` |
| Table III | `matlab -batch "addpath('scripts');table_III"` |
| Table IV | `python scripts/table_IV.py` |
| Section V-A coverage and statistics | `matlab -batch "addpath('scripts');section_V_A"` |
| Section V-C basis and quality | `python scripts/section_V_C_entry.py` |
| Section V-D window rules | `matlab -batch "addpath('scripts');section_V_D"` |
| STRaP-OS first-round budget and statistics | `matlab -batch "addpath('scripts');recompute_own_budget"` |
| Three-log walkthrough | `matlab -batch "addpath('scripts','walkthrough');walkthrough"` |

Use `preprocess(2)` and `preprocess(3)` for the other datasets. The walkthrough
traces example logs through feature extraction, window selection, covariance
construction, quality scoring, basis selection, bucket assignment and final
rank.

## Code map

| Item | File |
|---|---|
| VoD-Sampler | `VoD_Sampler_core.m`, `function/feature_space_greedy_dpp.m` |
| SO-SDC and MO-SDC | `function/baseline_SDC_GA.m` |
| STRaP-IR | `function/baseline_SBT.m` |
| STRaP-OS entry | `scripts/rank_full.py`, `strap_os/launch_fullrank.py` |
| STRaP-OS ranking | `strap_os/fullrank/strapos_fullrank.py`, `strap_os/fullrank/run_strapos_repeated.py` |
| STRaP-OS input adapter | `strap_os/fullrank/run_strapos_fullrank.py`, `strap_os/semantics.py`, `strap_os/grouping.py`, `strap_os/data.py` |
| STRaP-OS variants | `scripts/rank_strapos_variants.py`, `scripts/export_strapos_variants.m` |
| BehAVExplor | `function/baseline_BehAVExplor.m` |
| Random | `scripts/run_benchmarks.m` |
| CDW-APFD, FD@K%, CCR@K% and ΔGS | `function/compute_CDWAPFD.m`, `function/compute_FaultCurve.m`, `function/compute_CCR.m`, `function/compute_GeomSpread.m` |
| Rank-sum and Â12 | `function/compute_StatTest.m` |
| Figures | `scripts/figure3.m`, `scripts/figure4.m`, `scripts/figure5.m` |
| Tables | `scripts/table_I.m`, `scripts/table_II.m`, `scripts/table_III.m`, `scripts/table_IV.py` |
| Feature extraction | `feature extraction/extract_feature_random.m`, `extract_feature_Greedy.m`, `extract_feature_RL.m` |

Run the STRaP-OS tests after setup:

```text
python strap_os/fullrank/test_strapos_fullrank.py outputs/strap --stubs
```
