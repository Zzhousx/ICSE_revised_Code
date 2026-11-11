"""
strapos_fullrank.py

STRaP-OS ranking in which every log receives a rank (VoD-Sampler revision, ICSE 2027 #1929).

Only the official STRaP functions are used for STRaP's own stages (pinned clone, commit
1ba86ff54e77ff61548f458ea7559578564bce70, files unchanged):
  * segment_split.record2segments   -> split_record, the 1-second minimum, clipping
                                       and the WITHIN-recording duplicate reduction
  * prioritization.prioritize_by_rsc / prioritization.prioritize_by_semantic_coverage
Everything in this file is either plumbing around those calls or the mapping from STRaP's
segments to a ranking of logs.

Two orderings are provided:
  * rank_logs           the single-pass setting (kept segments first, then the remaining logs
                        by score or by input order); used for the checks
  * rank_logs_repeated  repeated STRaP (Section 7): STRaP's reduction and prioritization are
                        applied again to the segments not yet ranked, until every log is ranked

What this file does NOT produce (reuse the code of the earlier STRaP runs for it):
  * record_data of every recording: the dict that the existing adapter + official
    get_alignment_messages / create_actor_vecs / vec_denoise_v3 produce. It must contain
        'actor_vec'          smoothed frame vectors (list of lists of 8 ints), one per SMOOTHED frame
        'time'               seconds since the first raw frame, one float per RAW frame
        'localization_msgs'  one message per RAW frame, each with .header.timestamp_sec
    Exactly as in the official record2vec, smoothed frame i is aligned with raw frame i
    (record2segments indexes record_data['time'] with smoothed indices).
  * frame_log of every recording: the log id of every RAW frame (len == len(record_data['time'])).
  * the input orders of the 30 runs.

Python 3.8+, numpy.
"""

import contextlib
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
import pickle
import sys
import types
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

PINNED_COMMIT = "1ba86ff54e77ff61548f458ea7559578564bce70"
PINNED_SHA256 = {
    "segment_split.py": "e82196720f7a5223ba7439f28b040a896f4528742be917ef16ac56acb982b867",
    "prioritization.py": "66d6f05a0a696b2fbdb18f8a1e1238b3f6584f082cafe8d391a429bed0909b3d",
    "utils.py": "17bf2e6f5f3610b278bb75e8a4de2a6c3d045beefb1c3d194ca62f8959bf98ac",
}


# --------------------------------------------------------------------------------------
# 1. Importing the official code
# --------------------------------------------------------------------------------------

class _Dummy:
    """Placeholder for Apollo/protobuf/shapely names that the functions used here never touch."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, prefixes):
        self.prefixes = tuple(prefixes)

    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def _getattr(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Dummy

        module.__getattr__ = _getattr
        return module

    def exec_module(self, module):
        pass


def install_import_stubs():
    """Only needed if the Apollo/cyber dependencies of the official files are NOT importable.

    record2segments, split_record, clip_segment, prioritize_by_rsc and
    prioritize_by_semantic_coverage use only numpy, pickle and datetime, so stubbing the
    other imports does not change their behaviour. If the earlier runs already had a way to
    import the official files, use that instead and do not call this function.
    """
    candidates = ["cyber_py", "modules", "common", "shapely", "google.protobuf",
                  "matplotlib", "scipy"]
    missing = []
    for name in candidates:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        sys.meta_path.insert(0, _StubFinder(missing))
    return missing


def verify_official_files(strap_dir):
    """Stop if any official file differs from the pinned commit."""
    for fname, expected in PINNED_SHA256.items():
        with open(os.path.join(strap_dir, fname), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != expected:
            raise RuntimeError(f"{fname} differs from the pinned commit {PINNED_COMMIT}: {got}")


def import_official(strap_dir, use_stubs=False):
    verify_official_files(strap_dir)
    if use_stubs:
        install_import_stubs()
    strap_dir = os.path.abspath(strap_dir)
    if strap_dir not in sys.path:
        sys.path.insert(0, strap_dir)
    import segment_split  # noqa: E402  (official, unchanged)
    import prioritization  # noqa: E402  (official, unchanged)
    return segment_split, prioritization


# --------------------------------------------------------------------------------------
# 2. Data structures
# --------------------------------------------------------------------------------------

@dataclass
class Recording:
    rec_id: str            # e.g. "<folder>/<execution>/piece_<j>"
    record_data: dict      # see the module docstring
    frame_log: list        # log id of every RAW frame, in time order
    logs: list             # the recording's log ids, in order of k


@dataclass
class SegInfo:
    # eligible segments (those that pass the official 1-second minimum), in time order:
    # (idx, start, end_exclusive, vec_as_returned_by_record2segments)
    elig: list
    # kept segments of the official WITHIN-recording reduction: idx -> (start, end) as returned
    # in record2segments' 'segments_reduced' (clipped if longer than kept_length)
    reduced: dict
    reduced_order: list    # idx of the kept segments, in the order record2segments returns them
    reduced_vec: dict      # idx -> tuple(vec) of the kept segments


def check_recording(rec):
    n_raw = len(rec.record_data["time"])
    if len(rec.frame_log) != n_raw:
        raise ValueError(f"{rec.rec_id}: len(frame_log)={len(rec.frame_log)} != len(time)={n_raw}")
    if len(rec.record_data["localization_msgs"]) != n_raw:
        raise ValueError(f"{rec.rec_id}: localization_msgs and time have different lengths")
    if len(rec.record_data["actor_vec"]) >= n_raw:
        raise ValueError(f"{rec.rec_id}: actor_vec must be the SMOOTHED vectors (shorter than time)")
    order = []
    for lg in rec.frame_log:
        if not order or order[-1] != lg:
            order.append(lg)
    if order != list(rec.logs):
        raise ValueError(f"{rec.rec_id}: frame_log is not the logs concatenated in order of k")


# --------------------------------------------------------------------------------------
# 3. Calls to the official functions
# --------------------------------------------------------------------------------------

def segment_recording(seg_mod, rec, kept_length=3):
    """Official record2segments on one recording. Independent of the input order, so cache it."""
    if len(rec.record_data["actor_vec"]) == 0:
        # too short to survive the official smoothing: no segment (the official split_record
        # would fail on an empty list); its logs end up in Tier C
        return SegInfo([], {}, [], {})
    sd = seg_mod.record2segments(rec.record_data, "obstacle", kept_length=kept_length)
    elig = [(idx, s, e, v) for (idx, (s, e)), v in zip(sd["segments"], sd["vecs"])]
    reduced, reduced_order, reduced_vec = {}, [], {}
    for (idx, (s, e)), v in zip(sd["segments_reduced"], sd["segments_vecs_reduced"]):
        reduced[idx] = (s, e)
        reduced_order.append(idx)
        reduced_vec[idx] = tuple(v)
    return SegInfo(elig, reduced, reduced_order, reduced_vec)


@contextlib.contextmanager
def _in_dir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def official_order(prio_mod, vecs, criterion, workdir):
    """Official prioritization on the pooled eligible segments.

    The official functions read 'data/record_data/original/{map}_segments.pickle' relative to
    the current directory, so the pooled list is written there under the name 'pool'.
    """
    target = os.path.join(workdir, "data", "record_data", "original")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "pool_segments.pickle"), "wb") as fh:
        pickle.dump({"obstacle": {"vecs": vecs}}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    fn = {"rsc": prio_mod.prioritize_by_rsc,
          "sc": prio_mod.prioritize_by_semantic_coverage}[criterion]
    with _in_dir(workdir), contextlib.redirect_stdout(io.StringIO()):
        order = fn("pool", "obstacle")
    return [int(p) for p in order]


def segment_scores(vecs, criterion):
    """The segment scores of the official functions, with the same arithmetic
    (prioritization.py, prioritize_by_rsc and prioritize_by_semantic_coverage).
    The official functions return only the order, and Tier B needs the scores themselves."""
    vec_total = np.sum(np.array(vecs), axis=0)
    if criterion == "rsc":
        vec_weight = []
        for num in vec_total:
            if num != 0:
                vec_weight.append(float(np.sum(vec_total)) / num)
            else:
                vec_weight.append(0)
        vec_weight_new = []
        for weight in vec_weight:
            vec_weight_new.append(weight / sum(vec_weight))
        vec_weight = np.array(vec_weight_new)
    elif criterion == "sc":
        vec_weight = np.array([1] * len(vec_total))
    else:
        raise ValueError(criterion)
    return [np.dot(vec_weight, np.array(vec)) for vec in vecs]


def check_scores_against_order(scores, order):
    """The official order must be a permutation and sorted by our recomputed scores."""
    if sorted(order) != list(range(len(scores))):
        raise RuntimeError("official order is not a permutation of the pooled segments")
    for a, b in zip(order, order[1:]):
        if scores[a] < scores[b]:
            raise RuntimeError("recomputed scores disagree with the official order")


# --------------------------------------------------------------------------------------
# 4. From segments to a ranking of every log
# --------------------------------------------------------------------------------------

def _logs_in(frame_log, start, stop):
    """Log ids of the frames start..stop-1, in time order, without repeats."""
    out = []
    for i in range(max(start, 0), min(stop, len(frame_log))):
        lg = frame_log[i]
        if not out or out[-1] != lg:
            if lg not in out:
                out.append(lg)
    return out


def rank_logs(recs, seginfo, rec_order, prio_mod, workdir, criterion="rsc",
              pooled_reduction=False, tail="score", kept_interval="closed"):
    """Rank every log of the dataset for one run.

    recs             list of Recording
    seginfo          list of SegInfo, seginfo[r] = segment_recording(seg_mod, recs[r])
    rec_order        this run's input order: a permutation of range(len(recs))
    criterion        'rsc' (main) or 'sc' (semantic-coverage variant)
    pooled_reduction False (main): only the official within-recording reduction.
                     True (cross-recording variant and the checks against the earlier runs):
                     additionally keep only the first segment of each vector across
                     recordings, in input order.
    tail             'score' (main): Tier B ranked by the best eligible-segment score.
                     'input' (checks against the earlier runs only): Tier B skipped, the rest
                     in input order.
    kept_interval    frames owned by a kept segment (start, end) from 'segments_reduced':
                     'closed' = start..end, 'half_open' = start..end-1. Fix it once with the
                     check against the earlier recording-level run, then keep it for every
                     configuration.

    Returns (ranking, info): ranking = list of log ids, info = per-log tier and score.
    """
    # input order of the logs: recordings in rec_order, logs in order of k
    all_logs, input_pos = [], {}
    for r in rec_order:
        for lg in recs[r].logs:
            if lg in input_pos:
                raise ValueError(f"log {lg} appears in two recordings")
            input_pos[lg] = len(all_logs)
            all_logs.append(lg)

    # pooled eligible segments, in input order (recordings in rec_order, segments in time order)
    pool, vecs = [], []
    for r in rec_order:
        for idx, s, e, v in seginfo[r].elig:
            pool.append((r, idx, s, e))
            vecs.append(v)
    if not pool:
        return list(all_logs), {lg: ("C", None) for lg in all_logs}

    order = official_order(prio_mod, vecs, criterion, workdir)
    scores = segment_scores(vecs, criterion)
    check_scores_against_order(scores, order)

    # kept segments
    if pooled_reduction:
        seen_vec, kept = set(), set()
        for r in rec_order:
            for idx in seginfo[r].reduced_order:
                v = seginfo[r].reduced_vec[idx]
                if v not in seen_vec:
                    seen_vec.add(v)
                    kept.add((r, idx))
    else:
        kept = {(r, idx) for r in rec_order for idx in seginfo[r].reduced_order}

    ranking, info = [], {}

    # Tier A: logs owned by kept segments, in the official order of the kept segments
    for p in order:
        r, idx, _, _ = pool[p]
        if (r, idx) not in kept:
            continue
        ks, ke = seginfo[r].reduced[idx]
        stop = ke + 1 if kept_interval == "closed" else ke
        for lg in _logs_in(recs[r].frame_log, ks, stop):
            if lg not in info:
                info[lg] = ("A", scores[p])
                ranking.append(lg)

    # Tier B: every other log covered by an eligible segment, by its best segment score
    if tail == "score":
        best = {}
        for p, (r, idx, s, e) in enumerate(pool):
            sc = scores[p]
            for lg in _logs_in(recs[r].frame_log, s, e):
                if lg not in best or sc > best[lg]:
                    best[lg] = sc
        rest = [lg for lg in all_logs if lg not in info and lg in best]
        rest.sort(key=lambda lg: (-best[lg], input_pos[lg]))  # ties: input order
        for lg in rest:
            info[lg] = ("B", best[lg])
            ranking.append(lg)
    elif tail != "input":
        raise ValueError(tail)

    # Tier C: the rest, in input order
    for lg in all_logs:
        if lg not in info:
            info[lg] = ("C", None)
            ranking.append(lg)

    if len(ranking) != len(all_logs) or set(ranking) != set(all_logs):
        raise RuntimeError("the ranking is not a permutation of the logs")
    return ranking, info


# --------------------------------------------------------------------------------------
# 5. Diagnostics
# --------------------------------------------------------------------------------------

def diagnostics(recs, seginfo, ranking, info, n_budget_5):
    """n_budget_5 = number of logs inspected at the 5% budget, as the authors' metric code counts it."""
    tiers = Counter(t for t, _ in info.values())
    b_scores = [s for t, s in info.values() if t == "B"]
    b_groups = Counter(b_scores)
    a_groups = Counter(s for t, s in info.values() if t == "A")
    # a position is decided by the input order if its log shares its tier and score with
    # another log, or if it is in Tier C
    order_decided = 0
    for t, s in info.values():
        if t == "C":
            order_decided += 1
        elif t == "A" and a_groups[s] > 1:
            order_decided += 1
        elif t == "B" and b_groups[s] > 1:
            order_decided += 1
    frame_vectors = set()
    for rec in recs:
        frame_vectors.update(tuple(v) for v in rec.record_data["actor_vec"])
    return {
        "recordings": len(recs),
        "distinct_frame_vectors": len(frame_vectors),
        "eligible_segments": sum(len(si.elig) for si in seginfo),
        "kept_segments_within_recording": sum(len(si.reduced_order) for si in seginfo),
        "tier_A_logs": tiers.get("A", 0),
        "tier_B_logs": tiers.get("B", 0),
        "tier_C_logs": tiers.get("C", 0),
        "distinct_scores_tier_A": len(a_groups),
        "distinct_scores_tier_B": len(b_groups),
        "largest_tie_group_tier_B": max(b_groups.values()) if b_groups else 0,
        "positions_decided_by_input_order_pct": 100.0 * order_decided / len(ranking),
        "tier_A_logs_in_first_5pct": sum(1 for lg in ranking[:n_budget_5] if info[lg][0] == "A"),
    }


# --------------------------------------------------------------------------------------
# 6. One configuration, 30 runs
# --------------------------------------------------------------------------------------

def run_configuration(recs, orders, seg_mod, prio_mod, workdir, kept_length=3, **kwargs):
    """orders: list of 30 input orders (permutations of range(len(recs))).
    Returns a list of (ranking, info) per run, plus the cached SegInfo."""
    for rec in recs:
        check_recording(rec)
    seginfo = [segment_recording(seg_mod, rec, kept_length) for rec in recs]
    runs = []
    for rec_order in orders:
        runs.append(rank_logs(recs, seginfo, rec_order, prio_mod, workdir, **kwargs))
    return runs, seginfo


# ======================================================================================
# 7. Repeated STRaP (the main configuration)
# ======================================================================================
#
# STRaP ranks only the segments that survive its duplicate reduction, one per distinct
# semantic vector. To rank every log, STRaP's two steps are applied repeatedly to the
# segments not yet ranked:
#
#   round 1   STRaP's own output: the official reduction rule (keep the first segment of
#             each vector, among segments whose official length is >= 1 s) over the whole
#             suite, ordered by the official prioritization;
#   round k   the same two steps on the segments not ranked in rounds 1..k-1;
#   until     every such segment is ranked.
#
# The rarity (or coverage) weights are computed once, by one official call over all
# eligible segments of the suite, exactly as in the single-pass setting. Inside a round,
# the members are ordered by their position in that official order, which is how the
# official evaluation orders the reduced segments. Each log takes the position of the
# first ranked segment that owns one of its frames. Logs owned by no ranked segment come
# last, in input order.

@dataclass
class SegInfoFull:
    # every eligible segment of the recording (official 1-second filter), in time order:
    # (idx, start, end_exclusive, vec, kept_start, kept_end, testable)
    #   kept_*   the interval record2segments stores in 'segments_kept' (clipped if longer
    #            than kept_length)
    #   testable True if the official reduction would accept it: length >= 1 s, with length
    #            computed by the official get_sequence_length exactly as in record2segments
    segs: list


def segment_recording_full(seg_mod, rec, kept_length=3):
    """Official record2segments on one recording, keeping every eligible segment."""
    if len(rec.record_data["actor_vec"]) == 0:
        return SegInfoFull([])
    sd = seg_mod.record2segments(rec.record_data, "obstacle", kept_length=kept_length)
    kept = {idx: (s, e) for idx, (s, e) in sd["segments_kept"]}
    lm = rec.record_data["localization_msgs"]
    segs = []
    for (idx, (s, e)), v in zip(sd["segments"], sd["vecs"]):
        length = seg_mod.get_sequence_length(lm[s].header.timestamp_sec,
                                             lm[e - 1].header.timestamp_sec)
        ks, ke = kept[idx]
        segs.append((idx, s, e, v, ks, ke, length >= 1))
    return SegInfoFull(segs)


def rank_logs_repeated(recs, seginfo, rec_order, prio_mod, workdir, criterion="rsc",
                       kept_interval="half_open"):
    """Rank every log of the dataset for one run with repeated STRaP.

    Returns (ranking, info, seg_ranking):
      ranking      list of log ids
      info         log id -> ("R", round, tuple(vec), score) for ranked logs,
                             ("C", None, None, None) for logs owned by no ranked segment
      seg_ranking  the ranked segments in order: (recording index, idx, round)
    """
    all_logs, input_pos = [], {}
    for r in rec_order:
        for lg in recs[r].logs:
            if lg in input_pos:
                raise ValueError(f"log {lg} appears in two recordings")
            input_pos[lg] = len(all_logs)
            all_logs.append(lg)

    # pooled eligible segments in input order (recordings in rec_order, segments in time order)
    pool, vecs = [], []
    for r in rec_order:
        for seg in seginfo[r].segs:
            pool.append((r, seg))
            vecs.append(seg[3])
    if not pool:
        return list(all_logs), {lg: ("C", None, None, None) for lg in all_logs}, []

    order = official_order(prio_mod, vecs, criterion, workdir)
    scores = segment_scores(vecs, criterion)
    check_scores_against_order(scores, order)
    rank_in_order = [0] * len(order)
    for pos, p in enumerate(order):
        rank_in_order[p] = pos

    # one queue per vector: the testable segments of that vector, in pool order
    queues = {}
    for p, (r, seg) in enumerate(pool):
        if seg[6]:
            queues.setdefault(tuple(seg[3]), []).append(p)

    # rounds: the k-th testable segment of every vector; inside a round, official order
    seg_ranking_p = []
    n_rounds = max((len(q) for q in queues.values()), default=0)
    for k in range(n_rounds):
        members = [q[k] for q in queues.values() if k < len(q)]
        members.sort(key=lambda p: rank_in_order[p])
        seg_ranking_p.extend((p, k + 1) for p in members)

    ranking, info = [], {}
    for p, rnd in seg_ranking_p:
        r, seg = pool[p]
        ks, ke = seg[4], seg[5]
        stop = ke + 1 if kept_interval == "closed" else ke
        for lg in _logs_in(recs[r].frame_log, ks, stop):
            if lg not in info:
                info[lg] = ("R", rnd, tuple(seg[3]), scores[p])
                ranking.append(lg)
    for lg in all_logs:
        if lg not in info:
            info[lg] = ("C", None, None, None)
            ranking.append(lg)

    if len(ranking) != len(all_logs) or set(ranking) != set(all_logs):
        raise RuntimeError("the ranking is not a permutation of the logs")
    seg_ranking = [(pool[p][0], pool[p][1][0], rnd) for p, rnd in seg_ranking_p]
    return ranking, info, seg_ranking


def diagnostics_repeated(recs, seginfo, ranking, info, seg_ranking, n_budget_5):
    """Diagnostics of one repeated-STRaP run."""
    testable = [seg for si in seginfo for seg in si.segs if seg[6]]
    class_size = Counter(tuple(seg[3]) for seg in testable)
    round1_logs = sum(1 for t, rnd, _, _ in info.values() if t == "R" and rnd == 1)
    ranked = sum(1 for t, _, _, _ in info.values() if t == "R")
    first5 = ranking[:n_budget_5]
    first5_by_vec = Counter(info[lg][2] for lg in first5 if info[lg][0] == "R")
    rounds_in_first5 = max((info[lg][1] for lg in first5 if info[lg][0] == "R"), default=0)
    # the log at a position is chosen by input order whenever its vector has more than one
    # testable segment (STRaP keeps the first occurrence)
    chosen_by_order = sum(1 for lg in ranking
                          if info[lg][0] == "C" or class_size[info[lg][2]] > 1)
    durations = []
    for rec in recs:
        t = rec.record_data["time"]
        if len(t) > 1:
            durations.append(t[-1] - t[0])
    return {
        "recordings": len(recs),
        "median_recording_duration_s": float(np.median(durations)) if durations else 0.0,
        "eligible_segments": sum(len(si.segs) for si in seginfo),
        "testable_segments": len(testable),
        "distinct_vectors_testable": len(class_size),
        "largest_class_segments": max(class_size.values()) if class_size else 0,
        "rounds": max(class_size.values()) if class_size else 0,
        "round1_segments": sum(1 for _, _, rnd in seg_ranking if rnd == 1),
        "round1_logs": round1_logs,
        "ranked_logs": ranked,
        "unranked_logs": len(ranking) - ranked,
        "rounds_within_first_5pct": rounds_in_first5,
        "positions_chosen_by_input_order_pct": 100.0 * chosen_by_order / len(ranking),
        "class_sizes": json.dumps({"".join(map(str, k)): v for k, v in sorted(class_size.items())}),
        "first_5pct_by_vector": json.dumps({"".join(map(str, k)): v
                                            for k, v in sorted(first5_by_vec.items())}),
    }


def run_configuration_repeated(recs, orders, seg_mod, prio_mod, workdir, kept_length=3,
                               criterion="rsc", kept_interval="half_open"):
    for rec in recs:
        check_recording(rec)
    seginfo = [segment_recording_full(seg_mod, rec, kept_length) for rec in recs]
    runs = []
    for rec_order in orders:
        runs.append(rank_logs_repeated(recs, seginfo, rec_order, prio_mod, workdir,
                                       criterion=criterion, kept_interval=kept_interval))
    return runs, seginfo
