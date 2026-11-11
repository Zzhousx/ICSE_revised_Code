"""
Tests of strapos_fullrank.py on hand-made recordings, with the official STRaP functions.

Run:  python3 test_strapos_fullrank.py /path/to/official/STRAP [--stubs]
Use --stubs only if the Apollo/cyber dependencies of the official files are not importable.
Every test prints PASS or raises.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import strapos_fullrank as fr

T0 = 1600000000.0          # epoch seconds of the first raw frame
DT = 0.5                   # DeepScenario sampling step

VA = [1, 0, 0, 0, 0, 0, 0, 0]   # vehicle present
VB = [1, 0, 1, 0, 0, 0, 0, 0]   # vehicle + pedestrian present
VC = [1, 0, 0, 0, 0, 0, 1, 0]   # vehicle + obstacle heading opposite


def make_recording(rec_id, logs_frames, smoothed):
    """logs_frames: list of (log_id, number_of_raw_frames); smoothed: the smoothed actor_vec."""
    frame_log = [lg for lg, n in logs_frames for _ in range(n)]
    n_raw = len(frame_log)
    msgs = [SimpleNamespace(header=SimpleNamespace(timestamp_sec=T0 + DT * i)) for i in range(n_raw)]
    record_data = {
        "actor_vec": [list(v) for v in smoothed],
        "time": [DT * i for i in range(n_raw)],
        "localization_msgs": msgs,
    }
    return fr.Recording(rec_id, record_data, frame_log, [lg for lg, _ in logs_frames])


def toy_dataset():
    # R0: logs L0..L2 (6 raw frames each); smoothed frames: 5 x VA, 5 x VB, 5 x VA
    r0 = make_recording("R0", [("L0", 6), ("L1", 6), ("L2", 6)], [VA] * 5 + [VB] * 5 + [VA] * 5)
    # R1: logs L3, L4 (6 raw frames each); smoothed frames: 4 x VC, 5 x VA
    r1 = make_recording("R1", [("L3", 6), ("L4", 6)], [VC] * 4 + [VA] * 5)
    # R2: logs L5, L6 (10 raw frames each); smoothed frames: 17 x VB -> one segment of 8 s, clipped to 3 s
    r2 = make_recording("R2", [("L5", 10), ("L6", 10)], [VB] * 17)
    return [r0, r1, r2]


def main():
    strap_dir = sys.argv[1]
    seg_mod, prio_mod = fr.import_official(strap_dir, use_stubs="--stubs" in sys.argv)
    recs = toy_dataset()
    for rec in recs:
        fr.check_recording(rec)
    seginfo = [fr.segment_recording(seg_mod, rec, kept_length=3) for rec in recs]

    # --- official segmentation on the toy data
    assert [(i, s, e) for i, s, e, _ in seginfo[0].elig] == [(0, 0, 5), (1, 5, 10), (2, 10, 15)]
    assert seginfo[0].reduced_order == [0, 1]                       # second VA segment removed
    assert seginfo[1].reduced_order == [0, 1]
    assert seginfo[2].reduced == {0: (0, 6)}                        # clipped to 3 s
    print("PASS official segmentation, within-recording reduction and clipping")

    with tempfile.TemporaryDirectory() as wd:
        for rec_order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
            # --- main configuration
            ranking, info = fr.rank_logs(recs, seginfo, rec_order, prio_mod, wd)
            assert sorted(ranking) == sorted(f"L{i}" for i in range(7))
            tiers = {lg: t for lg, (t, _) in info.items()}
            # kept segments: R0 VA(0-5), R0 VB(5-10), R1 VC(0-4), R1 VA(4-9), R2 VB(0-6)
            assert tiers == {"L0": "A", "L1": "A", "L2": "B", "L3": "A", "L4": "A",
                             "L5": "A", "L6": "B"}, tiers
            # Tier A before Tier B before Tier C, and scores non-increasing inside A and B
            seq = [info[lg] for lg in ranking]
            assert [t for t, _ in seq] == sorted((t for t, _ in seq))
            for tier in ("A", "B"):
                sc = [s for t, s in seq if t == tier]
                assert all(a >= b for a, b in zip(sc, sc[1:]))
            # L6 is covered only by the unclipped part of R2's segment -> best score = VB
            assert info["L6"][1] == info["L5"][1]
            # L2 is covered by R0's removed VA segment (frames 10-14) -> score of VA
            assert info["L2"][1] < info["L6"][1]

            # --- setting of the earlier runs: pooled reduction, rest in input order
            ranking_p, info_p = fr.rank_logs(recs, seginfo, rec_order, prio_mod, wd,
                                             pooled_reduction=True, tail="input")
            tiers_p = {lg: t for lg, (t, _) in info_p.items()}
            n_a = sum(1 for t in tiers_p.values() if t == "A")
            assert n_a < len(ranking_p)
            # the C tier is in input order
            c_logs = [lg for lg in ranking_p if tiers_p[lg] == "C"]
            all_in_order = [lg for r in rec_order for lg in recs[r].logs]
            assert c_logs == [lg for lg in all_in_order if lg in c_logs]

            # --- semantic coverage criterion also runs
            fr.rank_logs(recs, seginfo, rec_order, prio_mod, wd, criterion="sc")

            d = fr.diagnostics(recs, seginfo, ranking, info, n_budget_5=1)
            assert d["tier_A_logs"] + d["tier_B_logs"] + d["tier_C_logs"] == 7
        print("PASS tiers, ordering, clipping ownership, setting of the earlier runs, SC criterion")
        print("diagnostics example:", d)

        test_repeated(seg_mod, prio_mod, wd)


def per_log_toy():
    """Seven logs, each its own recording (6-8 raw frames at 0.5 s)."""
    return [
        make_recording("L0", [("L0", 8)], [VA] * 5),
        make_recording("L1", [("L1", 8)], [VA] * 5),
        make_recording("L2", [("L2", 8)], [VB] * 5),
        make_recording("L3", [("L3", 8)], [VC] * 5),
        # VA part lasts 0.5 s by the official reduction rule (not testable), VB part 1.0 s
        make_recording("L4", [("L4", 8)], [VA] * 2 + [VB] * 3),
        make_recording("L5", [("L5", 8)], [VB] * 5),
        # too short: no segment passes the official 1-second filter
        make_recording("L6", [("L6", 4)], [VA] * 1),
    ]


def test_repeated(seg_mod, prio_mod, wd):
    recs = per_log_toy()
    for rec in recs:
        fr.check_recording(rec)
    full = [fr.segment_recording_full(seg_mod, rec, kept_length=3) for rec in recs]
    assert full[6].segs == []
    assert [(s[0], s[1], s[2], s[6]) for s in full[4].segs] == [(0, 0, 2, False), (1, 2, 5, True)]

    # expected rankings worked out by hand (rarity: VC > VB > VA)
    expected = {
        (0, 1, 2, 3, 4, 5, 6): ["L3", "L2", "L0", "L4", "L1", "L5", "L6"],
        (6, 5, 4, 3, 2, 1, 0): ["L3", "L5", "L1", "L4", "L0", "L2", "L6"],
    }
    for order, exp in expected.items():
        ranking, info, seg_ranking = fr.rank_logs_repeated(recs, full, list(order), prio_mod, wd)
        assert ranking == exp, (order, ranking)
        rounds = {lg: info[lg][1] for lg in ranking}
        assert [rounds[lg] for lg in exp[:-1]] == [1, 1, 1, 2, 2, 3] and info["L6"][0] == "C"
        # at most one segment per vector in every round, scores non-increasing inside a round
        by_round = {}
        for r, idx, rnd in seg_ranking:
            vec = tuple(next(s[3] for s in full[r].segs if s[0] == idx))
            by_round.setdefault(rnd, []).append(vec)
        assert all(len(v) == len(set(v)) for v in by_round.values())

        # round 1 is STRaP's own output over the suite: the same logs, in the same order, as
        # the kept segments of the single-pass setting with suite-level reduction
        seginfo = [fr.segment_recording(seg_mod, rec, kept_length=3) for rec in recs]
        r1, i1 = fr.rank_logs(recs, seginfo, list(order), prio_mod, wd, pooled_reduction=True,
                              tail="input", kept_interval="half_open")
        tier_a = [lg for lg in r1 if i1[lg][0] == "A"]
        assert ranking[:len(tier_a)] == tier_a
        assert all(info[lg][1] == 1 for lg in tier_a)

        d = fr.diagnostics_repeated(recs, full, ranking, info, seg_ranking, n_budget_5=3)
        assert d["round1_logs"] == 3 and d["ranked_logs"] == 6 and d["unranked_logs"] == 1
        assert d["rounds"] == 3 and d["distinct_vectors_testable"] == 3

    # one recording: round 1 equals the official within-recording reduction exactly
    one = make_recording("R", [("A", 6), ("B", 6), ("C", 6), ("D", 6)],
                         [VA] * 4 + [VB] * 4 + [VA] * 4 + [VC] * 4 + [VB] * 4)
    fr.check_recording(one)
    sd = seg_mod.record2segments(one.record_data, "obstacle", kept_length=3)
    official_round1 = {idx for idx, _ in sd["segments_reduced"]}
    full1 = [fr.segment_recording_full(seg_mod, one, kept_length=3)]
    _, _, seg_ranking = fr.rank_logs_repeated([one], full1, [0], prio_mod, wd)
    assert {idx for _, idx, rnd in seg_ranking if rnd == 1} == official_round1 == {0, 1, 3}
    assert [idx for _, idx, rnd in seg_ranking if rnd == 2] == [4, 2]   # VB before VA

    # joined recording with one long segment: only the logs inside STRaP's clipped 3 s are
    # owned; the rest of the drive stays unranked (reported as unranked logs)
    long_rec = make_recording("J", [("J1", 10), ("J2", 10)], [VB] * 17)
    full2 = [fr.segment_recording_full(seg_mod, long_rec, kept_length=3)]
    ranking, info, _ = fr.rank_logs_repeated([long_rec], full2, [0], prio_mod, wd)
    assert info["J1"][0] == "R" and info["J2"][0] == "C"
    print("PASS repeated STRaP: rounds, round 1 = STRaP's own output, ownership, diagnostics")


if __name__ == "__main__":
    main()
