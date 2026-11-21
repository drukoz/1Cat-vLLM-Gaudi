# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tools/flash_next/replay/place_workers.py: the plan (shares, mains, siblings) and one pass of the
placement on fake threads."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("place_workers", ROOT / "tools/flash_next/replay/place_workers.py")
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)

BUILD = {18, 37, 56, 75, 94, 113, 132, 151}
JOB_CPUS = [c for c in range(152) if c not in BUILD]  # an all-card job's 144 CPUs


def test_sibling_rules():
    assert pw.sibling(1, "a+N/2", 152) == 77 and pw.sibling(77, "a+N/2", 152) == 1
    assert pw.sibling(20, "a^1", 152) == 21
    assert pw.sibling(5, "", 152) is None


def test_shares_match_the_pin_flag():
    parts = pw.shares(JOB_CPUS, 8)
    assert [p[0] for p in parts] == [19 * r for r in range(8)]
    assert all(len(p) == 18 for p in parts)


def _check(plan, rule):
    mains = list(plan["main"].values())
    assert len(mains) == 8 and len(set(mains)) == 8
    for c in mains:
        sib = pw.sibling(c, rule, 152)
        assert sib not in mains
        assert c in plan["reserved"] and (sib is None or sib in plan["reserved"])
    for rank, cpus in plan["helpers"].items():
        assert plan["main"][rank] not in cpus
        assert not set(cpus) & set(plan["reserved"])


def test_plan_half_rule_keeps_mains_off_each_others_siblings():
    plan = pw.plan(range(8), JOB_CPUS, "a+N/2", 152)
    _check(plan, "a+N/2")
    # rank 0's main would be 1 (sibling 77, rank 4's second CPU): rank 4 must take another one
    assert plan["main"][0] == 1 and plan["main"][4] != 77


def test_plan_xor_rule_and_unknown_rule():
    _check(pw.plan(range(8), JOB_CPUS, "a^1", 152), "a^1")
    plan = pw.plan(range(8), JOB_CPUS, "", 152)
    assert sorted(plan["reserved"]) == sorted(plan["main"].values())


def test_apply_moves_threads():
    plan = pw.plan([0, 1], [0, 1, 2, 3, 4, 5, 6, 7], "a^1", 8)
    # rank 0: share 0-3, main 1 (sibling 0); rank 1: share 4-7, main 5 (sibling 4)
    assert plan["main"] == {0: 1, 1: 5} and plan["reserved"] == [0, 1, 4, 5]
    aff = {10: {0, 1, 2, 3}, 11: {0, 1, 2, 3}, 20: {4, 5, 6, 7}, 21: {4, 5, 6, 7}, 30: set(range(8))}
    threads = {10: [10, 11], 20: [20, 21], 30: [30]}
    calls = []

    def setaffinity(tid, cpus):
        calls.append(tid)
        aff[tid] = set(cpus)

    counts = pw.apply(plan, {0: 10, 1: 20}, setaffinity=setaffinity, getaffinity=lambda t: aff[t],
                      tids=lambda pid: threads[pid], pids=lambda: [10, 20, 30], allowed=range(8))
    assert aff[10] == {1} and aff[11] == {2, 3}
    assert aff[20] == {5} and aff[21] == {6, 7}
    assert aff[30] == {2, 3, 6, 7}
    assert counts == {"moved": 5, "failed": 0}
    # a second pass changes nothing
    assert pw.apply(plan, {0: 10, 1: 20}, setaffinity=setaffinity, getaffinity=lambda t: aff[t],
                    tids=lambda pid: threads[pid], pids=lambda: [10, 20, 30], allowed=range(8))["moved"] == 0
