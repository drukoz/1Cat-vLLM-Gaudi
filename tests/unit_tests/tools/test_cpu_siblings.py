# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tools/flash_next/replay/cpu_siblings.py: the partner rules and the rule inference (no C compiler,
no loop runs)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("cpu_siblings", ROOT / "tools/flash_next/replay/cpu_siblings.py")
sib = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sib)


def test_partner_rules_on_152_vcpus():
    rules = sib.candidate_rules(152)
    assert rules["a+N/2"](0) == 76 and rules["a+N/2"](75) == 151 and rules["a+N/2"](76) == 0
    assert rules["a+N/2"](151) == 75
    assert rules["a^1"](18) == 19 and rules["a^1"](19) == 18
    assert rules["a+1"](5) == 6 and rules["a-1"](5) == 4


def test_slowdown():
    assert sib.slowdown(100.0, 60.0) == 0.4
    assert sib.slowdown(100.0, 105.0) == 0.0
    assert sib.slowdown(0.0, 10.0) == 0.0


def _r(rule, a, b, sa, sb):
    return {"rule": rule, "a": a, "b": b, "slow_a": sa, "slow_b": sb}


def test_infer_rule_one_fits():
    res = [_r("a+N/2", 0, 76, 0.42, 0.40), _r("a^1", 0, 1, 0.02, 0.01), _r("a+N/2", 10, 86, 0.38, 0.41),
           _r("a^1", 10, 11, 0.03, 0.0)]
    rule, explained = sib.infer_rule(res, 0.2)
    assert rule == "a+N/2"
    assert explained == {"a+N/2": 2}


def test_infer_rule_none_when_a_pair_is_weak():
    res = [_r("a+N/2", 0, 76, 0.42, 0.40), _r("a+N/2", 10, 86, 0.05, 0.04), _r("a^1", 0, 1, 0.0, 0.0)]
    rule, explained = sib.infer_rule(res, 0.2)
    assert rule is None
    assert explained == {"a+N/2": 1}


def test_infer_rule_none_when_one_side_only():
    # a slowdown on one side only (a noisy neighbour) is not a sibling pair
    res = [_r("a^1", 0, 1, 0.5, 0.01)]
    assert sib.infer_rule(res, 0.2)[0] is None


def test_infer_rule_relative_on_job_00193():
    # 00193: one dependent chain per loop; a^1 pairs 0.07-0.09, everything else 0.00-0.01 (a+1 = a^1 for even a)
    res = [_r("a+N/2", 0, 76, 0.01, 0.0), _r("a+N/2", 12, 88, 0.018, 0.003), _r("a^1", 0, 1, 0.083, 0.085),
           _r("a^1", 12, 13, 0.103, 0.088), _r("a^1", 25, 24, 0.101, 0.086), _r("a+1", 0, 1, 0.083, 0.084),
           _r("a+1", 25, 26, 0.068, 0.013), _r("a-1", 12, 11, 0.003, 0.0), _r("a-1", 25, 24, 0.13, 0.118)]
    rule, weakest = sib.infer_rule_relative(res)
    assert rule == "a^1"
    assert weakest["a^1"] == 0.083


def test_infer_rule_relative_none_when_flat():
    res = [_r("a+N/2", 0, 76, 0.01, 0.0), _r("a^1", 0, 1, 0.02, 0.01), _r("a+1", 0, 1, 0.02, 0.01)]
    assert sib.infer_rule_relative(res)[0] is None
