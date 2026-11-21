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
