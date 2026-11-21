# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Flash-Next MTP draft-rank logging (M3b).

Covers ``DraftRankStats`` (the rank histogram of the target token among the
drafter's top-4 at the first rejected position) and the proposer's gate: with
``VLLM_FLASH_NEXT_DRAFT_RANK_LOG`` unset the proposer never reaches ``topk``
(``draft_rank_stats is None``, every ``.topk(`` behind the gate).
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps as steps  # noqa: E402
from vllm_gaudi.v1.spec_decode.flash_next_mtp_steps import (DRAFT_RANK_LOG_ENV, DRAFT_RANK_TOP, DraftRankStats,
                                                            draft_rank_log_enabled)  # noqa: E402


class _HabanaStubFinder:
    """Answer every habana_frameworks import with a MagicMock module.

    ``hpu_flash_next_mtp.py`` imports the HPU runtime at module level; only
    ``_init_draft_rank_log`` (pure python) is exercised here, so the runtime
    modules are stubbed to let the import succeed on CPU.
    """

    class _Loader:

        def create_module(self, spec):
            return MagicMock(name=spec.name)

        def exec_module(self, module):
            pass

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "habana_frameworks":
            return importlib.machinery.ModuleSpec(fullname, self._Loader(), is_package=True)
        return None


if not importlib.util.find_spec("habana_frameworks"):
    sys.meta_path.insert(0, _HabanaStubFinder())

from vllm_gaudi.v1.spec_decode.hpu_flash_next_mtp import HpuFlashNextMtpProposer  # noqa: E402

PROPOSER_SRC = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "spec_decode" / "hpu_flash_next_mtp.py"


def _row(top4):
    """[1, 1, DRAFT_RANK_TOP] top-4 tensor for a single k == 1 draft."""
    assert len(top4) == DRAFT_RANK_TOP
    return torch.tensor([top4]).unsqueeze(0)


def test_gate_reads_the_env(monkeypatch):
    monkeypatch.delenv(DRAFT_RANK_LOG_ENV, raising=False)
    assert not draft_rank_log_enabled()
    for value in ("1", "true", "TRUE", " 1 "):
        monkeypatch.setenv(DRAFT_RANK_LOG_ENV, value)
        assert draft_rank_log_enabled()
    for value in ("0", "", "yes"):
        monkeypatch.setenv(DRAFT_RANK_LOG_ENV, value)
        assert not draft_rank_log_enabled()


def test_counts_the_target_rank_one_to_four_at_the_first_rejected_position():
    stats = DraftRankStats()
    for i, target in enumerate([10, 11, 12, 13]):
        stats.remember([f"r{i}"], _row([10, 11, 12, 13]))
        # a k == 1 draft rejected at index 0: ids == [target] (the target's own token)
        stats.observe([f"r{i}"], [[target]])
    assert stats.counts == [1, 1, 1, 1, 0]
    assert stats.all_accepted == 0


def test_counts_beyond_four_when_the_target_is_not_a_candidate():
    stats = DraftRankStats()
    stats.remember(["r"], _row([10, 11, 12, 13]))
    stats.observe(["r"], [[99]])
    assert stats.counts == [0, 0, 0, 0, 1]


def test_all_accepted_when_the_sampled_stream_has_k_plus_one_tokens():
    stats = DraftRankStats()
    # k = 2 drafts, ids = 2 accepted drafts + the bonus token => L == k + 1
    stats.remember(["r"], torch.tensor([[[10, 11, 12, 13], [20, 21, 22, 23]]]))
    stats.observe(["r"], [[10, 20, 55]])
    assert stats.all_accepted == 1
    assert sum(stats.counts) == 0


def test_rejection_uses_the_row_of_the_first_rejected_draft_index():
    stats = DraftRankStats()
    # k = 3 drafts, ids = 1 accepted draft + the bonus: the draft at index 1 was rejected
    stats.remember(["r"], torch.tensor([[[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]]))
    stats.observe(["r"], [[10, 22]])
    assert stats.counts == [0, 0, 1, 0, 0]  # 22 is the rank-3 candidate of row 1


def test_requests_without_a_remembered_row_are_ignored():
    stats = DraftRankStats()
    stats.observe(["ghost"], [[7]])
    stats.observe(["ghost"], [[7, 8]])
    assert stats.counts == [0, 0, 0, 0, 0]
    assert stats.all_accepted == 0
    assert stats.observations == 0


def test_observe_pops_the_row_and_a_second_observe_does_nothing():
    stats = DraftRankStats()
    stats.remember(["r"], _row([10, 11, 12, 13]))
    stats.observe(["r"], [[10]])
    stats.observe(["r"], [[10]])
    assert stats.counts == [1, 0, 0, 0, 0]
    assert stats.observations == 1


def test_empty_sampled_ids_are_skipped_and_drop_the_row():
    stats = DraftRankStats()
    stats.remember(["r"], _row([10, 11, 12, 13]))
    stats.observe(["r"], [[]])
    assert stats.counts == [0, 0, 0, 0, 0]
    assert stats.observations == 0
    stats.observe(["r"], [[10]])  # the row was popped: nothing is counted
    assert stats.observations == 0


def test_reordered_req_ids_between_remember_and_observe():
    stats = DraftRankStats()
    stats.remember(["a", "b"], torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]))
    # the batch order flipped: b's target 6 is the rank-2 candidate of b's row,
    # a's target 99 is not a candidate of a's row
    stats.observe(["b", "a"], [[6], [99]])
    assert stats.counts == [0, 1, 0, 0, 1]


def test_remember_rejects_a_length_mismatch():
    stats = DraftRankStats()
    with pytest.raises(ValueError):
        stats.remember(["a", "b"], torch.tensor([[[1, 2, 3, 4]]]))
    with pytest.raises(ValueError):
        stats.remember(["a"], torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]))
    with pytest.raises(ValueError):
        stats.remember([], torch.tensor([[[1, 2, 3, 4]]]))


def test_summary_line_format():
    stats = DraftRankStats()
    stats.counts = [1, 2, 3, 4, 5]
    stats.all_accepted = 6
    assert stats.summary_line() == ("Flash-Next draft rank at the first rejected position: "
                                    "rank1=1 rank2=2 rank3=3 rank4=4 beyond4=5 all_accepted=6")


def test_summary_is_logged_every_log_every_observations(monkeypatch):
    logged = []
    monkeypatch.setattr(steps, "logger", MagicMock(info=lambda line: logged.append(line)))
    stats = DraftRankStats(log_every=2)
    for i in range(5):
        stats.remember([f"r{i}"], _row([10, 11, 12, 13]))
        stats.observe([f"r{i}"], [[99]])
        assert len(logged) == (i + 1) // 2
    assert logged == [
        "Flash-Next draft rank at the first rejected position: rank1=0 rank2=0 rank3=0 rank4=0 beyond4=2 "
        "all_accepted=0",
        "Flash-Next draft rank at the first rejected position: rank1=0 rank2=0 rank3=0 rank4=0 beyond4=4 "
        "all_accepted=0",
    ]


def _bare_proposer(monkeypatch, env):
    if env is None:
        monkeypatch.delenv(DRAFT_RANK_LOG_ENV, raising=False)
    else:
        monkeypatch.setenv(DRAFT_RANK_LOG_ENV, env)
    proposer = object.__new__(HpuFlashNextMtpProposer)
    proposer._init_draft_rank_log()
    return proposer


def test_proposer_gate_off_by_default(monkeypatch):
    proposer = _bare_proposer(monkeypatch, None)
    assert proposer.draft_rank_stats is None
    assert proposer.last_draft_top_ids is None


def test_proposer_gate_on_builds_the_stats(monkeypatch):
    proposer = _bare_proposer(monkeypatch, "1")
    assert isinstance(proposer.draft_rank_stats, DraftRankStats)
    assert proposer.last_draft_top_ids is None


def test_proposer_only_reaches_topk_under_the_gate():
    # The gate-off path never reaches topk: the proposer's one .topk( call (``_draft_tokens``, which the k == 1
    # exit, draft step 0 and the loop steps all go through since the collective cuts) is the else branch of a
    # `draft_rank_stats is None` test on its own line, and its rows are kept only when they are not None.
    lines = PROPOSER_SRC.read_text().splitlines()
    topk_lines = [i for i, line in enumerate(lines) if ".topk(" in line]
    assert len(topk_lines) == 1
    for i in topk_lines:
        assert "None if self.draft_rank_stats is None else" in lines[i]
    # draft step 0 (and the k == 1 exit), the loop, and the LM head of the one-graph loop (VLLM_HPU_MTP_DRAFT_GRAPH=1,
    # which _draft_graph_enabled admits only while draft_rank_stats is None: its rows are dropped, never top-4 rows)
    assert sum("self._draft_tokens(" in line for line in lines) == 3
    gate = "\n".join(lines[next(i for i, line in enumerate(lines) if "def _draft_graph_enabled(" in line):][:6])
    assert "self.draft_rank_stats is None" in gate
    assert sum("return self._draft_tokens(hidden_states)[0]" in line for line in lines) == 1
    assert sum("if top_ids is not None:" in line for line in lines) == 3
    # the collected rows are stacked only under the gate
    assert "torch.stack(draft_top_ids, dim=1)" in "\n".join(lines)
