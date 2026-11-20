# SPDX-License-Identifier: Apache-2.0
"""The collectives of a Flash-Next step and the cuts that remove them, on 1/2/4/8 gloo CPU ranks.

``collective_ranks.collective_rank`` builds the 8-head tiny model sharded at TP<n> + EP<n> through vLLM's real
loader, on the plugin's MoE path and with the plugin's ``HpuCommunicator`` on the tensor-parallel group, and
records every collective of one prompt chunk, one decode step, one speculative verification step and the
drafter's forwards. ``chain`` is a greedy stream: a prefill, six single-token decode steps, then one
speculative step (k drafts and their verification). The model is bf16, as served; the tiny model's logits are
flat, so its arg max is full of exact ties, which is what the token comparisons need.

Each cut is one flag (default 0, ``vllm_gaudi/envs.py``); a test runs the same stream with the flag off and
on and compares the tokens and the bits of the hidden states (``docs/design/flash-next/collectives.md``).
"""

import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.unit_tests.models.flash_next.collective_ranks import collective_rank  # noqa: E402

from harness import run_ranks  # noqa: E402

from tools.flash_next.collective_count import classify, summarize  # noqa: E402

TP_SIZES = [1, 2, 4, 8]
K = 3
TINY_LAYERS = 4
PHASES = ("prefill", "decode", "verify", "draft", "chain")
LOGITS = {"VLLM_HPU_VOCAB_PARALLEL_LOGITS": "1"}
CANDIDATES = {"VLLM_HPU_VOCAB_PARALLEL_LOGITS": "2"}
EMBEDDING = {"VLLM_HPU_REPLICATED_EMBEDDING": "1"}
MTP_FC = {"VLLM_HPU_MTP_REPLICATED_FC": "1"}
_RUN_CACHE: dict[tuple, list] = {}


def _ranks(checkpoint: Path, tp: int, env: dict, **extra) -> list:
    key = (tp, tuple(sorted(env.items())), tuple(sorted(extra.items())))
    if key not in _RUN_CACHE:
        opts = {"env": dict(env), "k": K, "phases": PHASES, **extra}
        _RUN_CACHE[key] = run_ranks(tp, collective_rank, str(checkpoint), tp, opts, timeout_s=900)
    return _RUN_CACHE[key]


def _kinds(result: dict, phase: str) -> list[str]:
    return [classify(e) for e in result["entries"] if e["phase"] == phase]


def _entries(result: dict, phase: str, kind: str) -> list[dict]:
    return [e for e in result["entries"] if e["phase"] == phase and classify(e) == kind]


def _same_stream(a: dict, b: dict, tp: int, what: str) -> None:
    """The chain of run ``a`` and of run ``b``: the same tokens, drafts and verified tokens, the same hidden bits."""
    assert a["chain"]["tokens"] == b["chain"]["tokens"], f"TP{tp} {what}: decode tokens differ"
    assert a["chain"]["drafts"] == b["chain"]["drafts"], f"TP{tp} {what}: drafted tokens differ"
    assert a["chain"]["verified"] == b["chain"]["verified"], f"TP{tp} {what}: verified tokens differ"
    np.testing.assert_array_equal(a["chain"]["hidden"], b["chain"]["hidden"], err_msg=f"TP{tp} {what}: hidden")
    np.testing.assert_array_equal(a["chain"]["verify_hidden"], b["chain"]["verify_hidden"],
                                  err_msg=f"TP{tp} {what}: verification hidden")


# -- step 1: the count --------------------------------------------------------------------------------------
def test_baseline_count_at_tp8(tiny_tp8_checkpoint):
    """Flags off: a target step is embedding + PLE table + (mixer, MoE) per layer + the logits all-gather."""
    results = _ranks(tiny_tp8_checkpoint, 8, {})
    target = ["embedding", "mixer", "moe", "ple_table", "mixer", "moe", "mixer", "moe", "mixer", "moe", "logits"]
    for phase in ("prefill", "decode", "verify"):
        assert _kinds(results[0], phase) == target, phase
    # the drafter: six collectives for every drafted token
    assert _kinds(results[0], "draft") == ["embedding", "mtp_fc", "mtp_fc", "mixer", "moe", "logits"] * K
    summary = summarize(results, real_layers=48, k=K)
    for phase in ("prefill", "decode", "verify"):
        assert summary["phases"][phase]["real_count"] == 99
        assert summary["phases"][phase]["real_counts"] == {
            "embedding": 1, "mixer": 48, "moe": 48, "ple_table": 1, "logits": 1}
    assert summary["phases"]["draft"]["real_count"] == 6 * K
    layout = results[0]["layout"]
    assert layout["hpu_communicator"] and layout["embed_class"] == "HPUVocabParallelEmbedding"
    assert (layout["ep_size"], layout["local_experts"], layout["embed_tp_size"]) == (8, 2, 8)


def test_baseline_payloads_at_tp8(tiny_tp8_checkpoint):
    """Who sends what: one rank owns an embedding row, every rank contributes to a block output and to the logits."""
    results = _ranks(tiny_tp8_checkpoint, 8, {})
    summary = summarize(results, real_layers=48, k=K)
    decode = {row["kind"]: row for row in summary["phases"]["decode"]["collectives"]}
    assert decode["embedding"]["ranks_with_payload"] == 1  # the other seven ranks send zeros
    for kind in ("mixer", "moe", "logits"):
        assert decode[kind]["ranks_with_payload"] == 8 and not decode[kind]["replicated"], kind
    # the real geometry: 2560 bf16 values per row for the embedding, a block output and the 16 x 160 PLE rows
    for kind in ("embedding", "mixer", "moe", "ple_table"):
        assert decode[kind]["sent_bytes_per_row"] == decode[kind]["held_bytes_per_row"] == 5120, kind
    assert decode["logits"]["sent_bytes_per_row"] == 31040 * 2
    assert decode["logits"]["held_bytes_per_row"] == 248320 * 2  # 497 KB per row on every rank
    verify = summary["phases"]["verify"]["collectives"]
    assert verify[-1]["tiny_shape"] == [1 + K, 31040]  # every verification row gets its logits gathered


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_collective_order_is_the_same_on_every_rank(tp, tiny_tp8_checkpoint):
    for env in ({}, {**LOGITS, **EMBEDDING, **MTP_FC}):
        results = _ranks(tiny_tp8_checkpoint, tp, env)
        first = [(e["phase"], e["method"], e["shape"], e["dtype"], e["site"]) for e in results[0]["entries"]]
        assert first, "no collective recorded"
        for rank, result in enumerate(results):
            got = [(e["phase"], e["method"], e["shape"], e["dtype"], e["site"]) for e in result["entries"]]
            assert got == first, f"TP{tp} rank {rank} {env}"


def test_tp1_has_no_collective(tiny_tp8_checkpoint):
    for env in ({}, {**LOGITS, **EMBEDDING, **MTP_FC}):
        assert _ranks(tiny_tp8_checkpoint, 1, env)[0]["entries"] == []


# -- cut: no logits all-gather for arg-max consumers ----------------------------------------------------------
@pytest.mark.parametrize("tp", TP_SIZES)
def test_vocab_parallel_argmax_picks_the_same_tokens(tp, tiny_tp8_checkpoint):
    """Greedy decode tokens, drafted tokens and verified tokens are the same with and without the gather."""
    plain = _ranks(tiny_tp8_checkpoint, tp, {})
    cut = _ranks(tiny_tp8_checkpoint, tp, LOGITS)
    for rank in range(tp):
        _same_stream(plain[rank], cut[rank], tp, f"rank {rank} vocab-parallel arg max")
        assert cut[rank]["chain"]["tokens"] == cut[0]["chain"]["tokens"], f"TP{tp}: rank {rank} != rank 0"
    assert len(set(plain[0]["chain"]["tokens"])) > 1  # the stream is not one repeated token


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_vocab_parallel_argmax_gathers_two_values_per_row(tp, tiny_tp8_checkpoint):
    cut = _ranks(tiny_tp8_checkpoint, tp, LOGITS)[0]
    for phase, rows in (("prefill", 1), ("decode", 1), ("verify", 1 + K), ("draft", 1)):
        assert _entries(cut, phase, "logits") == [], phase
        top = _entries(cut, phase, "top")
        assert len(top) == (K if phase == "draft" else 1), phase
        for entry in top:
            assert (entry["method"], entry["shape"], entry["dtype"]) == ("all_gather", (rows, 2), "float32")


@pytest.mark.parametrize("tp", TP_SIZES)
def test_vocab_parallel_top_k_values_are_the_gathered_top_k(tp, tiny_tp8_checkpoint):
    """Mode 2: the k largest values are exact, and every returned id holds its value."""
    cut = _ranks(tiny_tp8_checkpoint, tp, CANDIDATES, top_k=64)
    for rank in range(tp):
        top = cut[rank]["chain"]["top_k"]
        assert top["values"].shape == (1 + K, 64) and top["ids"].shape == (1 + K, 64)
        np.testing.assert_array_equal(top["values"], top["gathered_values"], err_msg=f"TP{tp} rank {rank}")
        assert top["ids_hold_values"], f"TP{tp} rank {rank}: an id does not hold its value"
        assert int(top["ids"].min()) >= 0 and int(top["ids"].max()) < 248320
        assert all(len(set(row.tolist())) == 64 for row in top["ids"]), "an id is returned twice"
    if tp > 1:
        gathers = [e for e in _entries(cut[0], "chain", "top") if e["shape"] == (1 + K, 2 * 64)]
        assert len(gathers) == 1  # values and ids travel in one gather
        assert (gathers[0]["method"], gathers[0]["dtype"]) == ("all_gather", "float32")


# -- cut: replicated token embedding ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp", TP_SIZES)
def test_replicated_embedding_keeps_every_bit(tp, tiny_tp8_checkpoint):
    plain = _ranks(tiny_tp8_checkpoint, tp, {})
    cut = _ranks(tiny_tp8_checkpoint, tp, EMBEDDING)
    for rank in range(tp):
        _same_stream(plain[rank], cut[rank], tp, f"rank {rank} replicated embedding")
        layout = cut[rank]["layout"]
        assert (layout["embed_rows"], layout["embed_tp_size"]) == (248320, 1)
        assert layout["lm_head_rows"] == 248320 // tp  # the head stays sharded over the vocabulary
    if tp > 1:
        for phase in ("prefill", "decode", "verify", "draft"):
            assert _entries(cut[0], phase, "embedding") == [], phase
            assert len(_kinds(cut[0], phase)) == len(_kinds(plain[0], phase)) - (K if phase == "draft" else 1)


# -- cut: unsharded MTP entry linears ----------------------------------------------------------------------------
@pytest.mark.parametrize("tp", TP_SIZES)
def test_replicated_mtp_fc_drafts_the_same_tokens(tp, tiny_tp8_checkpoint):
    plain = _ranks(tiny_tp8_checkpoint, tp, {})
    cut = _ranks(tiny_tp8_checkpoint, tp, MTP_FC)
    for rank in range(tp):
        _same_stream(plain[rank], cut[rank], tp, f"rank {rank} unsharded MTP linears")
        assert (cut[rank]["layout"]["draft_fc_tp_size"], cut[rank]["layout"]["draft_fc_rows"]) == (1, 128)
    if tp > 1:
        assert _entries(plain[0], "draft", "mtp_fc") != [] and _entries(cut[0], "draft", "mtp_fc") == []
        assert len(_kinds(cut[0], "draft")) == len(_kinds(plain[0], "draft")) - 2 * K
        for phase in ("prefill", "decode", "verify"):  # the target is untouched
            assert _kinds(cut[0], phase) == _kinds(plain[0], phase)


# -- the cuts together -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tp", TP_SIZES)
def test_all_cuts_together(tp, tiny_tp8_checkpoint):
    plain = _ranks(tiny_tp8_checkpoint, tp, {})
    cut = _ranks(tiny_tp8_checkpoint, tp, {**LOGITS, **EMBEDDING, **MTP_FC})
    for rank in range(tp):
        _same_stream(plain[rank], cut[rank], tp, f"rank {rank} all cuts")
    if tp == 8:
        summary = summarize(cut, real_layers=48, k=K)
        for phase in ("prefill", "decode", "verify"):
            assert summary["phases"][phase]["real_counts"] == {"mixer": 48, "moe": 48, "ple_table": 1, "top": 1}
        assert summary["phases"]["draft"]["real_counts"] == {"mixer": K, "moe": K, "top": K}
