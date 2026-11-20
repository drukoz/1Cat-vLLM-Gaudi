# SPDX-License-Identifier: Apache-2.0
"""The verify op's node limit and the verify kernel's vector local memory, checked without the TPC toolchain.

RUK-3105: ``csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c`` kept 24576 bytes of ``__local__`` vectors, more
than the 16 KiB of vector local memory a kernel with lookup-table functions has, and was compiled with
``-vlm 32``, which switches the compiler's check off.  The simulator passed; the op segfaulted on a card.  These
tests read the sources: the arrays fit a fixed budget, no build rule passes a local-memory override, every place
that states the maximum ``T`` says the same number, and one node more is rejected before any kernel runs.  The
compiler's own count and the real build are checked in ``tests/standalone/gdn_core_decode`` (tpcsim container).
"""

import pathlib
import re
import types

import pytest
import torch

from flashinfer_gaudi.gdn_core_verify import GDN_CORE_VERIFY_MAX_NODES, gdn_core_verify, gdn_core_verify_reference
from tests.unit_tests.gdn_core_decode.verify_sim_case import (MAX_NODES, VERIFY_ARG_ORDER, VERIFY_CASES,
                                                              VERIFY_HPU_CASES, VERIFY_TOO_MANY_NODES_CASE,
                                                              make_verify_inputs, verify_scalars)

REPO = pathlib.Path(__file__).resolve().parents[3]
GDN = REPO / "csrc" / "gdn_fused"
KERNEL = GDN / "kernels" / "gdn_core_verify_gaudi2.c"
PARAMS = GDN / "gdn_core_decode_params.h"
GLUE = GDN / "host" / "gdn_core_verify_gaudi2.cpp"
OP = GDN / "pytorch" / "hpu_gdn_core_decode.cpp"
CMAKE = GDN / "CMakeLists.txt"
DOC = REPO / "docs" / "design" / "flash-next" / "gdn-core-decode.md"
LAYER = REPO / "vllm_gaudi" / "models" / "qwen3_5.py"

VECTOR_BYTES = 256  # one float64 (64 fp32 lanes) in vector local memory
VLM_LIMIT = 16384  # vector local memory of a kernel that calls lookup-table functions
# For the __local__ vectors: what is left after 16 spilled vectors (4096 bytes; tpc-clang 2.7.33 spills 13) and
# 3 KiB of headroom.  It is exactly 6 vectors for each of the 6 nodes (q, k and v/o, two halves each).
STATIC_BUDGET = VLM_LIMIT - 4096 - 3072


def _kernel_max_nodes() -> int:
    match = re.search(r"^#define GDN_VERIFY_MAX_NODES (\d+)$", KERNEL.read_text(), re.M)
    assert match, f"no '#define GDN_VERIFY_MAX_NODES <n>' in {KERNEL}"
    return int(match.group(1))


def test_every_statement_of_the_maximum_agrees():
    assert GDN_CORE_VERIFY_MAX_NODES == 6  # T = 1 + k, k = 5 draft tokens
    assert MAX_NODES == GDN_CORE_VERIFY_MAX_NODES
    assert VERIFY_TOO_MANY_NODES_CASE.T == 7
    assert _kernel_max_nodes() == GDN_CORE_VERIFY_MAX_NODES
    params = re.search(r"kGdnCoreVerifyMaxNodes = (\d+);", PARAMS.read_text())
    assert params and int(params.group(1)) == GDN_CORE_VERIFY_MAX_NODES
    # The glue and the op callback compare with the constant, not with a number of their own.
    assert "nodes <= kGdnCoreVerifyMaxNodes" in GLUE.read_text()
    op = OP.read_text()
    assert "ssm_store_indices.size(1) <= kGdnCoreVerifyMaxNodes" in op
    assert not re.search(r"1<=T<=\d", op)
    doc = DOC.read_text()
    n = GDN_CORE_VERIFY_MAX_NODES
    assert f"`1 <= T <= {n}`" in doc
    assert f"`kGdnCoreVerifyMaxNodes = {n}`" in doc
    assert f"#define GDN_VERIFY_MAX_NODES {n}" in doc
    assert not re.search(r"T <= 8|MaxNodes = 8|MAX_NODES 8", doc)


def test_no_file_states_an_older_maximum():
    """The sources, the tests and the design note say 6 wherever they state the limit (it was 8, then 4)."""
    stale = re.compile(r"T <= [45]\b|1<=T<=[45]\b|MaxNodes = [45]\b|MAX_NODES\W* (?:= )?\(?[45]\b|"
                       r"\[1, [45]\]|largest `T` is [45]\b|`T = [56]`? is rejected|What happens for `T = [56]`")
    files = [KERNEL, KERNEL.with_name("gdn_core_verify_recurrence.h"), PARAMS, GLUE, OP, DOC,
             REPO / "flashinfer_gaudi" / "gdn_core_verify.py",
             REPO / "tests" / "unit_tests" / "gdn_core_decode" / "verify_sim_case.py",
             REPO / "tests" / "standalone" / "gdn_core_decode" / "test_simulation.py",
             REPO / "tests" / "standalone" / "gdn_core_decode" / "verify_sim_host.cpp"]
    found = [f"{path.relative_to(REPO)}:{number}: {line.strip()}" for path in files
             for number, line in enumerate(path.read_text().splitlines(), 1)
             if stale.search(line) and "->" not in line]  # "8 -> 4 -> 6" is the history, not a statement
    assert not found, "\n".join(found)


def test_layer_gate_uses_the_same_maximum():
    """The layer sends only ``1 <= T <= maximum`` to the op; anything else takes the torch path."""
    gates = re.findall(r"1 <= T <= (\w+) and T \+ 2 <= R", LAYER.read_text())
    assert len(gates) == 1, f"expected one 'if not (1 <= T <= <maximum> and T + 2 <= R)' gate in {LAYER}: {gates}"
    assert gates[0] in ("GDN_CORE_VERIFY_MAX_NODES", str(GDN_CORE_VERIFY_MAX_NODES)), (
        f"{LAYER.name} gates the verify op at T <= {gates[0]}, the op takes T <= {GDN_CORE_VERIFY_MAX_NODES}")


def test_layer_gate_takes_the_torch_path_above_the_maximum():
    """``_maybe_gdn_core_verify`` with a ``[B, T]`` lattice: T = maximum reaches the layer's precondition check,
    T = maximum + 1 returns ``None`` (the torch path) before it, with a conv pool that has rows for both."""
    from vllm_gaudi.models import qwen3_5
    from flashinfer_gaudi.gdn_core_verify import GdnCoreVerifyMode

    assert qwen3_5.GDN_CORE_VERIFY_MAX_NODES == GDN_CORE_VERIFY_MAX_NODES
    B, C = 2, 8

    class ReachedTheCheck(Exception):
        pass

    def reached(conv_pool, ssm_pool):
        raise ReachedTheCheck

    def call(T):
        conv_pool = torch.zeros(4, GDN_CORE_VERIFY_MAX_NODES + 3, C)  # R = T + 2 rows for T = maximum + 1
        layer = types.SimpleNamespace(gdn_core_verify_mode=GdnCoreVerifyMode.FUSED,
                                      _hpu_defer_ssm_writeback=False,
                                      kv_cache=(conv_pool, torch.zeros(4, 1, 1, 1)),
                                      _gdn_core_verify_ok=None,
                                      _gdn_core_verify_reason=reached)
        load = torch.zeros(B, T, dtype=torch.int32)
        qsl = torch.arange(B + 1, dtype=torch.int32) * T
        accepted = torch.ones(B, dtype=torch.int32)
        return qwen3_5.HPUGatedDeltaNetAttention._maybe_gdn_core_verify(layer, torch.zeros(B * T, C), None, None, None,
                                                                         load, None, B * T, accepted, qsl, False, False)

    with pytest.raises(ReachedTheCheck):
        call(GDN_CORE_VERIFY_MAX_NODES)
    assert call(GDN_CORE_VERIFY_MAX_NODES + 1) is None


def test_kernel_arrays_fit_the_vector_local_memory_budget():
    """The ``__local__`` vectors of the verify kernel take at most ``STATIC_BUDGET`` of the 16 KiB: 6 vectors per
    node.  One more per node (or a seventh node) does not fit it; -vlm is never the answer."""
    source = re.sub(r"//[^\n]*", "", KERNEL.read_text())
    declarations = re.findall(r"^__local__\s+(\w+)\s+(\w+)\[([^\]]+)\];", source, re.M)
    assert len(declarations) == source.count("__local__"), "a __local__ declaration this test cannot parse"
    max_nodes = _kernel_max_nodes()
    vectors = 0
    for ctype, name, size in declarations:
        assert ctype in ("float64", "int"), f"{name}: add the size of {ctype} to this test"
        factors = [part.strip() for part in size.split("*")]  # "2 * GDN_VERIFY_MAX_NODES" or "GDN_VERIFY_MAX_NODES"
        assert all(f == "GDN_VERIFY_MAX_NODES" or f.isdigit() for f in factors), (name, size)
        count = 1
        for factor in factors:
            count *= max_nodes if factor == "GDN_VERIFY_MAX_NODES" else int(factor)
        if ctype == "float64":
            vectors += count
    assert vectors == 6 * max_nodes == 36
    assert vectors * VECTOR_BYTES <= STATIC_BUDGET, (
        f"{vectors} __local__ vectors = {vectors * VECTOR_BYTES} bytes, budget {STATIC_BUDGET} of {VLM_LIMIT}")


def test_build_rules_pass_no_local_memory_override():
    """Every tpc-clang command of the build uses the one flag list, and nothing raises a local-memory limit."""
    text = re.sub(r"#[^\n]*", "", CMAKE.read_text())
    assert not re.search(r"-(vlm|slm|spill-vlm)\b", text)
    flags = re.search(r"set\(GDN_TPC_FLAGS ([^)]*)\)", text)
    assert flags and flags.group(1).split() == ["-Wall", "-Werror", "-O2", "-march=gaudi2"]
    commands = re.findall(r'COMMAND "\$\{TPC_COMPILER\}"([^\n]*)', text)
    assert len(commands) == 4  # decode, recurrent, verify, and the configure-time try of the verify compile
    for arguments in commands:
        assert arguments.split()[0] == "${GDN_TPC_FLAGS}", arguments
    # The diagnostic that strips the verify ELF's bitcode is off unless asked for.
    assert re.search(r"option\(GDN_FUSED_VERIFY_STRIP_BITCODE [^)]* OFF\)", text)


def _args(case):
    inputs = make_verify_inputs(case)
    return [inputs[name].clone() for name in VERIFY_ARG_ORDER] + list(verify_scalars(case))


@pytest.mark.parametrize("run",
                         [gdn_core_verify_reference, torch.ops.flashinfer_gaudi.gdn_core_verify, gdn_core_verify],
                         ids=["reference", "op", "wrapper"])
def test_too_many_nodes_is_rejected(run):
    """T = maximum + 1 raises ``ValueError`` and leaves both state pools untouched."""
    case = VERIFY_TOO_MANY_NODES_CASE
    assert case.T == GDN_CORE_VERIFY_MAX_NODES + 1 and case.R >= case.T + 2
    args = _args(case)
    conv_before, ssm_before = args[0].clone(), args[1].clone()
    with pytest.raises(ValueError, match=rf"T must be in \[1, {GDN_CORE_VERIFY_MAX_NODES}\], got {case.T}"):
        run(*args)
    assert torch.equal(args[0], conv_before)
    assert torch.equal(args[1], ssm_before)


def test_the_maximum_itself_is_accepted():
    case = next(c for c in VERIFY_HPU_CASES if c.T == GDN_CORE_VERIFY_MAX_NODES)
    out = gdn_core_verify(*_args(case))
    assert tuple(out.shape) == (case.B * case.T, case.HV, 128)


def test_the_shared_cases_cover_every_node_count():
    """T = 1 .. maximum each have a simulator case, and T = 5 and T = 6 have the TP8 and TP4 geometries."""
    assert {case.T for case in VERIFY_CASES} == set(range(1, GDN_CORE_VERIFY_MAX_NODES + 1))
    for T in (5, 6):
        shapes = {(case.H, case.HV, case.B) for case in VERIFY_CASES if case.T == T}
        assert {(2, 6, 1), (2, 6, 16), (2, 6, 4), (4, 12, 2)} <= shapes, (T, shapes)
        assert all(case.R >= T + 2 for case in VERIFY_CASES if case.T == T)
    chain = next(case for case in VERIFY_CASES if case.name == "ver_tp8_t6_chain")
    assert chain.T == 6 and chain.R == 8 and chain.rows[0].conv_offset == 5 and chain.rows[0].ssm_load == 6
