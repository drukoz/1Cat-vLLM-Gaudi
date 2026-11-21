# SPDX-License-Identifier: Apache-2.0
"""``VLLM_HPU_GDN_VERIFY_VARIANT``: the flag, its documentation and the sources that implement it agree.

The variant builds of the interleaved verify kernel (1 request-major index space, 2 lean broadcasts, 3 both, 6 the
fold recurrence; docs/design/flash-next/hc-kernels.md, "GDN verify") are proven bit-identical to the interleaved kernel in the
Gaudi2 simulator (``tests/standalone/gdn_core_decode/test_simulation.py``, ``test_verify_variant_*``). What a CPU can
check: the plugin declares the flag (default 0), the documentation names it, the glue reads exactly that variable
and only with the interleave variable, every mapping of the glue to an index dimension goes through the two
dimension names, each variant build is the interleaved kernel with its macros (the fold build: the verify kernel with
the fold recurrence), and the build lists all four.
"""

import re
from pathlib import Path

import vllm_gaudi.envs as gaudi_envs

ROOT = Path(__file__).resolve().parents[3]
FLAG = "VLLM_HPU_GDN_VERIFY_VARIANT"
GDN = ROOT / "csrc" / "gdn_fused"
FOLD = "gdn_core_verify_interleaved_fold_gaudi2"
VARIANTS = {
    "gdn_core_verify_interleaved_rmajor_gaudi2": ["#define GDN_VERIFY_REQUEST_MAJOR 1"],
    "gdn_core_verify_interleaved_lean_gaudi2": ["#define GDN_VERIFY_LEAN_BROADCAST 1"],
    "gdn_core_verify_interleaved_rmajor_lean_gaudi2": ["#define GDN_VERIFY_REQUEST_MAJOR 1",
                                                       "#define GDN_VERIFY_LEAN_BROADCAST 1"],
}


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_GDN_VERIFY_VARIANT == 0
    monkeypatch.setenv(FLAG, "3")
    assert gaudi_envs.VLLM_HPU_GDN_VERIFY_VARIANT == 3
    assert f"    {FLAG}: int = 0" in (ROOT / "vllm_gaudi" / "envs.py").read_text()


def test_flag_is_documented():
    text = (ROOT / "docs" / "configuration" / "env_variables.md").read_text()
    assert f"`{FLAG}`" in text and "bit-identical" in text


def test_the_glue_reads_exactly_this_variable_with_the_interleave_variable():
    header = (GDN / "host" / "gdn_kernel_variant.hpp").read_text()
    assert f'std::getenv("{FLAG}")' in header
    for value in ("1", "2", "3", "4", "5", "6", "7"):
        assert f'std::strcmp(value, "{value}") == 0) return {value};' in header
    glue = (GDN / "host" / "gdn_core_verify_gaudi2.cpp").read_text()
    assert "const int asked = GdnKernelInterleaveRequested() ? GdnVerifyVariantRequested() : 0;" in glue
    assert "const bool requestMajor = variant && (asked == 1 || asked == 3);" in glue
    assert "const bool splitKeyHeads = variant && (asked == 4 || asked == 5 || asked == 7);" in glue
    assert "} else if (asked == 6 || asked == 7) {" in glue
    assert "instance->preferredSplitDim = 1;" in glue and "dim.allRequired = true;" in glue
    for name in (*VARIANTS, FOLD):
        for edge in ("start", "end"):
            assert f"extern unsigned char _binary_{name}_o_{edge} __attribute__((weak));" in glue


def test_every_index_mapping_of_the_glue_names_its_dimension():
    """A mapping with a != 0 must go through keyHeadDim or requestDim; one with a == 0 reads the whole range."""
    glue = (GDN / "host" / "gdn_core_verify_gaudi2.cpp").read_text()
    calls = re.findall(r"MapDimension\((\w+), (\d+), (\w+), ([^,]+), ", " ".join(glue.split()))
    assert len(calls) >= 20
    for pattern, tensor_dim, index_dim, a in calls:
        if a.strip() != "0":
            assert index_dim in ("keyHeadDim", "requestDim"), (pattern, tensor_dim, index_dim, a)
    assert "instance->indexSpaceGeometry[keyHeadDim] = keyHeads;" in glue
    assert "instance->indexSpaceGeometry[requestDim] = batchSize;" in glue


def test_each_variant_build_is_the_interleaved_kernel_with_its_macros():
    for name, macros in VARIANTS.items():
        wrapper = (GDN / "kernels" / f"{name}.c").read_text()
        lines = [line for line in wrapper.splitlines() if line and not line.startswith("//")]
        assert lines == macros + ['#include "gdn_core_verify_interleaved_gaudi2.c"'], name
    kernel = (GDN / "kernels" / "gdn_core_verify_gaudi2.c").read_text()
    assert "index_start[GDN_VERIFY_REQUEST_DIM]" in kernel and "index_start[GDN_VERIFY_KEY_HEAD_DIM]" in kernel
    assert "index_start[1]" not in kernel and "index_start[0]" not in kernel
    recurrence = (GDN / "kernels" / "gdn_core_verify_recurrence_interleaved.h").read_text()
    assert recurrence.count("#ifdef GDN_VERIFY_LEAN_BROADCAST") == 2
    cmake = (GDN / "CMakeLists.txt").read_text()
    assert ("list(APPEND GDN_INTERLEAVE_CANDIDATES gdn_core_verify_interleaved_rmajor_gaudi2\n"
            "         gdn_core_verify_interleaved_lean_gaudi2 gdn_core_verify_interleaved_rmajor_lean_gaudi2)") in cmake
    assert f"list(APPEND GDN_INTERLEAVE_CANDIDATES {FOLD})" in cmake
    assert '"${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_recurrence_fold.h"' in cmake


def test_the_fold_build_is_the_verify_kernel_with_the_fold_recurrence():
    wrapper = (GDN / "kernels" / f"{FOLD}.c").read_text()
    lines = [line for line in wrapper.splitlines() if line and not line.startswith("//")]
    assert lines == ["#define GDN_VERIFY_BATCHED_GATES 1",
                     '#define GDN_CORE_VERIFY_RECURRENCE_HEADER "gdn_core_verify_recurrence_fold.h"',
                     '#include "gdn_core_verify_gaudi2.c"']
    kernel = (GDN / "kernels" / "gdn_core_verify_gaudi2.c").read_text()
    # the batched gates exist only under the fold build's macro: the other builds compile what they compiled
    assert kernel.count("GDN_VERIFY_BATCHED_GATES") == 1 and "#ifdef GDN_VERIFY_BATCHED_GATES" in kernel
    fold = (GDN / "kernels" / "gdn_core_verify_recurrence_fold.h").read_text()
    # the contract of the recurrence header: the function the kernel calls, the arrays it reads and writes
    assert "static inline void gdn_core_verify_recurrence_head(" in fold
    for array in ("verify_q[", "verify_k[", "verify_vo[", "verify_store_slot["):
        assert array in fold, array
    # the fold keeps v_f32_reduce_add's additions: lane bit 4, then 5, then 3, then 1, 2, 4 inside a group
    order = [fold.index(marker) for marker in ("1, 0, 3, 2, MkWrA(3, 3, 3, 3)", "2, 3, 0, 1, MkWrA(3, 3, 3, 3)",
                                              "v_f32_mov_group_b(x, 0xffffffff, 63", "shuffle_b(x, xor1",
                                              "shuffle_b(x, xor2", "shuffle_b(x, xor4")]
    assert order == sorted(order)
    assert "v_f32_reduce_add(" not in fold.split("static inline float64 gdn_verify_fold4")[1]
