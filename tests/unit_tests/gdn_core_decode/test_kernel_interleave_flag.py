# SPDX-License-Identifier: Apache-2.0
"""``VLLM_HPU_GDN_KERNEL_INTERLEAVE``: the flag, its documentation and the sources that implement it agree.

The interleaved kernels (four value rows per iteration of the recurrence, idea I27) are proven bit-identical to
today's kernels in the Gaudi2 simulator (``tests/standalone/gdn_core_decode/test_simulation.py``,
``test_interleaved_*``). What a CPU can check: the plugin declares the flag (default 0), the documentation names
it, the kernel library's glue reads exactly that variable and exactly "1", and the build and the kernel sources
are wired so that the interleaved build is today's kernel with only the recurrence header replaced.
"""

import re
from pathlib import Path

import vllm_gaudi.envs as gaudi_envs

ROOT = Path(__file__).resolve().parents[3]
FLAG = "VLLM_HPU_GDN_KERNEL_INTERLEAVE"
GDN = ROOT / "csrc" / "gdn_fused"
KERNELS = GDN / "kernels"


def test_flag_default_and_value(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert gaudi_envs.VLLM_HPU_GDN_KERNEL_INTERLEAVE == 0
    monkeypatch.setenv(FLAG, "1")
    assert gaudi_envs.VLLM_HPU_GDN_KERNEL_INTERLEAVE == 1
    assert f"    {FLAG}: int = 0" in (ROOT / "vllm_gaudi" / "envs.py").read_text()


def test_flag_is_documented():
    text = (ROOT / "docs" / "configuration" / "env_variables.md").read_text()
    assert f"`{FLAG}`" in text and "bit-identical" in text


def test_the_glue_reads_exactly_this_variable_and_exactly_one():
    header = (GDN / "host" / "gdn_kernel_variant.hpp").read_text()
    assert f'std::getenv("{FLAG}")' in header
    assert 'std::strcmp(value, "1") == 0' in header
    for glue, name in (("gdn_core_decode_gaudi2.cpp", "gdn_core_decode"), ("gdn_core_verify_gaudi2.cpp",
                                                                            "gdn_core_verify")):
        source = (GDN / "host" / glue).read_text()
        assert '#include "gdn_kernel_variant.hpp"' in source
        for edge in ("start", "end"):
            assert f"extern unsigned char _binary_{name}_interleaved_gaudi2_o_{edge} __attribute__((weak));" in source
        assert "GdnSelectKernelElf(" in source and "std::memcpy(instance->kernel.kernelElf, elf.start, isaSize);" in source
        assert f"&_binary_{name}_gaudi2_o_start, isaSize" not in source  # the ELF is the selected one


def test_the_interleaved_builds_replace_only_the_recurrence_header():
    for kind, macro in (("decode", "GDN_CORE_DECODE_RECURRENCE_HEADER"), ("verify", "GDN_CORE_VERIFY_RECURRENCE_HEADER")):
        base = (KERNELS / f"gdn_core_{kind}_gaudi2.c").read_text()
        assert f"#ifdef {macro}\n#include {macro}\n#else\n#include \"gdn_core_{kind}_recurrence.h\"\n#endif" in base
        wrapper = (KERNELS / f"gdn_core_{kind}_interleaved_gaudi2.c").read_text()
        lines = [line for line in wrapper.splitlines() if line and not line.startswith("//")]
        assert lines == [f'#define {macro} "gdn_core_{kind}_recurrence_interleaved.h"',
                         f'#include "gdn_core_{kind}_gaudi2.c"']
        today = (KERNELS / f"gdn_core_{kind}_recurrence.h").read_text()
        interleaved = (KERNELS / f"gdn_core_{kind}_recurrence_interleaved.h").read_text()
        functions = re.compile(r"static inline \w+ (\w+)\(")
        assert functions.findall(interleaved) == functions.findall(today)


def test_the_build_compiles_and_embeds_both_interleaved_kernels():
    cmake = (GDN / "CMakeLists.txt").read_text()
    assert 'option(GDN_FUSED_INTERLEAVE "Build the interleaved decode and verify kernels into the library" ON)' in cmake
    assert "set(GDN_INTERLEAVE_CANDIDATES gdn_core_decode_interleaved_gaudi2)" in cmake
    assert "list(APPEND GDN_INTERLEAVE_CANDIDATES gdn_core_verify_interleaved_gaudi2)" in cmake
    assert "${GDN_INTERLEAVE_LIBRARY_SOURCES}" in cmake
    # the same flag list as every other kernel, no local-memory override
    block = cmake[cmake.index("option(GDN_FUSED_INTERLEAVE"):cmake.index("add_library(")]
    assert block.count("${GDN_TPC_FLAGS}") == 2 and not re.search(r"-(vlm|slm|spill-vlm)\b", block)


def test_the_interleaved_recurrences_carry_every_row():
    """Four rows per iteration: 32 iterations of rows lane, lane + 32, 64 + lane and 96 + lane cover the 128."""
    for kind in ("decode", "verify"):
        text = (KERNELS / f"gdn_core_{kind}_recurrence_interleaved.h").read_text()
        assert "for (int lane = 0; lane < 32; ++lane)" in text
        rows = sorted({row.strip() for row in re.findall(r"\{0, ((?:row_\w+)|(?:[^,{}]*lane[^,{}]*)), value_head", text)})
        offsets = set()
        for row in re.findall(r"(?:const int row_\w+ = |\{0, )((?:64 \+ )?lane \+ \d+)", text):
            base = 64 if row.startswith("64 + ") else 0
            offsets.add(base + int(row.rsplit("+ ", 1)[1]))
        assert offsets == {0, 32, 64, 96}, (kind, rows, offsets)
