# SPDX-License-Identifier: Apache-2.0
"""Text checks that tie the MoE glue pieces of ``csrc/hc_fused`` together (no toolchain needed).

The GUIDs, the node-parameter ABI, the kernel scalar order, the limits and the build switches are
written in C++ headers, TPC-C kernels, the CMake project, the PyTorch op file and the Python modules; a
drift between them would only show on a card.  These checks read the sources.
"""

import re
import struct
from pathlib import Path

from tests.unit_tests.hc_fused import moe_ops
from tests.unit_tests.hc_fused.moe_case import GUIDS, KINDS, MAX_TOP_K, OP_NAME, SIM_CASES, kernel_params
from tests.unit_tests.hc_fused.sim_build import MOE_KERNELS

REPO = Path(__file__).resolve().parents[3]
CSRC = REPO / "csrc" / "hc_fused"
KERNEL_FILE = {kind: f"moe_glue_{kind}_gaudi2" for kind in KINDS}
GUID_CONSTANT = {
    "route": "kMoeGlueRouteGuid",
    "silu_mul": "kMoeGlueSiluMulGuid",
    "combine": "kMoeGlueCombineGuid",
    "row_quant": "kMoeGlueRowQuantGuid",
}
BUILD_DEFINE = {kind: f"HC_FUSED_HAS_MOE_GLUE_{kind.upper()}" for kind in KINDS}
SCALAR_TAIL = "int experts, int top_k, int gate_column, float inv_fp8_max)"
TENSORS = {"route": (1, 2), "silu_mul": (1, 1), "combine": (3, 1), "row_quant": (1, 2)}


def _code(path: Path) -> str:
    """The file without // comments."""
    return "\n".join(line.split("//")[0] for line in path.read_text().splitlines())


def test_params_header_has_the_python_guids_and_limits():
    text = (CSRC / "hc_fused_params.h").read_text()
    found = dict(re.findall(r'inline constexpr const char\* (kMoeGlue\w+Guid) = "([^"]+)";', text))
    assert found == {GUID_CONSTANT[kind]: GUIDS[kind] for kind in KINDS}
    assert f"kMoeGlueGuidCount = {len(KINDS)};" in text
    assert all(len(guid) < 64 for guid in GUIDS.values())  # MAX_NODE_NAME
    assert len(set(GUIDS.values())) == len(GUIDS)
    assert f"kMoeGlueMaxTopK = {moe_ops.moe_glue_module().MAX_TOP_K};" in text and MAX_TOP_K == 64
    assert f"kMoeGlueMaxExperts = {moe_ops.moe_glue_module().MAX_EXPERTS};" in text
    assert f"kMoeGlueFp8Max = {moe_ops.moe_glue_module().GAUDI2_FP8_MAX}f;" in text
    assert "kHcFusedGuidCount = 6;" in text  # the hyper-connection GUIDs keep their six places at the front


def test_node_parameter_abi():
    text = (CSRC / "hc_fused_params.h").read_text()
    body = re.search(r"struct MoeGlueParams \{(.*?)\};", text, re.S).group(1)
    fields = re.findall(r"^\s*(float|int32_t)\s+(\w+);", body, re.M)
    assert fields == [("int32_t", "experts"), ("int32_t", "top_k"), ("int32_t", "gate_column"), ("float", "fp8_max")]
    assert struct.calcsize("<iiif") == 16 and "sizeof(MoeGlueParams) == 16" in text
    args = re.search(r"struct MoeGlueKernelArgs \{(.*?)\};", text, re.S).group(1)
    assert re.findall(r"^\s*(float|int32_t)\s+(\w+);", args, re.M) == [("int32_t", "experts"), ("int32_t", "top_k"),
                                                                        ("int32_t", "gate_column"),
                                                                        ("float", "inv_fp8_max")]
    # the 16 bytes differ from the 12 of the hyper-connection kernels: shape inference tells the families apart
    assert "sizeof(HcFusedParams) == 12" in text
    for case in SIM_CASES:
        assert len(struct.pack("<iiif", *kernel_params(case))) == 16


def test_every_kernel_main_takes_the_scalar_arguments_in_abi_order():
    for kind in KINDS:
        source = (CSRC / "kernels" / f"{KERNEL_FILE[kind]}.c").read_text()
        mains = re.findall(r"void main\((.*?)\)\s*\n", source, re.S)
        assert len(mains) == 1, kind
        assert " ".join((mains[0] + ")").split()).endswith(SCALAR_TAIL), kind


def test_kernel_tensor_counts_match_the_glue_table():
    glue = (CSRC / "host" / "moe_glue_gaudi2.cpp").read_text()
    rows = dict((constant, (int(inputs), int(outputs)))
                for constant, inputs, outputs in re.findall(r"\{Kind::\w+, (kMoeGlue\w+Guid), (\d), (\d)\}", glue))
    assert rows == {GUID_CONSTANT[kind]: TENSORS[kind] for kind in KINDS}
    for kind in KINDS:
        main = re.search(r"void main\((.*?)\)", _code(CSRC / "kernels" / f"{KERNEL_FILE[kind]}.c"), re.S).group(1)
        tensors = len(re.findall(r"\btensor\b", main))
        assert tensors == sum(TENSORS[kind]), kind
        assert tensors <= 16  # a TPC-C program can access at most 16 tensors


def test_kernels_round_like_the_oracle_and_keep_no_local_memory():
    for kind in KINDS:
        code = _code(CSRC / "kernels" / f"{KERNEL_FILE[kind]}.c")
        assert "#pragma clang fp contract(off)" in code, kind
        assert "v_f32_mac_b" not in code and "v_sigmoid_f32" not in code and "__local__" not in code, kind
        assert not re.search(r"\b(bfloat128|float64|float128|short128|int64|minifloat256)\s+\w+\[", code), kind
    common = _code(CSRC / "kernels" / "moe_glue_common.h")
    assert "__local__" not in common and "SW_RHNE" in common
    route = (CSRC / "kernels" / "moe_glue_route_gaudi2.c").read_text()
    assert "v_i16_sel2_geq_bf16_vb" in route and "v_i16_reduce_min" in route  # the lowest expert id among equal logits


def test_cmake_builds_every_kernel_optionally_and_without_a_memory_override():
    lines = (CSRC / "CMakeLists.txt").read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    assert not re.search(r"-+(vlm|slm|spill-vlm)\b", code)
    assert code.count("-Wall -Werror -O2 -march=gaudi2") == 3  # the six kernels, the MoE probe, the MoE build
    assert tuple(re.search(r"set\(MOE_GLUE_KERNELS(.*?)\)", code, re.S).group(1).split()) == MOE_KERNELS
    assert MOE_KERNELS == tuple(KERNEL_FILE[kind] for kind in KINDS)
    assert "option(HC_FUSED_MOE_GLUE " in code and "execute_process(" in code and "continue()" in code
    assert 'list(APPEND MOE_GLUE_DEFINITIONS "HC_FUSED_HAS_${kernel_define}=1")' in code
    assert "host/moe_glue_gaudi2.cpp" in code
    glue = (CSRC / "host" / "moe_glue_gaudi2.cpp").read_text()
    for kind in KINDS:
        assert glue.count(f"#ifdef {BUILD_DEFINE[kind]}\n") == 3, kind  # the ELF symbols, the table row, the ELF lookup
        assert f"_binary_{KERNEL_FILE[kind]}_o_start" in glue
    # the hyper-connection kernels do not depend on any of it
    hc_glue = (CSRC / "host" / "hc_fused_gaudi2.cpp").read_text()
    assert "moe" not in hc_glue.lower()


def test_op_file_registers_each_schema_with_its_guid():
    text = (CSRC / "pytorch" / "hpu_hc_fused.cpp").read_text()
    flat = " ".join(line.strip().lstrip("/ ") for line in text.splitlines())
    for kind in KINDS:
        op = OP_NAME[kind]
        assert f'"flashinfer_gaudi::{op}"' in text
        assert f'm.impl("{op}", run_{op});' in text
        assert re.search(rf"k\w+Schema, {GUID_CONSTANT[kind]},", text), kind
        assert moe_ops.SCHEMAS[op].split(" -> ")[0] in flat, op
        assert moe_ops.GUIDS[op] == GUIDS[kind]
    assert "m.def(" not in text and "TORCH_LIBRARY_FRAGMENT" not in text  # the schemas are defined in Python
    assert "at::kInt, {gates.size(0), top_k}" in text  # the ids are int32: the layer widens them for the vendor op
    assert "at::ScalarType::Float8_e4m3fn, x.sizes().vec()" in text


def test_entry_points_serve_both_families():
    text = (CSRC / "host" / "entry_points.cpp").read_text()
    assert "kHcFusedGuidCount + moe_glue::KernelCount()" in text
    assert text.count("moe_glue::FindKernel(") >= 4  # instantiate, shape inference (two names), data layouts
    assert "moe_glue::KernelForShapes(params)" in text
