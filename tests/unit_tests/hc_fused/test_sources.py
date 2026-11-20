# SPDX-License-Identifier: Apache-2.0
"""Text checks that tie the pieces of ``csrc/hc_fused`` together (no toolchain needed).

The GUIDs, the node-parameter ABI and the kernel scalar order are written in C++ headers, TPC-C
kernels, the PyTorch op file and the Python test helpers; a drift between them would only show on
a card.  These checks read the sources.
"""

import re
import struct
from pathlib import Path

from tests.unit_tests.hc_fused import hc_ops
from tests.unit_tests.hc_fused.hc_case import GUIDS, KINDS, OP_NAME

REPO = Path(__file__).resolve().parents[3]
CSRC = REPO / "csrc" / "hc_fused"
KERNEL_FILE = {kind: f"hc_fused_{kind}_gaudi2" for kind in KINDS}
GUID_CONSTANT = {
    "norm": "kHcFusedNormGuid",
    "silu": "kHcFusedSiluGuid",
    "silu_split": "kHcFusedSiluSplitGuid",
    "gate_mix": "kHcFusedGateMixGuid",
    "combine": "kHcFusedCombineGuid",
    "combine_norm": "kHcFusedCombineNormGuid",
}
SCALAR_TAIL = "float eps, float inv_group, float inv_hc, int hc_count)"


def _code(path: Path) -> str:
    """The file without // comments."""
    return "\n".join(line.split("//")[0] for line in path.read_text().splitlines())


def test_params_header_has_the_python_guids():
    text = (CSRC / "hc_fused_params.h").read_text()
    found = dict(re.findall(r'inline constexpr const char\* (kHcFused\w+Guid) = "([^"]+)";', text))
    assert found == {GUID_CONSTANT[kind]: GUIDS[kind] for kind in KINDS}
    assert f"kHcFusedGuidCount = {len(KINDS)};" in text
    assert all(len(guid) < 64 for guid in GUIDS.values())  # MAX_NODE_NAME
    assert len(set(GUIDS.values())) == len(GUIDS)


def test_node_parameter_abi():
    text = (CSRC / "hc_fused_params.h").read_text()
    body = re.search(r"struct HcFusedParams \{(.*?)\};", text, re.S).group(1)
    fields = re.findall(r"^\s*(float|int32_t)\s+(\w+);", body, re.M)
    assert fields == [("float", "eps"), ("int32_t", "hc_count"), ("int32_t", "lora_rank")]
    assert struct.calcsize("<fii") == 12 and "sizeof(HcFusedParams) == 12" in text
    args = re.search(r"struct HcFusedKernelArgs \{(.*?)\};", text, re.S).group(1)
    assert re.findall(r"^\s*(float|int32_t)\s+(\w+);", args, re.M) == [("float", "eps"), ("float", "inv_group"),
                                                                        ("float", "inv_hc"), ("int32_t", "hc_count")]


def test_every_kernel_main_takes_the_scalar_arguments_in_abi_order():
    for kind in KINDS:
        source = (CSRC / "kernels" / f"{KERNEL_FILE[kind]}.c").read_text()
        if "hc_fused_silu.h" in source:
            source += (CSRC / "kernels" / "hc_fused_silu.h").read_text()
        mains = re.findall(r"void main\((.*?)\)\s*\n", source, re.S)
        assert mains, kind
        for main in mains:
            assert " ".join((main + ")").split()).endswith(SCALAR_TAIL), kind


def test_kernel_tensor_counts_match_the_glue_table():
    glue = (CSRC / "host" / "hc_fused_gaudi2.cpp").read_text()
    rows = dict(
        (constant, (int(inputs), int(outputs)))
        for constant, inputs, outputs in re.findall(r"\{Kind::\w+, (kHcFused\w+Guid), (\d), (\d)\}", glue))
    silu = _code(CSRC / "kernels" / "hc_fused_silu.h")
    split_main, plain_main = re.findall(r"void main\((.*?)\)", silu, re.S)
    for kind in KINDS:
        if kind == "silu":
            main = plain_main
        elif kind == "silu_split":
            main = split_main
        else:
            main = re.search(r"void main\((.*?)\)", _code(CSRC / "kernels" / f"{KERNEL_FILE[kind]}.c"), re.S).group(1)
        tensors = len(re.findall(r"\btensor\b", main))
        assert tensors == sum(rows[GUID_CONSTANT[kind]]), kind
        assert tensors <= 16  # a TPC-C program can access at most 16 tensors


def test_kernels_use_no_fused_multiply_add_and_no_tanh_sigmoid():
    """The oracle rounds every product and sum; v_sigmoid_f32 is the tanh form that fails the stress cases."""
    for path in sorted((CSRC / "kernels").iterdir()):
        code = _code(path)
        assert "v_f32_mac_b" not in code, path.name
        assert "v_sigmoid_f32" not in code, path.name
        assert "__local__" not in code, path.name
        if path.suffix == ".c" and "hc_fused_silu.h" not in code:
            assert "#pragma clang fp contract(off)" in code, path.name
    assert "#pragma clang fp contract(off)" in (CSRC / "kernels" / "hc_fused_silu.h").read_text()


def test_cmake_builds_every_kernel_without_vlm():
    code = "\n".join(line for line in (CSRC / "CMakeLists.txt").read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    assert "-vlm" not in code
    assert "-Wall -Werror -O2 -march=gaudi2" in code
    for kind in KINDS:
        assert KERNEL_FILE[kind] in code
        assert f"_binary_{KERNEL_FILE[kind]}_o_start" in (CSRC / "host" / "hc_fused_gaudi2.cpp").read_text()
    assert "add_library(\n    hc_fused_gaudi2\n    SHARED" in code
    assert "gdn_fused" not in code  # its own library: a failure here cannot remove the GDN kernels


def test_op_file_registers_each_schema_with_its_guid():
    text = (CSRC / "pytorch" / "hpu_hc_fused.cpp").read_text()
    for kind in KINDS:
        op = OP_NAME[kind]
        assert f'"flashinfer_gaudi::{op}"' in text
        assert f'm.impl("{op}", run_{kind});' in text
        assert re.search(rf"k\w+Schema, {GUID_CONSTANT[kind]},", text), kind
        assert hc_ops.SCHEMAS[op] in " ".join(line.strip().lstrip("/ ") for line in text.splitlines())
    assert "m.def(" not in text and "TORCH_LIBRARY_FRAGMENT" not in text  # the schemas are defined in Python


def test_entry_points_answer_supported_data_layouts():
    text = (CSRC / "host" / "entry_points.cpp").read_text()
    for name in ("GetKernelGuids", "InstantiateTpcKernel", "GetShapeInference", "GetSupportedDataLayouts",
                 "GetSuggestedManipulation"):
        assert re.search(rf'extern "C" tpc_lib_api::GlueCodeReturn {name}\(', text), name
    assert "'x'" in text


def test_extension_setup_links_only_the_backend_library():
    text = (CSRC / "pytorch" / "setup.py").read_text()
    assert 'name="hc_fused_ops"' in text and '"hpu_hc_fused.cpp"' in text
    assert "libraries=[BACKEND_LIBRARY]" in text
    assert "habana_pytorch2_plugin" not in text.split("setup(")[1]
