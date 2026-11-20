# SPDX-License-Identifier: Apache-2.0
"""Every metadata field the Flash-Next layers read survives ``trim_attn_metadata``.

Under HPU graphs the model does not see ``HPUAttentionMetadataV1``: ``HpuModelAdapter`` hands it the
``TrimmedAttentionMetadata`` tuple built from the whitelist in ``trim_attn_metadata``
(``vllm_gaudi/v1/worker/hpu_model_runner.py``). A field the model reads and the whitelist lacks is an
``AttributeError`` on the first forward that reaches the read (``md.block_table`` in the sparse QSA decode), or,
behind a ``getattr`` default, a value that is silently always the default (``lattice_store_differs``).

The read set is collected from the source of the Flash-Next layers and of the GDN layer they share
(``md.<name>``, ``attn_metadata.<name>``, ``getattr(md, "<name>", ...)`` and helper calls that take the metadata
and a field name), so the next field a layer starts to read fails here, on CPU, until it is whitelisted.
"""

from tests.unit_tests import hpu_cpu_stubs  # noqa: F401  (before any vllm / vllm_gaudi import)

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1  # noqa: E402
from vllm_gaudi.v1.worker.hpu_model_runner import trim_attn_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODEL_FILES = sorted((ROOT / "vllm_gaudi" / "models" / "qwen4_exp").glob("*.py")) + [
    ROOT / "vllm_gaudi" / "models" / "qwen3_5.py"
]
# The names the layers bind the forward's attention metadata to.
METADATA_NAMES = frozenset({"md", "attn_metadata", "metadata", "attn_meta"})
# ``md.get(...)`` is the per-layer dict form of the metadata (``isinstance(md, dict)``), not a field.
NOT_FIELDS = frozenset({"get"})
# Read in the sources, on the metadata, by at least one layer: a scan that stops finding these is broken.
KNOWN_READS = frozenset({
    "is_prompt", "slot_mapping", "block_size", "block_list", "block_table", "context_lens_tensor",
    "load_indices_tensor", "store_indices_tensor", "query_start_loc_p", "has_initial_states_p", "padding_mask_flat",
    "num_accepted_tokens", "lattice_store_differs", "direct_gdn_state", "dflash_full_query",
})


def metadata_reads(path: Path) -> dict[str, int]:
    """``{field: first line}`` of the metadata fields ``path`` reads."""
    reads: dict[str, int] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        name = None
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)
                and node.value.id in METADATA_NAMES):
            name = node.attr  # md.block_table
        elif (isinstance(node, ast.Call) and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)
              and node.args[0].id in METADATA_NAMES and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            name = node.args[1].value  # getattr(md, "x", None), self._state_indices(md, "x")
        if name is not None and name not in NOT_FIELDS:
            reads.setdefault(name, node.lineno)
    return reads


def all_reads() -> dict[str, str]:
    """``{field: "file:line"}`` over the Flash-Next layers and the GDN layer."""
    found: dict[str, str] = {}
    for path in MODEL_FILES:
        for name, line in metadata_reads(path).items():
            found.setdefault(name, f"{path.relative_to(ROOT)}:{line}")
    return found


def decode_metadata(**overrides) -> HPUAttentionMetadataV1:
    kwargs = dict(
        block_list=torch.zeros(2, dtype=torch.int32),
        block_usage=torch.zeros(2, dtype=torch.int32),
        block_groups=torch.zeros(2, dtype=torch.int32),
        input_positions=None,
        slot_mapping=torch.zeros(2, 1, dtype=torch.int64),
        block_size=128,
        window_block_list=None,
        window_block_usage=None,
        window_block_groups=None,
        chunked_block_list=None,
        chunked_block_usage=None,
        chunked_block_groups=None,
        load_indices_tensor=torch.zeros(1, 2, dtype=torch.int32),
        store_indices_tensor=torch.zeros(1, 2, dtype=torch.int32),
        num_accepted_tokens=None,
        seq_lens_tensor=torch.ones(2, dtype=torch.int32),
        query_start_loc=torch.arange(3, dtype=torch.int32),
        block_table=torch.zeros(2, 32, dtype=torch.int32),
        lattice_store_differs=False,
    )
    kwargs.update(overrides)
    return HPUAttentionMetadataV1.make_decode_metadata(**kwargs)


def prefill_metadata() -> HPUAttentionMetadataV1:
    return HPUAttentionMetadataV1.make_prefill_metadata(
        attn_bias=None,
        block_list=None,
        context_lens_tensor=torch.zeros(1, dtype=torch.int32),
        seq_lens_tensor=torch.full((1, ), 128, dtype=torch.int32),
        slot_mapping=torch.zeros(1, 128, dtype=torch.int64),
        block_size=128,
        load_indices_tensor=torch.zeros(1, 1, dtype=torch.int32),
        store_indices_tensor=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 128], dtype=torch.int32),
        padding_mask_flat=torch.ones(128, 1, dtype=torch.bfloat16),
    )


def test_the_scan_finds_the_known_reads():
    found = all_reads()
    missing = sorted(KNOWN_READS - set(found))
    assert not missing, f"the source scan no longer finds the metadata reads {missing}; fix the scan"


def test_every_field_the_layers_read_is_whitelisted():
    trimmed = trim_attn_metadata(decode_metadata())
    found = all_reads()
    missing = {name: where for name, where in sorted(found.items()) if name not in trimmed._fields}
    assert not missing, (
        f"the model reads metadata fields that trim_attn_metadata drops (field: first read): {missing}. "
        "Add them to the whitelist in vllm_gaudi/v1/worker/hpu_model_runner.py (tensors, None or a two-valued "
        "bool only; see the comment above the whitelist)")


def test_the_untrimmed_metadata_carries_every_field_the_layers_read():
    for metadata in (decode_metadata(), prefill_metadata()):
        missing = sorted(name for name in all_reads() if not hasattr(metadata, name))
        assert not missing, f"HPUAttentionMetadataV1 (is_prompt={metadata.is_prompt}) has no field {missing}"


def test_decode_block_table_and_lattice_flag_pass_through():
    table = torch.arange(64, dtype=torch.int32).view(2, 32)
    trimmed = trim_attn_metadata(decode_metadata(block_table=table, lattice_store_differs=True))
    assert trimmed.block_table is table
    assert trimmed.lattice_store_differs is True
    assert trim_attn_metadata(decode_metadata()).lattice_store_differs is False


def test_prefill_metadata_trims_with_the_same_fields():
    trimmed = trim_attn_metadata(prefill_metadata())
    assert trimmed.block_table is None
    assert trimmed.lattice_store_differs is False
    assert trimmed.is_prompt is True


def test_whitelisted_values_follow_the_hashing_rule():
    """The PT bridge hashes a tensor by its metadata and a primitive by its VALUE (the comment above the
    whitelist): a field may be a tensor, ``None`` or a bool that selects a graph branch; ``block_size`` is the
    one integer, constant for a server. ``lattice_store_differs`` is read by the model in a Python ``if``, so it
    must stay a bool (hashed by value: one graph per branch), never a tensor read on the host."""
    for metadata in (decode_metadata(lattice_store_differs=True), prefill_metadata()):
        trimmed = trim_attn_metadata(metadata)
        for name in trimmed._fields:
            value = getattr(trimmed, name)
            if name == "block_size":
                assert isinstance(value, int)
                continue
            assert value is None or isinstance(value, (torch.Tensor, bool)), (
                f"{name} = {value!r} ({type(value).__name__}) is hashed by value: one HPU graph per value")
    assert isinstance(trim_attn_metadata(decode_metadata()).block_table, torch.Tensor)
    assert type(trim_attn_metadata(decode_metadata()).lattice_store_differs) is bool
