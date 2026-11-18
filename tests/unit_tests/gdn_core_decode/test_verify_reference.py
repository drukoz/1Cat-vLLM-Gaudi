# SPDX-License-Identifier: Apache-2.0
"""CPU tests for flashinfer_gaudi.gdn_core_verify.gdn_core_verify_reference.

Phase 2 contract (``docs/design/flash-next/gdn-core-decode.md``): ``T`` draft nodes per row, node ``t``'s input
state is node ``parent_idx[r, t]``'s output state.  The chain, tree, padding and unstored-node cases compare the
verify reference with sequential ``gdn_core_decode_reference`` paths; comparisons use ``torch.equal``, since both
sides run the same per-node ops in the same order.
"""

import pytest
import torch

from flashinfer_gaudi.gdn_core_decode import GdnGateActivation, gdn_core_decode_reference
from flashinfer_gaudi.gdn_core_verify import gdn_core_verify_reference

HEAD_COUNTS = [(16, 48), (8, 24), (4, 12), (2, 6)]
GATE = GdnGateActivation.SIGMOID

# torch.equal where both sides run the same ops on the same batch; at (2, 6) a batch-3 matmul inside
# gdn_core_decode_reference reduces in a different order than the batch-1 matmul of the 1-row reference paths
# (measured: outputs within 4.8e-7, SSM state within 6e-8), so the chain and tree comparisons use atol=1e-6.
# At (16, 48) those comparisons are bit-exact.
BATCH_TOL = dict(atol=1e-6, rtol=0)


def _make_inputs(H, HV, D, T, S, B=3, seed=0):
    torch.manual_seed(seed)
    C = 2 * H * D + HV * D
    bf = lambda *shape: torch.randn(*shape).to(torch.bfloat16)
    return {
        "conv_state": bf(S, 3, C),
        "ssm_state": (torch.randn(S, HV, D, D) * 0.1).float(),
        "mixed_qkv": bf(B, T, C),
        "z": bf(B, T, HV * D),
        "a": bf(B, T, HV),
        "b": bf(B, T, HV),
        "A_log": torch.randn(HV).float() * 0.5,
        "dt_bias": torch.randn(HV).float(),
        "conv_weight_t": bf(4, C) * 0.5,
        "norm_weight": bf(D) * 0.1 + 1,
        "scale": D ** -0.5,
        "eps": 1e-6,
        "parent_idx": torch.full((B, T), -1, dtype=torch.int32),
        "load_indices": torch.full((B, ), -1, dtype=torch.int32),
        "store_indices": torch.full((B, T), -1, dtype=torch.int32),
    }


def _verify_args(inp, conv=None, ssm=None, parent=None, load=None, store=None):
    """Positional args for gdn_core_verify_reference; conv/ssm default to the input pools (mutated)."""
    return [
        inp["conv_state"] if conv is None else conv,
        inp["ssm_state"] if ssm is None else ssm,
        inp["mixed_qkv"],
        inp["z"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        inp["conv_weight_t"],
        inp["norm_weight"],
        inp["parent_idx"] if parent is None else parent,
        inp["load_indices"] if load is None else load,
        inp["store_indices"] if store is None else store,
        inp["scale"],
        inp["eps"],
        GATE,
    ]


def _path_reference_row(inp, row, nodes, stores):
    """One row's gdn_core_decode_reference chain over the node ids in ``nodes`` (increasing): node k loads the
    slot node k-1 stored (initially ``load_indices[row]``) and stores into pool slot ``stores[k]``.  Uses its own
    pool clones.  Returns the stacked outputs ``[len(nodes), HV, D]`` and the pools."""
    conv, ssm = inp["conv_state"].clone(), inp["ssm_state"].clone()
    load = inp["load_indices"][row:row + 1].clone()
    outs = []
    for k, t in enumerate(nodes):
        store = torch.tensor([stores[k]], dtype=torch.int32)
        outs.append(
            gdn_core_decode_reference(conv, ssm, inp["mixed_qkv"][row:row + 1, t], inp["z"][row:row + 1, t],
                                      inp["a"][row:row + 1, t], inp["b"][row:row + 1, t], inp["A_log"],
                                      inp["dt_bias"], inp["conv_weight_t"], inp["norm_weight"], load, store,
                                      inp["scale"], inp["eps"], GATE))
        load = store
    return torch.cat(outs, dim=0), conv, ssm


@pytest.mark.parametrize("H,HV", [HEAD_COUNTS[0], HEAD_COUNTS[3]])
def test_t1_equals_decode_reference(H, HV):
    D = 128
    inp = _make_inputs(H, HV, D, T=1, S=16)
    inp["parent_idx"] = torch.full((3, 1), -1, dtype=torch.int32)
    inp["load_indices"] = torch.tensor([1, 5, 9], dtype=torch.int32)
    inp["store_indices"] = torch.tensor([[2], [6], [7]], dtype=torch.int32)

    conv_v, ssm_v = inp["conv_state"].clone(), inp["ssm_state"].clone()
    out_v = gdn_core_verify_reference(*_verify_args(inp, conv=conv_v, ssm=ssm_v))

    conv_d, ssm_d = inp["conv_state"].clone(), inp["ssm_state"].clone()
    out_d = gdn_core_decode_reference(conv_d, ssm_d, inp["mixed_qkv"][:, 0], inp["z"][:, 0], inp["a"][:, 0],
                                      inp["b"][:, 0], inp["A_log"], inp["dt_bias"], inp["conv_weight_t"],
                                      inp["norm_weight"], inp["load_indices"], inp["store_indices"][:, 0],
                                      inp["scale"], inp["eps"], GATE)

    assert torch.equal(out_v[:, 0], out_d)
    assert torch.equal(conv_v, conv_d)
    assert torch.equal(ssm_v, ssm_d)


@pytest.mark.parametrize("H,HV", [HEAD_COUNTS[0], HEAD_COUNTS[3]])
def test_chain_equals_sequential(H, HV):
    D, T = 128, 4
    inp = _make_inputs(H, HV, D, T=T, S=16)
    inp["parent_idx"] = torch.tensor([[-1, 0, 1, 2]] * 3, dtype=torch.int32)
    inp["load_indices"] = torch.tensor([0, 13, 14], dtype=torch.int32)
    inp["store_indices"] = torch.tensor([[1 + 4 * r + t for t in range(T)] for r in range(3)], dtype=torch.int32)

    conv_v, ssm_v = inp["conv_state"].clone(), inp["ssm_state"].clone()
    out_v = gdn_core_verify_reference(*_verify_args(inp, conv=conv_v, ssm=ssm_v))

    exp_out = []
    for r in range(3):
        stores = inp["store_indices"][r].tolist()
        outs, exp_conv, exp_ssm = _path_reference_row(inp, r, list(range(T)), stores)
        exp_out.append(outs)
        # Every stored node state, and the untouched load slot, must match the verify pools.  (The path pool only
        # touches row r's slots; verify also writes the other rows' store slots, so compare slot by slot.)
        for slot in [int(inp["load_indices"][r])] + stores:
            assert torch.equal(conv_v[slot], exp_conv[slot])
            torch.testing.assert_close(ssm_v[slot], exp_ssm[slot], **BATCH_TOL)
    torch.testing.assert_close(out_v, torch.stack(exp_out), **BATCH_TOL)


@pytest.mark.parametrize("H,HV", [HEAD_COUNTS[0], HEAD_COUNTS[3]])
def test_tree_branches_equal_their_paths(H, HV):
    D, T = 128, 5
    inp = _make_inputs(H, HV, D, T=T, S=24)
    inp["parent_idx"] = torch.tensor([[-1, 0, 0, 1, 2]] * 3, dtype=torch.int32)
    inp["load_indices"] = torch.tensor([0, 17, 18], dtype=torch.int32)
    inp["store_indices"] = torch.tensor([[1 + 5 * r + t for t in range(T)] for r in range(3)], dtype=torch.int32)
    dummy = 23

    conv_v, ssm_v = inp["conv_state"].clone(), inp["ssm_state"].clone()
    out_v = gdn_core_verify_reference(*_verify_args(inp, conv=conv_v, ssm=ssm_v))

    for r in range(3):
        s = inp["store_indices"][r].tolist()
        # Node 3 equals the sequential path 0 -> 1 -> 3 (output and stored state).
        outs, path_conv, path_ssm = _path_reference_row(inp, r, [0, 1, 3], [dummy, dummy, s[3]])
        torch.testing.assert_close(out_v[r, 0], outs[0], **BATCH_TOL)
        torch.testing.assert_close(out_v[r, 1], outs[1], **BATCH_TOL)
        torch.testing.assert_close(out_v[r, 3], outs[2], **BATCH_TOL)
        torch.testing.assert_close(ssm_v[s[3]], path_ssm[s[3]], **BATCH_TOL)
        assert torch.equal(conv_v[s[3]], path_conv[s[3]])
        # Node 4 equals the sequential path 0 -> 2 -> 4.
        outs, path_conv, path_ssm = _path_reference_row(inp, r, [0, 2, 4], [dummy, dummy, s[4]])
        torch.testing.assert_close(out_v[r, 2], outs[1], **BATCH_TOL)
        torch.testing.assert_close(out_v[r, 4], outs[2], **BATCH_TOL)
        torch.testing.assert_close(ssm_v[s[4]], path_ssm[s[4]], **BATCH_TOL)
        assert torch.equal(conv_v[s[4]], path_conv[s[4]])
        # Nodes 1 and 2 both start from node 0's state: the two-node paths pin them to it.
        outs1, _, _ = _path_reference_row(inp, r, [0, 1], [dummy, dummy])
        outs2, _, _ = _path_reference_row(inp, r, [0, 2], [dummy, dummy])
        torch.testing.assert_close(out_v[r, 1], outs1[1], **BATCH_TOL)
        torch.testing.assert_close(out_v[r, 2], outs2[1], **BATCH_TOL)


@pytest.mark.parametrize("H,HV", [HEAD_COUNTS[0], HEAD_COUNTS[3]])
def test_padding_row_and_unstored_nodes(H, HV):
    D, T = 128, 4
    inp = _make_inputs(H, HV, D, T=T, S=16)
    inp["parent_idx"] = torch.tensor([[-1, 0, 1, 2]] * 3, dtype=torch.int32)
    inp["load_indices"] = torch.tensor([1, 2, -1], dtype=torch.int32)  # row 2 is padding
    store_all = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]], dtype=torch.int32)
    inp["store_indices"] = store_all
    init_conv, init_ssm = inp["conv_state"].clone(), inp["ssm_state"].clone()

    conv_a, ssm_a = init_conv.clone(), init_ssm.clone()
    out_a = gdn_core_verify_reference(*_verify_args(inp, conv=conv_a, ssm=ssm_a))

    # The padding row: zero outputs, and no writes to the slots its store_indices name.
    assert torch.equal(out_a[2], torch.zeros(T, HV, D, dtype=torch.float32))
    for slot in (11, 12, 13, 14):
        assert torch.equal(conv_a[slot], init_conv[slot])
        assert torch.equal(ssm_a[slot], init_ssm[slot])
    assert not torch.equal(ssm_a[3], init_ssm[3])  # a real row does write its stored slots

    # Node 2 of row 0 with store -1: it keeps its output and writes nothing.
    store_b = store_all.clone()
    store_b[0, 2] = -1
    conv_b, ssm_b = init_conv.clone(), init_ssm.clone()
    out_b = gdn_core_verify_reference(*_verify_args(inp, conv=conv_b, ssm=ssm_b, store=store_b))
    assert torch.equal(out_b, out_a)
    assert torch.equal(conv_b[5], init_conv[5])
    assert torch.equal(ssm_b[5], init_ssm[5])

    # Every slot not named in store_b is unchanged (slots 11..14 are named by the padding row, checked above).
    named = {int(v) for v in store_b.reshape(-1) if int(v) >= 0}
    for slot in range(16):
        if slot not in named:
            assert torch.equal(conv_b[slot], init_conv[slot])
            assert torch.equal(ssm_b[slot], init_ssm[slot])


@pytest.mark.parametrize("H,HV", [HEAD_COUNTS[0], HEAD_COUNTS[3]])
def test_bad_parent_raises(H, HV):
    D, T = 128, 4
    inp = _make_inputs(H, HV, D, T=T, S=16)
    inp["parent_idx"] = torch.tensor([[-1, 0, 1, 2]] * 3, dtype=torch.int32)
    inp["load_indices"] = torch.tensor([1, 2, -1], dtype=torch.int32)  # row 2 is padding
    inp["store_indices"] = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [-1, -1, -1, -1]], dtype=torch.int32)

    for t, value in ((0, 0), (2, 2)):  # parent_idx[r, 0] = 0 and parent_idx[r, 2] = 2
        bad = inp["parent_idx"].clone()
        bad[0, t] = value
        with pytest.raises(ValueError):
            gdn_core_verify_reference(*_verify_args(inp, parent=bad))
        on_padding = inp["parent_idx"].clone()
        on_padding[2, t] = value
        gdn_core_verify_reference(*_verify_args(inp, parent=on_padding))  # no raise: row 2 is padding
