# SPDX-License-Identifier: Apache-2.0
"""No in-place write through a slice of a persistent state under a speculative config (card job 00021).

A speculative config widens the conv states of the PLE layer and of the GDN layers (the committed history in rows
``[0, state_len)``, candidate rows behind it). The prompt path and the plain decode path under such a config
wrote the first rows through a slice of the state tensor; on the card that is ``hpu::slice_insert`` inside an HPU
graph and the first prompt of a speculative server died with ``Neither storage attached to input tensor, not its
view``. ``slice_write_probe.SliceWriteProbe`` models the rule on CPU; these tests hold the three paths to it and
to the values the slice write gave (first rows updated, the rest of the row untouched).
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

from tests.unit_tests.models.flash_next.slice_write_probe import SliceWriteProbe  # noqa: E402
from tests.unit_tests.models.flash_next.test_ple_layer import (  # noqa: E402, F401 - ple_layer is a fixture
    CHANNELS, SLOTS, SPEC_STATE, STATE_LEN, TOL, _eos, _md, ple_layer,
)
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn, hpu_causal_conv1d_update  # noqa: E402

CANDIDATE = 7.0  # what the rows behind the committed history hold before the step


def test_the_probe_sees_the_card_failure_and_nothing_else():
    rows = torch.tensor([1, 3])

    def violations(write):
        state = SliceWriteProbe.wrap(torch.zeros(6, 5, 4))
        write(state)
        return list(SliceWriteProbe.violations)

    def assign(state, last):
        state[rows, 0:last, :] = torch.ones(2, last, 4)

    # the card failure, and the fork's indexed assignment on a widened state
    assert violations(lambda s: s[:, :3, :].index_copy_(0, rows, torch.ones(2, 3, 4)))
    assert violations(lambda s: assign(s, 3))
    assert violations(lambda s: s[:, 0:3, :].copy_(torch.ones(6, 3, 4)))
    # whole rows, a whole-dimension assignment (the non-speculative form), whole-tensor views, reads
    assert not violations(lambda s: s.index_copy_(0, rows, torch.ones(2, 5, 4)))
    assert not violations(lambda s: assign(s, 5))
    assert not violations(lambda s: s.transpose(-1, -2).index_copy_(0, rows, torch.ones(2, 4, 5)))
    assert not violations(lambda s: s.view(-1, 4).index_copy_(0, rows, torch.ones(2, 4)))
    assert not violations(lambda s: s.index_copy_(0, rows, torch.cat((torch.ones(2, 3, 4),
                                                                      s.index_select(0, rows)[:, 3:, :]), dim=1)))


def _ple_steps(vllm_config, layer, ids, hidden, eos, slot):
    """A 10-token prompt, then two plain decode steps, into ``slot`` of the layer's state."""
    def run(step_ids, step_hidden, qsl, ctx, md):
        with torch.no_grad(), set_forward_context(md, vllm_config):
            return layer(step_hidden, step_ids, qsl, ctx)

    outs = [run(ids[:10], hidden[:10], torch.tensor([0, 10], dtype=torch.int32),
                torch.full((1, 2), eos, dtype=torch.int32), _md(True, [slot], has_initial=[0]))]
    for pos in (10, 11):
        outs.append(run(ids[pos:pos + 1], hidden[pos:pos + 1], torch.tensor([0, 1], dtype=torch.int32),
                        ids[pos - 2:pos].unsqueeze(0).to(torch.int32), _md(False, [slot])))
    return outs


def test_ple_prompt_and_plain_decode_write_whole_rows_of_a_widened_state(ple_layer):
    vllm_config, layer = ple_layer
    eos = _eos(vllm_config)
    torch.manual_seed(5)
    ids = torch.randint(0, 1000, (12, ), dtype=torch.int64)
    hidden = torch.randn(12, CHANNELS)

    layer.kv_cache = (torch.zeros(SLOTS, STATE_LEN, CHANNELS), )
    plain_outs = _ple_steps(vllm_config, layer, ids, hidden, eos, slot=2)
    plain_state = layer.kv_cache[0].clone()

    wide = torch.zeros(SLOTS, SPEC_STATE, CHANNELS)
    wide[:, STATE_LEN:, :] = CANDIDATE
    layer.kv_cache = (SliceWriteProbe.wrap(wide), )
    wide_outs = _ple_steps(vllm_config, layer, ids, hidden, eos, slot=2)

    assert SliceWriteProbe.violations == []
    for plain, widened in zip(plain_outs, wide_outs):
        torch.testing.assert_close(widened.as_subclass(torch.Tensor), plain, **TOL)
    state = layer.kv_cache[0].as_subclass(torch.Tensor)
    torch.testing.assert_close(state[:, :STATE_LEN, :], plain_state, **TOL)
    assert torch.equal(state[:, STATE_LEN:, :], torch.full((SLOTS, SPEC_STATE - STATE_LEN, CHANNELS), CANDIDATE))


def _gdn_states(slots, state_len, dim, extra):
    torch.manual_seed(11)
    narrow = torch.randn(slots, state_len, dim)
    wide = torch.cat((narrow, torch.full((slots, extra, dim), CANDIDATE)), dim=1)
    return narrow, wide


def test_gdn_conv_prefill_writes_whole_rows_of_a_widened_state():
    dim, width, batch, seq, slots = 6, 4, 3, 5, 7
    state_len = width - 1
    torch.manual_seed(1)
    x = torch.randn(dim, batch * seq)
    weight = torch.randn(dim, width)
    qsl = torch.tensor([0, 5, 8, 8], dtype=torch.int32)  # 5 and 3 real tokens, one padding row
    cache = torch.tensor([4, 1, -1])  # the padding row writes the last (garbage) slot
    has_initial = torch.tensor([1, 0, 0])
    narrow, wide = _gdn_states(slots, state_len, dim, extra=3)

    plain_state = narrow.clone()
    plain = hpu_causal_conv1d_fn(x, weight, None, plain_state, qsl, cache_indices=cache,
                                 has_initial_state=has_initial)
    probe = SliceWriteProbe.wrap(wide.clone())
    widened = hpu_causal_conv1d_fn(x, weight, None, probe, qsl, cache_indices=cache, has_initial_state=has_initial)

    assert SliceWriteProbe.violations == []
    torch.testing.assert_close(widened.as_subclass(torch.Tensor), plain)
    state = probe.as_subclass(torch.Tensor)
    real = slots - 1  # every slot but the garbage slot
    torch.testing.assert_close(state[:real, :state_len, :], plain_state[:real])
    assert torch.equal(state[:real, state_len:, :], wide[:real, state_len:, :])
    assert not torch.equal(plain_state[4], narrow[4])  # the step did write


def test_gdn_conv_plain_decode_writes_whole_rows_of_a_widened_state():
    dim, width, batch, slots = 6, 4, 3, 7
    state_len = width - 1
    torch.manual_seed(2)
    x = torch.randn(batch, dim)  # token-major, one token per request, as the GDN layer hands it over
    qsl = torch.arange(batch + 1, dtype=torch.int32)
    weight = torch.randn(dim, width)
    load = torch.tensor([4, 1, -1])
    store = torch.tensor([5, 1, -1])  # prefix caching: one request stores to another slot
    narrow, wide = _gdn_states(slots, state_len, dim, extra=3)

    plain_state = narrow.clone()
    plain = hpu_causal_conv1d_update(x, plain_state, weight, None, "silu", conv_state_indices=load,
                                     query_start_loc=qsl, store_cache_indices=store)
    probe = SliceWriteProbe.wrap(wide.clone())
    widened = hpu_causal_conv1d_update(x, probe, weight, None, "silu", conv_state_indices=load,
                                       query_start_loc=qsl, store_cache_indices=store)

    assert SliceWriteProbe.violations == []
    torch.testing.assert_close(widened.as_subclass(torch.Tensor), plain)
    state = probe.as_subclass(torch.Tensor)
    real = slots - 1
    torch.testing.assert_close(state[:real, :state_len, :], plain_state[:real])
    assert torch.equal(state[:real, state_len:, :], wide[:real, state_len:, :])
    assert not torch.equal(plain_state[5], narrow[5])
