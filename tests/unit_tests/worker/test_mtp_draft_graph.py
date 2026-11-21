# SPDX-License-Identifier: Apache-2.0
"""The k Flash-Next MTP draft passes of a decode propose in one forward (``MtpDraftLoop``, VLLM_HPU_MTP_DRAFT_GRAPH).

Today every draft pass is its own HPU-graph replay plus a lazy LM head and arg max between the replays, and the
host builds each later pass's attention metadata between them: at 32 streams on TP8 the draft costs 16 to 18 ms of
a 104 ms step and is host-bound (card job 00178, the MTP step timer). ``MtpDraftLoop`` runs pass 0, the LM head,
the arg max and the passes 1 .. k-1 inside one forward, so one replay issues them all; the metadata of every later
pass is built before it (the host knows every pass's positions).

These tests hold the loop to a transcription of the per-pass proposer (``HpuFlashNextMtpProposer.propose``): the
same calls in the same order with the same inputs, the same draft ids, and nothing in the forward that reads a
device value back to the host (inside an HPU graph capture such a read sees no data).
"""
import os

os.environ.setdefault("VLLM_PLUGINS", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_gaudi.v1.spec_decode import flash_next_mtp_steps as steps  # noqa: E402

H, HC, VOCAB = 4, 2, 23
W = H * HC


class FakeDraft:
    """A deterministic stand-in of the draft adapter's forward and the proposer's LM head + arg max.

    The pass's output depends on every input (ids, positions, hidden, metadata tag, step index), so a loop that
    hands a pass the wrong input gives other ids.
    """

    def __init__(self, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.embed = torch.randn(VOCAB, H, generator=gen)
        self.mix = torch.randn(W, H, generator=gen)
        self.spread = torch.randn(H, W, generator=gen)
        self.lm_head = torch.randn(H, VOCAB, generator=gen)
        self.calls = []

    def pass_forward(self, *, input_ids, positions, hidden_states, inputs_embeds, attn_metadata, spec_step_idx):
        assert inputs_embeds is None
        self.calls.append({"ids": input_ids.clone(), "positions": positions.clone(), "hidden": hidden_states.clone(),
                           "md": attn_metadata, "step": spec_step_idx})
        ids = input_ids.reshape(-1).long()
        pos = positions.reshape(-1).to(torch.float32)
        hidden = hidden_states.reshape(-1, W)
        sample = self.embed.index_select(0, ids) + hidden @ self.mix + 0.01 * pos[:, None] + attn_metadata["tag"]
        sample = sample + 0.1 * spec_step_idx
        multi = torch.tanh(sample @ self.spread)
        return sample.view(-1, 1, H), multi.view(-1, 1, W)

    def draft_tokens(self, hidden):
        return (hidden @ self.lm_head).argmax(dim=-1)


def reference_drafts(draft, k, max_model_len, target_token_ids, target_positions, target_hidden_states,
                     last_token_indices, attn_metadata, step_metadata):
    """``HpuFlashNextMtpProposer.propose`` with ``self.model(...)`` as ``draft.pass_forward`` (the per-pass path)."""
    batch_size = last_token_indices.shape[0]
    ret = draft.pass_forward(input_ids=target_token_ids, positions=target_positions,
                             hidden_states=target_hidden_states, inputs_embeds=None, attn_metadata=attn_metadata,
                             spec_step_idx=0)
    last_hidden_states, hidden_states = steps.split_draft_output(ret)
    last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
    sample_hidden_states = torch.index_select(last_hidden_states, 0, last_token_indices)
    draft_token_ids = draft.draft_tokens(sample_hidden_states)
    if k == 1:
        return draft_token_ids.view(-1, 1)
    target_positions = target_positions.view(-1)
    positions = torch.index_select(target_positions, 0, last_token_indices)
    hidden_states = steps.next_step_hidden(hidden_states, last_token_indices)
    draft_token_ids_list = [draft_token_ids]
    for token_index in range(k - 1):
        input_ids = draft_token_ids_list[-1].int()
        positions += 1
        exceeds_max_model_len = positions >= max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
        ret = draft.pass_forward(input_ids=input_ids.view(-1, 1), positions=clamped_positions.view(-1, 1),
                                 hidden_states=hidden_states.view(-1, 1, hidden_states.shape[-1]), inputs_embeds=None,
                                 attn_metadata=step_metadata[token_index], spec_step_idx=token_index + 1)
        last_hidden_states, hidden_states = steps.split_draft_output(ret)
        last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[:batch_size]
        draft_token_ids_list.append(draft.draft_tokens(last_hidden_states[:batch_size]))
    return torch.stack(draft_token_ids_list, dim=1)


def decode_inputs(batch_size, k, max_model_len, seed=1):
    """A decode propose's inputs: ``1 + k`` rows per request, one accepted row each, positions near the end."""
    gen = torch.Generator().manual_seed(seed)
    rows = batch_size * (1 + k)
    target_token_ids = torch.randint(0, VOCAB, (rows, 1), generator=gen, dtype=torch.int32)
    base = torch.randint(0, max_model_len, (batch_size, ), generator=gen)
    base[0] = max_model_len - 2  # the later passes of request 0 cross max_model_len: their positions clamp to 0
    target_positions = (base[:, None] + torch.arange(1 + k)[None, :]).reshape(rows, 1)
    target_hidden_states = torch.randn(rows, 1, W, generator=gen)
    accepted = torch.randint(0, 1 + k, (batch_size, ), generator=gen)
    last_token_indices = torch.arange(batch_size) * (1 + k) + accepted
    attn_metadata = {"tag": 0.5}
    step_metadata = tuple({"tag": 0.25 * (j + 1)} for j in range(k - 1))
    return target_token_ids, target_positions, target_hidden_states, last_token_indices, attn_metadata, step_metadata


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_the_loop_gives_the_drafts_of_the_per_pass_proposer(k, batch_size):
    max_model_len = 64
    inputs = decode_inputs(batch_size, k, max_model_len, seed=k * 10 + batch_size)
    ref_draft, loop_draft = FakeDraft(), FakeDraft()
    want = reference_drafts(ref_draft, k, max_model_len, *(x.clone() if torch.is_tensor(x) else x for x in inputs))
    loop = steps.MtpDraftLoop(loop_draft.pass_forward, loop_draft.draft_tokens, k, max_model_len)
    got = loop(*inputs)
    assert got.shape == (batch_size, k)
    assert got.dtype == want.dtype
    assert torch.equal(got, want)
    # the same passes, in the same order, with the same inputs
    assert [c["step"] for c in loop_draft.calls] == list(range(k))
    assert [c["md"] for c in loop_draft.calls] == [inputs[4], *inputs[5]]
    for ref_call, loop_call in zip(ref_draft.calls, loop_draft.calls):
        for key in ("ids", "positions", "hidden"):
            assert ref_call[key].dtype == loop_call[key].dtype, key
            assert ref_call[key].shape == loop_call[key].shape, key
            assert torch.equal(ref_call[key], loop_call[key]), key


def test_later_passes_take_the_previous_ids_as_int32_and_the_next_positions():
    k, batch_size, max_model_len = 4, 3, 64
    inputs = decode_inputs(batch_size, k, max_model_len)
    draft = FakeDraft()
    got = steps.MtpDraftLoop(draft.pass_forward, draft.draft_tokens, k, max_model_len)(*inputs)
    _, target_positions, _, last_token_indices, _, _ = inputs
    base = target_positions.view(-1).index_select(0, last_token_indices)
    for j, call in enumerate(draft.calls[1:], start=1):
        assert call["ids"].dtype == torch.int32 and call["ids"].shape == (batch_size, 1)
        assert torch.equal(call["ids"].view(-1).long(), got[:, j - 1])
        want = torch.where(base + j >= max_model_len, torch.zeros_like(base), base + j)
        assert torch.equal(call["positions"].view(-1), want)
        assert call["hidden"].shape == (batch_size, 1, W)
    # request 0 sits at max_model_len - 2: its pass 2 and later attend from position 0, as the proposer's clamp does
    assert int(draft.calls[2]["positions"].view(-1)[0]) == 0


def test_the_step_metadata_must_match_k():
    k, batch_size, max_model_len = 3, 2, 64
    inputs = list(decode_inputs(batch_size, k, max_model_len))
    draft = FakeDraft()
    loop = steps.MtpDraftLoop(draft.pass_forward, draft.draft_tokens, k, max_model_len)
    inputs[5] = inputs[5][:1]
    with pytest.raises(ValueError, match="2 later passes"):
        loop(*inputs)
    with pytest.raises(ValueError):
        steps.MtpDraftLoop(draft.pass_forward, draft.draft_tokens, 0, max_model_len)


class HostReadGuard:
    """Make every host read of a tensor value fail while it is active (inside an HPU graph capture such a read
    sees no data: the capture of a disable_tensor_cache graph is a dry run)."""

    NAMES = ("item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__")

    def __enter__(self):
        self.saved = {name: getattr(torch.Tensor, name) for name in self.NAMES}

        def refuse(name):

            def method(*args, **kwargs):
                raise AssertionError(f"host read inside the draft graph: Tensor.{name}")

            return method

        for name in self.NAMES:
            setattr(torch.Tensor, name, refuse(name))
        return self

    def __exit__(self, *exc):
        for name, method in self.saved.items():
            setattr(torch.Tensor, name, method)
        return False


@pytest.mark.parametrize("k", [1, 3, 5])
def test_the_loop_reads_no_device_value_back(k):
    max_model_len = 64
    inputs = decode_inputs(4, k, max_model_len)
    draft = FakeDraft()
    loop = steps.MtpDraftLoop(draft.pass_forward, draft.draft_tokens, k, max_model_len)
    with HostReadGuard():
        got = loop(*inputs)
    assert got.shape == (4, k)


def test_the_loop_is_a_module_whose_forward_a_graph_wrapper_can_replace():
    # wrap_in_hpu_graph replaces module.forward with its capture/replay closure; the loop must hold no submodule
    # (the draft model stays the proposer's: its parameters are not registered twice)
    draft = FakeDraft()
    loop = steps.MtpDraftLoop(draft.pass_forward, draft.draft_tokens, 2, 64)
    assert isinstance(loop, torch.nn.Module)
    assert list(loop.parameters()) == [] and list(loop.children()) == []
    calls = []
    original = loop.forward

    def wrapped(*args, **kwargs):
        calls.append(len(args))
        return original(*args, **kwargs)

    loop.forward = wrapped
    loop(*decode_inputs(2, 2, 64))
    assert calls == [6]
