# SPDX-License-Identifier: Apache-2.0
"""The tiny Flash-Next model with the PLE n-gram ids from the host (``VLLM_HPU_PLE_HOST_NGRAM_IDS``), on CPU.

``test_ple_host_ngram_ids.py`` checks the ids and the PLE layer. Here the whole model generates:

1. the committed reference prompts (dense attention, the harness of ``test_decode_parity.py``): a prefill and 15
   free-running greedy decode steps give HF's tokens with the flag 0 and with the flag 1, and the two runs'
   logits are equal bit for bit;
2. prompts in the HPU runner's layout (``hpu_bucket.py``: the real sparse-QSA wrapper, a padded prompt bucket
   whose padding ids are ``-1``, decode steps in a padded batch), in one piece and as two chunks with the
   n-gram context of the second chunk re-fed by the runner, and the 2600-token reference prompt (the one sparse
   attention matters for): every row of every call is equal bit for bit between the two values of the flag,
   and the generated tokens are HF's;
3. speculative decoding: the committed MTP and n-gram end-to-end flows (``test_mtp_end_to_end.py``,
   ``test_ngram_end_to_end.py``: verification steps of ``1 + k`` tokens for k = 3, 4 and 5, every accepted
   count, ragged rows, the ``-1`` placeholder of an empty draft, a forced rejection) pass unchanged when every
   target forward gets its ids from the host. The same flows run with the flag 0 in their own files.

With the flag 1 the model receives the ready ids as the ``ple_ngram_ids`` input (``hpu_bucket.runner_ple_kwargs``
does what the runner does) and the device id computation, ``compute_ngram_ids``, must not run at all.
"""

import json
import os

os.environ.setdefault("VLLM_PLUGINS", "")
os.environ["TMPDIR"] = "/tmp"

import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

import vllm_gaudi.models.qwen4_exp as flash_next  # noqa: E402
from vllm_gaudi.models.qwen4_exp.ple_layer import HpuQwen4ExpNGramEmbedding  # noqa: E402
from vllm_gaudi.models.qwen4_exp.qsa import QsaMode  # noqa: E402

from tests.unit_tests.models.flash_next.cpu_model import (  # noqa: E402
    cpu_model, dense_reference_prompts, eos_token_id, fp32_logits, ple_kwargs, prefill_metadata, reset_cpu_states)
from tests.unit_tests.models.flash_next.hpu_bucket import (  # noqa: E402
    bucket_decode, bucket_prefill, clamp_rotary_padding, install_hpu_sparse_wrapper, prompt_bucket,
    runner_ple_kwargs)
import tests.unit_tests.models.flash_next.test_mtp_end_to_end as mtp_flows  # noqa: E402
import tests.unit_tests.models.flash_next.test_ngram_end_to_end as ngram_flows  # noqa: E402
from tests.unit_tests.models.flash_next.test_decode_parity import (  # noqa: E402
    decode_metadata, install_cached_attention)

FLAG = "VLLM_HPU_PLE_HOST_NGRAM_IDS"
MAX_MODEL_LEN = 4096
BLOCK = 128
PAD_BLOCK = MAX_MODEL_LEN // BLOCK - 1
NUM_SLOTS = 6
SLOT = 1
PAD_STATE = NUM_SLOTS - 1


@pytest.fixture()
def device_id_calls(monkeypatch):
    """Counts the device id computations (``HpuQwen4ExpNGramEmbedding.compute_ngram_ids``) of the test."""
    calls = []
    original = HpuQwen4ExpNGramEmbedding.compute_ngram_ids

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HpuQwen4ExpNGramEmbedding, "compute_ngram_ids", counted)
    return calls


def load_reference(tiny_checkpoint) -> dict:
    return json.loads((tiny_checkpoint / "reference.json").read_text(encoding="utf-8"))


def generate_dense(model, vllm_config, atts, prompt_ids, eos, steps: int) -> tuple[list[int], torch.Tensor]:
    """Greedy generation of ``steps`` tokens (one prefill, ``steps - 1`` single-token decodes), free running.

    Returns ``(tokens, logits [steps, vocab])``; every model call takes its PLE inputs from
    ``runner_ple_kwargs``, so the flag decides where the n-gram ids are computed.
    """
    reset_cpu_states(model)
    for att in atts:
        att.reset()
        att.slot = SLOT
    seq = list(prompt_ids)
    ids = torch.tensor(seq)
    with torch.no_grad(), set_forward_context(prefill_metadata(len(seq), SLOT), vllm_config):
        hidden = model(input_ids=ids,
                       positions=torch.arange(len(seq)),
                       **runner_ple_kwargs(model, ids, ple_kwargs(len(seq), eos)))
    logits = [fp32_logits(model, hidden)[-1]]
    tokens = [int(logits[-1].argmax())]
    while len(tokens) < steps:
        seq.append(tokens[-1])
        pos = len(seq) - 1
        ids = torch.tensor([seq[pos]])
        ple = {
            "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
            "ngram_context": torch.tensor([[seq[pos - 2] if pos >= 2 else eos, seq[pos - 1]]], dtype=torch.int32),
        }
        with torch.no_grad(), set_forward_context(decode_metadata(SLOT), vllm_config):
            hidden = model(input_ids=ids, positions=torch.tensor([pos]), **runner_ple_kwargs(model, ids, ple))
        logits.append(fp32_logits(model, hidden)[-1])
        tokens.append(int(logits[-1].argmax()))
    return tokens, torch.stack(logits)


def test_reference_prompts_generate_the_same_tokens_with_the_flag_0_and_1(tiny_checkpoint, monkeypatch,
                                                                          device_id_calls):
    reference = load_reference(tiny_checkpoint)
    steps = int(reference["max_new_tokens"])
    prompts = dense_reference_prompts(reference)
    assert [len(p["prompt_ids"]) for p in prompts] == [1, 7, 33, 100, 700]
    with cpu_model(tiny_checkpoint, fp32=True) as (vllm_config, model):
        eos = eos_token_id(vllm_config)
        atts = install_cached_attention(model)
        for prompt in prompts:
            length = len(prompt["prompt_ids"])

            monkeypatch.setenv(FLAG, "0")
            del device_id_calls[:]
            device_tokens, device_logits = generate_dense(model, vllm_config, atts, prompt["prompt_ids"], eos, steps)
            assert len(device_id_calls) == steps, "flag 0: one device id computation per forward of the PLE layer"

            monkeypatch.setenv(FLAG, "1")
            del device_id_calls[:]
            host_tokens, host_logits = generate_dense(model, vllm_config, atts, prompt["prompt_ids"], eos, steps)
            assert not device_id_calls, (f"flag 1, prompt of {length} tokens: the device id computation still ran "
                                         f"{len(device_id_calls)} times")

            assert device_tokens == prompt["generated_ids"], f"prompt of {length} tokens: flag 0 != HF reference"
            assert host_tokens == prompt["generated_ids"], (
                f"prompt of {length} tokens: with the host n-gram ids the model generated {host_tokens}, the "
                f"reference is {prompt['generated_ids']}")
            assert torch.equal(host_logits, device_logits), (
                f"prompt of {length} tokens: logits differ between the flag 1 and 0 by "
                f"{(host_logits - device_logits).abs().max().item():.3e}")


def run_bucket(model, vllm_config, batch) -> torch.Tensor:
    with torch.no_grad(), set_forward_context(batch.metadata, vllm_config):
        return model(input_ids=batch.input_ids,
                     positions=batch.positions,
                     **runner_ple_kwargs(model, batch.input_ids, batch.ple))


def generate_in_buckets(model, vllm_config, attns, prompt_ids, eos, steps: int, chunks: list[int], padded: bool,
                        decode_batch: int):
    """Greedy generation in the runner's layout; returns ``(tokens, hidden states of every call, all rows)``.

    ``chunks`` are the prefill chunk lengths (their sum is the prompt length): each chunk runs in its own prompt
    bucket (``padded``: the smallest bucket that holds it, padding ids ``-1``; else exactly its length), a later
    chunk with the context blocks of the earlier ones and the two tokens before it as its n-gram context. The
    decode steps run in a batch of ``decode_batch`` rows, all but the first padding rows (id 0, the pad block,
    the pad state row).
    """
    reset_cpu_states(model)
    for attn in attns:
        attn.reset()
    seq = list(prompt_ids)
    assert sum(chunks) == len(seq)
    calls = []
    start = 0
    for count in chunks:
        batch = bucket_prefill(seq[start:start + count],
                               slot=SLOT,
                               eos=eos,
                               bucket=prompt_bucket(count) if padded else None,
                               mrope=False,
                               context_len=start,
                               ngram_context=seq[start - 2:start] if start else None)
        if count < batch.bucket:
            assert int(batch.input_ids[-1]) == -1  # the runner's prompt padding reaches the model as -1
        calls.append(run_bucket(model, vllm_config, batch))
        start += count
    tokens = [int(fp32_logits(model, calls[-1][chunks[-1] - 1:chunks[-1]])[0].argmax())]
    while len(tokens) < steps:
        seq.append(tokens[-1])
        pos = len(seq) - 1
        step = bucket_decode([seq[pos]], [pos], [SLOT], [(seq[pos - 2], seq[pos - 1])],
                             batch=decode_batch,
                             eos=eos,
                             pad_state_slot=PAD_STATE,
                             pad_block=PAD_BLOCK,
                             mrope=False,
                             max_blocks=MAX_MODEL_LEN // BLOCK)
        calls.append(run_bucket(model, vllm_config, step))
        tokens.append(int(fp32_logits(model, calls[-1][:1])[0].argmax()))
    return tokens, calls


# EOS inside the prompt, on both sides of the 512-token chunk boundary and right before the first decoded token.
EOS_INSIDE = (100, 101, 350, 510, 512, 698)


@pytest.mark.parametrize("chunks,padded,decode_batch,eos_at", [([700], True, 4, ()), ([512, 188], True, 4, ()),
                                                               ([512, 188], True, 4, EOS_INSIDE),
                                                               ([2600], False, 1, ())],
                         ids=["700_in_one_padded_bucket", "700_in_two_chunks", "700_in_two_chunks_eos_inside",
                              "2600_sparse"])
def test_runner_buckets_generate_the_same_tokens_with_the_flag_0_and_1(tiny_checkpoint, monkeypatch, device_id_calls,
                                                                       chunks, padded, decode_batch, eos_at):
    """Reference prompts through the real sparse-QSA wrapper, in the runner's layout (see the module docstring).

    The 700-token prompt sits in the 768-token bucket, or as chunks of 512 (a full bucket) and 188 (the
    256-token bucket), with decode batches of 4 rows. The 2600-token prompt runs unpadded with a batch of 1,
    the layout in which the wrapper reproduces HF's tokens (``test_sparse_padded_bucket.py``: in a padded bucket
    the top-512 block selection breaks its ties differently). The reference prompts hold one EOS, their first
    token, which cuts no n-gram history; ``eos_inside`` writes EOS into the prompt (no HF reference for that
    prompt: the two values of the flag are compared with each other).
    """
    reference = load_reference(tiny_checkpoint)
    prompt = dict(next(p for p in reference["prompts"] if len(p["prompt_ids"]) == sum(chunks)))
    steps = int(reference["max_new_tokens"])
    with cpu_model(tiny_checkpoint,
                   fp32=True,
                   num_slots=NUM_SLOTS,
                   max_model_len=MAX_MODEL_LEN,
                   qsa_mode=QsaMode.SPARSE) as (vllm_config, model):
        attns = install_hpu_sparse_wrapper(model, monkeypatch, cache_slots=MAX_MODEL_LEN)
        clamp_rotary_padding(model, monkeypatch)
        eos = eos_token_id(vllm_config)
        if eos_at:
            prompt["prompt_ids"] = [eos if index in eos_at else token
                                    for index, token in enumerate(prompt["prompt_ids"])]

        monkeypatch.setenv(FLAG, "0")
        device_tokens, device_calls = generate_in_buckets(model, vllm_config, attns, prompt["prompt_ids"], eos,
                                                          steps, chunks, padded, decode_batch)
        assert len(device_id_calls) == len(device_calls) == len(chunks) + steps - 1

        del device_id_calls[:]
        monkeypatch.setenv(FLAG, "1")
        host_tokens, host_calls = generate_in_buckets(model, vllm_config, attns, prompt["prompt_ids"], eos, steps,
                                                      chunks, padded, decode_batch)
        assert not device_id_calls, f"flag 1: the device id computation still ran {len(device_id_calls)} times"

        assert host_tokens == device_tokens, f"flag 1 generated {host_tokens}, flag 0 {device_tokens}"
        if not eos_at:
            assert host_tokens == prompt["generated_ids"], f"{host_tokens} != HF reference {prompt['generated_ids']}"
        for index, (got, want) in enumerate(zip(host_calls, device_calls)):
            assert torch.equal(got, want), (f"call {index}: hidden states (padding rows included) differ between "
                                            f"the flag 1 and 0 by {(got - want).abs().max().item():.3e}")


@pytest.fixture()
def runner_feeds_host_ids(monkeypatch):
    """Stand in for the runner at the model boundary, with the flag on: every target forward that carries PLE
    inputs gets them from ``runner_ple_kwargs`` (the ids hashed on the host, no ``ngram_context``).

    Returns the list the fed forwards are counted in. The committed flows build their own model and call it with
    ``query_start_loc`` / ``ngram_context``; this makes them run as the runner would run them with the flag 1.
    """
    monkeypatch.setenv(FLAG, "1")
    fed = []
    target_cls = flash_next._MtpTargetOutputs  # the forward both Flash-Next target classes share
    original = target_cls.forward

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if "ngram_context" in kwargs:
            ple = {"query_start_loc": kwargs.pop("query_start_loc"), "ngram_context": kwargs.pop("ngram_context")}
            kwargs.update(runner_ple_kwargs(self, input_ids, ple))
            assert "ngram_context" not in kwargs
            fed.append(1)
        return original(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    monkeypatch.setattr(target_cls, "forward", forward)
    return fed


SPECULATIVE_FLOWS = [
    mtp_flows.test_dense_fp32_greedy_stream_equals_reference,  # the drafter, 6 verification steps of 1 + k tokens
    mtp_flows.test_every_accepted_count_matches_the_plain_path,  # a rejection at each draft position, state parity
    mtp_flows.test_forced_rejection_rolls_back,
    ngram_flows.test_ngram_ragged_chain_dense,  # ragged rows and an empty draft
    ngram_flows.test_placeholder_draft_does_not_crash,
    ngram_flows.test_ngram_ragged_sparse_ring,  # the 2600-token prompt, sparse
]


@pytest.mark.parametrize("flow", SPECULATIVE_FLOWS, ids=lambda flow: flow.__name__)
def test_speculative_flows_pass_with_the_ids_from_the_host(tiny_checkpoint, runner_feeds_host_ids, device_id_calls,
                                                           flow, spec_k):
    """Each flow at each speculative size (the flows take ``spec_k`` themselves; here the fixture supplies it)."""
    flow(tiny_checkpoint, spec_k)
    assert len(runner_feeds_host_ids) >= 3, "the flow ran fewer target forwards than a prefill and two steps"
    assert not device_id_calls, f"the device id computation ran {len(device_id_calls)} times with the flag 1"
