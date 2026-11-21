# Flash-Next decode: the fixed cost of a step, and what replays it (B10, C6)

Version 0.2, 2026-11-20 (track native-replay). Labels: **VERIFIED** (code at file:line, a card log, or a test),
**INFERRED** (reasoning on verified inputs), **ASSUMPTION** (not grounded; the range that matters is given).
This file is the design; measurements land in section 7 as the card jobs end.

## 0. Decision (2026-11-20 18:40 UTC; measurements in section 7)

1. **Decode already replays.** A Flash-Next decode step runs from a stock HPU graph (`wrap_in_hpu_graph`), so
   the host does not rebuild the graph every step (section 2). What B10 was meant to remove, graph building at
   about 16 us per op (section 7.1), is gone at decode.
2. **The fixed cost is the replay's host-issued launches.** A replay issues every captured recipe and collective
   from the caller's thread (`asynchronous=False`): about 80-120 us of host time per recipe and about 130-195 us
   per all-reduce at TP8, measured (section 7.2). A Flash-Next step replays about 146 recipes and 98 collectives
   with C1 on: 13-19 ms of collectives plus 2-4 ms of per-layer cuts, the bulk of the fixed term.
3. **The fork's native replay removes exactly that, but does not transfer.** It publishes pre-built compute and
   NIC command pages in one call (about 16 us per collective point at TP4), and it needs eager mode, a privately
   patched Bridge/Synapse/HCL, two or four ranks and at most 128 consumers per plan (section 3). Porting it to TP8
   lazy mode is not possible inside the rental.
4. **torch.compile is not cheaper** at TP8 for collectives or launches (section 7.2), and moving Flash-Next to it
   is a large, unproven change (section 4): rejected.
5. **Chosen path, O1 (stock replay with fewer launches):** (a) `VLLM_HPU_DECODE_CUT_INTERVAL=1000` removes 47
   recipe launches per decode step (built, CPU tests green; card parity in jobs 00027/00053); (b) fewer and cheaper
   collectives are the larger lever and belong to the collectives track and the layout decision E1 (section 5);
   (c) what an asynchronous replay and grouped all-reduces change is measured in job 00027 before anything is built.
6. **What is left:** after (a), about 13-19 ms of collectives per step remain at TP8 under stock replay, above the
   whole 5-10 ms budget. Reaching it needs the per-collective cost well under 100 us: TP4 collectives (E1), a
   cheaper collective form, or a native collective plan for TP8 (a long-term item beyond this rental).

7. **Update 2026-11-20 23:45 UTC (jobs 00065, 00066, 00067; sections 7.4-7.8).** No bridge, Synapse or HCCL
   setting lowers the host cost of a launch (22-45 us on one card, 32-76 us per rank at TP8) or of a collective;
   `HCCL_PRIM_COLLECTIVE_MASK=2` kills every rank. A served TP8 step pays about 150-170 us of host time per
   collective, the recipe split included (00073, 00012, 00065 at 32 rows). Asynchronous replay does not shorten an
   8-rank step: keep `VLLM_HPU_GRAPH_ASYNC_REPLAY` off. `VLLM_HPU_DECODE_CUT_INTERVAL=1000` keeps every token at TP1, TP4
   and TP8: safe to switch on (its real-model gain is measured in 00122; expected 47 x 32-90 us = 1.5-4 ms per TP8 step,
   INFERRED). The ranks' host times for the same replayed step differ by about 10 ms in the server (not in the
   benchmark), which is what `VLLM_HPU_WORKER_CPU_PIN` (build step 7, job 00122) targets.

8. **Update 2026-11-21 03:40 UTC (bridge job 00174; section 7.10).** Our own builds of the Gaudi bridge load at TP8 and
   keep every token (00160, 00174). On the real model at TP8 a replayed launch costs about 155 us of host time per
   rank (Synapse launch path 49, input handling 30, patching 22, post 15, launch bookkeeping 16; about 145 launches per
   1-stream step, so about 22 ms of the 38 ms step). The prepared replay (lean step 1, `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`)
   and the stock spinner-off setting (`PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`) each cut about 17 us per launch with the
   same tokens: 1-stream step 38.2 -> 30.8 ms (lean) and 34.6 ms (spinner off), one round each; at 32 streams the step is
   device-bound (-0.8 and -2.2 ms). Lean step 1 gave back 12 us in run_prep (fixed in lean2). Next: lean2 with the
   spinner off on the headline env, then lean3 (sub-timers, then four cuts of -20 to -35 us per launch; section 7.10).

## 1. The question

The target (32 streams, at most 10 ms per token per stream) needs the fixed cost of a step, Ft, at about 5 to 10 ms
(`reports/idea-sweep/target-model.md` s6). Ft is what a decode or verification step costs whatever its row count:
collectives, graph launches and boundaries, host work, the sampler. SPEED-UPGRADES.md lists the class of change that
removes it as B10 ("compiled decoder: a prepared recipe replayed with fixed buffers, so the host stops rebuilding and
hashing the graph every step"), and the fork reached 8.24 ms per token for a whole model at TP4 under its native
replay (`evidence/ROOFLINE.md:3`). The fork's fast collectives exist only in that native runtime (sweep G4-4), so C6
belongs here too.

Card baseline (coordinator job 00007, program head e238f84, fused GDN and HC kernels,
`VLLM_HPU_COLLECTIVE_MARK_STEP=0`, PLE host n-gram ids, short prompts, 256 tokens; compile pauses included):
51.5 ms per token at 1 stream, 48.8 at 8, 67.3 at 16, 83.6 to 102 at 32 (`reports/coord/RESULTS.md`). Step-anatomy
found live bucket compiles inside every round except the second at 32 streams, so the clean numbers are 83.6 ms at
32 streams and, from this morning, 31-35 ms (median gap) at one stream (`reports/step-anatomy/REPORT.md` section 2;
`ref/card-20261120/live/fused_gdn_hc_probe.txt`). A 32-stream step costs about 2.5 times a one-stream step.

## 2. What a decode step is today: lazy mode, replayed from an HPU graph

**B10 as SPEED-UPGRADES.md words it is already in place: decode does not rebuild its graph every step.**

| Fact | Status |
|---|---|
| In lazy mode the runner wraps the model: `htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(...), disable_tensor_cache=True)` | VERIFIED `vllm_gaudi/v1/worker/hpu_model_runner.py:1873-1876`, applied at load `:7089` (line numbers of e238f84) |
| The real-model card run is lazy, not eager, with lazy collectives: `enforce_eager=False`, `PT_HPU_LAZY_MODE = 1`, `PT_HPU_ENABLE_LAZY_COLLECTIVES = 1`, `Wrapping in HPUGraph` on all 8 workers | VERIFIED `ref/card-20261120/run050112/20261120T050112Z/A/server0.log:80, 315-322, 911-958` |
| A step bypasses the graph only for per-layer (Mapping) attention metadata, `enforce_eager`, or a prompt above `max_cudagraph_capture_size` | VERIFIED `hpu_model_runner.py:5663-5687, 7369-7382` |
| Flash-Next takes the legacy metadata path, so decode replays: its QSA layer gives the KV cache to a plain `Attention` child, GDN layers are `MambaBase`, the QSA side caches are runner tensors; the framework layout is chosen only for `AttentionLayerBase` layers that are none of these | VERIFIED `models/qwen4_exp/qsa.py:405-417, 506-514`; `hpu_model_runner.py:2818-2823, 9050-9056`; `qsa_side_caches.py:1-27` |
| The graph key is the bridge's `input_hash` over the inputs: tensors by metadata, Python scalars by value; Flash-Next's metadata fields are whitelisted for it | VERIFIED `hpu_model_runner.py:1909-1935` (`trim_attn_metadata`), `tests/unit_tests/worker/test_flash_next_trim_metadata.py` |
| The fused GDN decode op replays correctly from an HPU graph at TP8 geometry (1,000 steps, exact against eager) | VERIFIED `docs/design/flash-next/native-extensions.md:40` |
| On a replay the host does not run the Python forward: the wrapper calls the original `forward` only on a cache miss (capture), and on a hit hashes the inputs, copies the unmatched ones and calls `replayV3` with `asynchronous=False` unless asked; so `HpuModelAdapter.forward` (metadata processing, forward context, the per-layer hooks) runs only when a graph is captured | VERIFIED in the bridge 1.24.1.482 source dumped on the card by job 00012: `reports/native-replay/card/replay-bench-r8/bridge_src/habana_frameworks_torch_hpu_graphs.py:635-770` (`wrapped_hpugraph_forward`), `:840-905` (`wrap_in_hpu_graph`, `module.forward = forward` at `:905`), `:547-561` (`input_hash`) |
| Per step the runner copies 14 tensors to the device and reads the sampled tokens back once, at 1 and at 32 rows | VERIFIED by the engine-step track's CPU harness (`wt/engine-step`, commit 84aace3a message) |
| `compute_logits` (the logits gather, `lm_head`, the logits all-gather) and the sampler run after the replayed forward, outside the captured graph, as lazy ops every step | VERIFIED: only `forward` is wrapped (row above); `select_and_compute_logits` is called after it, `hpu_model_runner.py:1764-1767, 5752-5760` (e238f84) |

So a decode step on the card is: host-side input preparation, the wrapper's input hash and copies into the captured
inputs, the replay (the captured recipes and collectives launched in order), the logits and sampler as ordinary lazy
ops, one read of the sampled tokens, and the engine's own step. With speculation on, the HPU runner forbids async
scheduling (`hpu_model_runner.py:2361-2362`), so all host work is serial with the device.

### 2.1 Cuts inside the replayed decode graph

| Cut | Count per Flash-Next step | Status |
|---|---|---|
| `mark_step()` hook after every `VLLM_CONFIG_HIDDEN_LAYERS`-th `*DecoderLayer` (default 1) | 48 | VERIFIED `hpu_model_runner.py:880-897, 7074-7082`; upstream `Qwen4ExpDecoderLayer` (`vllm/models/qwen4_exp/nvidia/model.py:176`) |
| `mark_step()` before every all-reduce and all-gather when `VLLM_HPU_COLLECTIVE_MARK_STEP=1` (the program default; the card runs set 0) | 99 | VERIFIED `vllm_gaudi/distributed/device_communicators/hpu_communicator.py:49-55, 71-74`; `envs.py:538` |
| Each lazy collective itself: a Synapse recipe holds compute only, so the bridge executes the collective between two recipes | 98 in the graph | INFERRED (job 00012 measures a cut next to a collective) |
| Outside the graph, every step: the logits gather, `lm_head` and logits all-gather (`select_and_compute_logits`), two runner `mark_step`s around decode and two in `_run_sampling` | 1 collective, about 5 cuts | VERIFIED `hpu_model_runner.py:1764-1767, 5752-5760, 6102-6105, 6619, 6633` (e238f84) |

The 98 all-reduces of the graph: the embedding (1), the PLE table (1), the mixer out-projection of each layer (48)
and the MoE output of each layer (48) (VERIFIED `vllm_gaudi/ops/hpu_vocab_parallel_embedding.py:32`, upstream
`common/ple.py:82`, `models/qwen4_exp/qsa.py:483-490`, upstream `moe_runner.py:503-509`). With the program
defaults a captured decode graph therefore holds 146 cuts (48 + 98) and about 147 recipe segments, about three per
layer; with `VLLM_HPU_COLLECTIVE_MARK_STEP=0` the 98 explicit cuts go, the 98 collectives and 48 layer cuts stay
(INFERRED). The decode forward itself has no host read and no data-dependent shape: no `.item()`, `.cpu()`,
`.tolist()`, `nonzero` or tensor `print` in the GDN, QSA, HC, PLE or MoE decode code (VERIFIED by search; the
fixed-shape writes are `qsa.py:375-389`, `ple_layer.py:411-434`). Under MTP the target's verification forward has
the same cuts, the fused GDN decode op does not run unless `VLLM_HPU_GDN_CORE_VERIFY=1` (`num_accepted_tokens` is
set every step, `models/qwen3_5.py:318-319`), and the drafter adds three replays of about six collectives each.

## 3. The fork's native replay and prepared collectives

Abbreviations: PP.py = `vllm_gaudi/ops/tp2_prepared_plan.py`, PP.h = `tools/communication/tp2_prepared_plan.h`,
NDG = `tools/communication/tp2_native_decode_graph.h`, BR = `tools/communication/tp2_fused_ar_norm_bridge.cpp`,
AR = `vllm_gaudi/distributed/tp2_fused_ar_norm.py`, HCLp/SYNp/BRGp = `tools/communication/patches/native-runtime/`
`hcl.patch`/`synapse.patch`/`bridge.patch`, IDX = `evidence/20260928_tp4-decode-gap-1p5/INDEX.md`,
LED = `docs/dev_guide/deepseek_v41_tp4_optimization_ledger.md`.

### 3.1 What it is

| Layer | What it does | Evidence |
|---|---|---|
| Static group plan | A post-partitioner pass of torch.compile replaces each eligible compiled group by a `PreparedGroupModule`; its `HabanaGraphModule` partitions become prepared recipe launches with preallocated outputs, the collectives become exchange nodes. Replay is one lowering-thread task that launches each recipe and enqueues each collective: "host-side prepared replay, not a Synapse device graph API" | VERIFIED PP.py:697-846, PP.h:3, 232, 288-357, BRGp:47-77 |
| Native compute capture | Each static recipe launch inside `synNativeComputeGraphBeginCapture` becomes a segment owning program memory, workspace and its recorded SCAL commands (dynamic-shape recipes rejected, workspace capped at 2 GiB per plan) | VERIFIED SYNp:6179-6187, 7154-7206, 7975-8061, 7976-7983 |
| Collectives | HCL TP2/TP4 native graphs: warm-up submit, captured second submit, completion-ring wrap template; 64 ring phases per microstream | VERIFIED HCLp:825-1115, 96-97, 563-582 |
| Joint plan | Segment command buffers plus NIC wait packets become prepared command pages with patch sites; a replay stages changed inputs into fixed buffers, patches dependencies and publishes NIC and compute pages | VERIFIED SYNp:6366-6527, 5860-5936; NDG:1289-1463; HCLp:1444-1501, 3108-3140 |
| Dynamic inputs | Positions, hidden state, input ids, the packed metadata, block tables, slot mappings and rotary tables are copied into fixed buffers; KV and recurrent state are bound by address; routed experts are chosen inside the graph | VERIFIED `vllm_gaudi/ops/tp2_graph_inputs.py:63-102`, NDG:625-651, `docs/features/deepseek_v4_native_decode.md:33-36` |
| Completion and reuse | A synapse event per replay recorded on the execute thread; input/scheduled/executed epochs enforce one update per replay; metadata ring slots wait for completion before reuse | VERIFIED NDG:278-358, 741-768; `hpu_model_runner.py:4253-4256` |
| Invalidation | Communicator change, input contract or address change, profiler start/stop, cache or state view release, shutdown; an error after state mutation drops the plan with no retry | VERIFIED PP.py:93-108, 269-289, 402-414; `hpu_model_runner.py:1701-1716` |
| Runtime | Bridge `5176b1b2`, Synapse `6de1f66a` from a third-party public snapshot ("not a source release of the installed SDK"), HCL `11f9114b`; the loader refuses any runtime whose ABI fingerprints differ | VERIFIED `patches/native-runtime/manifest.json:5-21`, `README.md:8-12, 115-116`; AR:76-98 |

### 3.2 Why none of it runs here

1. **Mode.** Every path asserts eager mode: `PT_HPU_LAZY_MODE=0`, the eager pipeline and the eager collective
   pipeline, `ProcessGroupEagerHCCL`, and hooks torch.compile partitions (VERIFIED PP.h:172-176, AR:140-144, 167-171,
   BR:1970-1973, PP.py:957-960). In lazy mode the plugin turns dynamo off (VERIFIED `vllm_gaudi/platform.py:785-792`).
2. **Ranks.** The gates are hard-coded: AR refuses anything but a two-rank TP world (or the V4.1 graph-replay world)
   (AR:131-133); HCL native graph creation needs comm size 2 or 4 and batch replay accepts only 2 or 4 ranks
   (HCLp:251-262, 852-853, 903-909, 3076, 3115-3117); the TP2 exchange sends to `rank ^ 1` (HCLp:3084);
   `tp_peer_allgather` works at exactly 4 ranks (BR:775, 794, 1714) with a 32,768-element cap (AR:562-567,
   HCLp:903); the Qwen native graph requires `tensor_parallel_size == 2` (`platform.py:441-442`). All VERIFIED.
3. **Size of the plan.** The joint plan takes at most 128 consumers (VERIFIED SYNp:6429, 7246); a Flash-Next TP8
   step has 99 collectives, about 198 phases as reduce-scatter plus all-gather (INFERRED), above the cap.
4. **Model contract.** `_eligible` raises on any op it does not know (PP.py:599-615), so the EP8 MoE path would be
   refused (INFERRED); the Qwen adapter requires the direct GDN state update (PP.py:589-598), which excludes
   speculation and prefix caching (`docs/configuration/env_variables.md:384`), both of which this program needs.
5. **Rental.** The patched runtime is built from three source checkouts and fingerprinted; each node start here is a
   fresh machine (BRIEF 08:10), so it would be rebuilt every rental (INFERRED).

### 3.3 What it gained, and against what

| Case | Before | After | Evidence |
|---|---|---|---|
| Qwen3.8 FP8 TP2, one request | "original TP2 route" (compile mode) | about 30 % lower decode latency from the fused all-reduce + residual + norm boundary; about 45 % with 8-layer regional groups as well; stock HCCL transport, **not** native replay | VERIFIED `README.md:193-201` |
| Qwen3.8 TP2 native graph | | no number: "disabled and unqualified" | VERIFIED `docs/configuration/env_variables.md:23-35` |
| DSV4.1 TP4, 16-layer chain | 5.674 ms (compiled 4-layer groups, stock HCCL, 43 segments) | per-group native replay 6.048 ms (**6.6 % slower**); one whole-stage native graph with peer all-gather and shared compute 5.261 ms (7.3 % faster) | VERIFIED IDX:280, 298, 345, 351, 425; LED:273, 316 |
| DSV4.1 TP4, whole model | 14.62 ms/token | 11.08 ms/token, from shared C1 compute, N-card peer exchange and stage replay together; later 9.56 -> 8.24 under native replay | VERIFIED IDX:466; LED:9-13, 259-268; `evidence/ROOFLINE.md:3` |
| Device idle under native replay | | about 27-30 % of the step stayed non-compute (2.25 of 8.18 ms at the 8.24 baseline); about 16 us per producer/consumer handoff survived | VERIFIED `evidence/ROOFLINE.md:5-6, 20`; LED:293-298 |
| Recipe count under native replay | 73 segments | 41 segments: **0.13-0.20 ms slower** | VERIFIED `evidence/ROOFLINE.md:171-175, 188` |

Reading (INFERRED): the fork's measured gains came from fewer and fused collective boundaries and from shared
compute, much less from replaying launches; per-group replay alone lost. Its fast collective is a one-phase peer
all-gather with an ordered local sum (`evidence/ROOFLINE.md:60-64`), which is what C4 builds in stock software.
There is no measurement in the repository of stock HCCL small-message latency at 2, 4 or 8 ranks, nor of the cost
of a stock HPU-graph replay (`docs/design/flash-next/system-map.md:249`): job 00012 is the first.

## 4. Options for Flash-Next at TP8 + EP8

Three ways to replay a decode or verification step. What every one of them must keep dynamic is the same:

| Input | Changes every step | How a replay sees it |
|---|---|---|
| token ids, positions, `logits_indices` | values | fixed-shape int32 inputs of the bucket |
| slot mapping, block table (`[padded_bs * num_tokens, max_blocks]`), `block_list`/`block_usage`/`block_groups` | values | fixed-shape int32 inputs; the block table width is the bucket's, not the context's |
| GDN load/store state indices into the recurrent state pool | values | int32 inputs; the pool tensor itself never moves (allocated once at KV-cache init) |
| QSA indexer selections (top-k blocks) | values, computed on the device | produced inside the graph from the compressed key cache; no host involvement |
| routed experts | values, computed on the device | inside the vendor MoE op |
| PLE n-gram ids (`VLLM_HPU_PLE_HOST_NGRAM_IDS=1`) | values | int32 input built on the host |
| number of rows | per bucket | one captured graph per (padded batch, tokens per request, block bucket) |
| speculative rows (`1 + k` per request) | per k | its own graph per k; verification is a decode forward (`is_prompt=False`) |
| `lattice_store_differs`, `direct_gdn_state`, `is_prompt` | per branch | hashed by value: one graph per branch (`hpu_model_runner.py:1931-1935`) |

Fixed addresses are needed for the weights, the KV cache, the QSA side caches (compressed keys and the raw-key ring)
and the recurrent state pool. All of them are allocated once and stay put while a server runs; prefix caching reuses
pool slots through the same index inputs, so it changes values, not addresses (INFERRED from the pool design,
`docs/design/flash-next/plan.md` M2; the fork's plans are invalidated on exactly these events: cache reallocation,
weight reload, communicator change, `hpu_model_runner.py:1701-1716`, `invalidate_gdn_state_views`).

Completion and reuse under stock HPU graphs: the wrapper copies each step's inputs into the inputs it captured,
on the same stream as the replay, so the copies of step n+1 are ordered after the replay of step n; the runner's
read of the sampled tokens orders the host. Nothing more is needed today. Two cases would break it and must reset
the captured graphs: a reallocation of any captured state (KV cache, state pool, side caches; the defragmenter is
off for recurrent models, `vllm_gaudi/extension/defragmentation.py:48-66`), and, with async scheduling, any input
written by the host into memory the in-flight replay still reads (only if the wrapper's copies are bypassed, as in
step 4 below) (INFERRED).

**O1. Stock HPU graphs, tightened (what decode uses today, made smaller).** The bridge already captures the whole
forward: the fused GDN decode op, the QSA indexer and sparse attention (torch ops today, kernels from qsa-fused
later), the fused hyper-connection kernels, the vendor MoE op, the PLE layer, the final mixer, and all 98 layer and
embedding all-reduces. What O1 can still remove: (a) the 48 per-layer cuts in decode graphs (built:
`VLLM_HPU_DECODE_CUT_INTERVAL`, section 8); (b) the wrapper's per-call input hash and the copies of the 14 new
inputs into the captured ones, by keeping the inputs in persistent device buffers that the graph reads directly (the
engine-step track keeps the decode inputs on the device, D3; the replay side is a direct `HPUGraph` per bucket);
(c) the logits, the logits all-gather and greedy sampling outside the graph (`VLLM_HPU_FUSED_GREEDY_LOGITS` exists,
non-speculative only; D4); (d) any host read inside the step (MTP drafter `.cpu()`, I02). Collectives stay the stock
lazy HCCL ones, launched by the bridge between recipes; their cost under replay is measured, not assumed (job 00012).

**O2. torch.compile with `hpu_backend` (eager mode, `PT_HPU_LAZY_MODE=0`).** The runner has the path
(`_maybe_compile`, regional compilation for Qwen3-Next/Qwen3.5 layer groups, `skip_guard_eval_unsafe` after warmup);
the fork's Qwen3.8 TP2 decode gained about 30 % from a fused reduction boundary and about 45 % with 8-layer regional
groups, both against its own compile-mode baseline (README "Qwen TP2", PR #5). In compile mode there is no HPU-graph
replay ("no hpu graphs for t.compile?", `hpu_model_runner.py:5686-5687`): every step pays guard checks and one host
launch per compiled region, and collectives inside a compiled region are functional collectives lowered by the
backend. It would move every Flash-Next custom op (fused GDN, HC kernels, PLE ops, the vendor MoE op) to a path
that needs fake (meta) kernels, and prefill with it. O2 only pays if its collectives or launches are much cheaper
than O1's: job 00012 measures one all-reduce and one layer under both on the same body at TP8.

**O3. The fork's native joint replay, ported to TP8.** Section 3: it needs private patched Bridge, Synapse and HCL
builds pinned to an older Synapse source snapshot, it is written for TP2 (and TP4 for V4.1) and refuses other
groups, and every node start is a fresh machine here. Not portable to this program in the rental window; its numbers
serve as the ceiling of what a stock path could approach (about 16 us per collective point at TP4).

## 5. Expected fixed cost per step

One decode step at one stream, steady state 33-37 ms (section 1). What it is made of, with the basis of each term;
"open" terms are closed by job 00012 (replay internals) and step-anatomy's job 00013 (timer and trace of the real
step). Ft in target-model.md s6 is every term here except the bytes.

| Term | Today, 1 stream | Basis | What removes it | Owner |
|---|---|---|---|---|
| Device time of the graph ops | about 10.6 ms (5,300 ops x 2 us) | 2 us per op on the card (section 7.1), op count from `ref/study/report_draft.txt` | fewer device ops: QSA kernels (B3, B4), MoE glue (B5, B6), PLE (B8); some fusion in bigger recipes | qsa-fused, kernel-fixes |
| Weight and state bytes | 1.6-2 ms (about 3.2 GB per rank at one row) | target-model.md s1.1 and s2 | not a fixed cost; saturates at 32 rows (s2) | - |
| 98 + 1 collectives | 13-19 ms (98 x 130-195 us) | count VERIFIED (section 2.1); 130-195 us per all-reduce under replay at TP8, measured on the decode-shaped body (section 7.2) | fewer (C2, C3), cheaper per call (TP4 layout E1; a cheaper form, measured in 00019 and 00027); the fork's native collectives are not portable (section 3.2) | collectives, coord (E1) |
| Recipe boundaries in the replay | 2-4 ms for the 47 per-layer cuts (C1 on); the 98 collective boundaries are inside the term above | 45-90 us per cut under replay (section 7.2) | C1 (on in the card runs), `VLLM_HPU_DECODE_CUT_INTERVAL=1000` (-47) | native-replay |
| Step tail outside the graph | about 0.5-1.5 ms: tens of lazy ops at 16 us, about 5 `mark_step`s, the logits all-gather | 16 us per issued op (section 7.1), the sites (section 2.1) | the tail captured with the step: logits and greedy argmax in the replayed graph | native-replay with engine-step (D4) |
| Replay entry | input hash and copies of 14 inputs, then the first launch: ASSUMPTION 0.3-1 ms | 14 inputs (section 2) | persistent inputs read by a direct `HPUGraph` per bucket | native-replay with engine-step (D3) |
| Host input preparation | ASSUMPTION 2-5 ms, serial with the device | async scheduling is off for Flash-Next and for any speculation (`platform.py:578-585`, `hpu_model_runner.py:2361-2362`) | async scheduling with MTP (I idea 7, D2), device-side metadata (D3) | engine-step |
| Engine step, IPC to 8 workers, output | ASSUMPTION 1-3 ms, serial | as above | as above | engine-step |

Reading (INFERRED from the table): the term B10 was meant to remove, host graph building, is already gone at decode
(section 7.1, reading 1). The largest fixed term is now the collectives under replay (13-19 ms at TP8), then the
device time of the ops (about 10.6 ms), then the per-layer cuts (2-4 ms) and serial host work. This track's lever
(`VLLM_HPU_DECODE_CUT_INTERVAL`) takes the cuts: Ft from about 25-35 ms to about 21-33 ms, still far from 5-10 ms.
Reaching the budget needs all of: the kernels (device ops to about 3-5 ms), collectives at about 30 us or less per
call (about 3 ms for 98; today's TP8 stock replay is 4-6 times that), and host work hidden behind the device
(async scheduling with speculation). Whether E1's TP4 collectives or a cheaper call form get there is the open
measurement (00019, 00027).

Update 2026-11-21 03:40 UTC (section 7.10, MEASURED on the real model at TP8): a replayed launch costs about 155 us of
host time per rank and a 1-stream step issues about 145 of them, about 22 ms of the 38 ms step. The bridge-only lean
path takes the bridge's share: lean step 1 and the spinner-off setting about 17 us each, lean2 about 12 more
(INFERRED), lean3 -20 to -35 (INFERRED); what stays is about 50 us per launch (Synapse's own launch, allocations, the
collective hand-off), about 7 ms per 1-stream step.

## 6. The deciding experiments

The cheapest experiment that separates O1 from O2 and sizes what O1 can still remove does not need the model: a
decode-shaped body on the eight cards, timed in every regime on the same ranks
(`tools/flash_next/replay/bench_replay.py`, card job 00012 `replay-bench-r8`, about 20 minutes). The body has 48
layers of two sub-blocks (RMS norm, a GEMM 2560 -> 320, SiLU, a GEMM 320 -> 2560, an all-reduce of `[rows, 2560]`
bf16, a residual add): the 96 layer all-reduces of a Flash-Next step. Regimes: plain lazy (`lazy`), plus the
per-layer cut (`lazy_layer`), plus a cut before every collective (`lazy_coll`); the same three captured with
`wrap_in_hpu_graph(disable_tensor_cache=True)` and replayed with a new input each call (`replay*`), as the runner
does; `torch.compile(hpu_backend, fullgraph=True)` of the whole body and of one layer called 48 times
(`compile_full`, `compile_layer`). Each with and without the all-reduces, at 1, 32 and 128 rows.

Readings, registered before the run:

| Derived | Rule |
|---|---|
| one all-reduce under replay, alpha_R3 = (replay with - without) / 96 | x 99 is the collective part of Ft today. At or below 20 us: collectives do not gate Ft, drop C6. 20-45 us: C4 (one-shot gather + sum) is worth doing. Above 45 us: collectives are a top term; compare alpha_R4 |
| one cut under replay, c_cut_R3 = (replay_layer - replay) / 47 | x 47 is what `VLLM_HPU_DECODE_CUT_INTERVAL` can save per step. Below 20 us: not worth a card A/B |
| host share of a replay call (call / total) | near 1: the replay itself is host-bound, so fewer launches per step (the native-replay class) matter; small: the step is device-bound |
| alpha_R4 and compile_layer total against replay | O2 is worth an attempt only if it beats replay by several ms per step at 1 and 32 rows |
| lazy against replay | the host cost of building the graph that replay already saves (cross-check of the coordinator's lazy-op bench, job 00009) |

The real step is measured by other tracks' jobs, which this design reads instead of repeating: the coordinator's
`microbench` (00009: per-op, launch, `mark_step` and HCCL latency, plain lazy), step-anatomy's timer and profiler
trace of the real model at 1, 8 and 32 streams (00013) and its `VLLM_CONFIG_HIDDEN_LAYERS` 48/4 A/B (00016).

## 7. Measurements

### 7.1 Plain lazy mode, one card (coordinator job 00009, `bench_lazy_ops.py`, accumulation mode 0)

VERIFIED (`bin/farm log 00009`, 2026-11-20 17:23-17:48 UTC), median per iteration of a chain of `v = v * c + d`:

| ops | host issue ms | `mark_step` ms | device wait ms | first iteration, cold / warm recipe cache |
|---|---|---|---|---|
| 100 | 1.63 | 0.09 | 0.42 | 0.44 s / 8 ms |
| 300 | 4.84 | 0.20 | 0.68 | 1.5 s / 25 ms |
| 1,000 | 16.2 | 0.66 | 2.14 | 13.7 s / 75 ms |
| 3,000 | 48.9 | 1.98 | 5.44 | 131 s / 176 ms |

Readings (INFERRED from these and section 2):
1. The host issues a lazy op in about 16 us. A Flash-Next decode step rebuilt every step (about 5,300 ops) would
   spend about 85 ms issuing on the host; the measured steady step is 33-37 ms at one stream. Decode is replayed,
   as the code says (section 2), and replay is what keeps the step from being host-bound.
2. The device spends about 2 us per op (2.1 ms per 1,000, 1.8 ms per 1,000 at 3,000). That is the BRIEF's
   "2 us per graph op" slope: it is device time inside the replayed recipes, which no replay mechanism removes.
   Only fewer device ops do (the B3-B6 kernels, and the graph compiler's fusion inside bigger recipes).
3. A recipe's cold compile grows faster than its op count (13.7 s at 1,000 ops, 131 s at 3,000). Bigger recipes
   per decode bucket are only affordable if each one stays small: job 00012 shows whether a captured decode
   graph without the per-layer cuts is still split at each collective (first-call time of `lazy`/`replay`
   against `lazy_layer`/`replay_layer`).
4. The 2 us per op held inside one 3,000-op recipe (5.44 ms), so it is a per-node device cost, not a per-recipe
   one, and the graph compiler did not fuse that element-wise chain. Native replay does not change it either:
   under the fork's native replay about 27-30 % of the step stayed non-compute and fewer segments were slower
   (section 3.3).

### 7.2 Replay at TP8: per collective and per recipe (job 00012 `replay-bench-r8`)

VERIFIED (`reports/native-replay/card/replay-bench-r8/bench.log`, `replay_r8.json`; 8 ranks, farm-base image of
e238f84, bridge 1.24.1.482, torch 2.11.0a0; the slowest rank's median of 20 calls). The body: 48 layers, two
sub-blocks each, 96 all-reduces of `[rows, 2560]` bf16; "call" is the time until the call returns on the host.

| regime | rows | no collectives: total / call ms | with the 96 all-reduces: total / call ms | per all-reduce |
|---|---|---|---|---|
| `replay` (`wrap_in_hpu_graph`, as the runner) | 1 | 1.83 / 0.21 | 18.84 / 17.43 | 177 us |
| | 32 | 1.30 / 0.22 | 13.71 / 10.42 | 129 us |
| | 128 | 1.50 / 0.18 | 20.24 / 14.84 | 195 us |
| `replay_layer` (+ a cut after every layer) | 1 | 3.97 / 3.92 | 20.18 / 18.58 | |
| | 32 | 4.31 / 4.24 | 22.06 / 13.45 | |
| | 128 | 5.75 / 5.71 | 24.75 / 20.84 | |
| `compile_full` (torch.compile `hpu_backend`, eager mode) | 1 | 2.57 / 0.38 | 15.21 / 11.10 | 132 us |
| | 32 | 1.91 / 0.35 | 17.12 / 14.74 | 158 us |
| `compile_layer` (one layer compiled, 48 calls) | 1 | 6.65 / 6.40 | 18.11 / 14.62 | |
| | 32 | 15.74 / 9.09 | 22.89 / 20.65 | |
| `lazy` (op by op, no replay) | 1 | 36.72 / 33.95 | 45.29 / 44.71 | 89 us |
| | 32 | 27.11 / 24.78 | 42.88 / 42.41 | 164 us |

Readings:
1. **A collective costs 130-195 us under replay at TP8**, about the same in compile mode (132-158 us) and plain lazy
   mode (89-164 us). Replayed, the 96 all-reduces are 87-93 % of the body's time.
2. **The replay holds the host.** `wrap_in_hpu_graph` defaults to `asynchronous=False`
   (`bridge_src/habana_frameworks_torch_hpu_graphs.py:840-849`), so the caller's thread issues every captured
   launch inside `replayV3` (`:635-770`): a one-recipe replay returns in about 0.2 ms, the same body with 48
   recipes in 3.9-5.7 ms (about 80-120 us of host time per extra recipe), with the 96 collectives in 10-17 ms
   (about 110-180 us per collective). (Host time per launch VERIFIED by these calls; where inside the bridge or
   HCL it goes is not visible from Python.)
3. **A recipe boundary under replay costs 45-90 us** (`replay_layer` minus `replay` without collectives, over 47
   cuts). With the collectives the difference is noisier (28-178 us per cut).
4. **Compile mode is not cheaper** at TP8 for collectives or launches, and per-layer compiled regions pay their
   host call every layer (6.6-15.7 ms without collectives). O2 is rejected.
5. Plain lazy mode spends 25-35 ms of host time building this body every step; replay removes that (and the
   replayed output equals the lazy one bit for bit, `check` in the log).
6. **Removing the per-layer cuts does not make the recipes much bigger when collectives are present**, so the
   compile-time growth step-anatomy measured for bigger recipes (about quadratic in size) does not apply to
   `VLLM_HPU_DECODE_CUT_INTERVAL`: the first call (capture and compile) of the body with collectives took 103 ms
   without the per-layer cuts against 88 ms with them at 1 row, 364 against 64 ms at 32 rows and 391 against 46 ms
   at 128 rows (`lazy` against `lazy_layer`); without collectives, the uncut body is one recipe and its first call
   took 539 ms against 179 ms. In a Flash-Next decode graph the cut after a layer only separates the layer's
   residual add from the next layer's first segment, between two collectives (INFERRED).
7. Rank 0 of the lazy process was killed in the last case (`replay_coll`, 128 rows, with collectives) by
   "Synapse detected a device critical error ... [No progress error]" at 18:20:39 UTC on module 0 (PCI
   0000:0b:00.0, which logged the same "unexpected user error interrupt" at 17:48:02 during job 00009). The
   other seven ranks finished; the next process acquired all cards. Later jobs leave that regime out.

8. **The ranks did not report the same times (found 2026-11-20 19:55 UTC in `replay_r8.json`, per rank).** In
   `replay` with collectives at 32 rows rank 0 reported 10.4 ms per call and 13.7 ms to `synchronize`, the seven
   other ranks 0.5-0.7 ms and 0.8 ms, less than the body without collectives (1.2 ms); at 1 row all eight
   reported 16.8-18.8 ms; in `replay_layer` at 1 row rank 0 reported 7.9 ms and the others 20.2 ms. An all-reduce
   needs every rank, so a rank cannot finish a step 17 times faster than another: on a rank that is ahead the
   call returns after queueing and `torch.hpu.synchronize()` returns without waiting for the launches that sit
   behind a collective (INFERRED; VERIFIED only that the reported times differ). So "call" in the table above is
   not always the host's issue time and "total" not always the end of the step: the slowest rank's median, which
   the table uses, is the rank that the others waited for. The benchmark now ends every timed call with a read
   of one output element (`done`, commit `40b14e24`), as the runner's read of the sampled tokens does, records
   the calling thread's CPU time and context switches per call, and the 00065/00067/00068 jobs measure with it.

What this means for the real decode step (INFERRED): with C1 on, a step replays about 98 collectives and about
146 recipes. At about 150 us per collective that is 13-19 ms; the 47 per-layer cuts at 45-90 us add 2-4 ms. Together
they are the "about 22 ms that does not scale with the op count" of the BRIEF's fit; the device time of the ops
(about 2 us each, section 7.1) comes on top. The fork's native replay removed exactly these host-issued launches
and collectives (pre-built command pages, about 16 us per collective point at TP4, section 3.3); its runtime is
not portable here (section 3.2).

### 7.3 Ops a prepared decoder could compute once (CPU, value numbering)

`reports/native-replay/probes/const_ops.py` value-numbers every op of one decode forward of the tiny model (world 1,
sparse QSA, the plugin's HPU MoE path under CPU stubs, the torch hyper-connection path because the fused kernels
have no CPU implementation). VERIFIED by that run (`probes/const_ops_hc1_b1.json`), one row: 1,350 dispatched ops,
858 compute ops, of which **43 depend only on weights and buffers** (casts of norm weights, the Gemma `1 + w`,
weight clones in the GDN path) and **17 repeat a value already computed in the same step**; 465 views and 27
factory ops. Scaled to 36 GDN and 12 QSA layers that is about 300 weight-only compute ops per step, about 0.6 ms
at 2 us per op (INFERRED). The cross-layer repeats of the QSA indexer metadata are larger (about 700 ops, sweep
I20) and belong to the QSA kernel's once-per-step row table (qsa-fused). A generic "prepared constants" pass in
the replay path is therefore not worth building on its own; folding the few weight-only ops at load is a small
change in each module.

### 7.4 One launch and one collective apart (jobs 00067 on one card, 00065 at 8 ranks)

VERIFIED (`reports/native-replay/card/launch-r1/bench/replay_r1.log`, `knobs_r1.log`; `card/host-r8/bench/replay_r8.log`,
`busy_r8.log`, `knobs_r8.log`; image of `1de7c3de`; the slowest rank's median of 40 calls, each ended by a read of one
output element). "One launch" = (`replay_seg` - `replay` without collectives) / 96, the 48-layer body cut where each
all-reduce would be (96 recipes instead of one); "one all-reduce" = (`replay` with - without) / 96; "beyond its
boundary" = (`replay` with - `replay_seg`) / 96. Several values = several runs in the same job.

| | rows | one launch | one all-reduce | beyond its boundary | host share of the call |
|---|---|---|---|---|---|
| 1 card (00067) | 1 | 22, 39 us | - | - | 0.92-0.94 |
| | 32 | 29, 45 us | - | - | 0.93-0.95 |
| 8 ranks (00065) | 1 | 66, 76 us | 48, 74, 45 us | -18, -2 us | 0.85-0.93 |
| | 32 | 65, 32 us | 145, 109, 151 us | 81, 76 us | 0.94-0.96 |

Readings:
1. A launch under replay costs the host 22-45 us on one card and 32-76 us per rank at 8 ranks.
2. Per all-reduce, at 32 rows 109-151 us (00012: 129 us). At 1 row this job gives 45-74 us against 177 us in 00012,
   but its 1-row points disagree within the run (the segmented body without any collective took longer than the body
   with all 96: 8.9 against 6.6 ms per call), and the body's GEMMs (1.7-3.1 ms of device work) can hide part of the
   host issue. The collectives track's bench (job 00073, a body without GEMMs, so nothing hides) gives 164-172 us per
   collective at 8 ranks, flat from 1 to 128 rows (`reports/collectives/card/00073-c4-bench-r8-r4/c4_r8.json`).
   **What a served TP8 step pays per collective is about 150-170 us of host time, the recipe split it forces
   included** (INFERRED from 00073, 00012 and 00065 at 32 rows). The real step agrees: 98 x 165 us + 47 cuts x
   65-90 us is about 19-20 ms, the fastest rank's replay call in step-anatomy's 00013 and 00117.
3. The calling thread is on a CPU for 60-75 % of a replay call with collectives (thread CPU 3.4-6.7 ms in a 5-7 ms
   call at 1 row): the host time is mostly work, partly waiting.
4. A lead, not a finding: with the tensor cache kept (`replay_seg_cache`, `wrap_in_hpu_graph` without
   `disable_tensor_cache`) one launch cost 41 / 9 us at 1 / 32 rows against 72 / 65 us without it at 8 ranks
   (`replay_r8`), but nothing less on one card (00067). The runner wraps with `disable_tensor_cache=True`; with the
   cache kept the intermediates keep fixed addresses, which is also what a lean replay path (Handoff, item 3) needs.

### 7.5 What the host does inside a replay (1-card sampler, 00067)

VERIFIED (`card/launch-r1/bench/sampler/threads.txt`, `stacks.txt`; a loop of 3,000 `replay_seg` replays, a 4.2 s
window): the calling thread ran for 100 % of the window. Its three stack samples: `RecipeCacheLRU::get` ->
`RecipeArgumentSpecEqual` (memcmp: the bridge finds each launch's recipe by comparing argument specs),
`RecipeValueSpec::patch_launch_info` (patching the launch tensors' addresses), and a config stream write. The bridge's
`AccThreadPool` thread spun at 100 % of a CPU (`habana_lazy::AccThreadPool::main_loop`, in accumulation mode 0), and
a third thread ran at 35 % with about 25,000 voluntary context switches per second: **each worker keeps about 2.3
CPUs busy while it replays.** The 8-rank sampler took no stacks (its loop ended before the first).

### 7.6 Bridge, Synapse and HCCL settings (00067: 12 sets on one card; 00065: 9 sets at 8 ranks)

VERIFIED (`knobs_r1.log`, `knobs_r8.log`; in every set that ran, the replayed output equals the CPU reference with
the reductions):
- One card, per launch at 32 rows: 26-33 us under `PT_HPU_ENABLE_EXECUTION_THREAD=0`,
  `PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT=0` and `=1`, `PT_HPU_QUEUE_SYNLAUNCHES=0` and `=1`,
  `PT_HPU_ENABLE_VALID_DATA_RANGE_CHECK=0` with `PT_HPU_VALIDATE_COMPUTE_SHAPE=0`, `PT_HPU_ENABLE_GRAPH_RUNNING_HASH=0`,
  `PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`, `RUNTIME_SKIP_RECIPE_VALIDATION` (experimental flags), the calling
  thread on a CPU of its own; 39 us under `PT_HPU_ENABLE_DEBUG_NAMES=0`; 29-45 us without any. Twice as dear (55 us):
  `PT_HPU_ENABLE_STAGE_SUBMISSION=0`, and `ENABLE_STATS` with `STATS_PER_SYNLAUNCH`. (The sets' 1-row `replay_seg`
  points ran in another case order than the baseline's and are not compared.)
- 8 ranks, `replay` with the 96 all-reduces, ms at 1 / 32 rows: each rank on its module's CPUs 7.5 / 11.4, the calling
  thread on a CPU of its own 7.3 / 15.8, each rank on the other NUMA half 7.3 / 15.7;
  `PT_HPU_USE_LAUNCH_RECORD_STREAM=1` 10.2 / 14.3, `=0` 8.0 / 11.3; `PT_HPU_LAZY_COLLECTIVES_HOLD_TENSORS=0` 8.4 / 14.3,
  `=1` 7.4 / 13.6; skipped recipe validation 7.7 / 11.7. The three unpinned baselines of the job: 7.0, 7.4, 9.5 and
  12.5, 15.1, 16.1. No set is outside the run-to-run spread.
- `HCCL_PRIM_COLLECTIVE_MASK=2` **kills every rank** at the first all-reduce: `all_reduce.cpp:17 (ar_runPairwise): The
  condition [ params.m_count % commSize == 0 ] failed. ar_runPairwise does not support remainder in count`,
  `hcclAllReduce` returns `hcclInternalError`, `exception 15 Terminated`, and Synapse's failure analysis logs "Received
  unexpected user error interrupt" on all eight cards (21:42:09 and 21:42:37 UTC: the two failure-analysis lines per
  card in the job's ledger entry). A setting the stack refuses, not a card fault (`card/host-r8/habana_logs/hcl.log`,
  `dfa_log.txt`).

Reading: no bridge, Synapse or HCCL setting lowers the host cost of a launch or a collective. The fixed cost has no
switch.

### 7.7 Asynchronous replay, and token parity on the card

VERIFIED:
- `replay_async` at 8 ranks: the call returns in 0.56-0.64 ms instead of 6.6-15 ms, but the call and the read take
  12.0 against 7.4 ms at 1 row and 13.9 against 15.1 ms at 32 rows; with a sampler tail behind it
  (`replay_async_tail` against `replay_tail`) 17.9 against 16.0 ms at 1 row and 11.4 against 6.9 ms at 32 rows. One
  card: equal at 1 row, 1.4 against 3.5 ms at 32 rows. The tiny model at TP4 (00066): 10.7 ms per token with
  `VLLM_HPU_GRAPH_ASYNC_REPLAY=1` against 6.6 without (best of 3, the shortest prompt). An asynchronous replay hands
  the same launches to another thread; at 8 ranks it does not shorten a step. **Keep it off.**
- Token parity (tiny model, `--gdn hc`, 6 prompts x 16 tokens): `VLLM_HPU_DECODE_CUT_INTERVAL=1000` at TP1 (00067),
  TP4 (00066) and TP8 (00065), and with `VLLM_HPU_GRAPH_ASYNC_REPLAY=1` as well at TP4 (00066): every token equals
  every tiny run without these flags on the card (8 runs of 4 tracks at TP 1, 2, 4 and 8), including the whole
  sequences of the two prompts whose first token is a near tie (prompts 0 and 3, margins 2.7e-5 and 9.2e-4, the same
  tie in every run). Each server logged `a decode forward cuts after every 1000-th decoder layer` on every rank.
  **The cut flag is safe for every server.**

### 7.8 The ranks' unequal host times in the server, and where the workers run

VERIFIED:
- Benchmark (00065): the 8 ranks' host call for the body with 96 all-reduces at 1 row takes 5.2-6.6 ms unpinned
  (spread 1.4 ms); each rank on its module's 18 CPUs 5.1-6.6, the calling thread on a CPU of its own 5.9-6.5, each
  rank on the other NUMA half 5.5-7.1 ms; call and read 7.2-7.5 against 7.0-7.4 ms. The benchmark has almost no
  spread, and placement changes nothing in it.
- Server (step-anatomy's timer, forward = the replay call, p50 [min] per rank and round, 1 stream): 00117 at TP8,
  per round the fastest against the slowest rank 19.9 [18.3] against 36.8 [27.4], 21.0 [19.5] against 35.9 [23.8],
  19.7 [18.5] against 29.3 [26.4], 21.5 [20.1] against 35.7 [32.4]; 00113 at TP4, rank 3 the slowest in every 1-stream
  round (38.4 [36.5], 31.2 [24.4], 28.9 [26.9], 34.3 [33.1]) against 20.3-26.0 on the fastest.

INFERRED: the slow rank's fastest step of a round is above the fast ranks' median, so it is slow for the whole round,
not by jitter; which rank it is changes between rounds (TP8) or holds for a job (TP4). That is a condition that lasts
while a thread stays where the kernel put it: next to another busy thread, or away from its memory. The benchmark's
short-lived processes do not reproduce it. The VM shows no NUMA affinity for the cards (`numa_node` -1 for all eight
PCI devices, two NUMA nodes 0-75 and 76-151), so the farm's module-to-CPU table is an assumption; pinning each worker
to one slice still keeps its threads and its memory together. Build step 7 (`VLLM_HPU_WORKER_CPU_PIN`) and job 00122
test it on the real model.

### 7.9 The worker CPU pin and the hash switch at TP8 (integration track's job 00156, arm B against arm A)

MEASURED (integration's report; `~/farm/integration/results/00156-real8-build2b/{nospec,nospec-pin}/`; the CPU watcher
`reports/native-replay/card/cpuwatch-00156/`): TP8 + EP8, integration build 2, greedy, steady ms per step (round 1 /
round 2): 1 stream B (`VLLM_HPU_WORKER_CPU_PIN=1 PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`) 41.15 / 41.06 against A
(neither) 43.95 / 33.01; 32 streams B 68.45 / 69.01 against A 53.74 / 62.63. Server healthy after 811 s (B) against 477 s
(A), the weights at about 6 s against 2 s per shard. All 8 pin lines (rank r: share r of 8, CPUs 19r to 19r+17, 35
threads moved), no fallback line; B's 1-stream text equals A's. **No gain, worse at 32 streams and at load: both flags
stay off.**

The watcher (MEASURED): unpinned, each worker's main thread uses 82-94 % of a CPU at 1 stream and 67-77 % at 32 streams
and drifts over both NUMA halves; pinned, it stays in its slice at 83-97 % and 83-86 %, with the worker's AccThreadPool
thread spinning at 100 % on a neighbouring CPU of the same slice. At 32 streams the pinned main thread therefore spent
about 0.85 x 69 = 59 ms of CPU per step against about 0.70 x 58 = 41 ms unpinned (INFERRED): it runs about 45 % slower,
it does not wait more. Reading (INFERRED): the VM's vCPUs are host hyperthreads (lscpu hides it: "1 thread per core"),
and packing a worker's busy threads into 18 neighbouring vCPUs puts the main thread next to (or on the sibling of) a
spinner. A different pin cannot avoid that without the host's sibling map. The slower weight loading under the pin is
not explained (more loader threads than the slice's 18 CPUs, or NUMA).

The spinner (VERIFIED in the bridge source at 5176b1b2, `pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.cpp`):
`CreateAccThreadPool()` (line 35) picks `AccThreadPool` by default (`PT_HPU_ACC_THREAD_VERSION=0`), whose `main_loop`
(line 158) is `while (task_count_ == 0 && !stop_) {}`, a busy-wait without pause: one CPU per worker, forever, also in
accumulation mode 0 (51 % of all samples of the 1-card profile, section 7.5). Stock switches remove it:
`PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1` (`AccNoThread`, no thread) or `PT_HPU_ACC_THREAD_VERSION=1` (a blocking-queue
pool that sleeps when idle). On one card (00067, "accsync") the first gave right values and no change, as expected
on an otherwise idle host.

### 7.10 Where a replayed launch spends its host time: bridge job 00174 (real model at TP8)

MEASURED (the collectives track's job 00174 `real8-bridge-phase`, all cards, 02:30-03:06 UTC; our bridge builds
`timers-c281d8b53d08` and `lean-e1dbb4942fb8`, build steps 9 and 10; results
`reports/collectives/card/00174-real8-bridge-phase-<arm>/`; one `HPUGRAPH_REPLAY_TIMING` line per rank every 5000
replayed launches, 2000 on the tiny model). Base env of the collectives track's real runs (QSA mode 1, GDN core decode,
fused HC, recurrent state pool, no collective mark step, `VLLM_SKIP_WARMUP=true`, greedy; about 84 ms at 32 streams,
so read it relatively). Every arm ok. Tokens: on the tiny model at TP8, spinner off and lean equal timers prompt by
prompt; on the real model the 1-stream text is identical in all three arms.

Steady ms per step (the probe's median gap; 1 stream one round, 32 streams two rounds):

| arm | 1 stream | 32 streams |
|---|---|---|
| timers build (reference) | 38.22 | 84.24 / 84.73 |
| spinner off (`PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`) | 34.62 | 82.37 / 82.17 |
| prepared replay, lean step 1 (`PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`) | 30.76 | 83.71 / 83.56 |

Per replayed launch, us, the median over every line after each rank's first (the collectives track's parser
`reports/collectives/probes/replay_timing_table.py`, 287-322 lines from 8-9 pids; `probes/replay_timing.py` gives
means over the ranks, same direction):

| phase | timers | lean step 1 | spinner off |
|---|---|---|---|
| total | 155.4 | 137.8 | 138.8 |
| exec_setup | 6.78 | 1.52 | 5.83 |
| launch_hashes | 4.76 | 0 | 3.99 |
| create_launcher | 1.95 | 0 | 1.79 |
| run_spec | 0.13 | 0 | 0.13 |
| run_lookup | 4.08 | 0 | 3.59 |
| run_prep | 1.75 | 14.05 | 1.38 |
| run_patching | 22.4 | 16.9 | 19.2 |
| launcher_inputs | 29.7 | 33.3 | 29.6 |
| syn_launch | 48.9 | 41.0 | 43.7 |
| launcher_post | 14.6 | 13.0 | 12.8 |

Readings:
1. **The real model's split is not the tiny model's.** launcher_post is 15 us per launch here (tiny model: 118 us, 45 %,
   its recipes all end in a collective). On the real model the big phases are the Synapse launch path (syn_launch, 49),
   the input handling (launcher_inputs, 30), the patching (22) and the post (15); the phases lean step 1 removes are 16.
2. **The prepared path engages** (hashes, launcher, spec and lookup at 0 in the lean arm), and the thread-local state is
   shared between the two libraries as built (exec_setup, launch_hashes, create_launcher nonzero in the timers arm).
3. **Lean step 1's run_prep rose from 1.75 to 14.05 us.** Cause (code reading, INFERRED until lean2's timers): the kept
   launcher held the previous replay's tensors (output handles, input copies, intermediates) until its next replay,
   one step later, when it held the last references, so RunPrepared's clears freed them in run_prep. The stock
   launcher drops the same references at the end of each launch (its destructor, while the stack and the launch's own
   holders still own the tensors). Fixed in lean2 (build step 10): `ReleaseLaunchState()` right after each prepared
   launch and after the first kept one.
4. **The spinner-off setting lowers almost every phase by 10-20 %** (the AccThreadPool spinner's CPU contention,
   section 7.9). It is a stock setting; the coordinator put it into integration build 3.
5. **About 140-150 launches per 1-stream step** (rough: 5000-launch lines against the 10 s throughput lines of the
   log; step-anatomy counted 148), so the replay path is about 22 ms of the 38 ms 1-stream step on each rank.
6. **Lean's 1-stream gain (-7.5 ms) is larger than its timed saving explains** (17.6 us x about 145 = 2.6 ms). 1-stream
   TP8 rounds spread 33-44 ms between rounds (RESULTS.md) and this job measured one round: the next job's two rounds
   per level decide it. At 32 streams the step is device-bound (-0.8 ms).

What the big phases hold (VERIFIED in the bridge source at 5176b1b2):
- syn_launch, `backend/synapse_helpers/graph.cpp` `graph::launch`: two allocator walks over all ~17 addresses
  (`get_total_memory_required`; the second only feeds `habana_lazy::log_dev_mem_stats`, its argument evaluated even
  with logging off), the workspace check and buffer, `lock_addresses`, two trace-vector copies
  (`towl::emitRecipeLaunch`), then `synLaunchWithExternalEvents`.
- launcher_inputs, `backend/kernel/hpu_habana_cache.cpp` `RecipeLauncher::Launch`: the profile registry's per-launch
  insert (`RecipeRegistry::registerRecipe`), per input `add_wait_events_on_stream` (a producer-event lookup per
  address, and the wait for a collective's output still queued on the HCCL job thread), the intermediates' refs.
- launcher_post: the producer registration of the outputs, the resource holder, `GenericRecordStream` of inputs and
  outputs (stream allocator), the collective launch handed to the HCCL job thread.
- run_patching, `UpdatePatchingInformation`: `RecipeValueSpec::update_patching_table` (new outputs and intermediates
  allocated, every tensor info walked) and `patch_launch_info` (the whole launch info rebuilt).

**Step 2 (coordinator approved, 03:35 UTC): `lean3`** = lean2 plus sub-timers inside syn_launch, run_patching and
launcher_inputs first, then in this order, each measured (gains per launch INFERRED):
(a) a launch-info template per prepared launcher: each launch rewrites only the addresses of its new inputs, outputs
and intermediates, no walk over tensor infos or maps (allocation stays): -8 to -12 us;
(b) syn_launch's bridge part: the tensor-memory total cached per prepared recipe, the logging walk and the trace
copies skipped: -4 to -8 us;
(c) no producer-event lookup for inputs written on the same compute stream (stream order covers them), the
cross-stream waits on collective outputs kept: -5 to -10 us; plus one arm with the stock
`PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` (the collective issued inline, no future wait; never measured, sign unknown);
(d) one producer event per launch for all outputs, a lighter resource holder, no profile-registry insert when
profiling is off: -3 to -6 us.
Step 2 in all: -20 to -35 us per launch, about -3 to -5 ms per 1-stream step, little at 32 streams. What stays is
about 50 us per launch (Synapse's own launch, INFERRED 30-35 of syn_launch; allocations; the collective hand-off),
about 7 ms per 1-stream step; lower needs fewer launches per step (fewer collectives, bigger segments).

### 7.11 lean3t with sub-timers against the image's bridge: job 00181 (real model at TP8, integration build 3)

MEASURED (the collectives track's job 00181 `real8-bridge-headline2`, ended 04:56 UTC, all six arms ok; integration build
3's image and env, the spinner off (`PT_HPU_SYNCHRONOUS_ACC_QUEUE_FLUSHING=1`) in every arm; reference = the image's own,
vendor-built bridge; results `reports/collectives/card/00181-real8-bridge-headline2-*/`). Tokens: tiny-lean and
tiny-lean-inline equal tiny-stock prompt by prompt; the 1-stream texts of both rounds are equal in all three real arms.

Steady ms per step, two rounds (greedy 32 / 8 / 1 streams; sampled 32 / 8 / 1):

| arm | greedy 32 | greedy 8 | greedy 1 | sampled 32 | sampled 8 | sampled 1 |
|---|---|---|---|---|---|---|
| image bridge | 49.4, 56.4 | 53.3, 53.5 | 37.6, 37.2 | 57.6, 63.5 | 59.3, 57.7 | 36.2, 38.2 |
| lean3t (`lean3t-412153bbab3e`, flag 1) | 46.2, 47.0 | 63.1, 61.3 | 43.5, 43.6 | 47.9, 50.7 | 59.4, 59.6 | 44.5, 44.7 |
| lean3t + `PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` | 45.5, 46.4 | 61.1, 59.9 | 42.2, 43.4 | 51.8, 51.0 | 63.2, 60.5 | 44.0, 44.5 |

Per launch, us (medians over all lines after each rank's first; lean3t / inline): total 143.8 / 133.7; run_prep 2.14 /
2.20 (**the lean2 fix works**: 14.05 in 00174); unattributed 5.0 / 5.3 (the frees did not move); run_patching 24.4 /
24.4; launcher_inputs 40.1 / 6.0; syn_launch 43.6 / 48.3; launcher_post 13.7 / 35.9; exec_setup 1.5; hashes, launcher,
spec, lookup 0. Sub-timers (nested): in_registry 0.54, in_refs 0.52, **in_waits 39.0 of which in_future 34.2** (inline
0.0) and in_streamwait 3.5; syn_mem 3.1, syn_workspace 0.78, **syn_lock 9.9**, syn_trace 1.0, **syn_call 27.9**,
syn_after 0.32; **patch_alloc 14.4**, patch_table 22.8 (holds patch_alloc), patch_info 1.3; post_holder 1.05,
post_events 6.0, post_record 0, post_coll 6.1.

Readings:
1. **The collective futures are the biggest bridge item**: the main thread waits 34 us per launch for the HCCL job
   thread (`habana_helpers::JobThread`, a condition-variable sleeper, `pytorch_helpers/habana_helpers/job_thread.h`)
   to issue the previous launch's collective. Inline issue removes the wait and adds 22 us of issue cost to the post
   phase: net -10 us per launch, same tokens.
2. What is left is mostly outside the bridge's bookkeeping: the Synapse call (28 us), the allocator's address lock
   (10 us: handle to device address for every tensor), the allocations of outputs and intermediates (14 us).
3. **The step-time flips between jobs are the ranks, not lean.** In every real arm exactly two of the 8 ranks run about
   2x slower per launch for long stretches, and which two changes with every server start (real-lean 1-stream windows:
   six ranks 117-158 us, two 229-238; real-lean-inline: six 130-159, two 233-243). Every phase is 2x in the slow ranks: a
   slower CPU, not more bridge work. At 1 stream the slowest rank sets the step: about 150 x 90 us = 13 ms, more than
   any per-launch gain (lean about -30 us, inline -10 us: 1.5-4.5 ms). INFERRED: pairs of busy main threads on vCPUs that
   are hyperthread siblings on the host (the VM hides the sibling map; section 7.9).
4. Our rebuild is the vendor's binary in all but the patch: `libhabana_pytorch_backend.so` 58,176,608 bytes (image)
   against 58,158,528 (lean3a), `.text` within 0.002 %, the same GCC 13.4.0, no LTO in either (0 `lto_priv` symbols),
   the same number of `isra` clones; the plugin likewise. A slower build is unlikely; job 00188 measures it (real-rebuild
   against the image's bridge).

### 7.12 The host's hyperthread pairs: job 00193 (CPU only, all cards held)

MEASURED (`tools/flash_next/replay/cpu_siblings.py` in job 00193, 281 s, the 144 CPUs of an all-card job; results
`reports/native-replay/card/00193-cpu-sibs/cpu_siblings.json`): a pinned C loop per vCPU, alone and in pairs. The VM
reports 152 sockets of one core each (lscpu: KVM, Icelake, 2 NUMA nodes 0-75 / 76-151, `numa_balancing=1`), but
**vCPUs 2k and 2k+1 share a host core**: all 12 tested a^1 pairs (0-1, 12-13, 24-25, 38-39, 50-51, 62-63, 76-77, 88-89,
100-101, 114-115, 126-127, 138-139) slow each other by 7-9 % on both sides; a+-76 pairs 0.00 (12 of 12); a+-1 only when
it is a^1; a full scan of vCPUs 0, 12, 25 against all 143 others puts the a^1 partner at 0.07-0.10 and the rest at a
median 0.000, 95th percentile 0.02, max 0.06. The small size is the loop's (one dependent chain leaves most of a core
free; fixed to four chains in 61ad3d1+); two busy decode main threads lose far more (INFERRED from the 2x slow ranks of
section 7.11, slow only at 1 and 8 streams, where the main threads are busy all the step). The tool's own verdict
printed "rule: None" (its absolute threshold 0.2 was meant for a core-filling loop; now relative).

Use: `tools/flash_next/replay/place_workers.py` with rule a^1 (`~/farm/native-replay/late/place.sh`, md5
7e54f2de93590d2478bf5a94dbca7d75) runs in 00189's pin arm after health: each worker's main thread alone on one vCPU
of its share, the sibling vCPU kept free, the worker's other threads on the rest of the share, every other process off
the reserved vCPUs, re-applied every 2 s. Also a rule for every placement here: never put two busy threads on 2k and
2k+1; `VLLM_HPU_WORKER_CPU_PIN=2` puts the main thread on its share's first CPU (19r) and lets other threads use its
sibling (19r xor 1), so mode 2 needs the sibling kept free (a fix for the flag, after 00189 says it pays).

## 8. Build steps

| Step | What | Proof off the card | Card | State |
|---|---|---|---|---|
| 1 | `tools/flash_next/replay/bench_replay.py`: the experiment of section 6 | gloo self-test of every regime, replay equals lazy; `tests/unit_tests/tools/test_bench_replay.py` | job 00012 | committed `bd813c29` |
| 2 | `VLLM_HPU_DECODE_CUT_INTERVAL` (default 0): decode and verification forwards cut after every N-th decoder layer instead of every `VLLM_CONFIG_HIDDEN_LAYERS`-th; prompt forwards unchanged. Hooks are only re-registered when the flag is set, so 0 is the old code path | `tests/unit_tests/worker/test_decode_cut_interval.py` (11 tests) | token parity on the tiny model: PASSED at TP1, TP4, TP8 (00067, 00066, 00065; section 7.7); real-model A/B in job 00122 | committed `021f9bf6`; card-proven safe |
| 3 | Replay-entry regimes in the benchmark: `graph_raw` (direct `HPUGraph`, no input hash), `replay_async`, `replay_meta` (20 int32 tensors and 2 scalars as a namedtuple keyword, like the trimmed decode metadata) | gloo self-test; `test_bench_replay.py` (8 tests) | job 00027 step A | committed `aef003f4` |
| 4 | Only if 00012/00027 show the replay entry or the tail costs about 1 ms or more per step: a direct `HPUGraph` per decode bucket over persistent input buffers (with engine-step's D3) and the logits tail captured with the forward | CPU: same logits and tokens with and without, graph keys unchanged | real model at 1 and 32 streams | decided by the measurements |
| 5 | `VLLM_HPU_GRAPH_ASYNC_REPLAY` (default 0): the model's HPU graph wrapped with `asynchronous=True`, so the forward returns once the replay is queued | `tests/unit_tests/worker/test_hpu_graph_async_replay.py` (4 tests) | tokens equal at TP4 (00066); not faster at 8 ranks or on the tiny model at TP4 (section 7.7) | committed `6966226e`; keep off |
| 6 | Bench regimes for grouped collectives (`replay_par2`, `replay_fused2`, `replay_coalesce`), a sampler tail, a torch.profiler trace of a replay, the knob names of the bridge and HCL | `test_bench_replay.py` (11 tests); gloo self-test | job 00027 step A | committed `d85d9730`, `e2726c41` |
| 7 | `VLLM_HPU_WORKER_CPU_PIN` (default 0): 1 = every thread of a worker on one equal share of the server's CPUs, the share at its module's position among the visible modules, set right after `torch.hpu.set_device`; 2 = 1 and the model's thread on a CPU of its own after warmup (`vllm_gaudi/v1/worker/hpu_worker_cpus.py`) | `tests/unit_tests/worker/test_worker_cpu_pin.py` (29 tests); a Linux check with real threads is in `reports/native-replay/probes/pin/linuxcheck/` (the colima VM's Docker daemon did not answer, not run) | job 00122: TP4, base / cut / cut + pin at 1 and 32 streams, every rank's busy threads' CPUs recorded (`late/cpu_watch.py`) | committed `6884fc07` |
| 8 | Our own build of the Gaudi bridge (gaudi-pytorch-bridge `5176b1b2`, the head of `v1.24.1`): `tools/flash_next/bridge/` (build scripts, `package_out.py`; `bin/farm ... --bridge <build-id>` installs it in a job) | wheels and their sha256 in each build's `manifest.json` | 00160: `stock-5176b1b2` loads at TP8, tiny tokens equal to the image bridge's | published `stock-5176b1b2` |
| 9 | Phase timers, `PT_HPU_HPUGRAPH_REPLAY_TIMING=N` (`patches/timers.patch`; off unless set) | the strings in the wheel | 00174: the per-phase split on the tiny and the real model at TP8 (section 7.10) | published `timers-c281d8b53d08` |
| 10 | Prepared replay, `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1` (off unless set): lean step 1 (`patches/lean1.patch`) and 1b (`patches/lean2.patch`: the kept launcher releases its launch state after each launch; one write per timer line) | tiny TP8 tokens equal (00174) | 00174 lean step 1: -17.6 us per launch, same tokens, run_prep +12 (fixed in lean2); lean2 in the collectives track's lean + spinner-off job | `lean-e1dbb4942fb8` published 02:17 UTC; `lean2-81b9b47e3599` building from 03:19 UTC |
| 11 | lean3: sub-timers, then cuts (a)-(d) of section 7.10 | | | planned |

## Handoff (if this agent stops)

**08:20 UTC.** lean5's qualification is job **00210** `real8-bridge-reuse2` (collectives track, all cards, 120 min,
after async-sched's 00202/00203; replaces 00198): tiny-stock, tiny-lean4-nan, tiny-lean5-verify; real-ref, real-lean5,
real-lean5-verify, real-lean4 (fallback, last); text_check 1/32 (+ churn on the lean5 arms); plan_bad 0 and churn
texts equal fail the job otherwise. lean6 (`patches/lean6.patch`, PT_HPU_HPUGRAPH_LEAN_COLLECTIVE: the lazy collective
issued inline without the job thread's copies, about -3 to -4 us per launch) is deferred by the coordinator. Sent to
engine-step (bridge view of the census's 178 H2D copies and 13 copy-only graphs per 32-stream MTP step): a CPU->HPU
copy is an immediate DMA on the DMA_H2D sub-stream, never a graph; with `non_blocking=False` (the default) the bridge
spin-waits on the host until it completes (`backend/helpers/tensor_utils.cpp` `copy_data_to_device`); an HPU->HPU
copy outside a captured graph is a lazy op flushed as its own recipe at the next replay or mark_step (a full launch);
batching: one pinned host buffer and one persistent device buffer per dtype, one `non_blocking` copy per step, the
device views as the graph's static inputs (double-buffered under async scheduling: the bridge's H2D waits only for
the destination's last writer).

**08:05 UTC (by `date -u`; earlier entries of this session may be off by up to an hour): lean5 built and VLLM_HPU_WORKER_CPU_PIN=3 pushed.**
- **Mode 3** (773a7f7c, default off): `VLLM_HPU_WORKER_CPU_PIN=3`, `VLLM_HPU_WORKER_CPU_SIBLINGS` (xor1 default, none):
  after warmup the model thread on share[1], its host sibling share[0] free of the worker's and the engine core's
  threads, the worker's other threads on the rest, a daemon keeps later threads off both; 7 CPU tests.
- **lean5** (`lean5-984bf2249087`, `patches/lean5.patch`, published 08:01:58 UTC; bridge clone commit after lean4 on
  `native-replay-lean`): the static launch plan behind `PT_HPU_HPUGRAPH_STATIC_PLAN` (needs REUSE >= 1). After an
  unplanned prepared launch the launcher keeps, per ring slot, the launch info as graph::launch left it (physical
  addresses), the untranslated handles (graph::launch's `addresses`, copied into a thread-local only when the flag
  is set), the input and output buffers, the intermediates (kept) and the defragmentation generation (new counter,
  bumped at `device_memory::defragment_memory` entry). A hit (same input buffers, kept outputs and intermediates held
  by nobody else, same generation) skips update_patching_table, patch_launch_info, allocation, translation and the
  allocator walks: the tensor infos get the addresses update_patching_table would patch, outputs and in-place outputs
  are built as there (`create_outdup`), graph::launch only enters the defragmenter's critical section (an empty
  `lock_addresses`) and launches. A defragmentation between the check and the critical section, or a constant swap
  by the inference-mode scan (recorded by `update_syn_launch_info` during a planned launch), makes graph::launch
  translate the handles after all and drops the plan. `=2`: every plan re-translated before use, mismatches counted
  as `plan_bad`. Not planned (lean4 path): DMA inputs, shape tensors, host-to-device or external tensors,
  out-duplicates of intermediates or outputs, dynamic shapes. `PT_HPU_HPUGRAPH_STATIC_PLAN_BUMP_EVERY=N` forces
  invalidations. Timer line: `plan_try plan_hit plan_capture plan_bad defrag_generation`. Estimate (from 00181's
  split): about 85-95 us per planned launch with inline collectives against 134 (00181 inline arm). The build
  container `nr-bridge-dev` now runs on the new build cores 18-19, 56-57, 94-95, 132-133 (BRIEF 06:58).
- **00198** (collectives track, about 12:30-13:00 UTC): arms on the lean5 wheel: tiny-stock, tiny-lean4-nan,
  tiny-lean5-verify (STATIC_PLAN=2, REUSE=2, BUMP_EVERY=500); real-ref (all off), real-lean4 (REUSE=1),
  real-lean5-verify (REUSE=2, STATIC_PLAN=2, BUMP_EVERY=5000, inline collectives; text_check 1/32/churn),
  real-lean5 (REUSE=1, STATIC_PLAN=1, inline; text_check 1/32/churn). `text_check.py --churn` (node md5 daca88b9...):
  each prompt again alone after a 32-stream burst, a 20K-token prefill and an 8-stream burst; must equal texts_1.
- Next (coordinator 07:56): lean6, a leaner bridge-side collective issue (inline: no PtTensorInfo copies, no
  promise/future, cached 2-address translation; about -4 to -6 us per collective, -3 to -4 us per launch).
  Fewer launches cannot come from the bridge: a launch is a recipe plus the collectives that end it; lazy mode must
  cut at every collective (HCCL outside Synapse recipes; the next recipe needs its result); no multi-recipe launch.

**06:40 UTC: lean4 built** (`lean4-44fe39c26127`, `patches/lean4.patch`, published 06:28:34, VERIFIED in the wheel; the
coordinator approved it at 06:25 with a harder qualification). Output-buffer reuse behind
`PT_HPU_HPUGRAPH_PREPARED_REUSE` (only with `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`): the bridge allocates every launch's new
outputs twice over (patch_alloc 14.4 us: `create_empty_tensor` gives a handle; syn_lock 9.9 us: `lock_addresses` backs
each new handle with device memory under the pool mutex, after a third allocator walk; VERIFIED in
`backend/synapse_helpers/device_memory.cpp`), about 24 of 144 us per launch. lean4 keeps each prepared launcher's
outputs of the last two launches (recipe output order, at most `PT_HPU_HPUGRAPH_PREPARED_REUSE_MAX_KB`, default 4096)
and reuses the set of two launches ago through the bridge's allocated-outputs path when the launcher holds the only
reference to every tensor and storage (two back, because the graph's lazy outputs keep the last set until they are
rebound after the next launch; safe in this mode because every device user holds a reference until its work is done:
the recipe's and the collective's resource holders; stream allocator off, post_record 0). `=2`: also a NaN fill
(`synMemsetD16/D32Async` on the launch stream) of every reused floating-point buffer right before the launch. Counters
`reuse_try`, `reuse_hit`, `reuse_poison` per launch in the timer line. Qualification (coordinator): tiny TP8 tokens with
=2 and =1; real model, `tools/flash_next/replay/text_check.py` (6 prompts x 600 greedy tokens at 1 stream must equal
the REUSE=0 arm; 32 streams coherent; on the node `~/farm/native-replay/late/text_check.py`, md5 1d8cb0c9...), hit rate
and sub-timers per rank. Arms sent to the collectives track for the bridge job after 00189 (tiny-lean4-nan, tiny-lean4,
real-ref = lean4 REUSE=0, real-lean4-nan, real-lean4; place.sh in the real arms if 00189 shows the pin pays).

**06:20 UTC: pull request #21 merged** into `feat/flash-next-gaudi` (merge commit 42d70a7e; gate on 63693b83: 100 files,
2 red = the two rental tests that also fail on the program head's own snapshot; everything of this branch up to
f0c36b39 is on the program branch). 00193 found the sibling rule a^1 (section 7.12); `place.sh` (rule a^1) is staged
in 00189's pin arm by the collectives track.

**State at 2026-11-21 05:35 UTC (newest first).** 00181 is read (section 7.11: run_prep fixed; the collective
futures are the biggest bridge item; inline collectives -10 us per launch; **two slow ranks per server at 1 and 8
streams** decide the 1-stream step, not lean). In flight:
- **00189** `real8-bridge-vendor2` (collectives track, all cards, 95 min, build 4's image and no-speculation env, after
  integration's 00186/00187): tiny gates (vendor, lean3a flag 0, lean3a flag 1); vendor spinner on against off, the
  better one kept; then real-rebuild (`lean3a-8f6f7faf88dc`, `PT_HPU_HPUGRAPH_PREPARED_REPLAY=0`, timing 5000: our build
  on the stock path, per-rank timers), real-lean (flag 1), real-pin (flag 1 + `VLLM_HPU_WORKER_CPU_PIN=1`, or the
  sibling-aware placement below if 00193 finds a stable map before 00189 starts, about 07:10 UTC at the latest). The
  collectives parser now splits per-rank medians by level (N=1, 8, 32 from the metrics lines); headline = the per-rank
  spread. No bridge builds while 00189 runs (the build CPUs may be siblings of the chosen vCPUs).
- **00193** `cpu-sibs` (this track, all cards held but not opened, 15 min hold, right after 00186):
  `tools/flash_next/replay/cpu_siblings.py` (pinned C loops alone and in pairs; rules a+-N/2, a^1, a+-1; verify on all
  144 job CPUs or scan; stability re-run) -> `reports/native-replay/card/00193-cpu-sibs/cpu_siblings.json`. 00192 was the
  same job with a 15-second hold by mistake (bare MAX_HOLD is seconds), replaced.
- **Sibling-aware placement** `tools/flash_next/replay/place_workers.py` (038a32dd; CPU tests; checked in the dev
  container with two fake workers: main threads alone on their vCPUs, the other threads and processes moved off). For
  00189: generate `place.sh` (bash wrapper around the script with `PLACE_SIBLING_RULE` set to the rule 00193 finds),
  put it on the node under `~/farm/native-replay/late/`, send its path and md5 to the collectives track (agent
  a3344386b2bc11c38): it copies it to `~/farm/collectives/results/bench/place.sh` and sets `PIN_PLACE` in
  `~/farm/collectives/bridge/late/00189.env`; its `place_wrap.sh` runs it as root in the pin arm's container right
  after health (log `$FARM_OUT/placement.log`).
- **Pull request.** Gate on `75b1de32` (branch + program head d9791183): 88 files, 2 red = exactly the two rental
  tests that also fail on the program head's own snapshot (`test_make_rental_bundle`: "not a git repository" in a
  snapshot; `test_dry_run_defaults_and_bundle_build_options`); the coordinator agreed that counts as green (state it
  in the PR). The program head then moved to 2e9f9196 (#17, #19, #20); merged as `63693b83` with one conflict in
  `hpu_model_runner.py` (two functions added at the same place: both kept); my test files pass on it; its gate
  (`reports/native-replay/logs/gate_63693b83.log`) is queued. On the same two reds: push, `gh pr create --base
  feat/flash-next-gaudi --body-file reports/native-replay/probes/pr_body_2.txt` (GATE_RESULT filled in), `gh pr merge
  --merge`, the number to the coordinator.

**State at 2026-11-21 03:45 UTC (the older state below stays for its details).** 00174 is read (section
7.10). The coordinator approved the step-2 plan (03:35 UTC): lean2 (the run_prep fix) into the collectives track's
lean + spinner-off job, then lean3 with sub-timers first, then cuts (a)-(d), each measured, and the stock
`PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` as one arm when cheap. **Owner's rule (BRIEF.md 03:07 UTC): a track merges its own
pull request as soon as its gate is green and its change is card-proven or default-off, then tells the coordinator the
number** (this replaces "no pull request before the node is gone" below). Bridge source with the lean2 edits:
`reports/native-replay/bridge-src/` (uncommitted on 5176b1b2; `git diff` there = `patches/lean2.patch`). Near miss at
03:14 UTC: the first lean2 start found no patch on the node (wrong `bin/node-scp` target: the remote path starts with
`:`, not `node:`) and `dev_build.sh` silently built the stock tree under the lean2 id; stopped before anything was
published (package_out.py refused the missing patch; `out/` has no lean2 entry). `dev_build.sh` now refuses a missing
patch (exit 10).

State at 2026-11-20 23:50 UTC by the real clock (fourth agent of this track, Opus 5.5, started 23:15 UTC). Branch
`feat/flash-next-gaudi-native-replay` (worktree `/Volumes/Storage/Claude/gaudi-work/wt/native-replay`), pushed; the
program head `938dada4` is merged in (`09d72dc9`); nothing of this track is on `feat/flash-next-gaudi` yet. **No pull
request before the node is gone** (coordinator, BRIEF.md 23:45 UTC): the integration track builds from this branch
directly. This harness keeps the handoff here (earlier harnesses refused report files). Outside the repository, in
`/Volumes/Storage/Claude/gaudi-work/reports/native-replay/`: `late/` (the host scripts the queued job runs, also in
`~/farm/native-replay/late/` on the node), `probes/` (analysis helpers: `knob_table.py`, `per_rank.py`, `stacks.py`),
`logs/` (test and sync logs), `card/` (fetched results; the coordinator's mirror has every result folder under
`node-mirror/farm/native-replay/results/`).

### Done and verified (details in sections 7.4-7.8)

- Read jobs 00065, 00066, 00067 and sent the coordinator the digest (23:25 UTC); it is in BRIEF.md (00:00 entry; the
  coordinator's clock runs ahead) and RESULTS.md. Correction sent to the collectives track and to be sent to the
  coordinator: a served TP8 step pays about 150-170 us per collective, not the 45-74 us of 00065's 1-row points
  (section 7.4, reading 2).
- `VLLM_HPU_DECODE_CUT_INTERVAL=1000`: token parity on the card at TP1, TP4, TP8 (section 7.7). Safe for every
  server; the coordinator gave it to the integration track. `VLLM_HPU_GRAPH_ASYNC_REPLAY`: parity at TP4 but not
  faster: keep off. No bridge, Synapse or HCCL setting helps; `HCCL_PRIM_COLLECTIVE_MASK=2` kills every rank (the
  failure-analysis lines of 00065 on all eight cards).
- Build step 7, `VLLM_HPU_WORKER_CPU_PIN` (commit `6884fc07`, default 0): `vllm_gaudi/v1/worker/hpu_worker_cpus.py`,
  hooks in `HPUWorker.init_device` (after `torch.hpu.set_device`, before the communicators) and at the end of
  `compile_or_warm_up_model` (mode 2), the flag in `envs.py`, its row in `docs/configuration/env_variables.md`.
  Rank r uses the r-th visible module, ascending (VERIFIED in the server logs of 00113, 00065, 00066).
  Commit `0835754e`: a placement the kernel refuses never stops a worker (each hook falls back to unpinned with
  one warning line). The node runs `0835754e` (`~/farm/native-replay/current`, synced 23:50 UTC).
- The coordinator asked (23:33 UTC) whether the fork's native joint replay can be measured tonight on 4 cards. Answer
  sent: no number by 03:00 UTC (three patched source builds never done here, among them Synapse from a third-party
  snapshot of the whole stack; the snapshot's match with the node's SDK 1.24.1 driver and firmware is known only
  after the build; a harness through the static group plan in eager mode; 3.5-5 h in the best case, and an hour of
  compiling on the node's CPUs under host-bound measurements). Recommended: build the three libraries before the
  next rental, then one 30-minute 4-card job, stock against native replay on the same TP4 body.
- The coordinator also asked for the launch count per step in each arm of 00122: the server has no counter and the
  farm's probe no profiler hook, so 00122 measures the cut flag as step time only; a tiny-model TP4 job with the
  vLLM profiler window (ExecuteCachedGraph events per decode step, base against cut) is offered, not built.
- CPU tests: `tests/unit_tests/worker/test_worker_cpu_pin.py`, `test_decode_cut_interval.py`,
  `test_hpu_graph_async_replay.py`: 46 passed on the merged tree; `test_bench_replay.py` with the first two: 43 passed
  on the merge commit's snapshot. **The common gate (`bin/gate --tools`) has not run on this branch**: tonight the
  one-at-a-time gate has five tracks queued for hours, and the coordinator said to leave the queue unless one's gate
  already runs.

### Lean replay work on this node (from 00:40 UTC: the owner extended the node to about 00:00 UTC on 2026-11-22)

The coordinator started next-week item 1 (the bridge-only prepared replay, Pending item 3) on this node, as a
default-off bridge setting `PT_HPU_HPUGRAPH_PREPARED_REPLAY`, qualified by the tiny model's tokens and
`bench_replay.py` at TP8, then a real-model A/B at 1 and 32 streams; card jobs 8-card where possible (owner's rule).
State at 00:57 UTC:
- **Our own bridge build.** The UNPATCHED gaudi-pytorch-bridge at 5176b1b2 (head of `v1.24.1`) built in `farm-base` in
  container `nr-bridge-build` (CPUs 18,37,56,75, nice 19, no card; `late/bridge-build/build_bridge.sh`; 840 build steps,
  23:43 to about 00:58 UTC). `keep_tree.sh` snapshots the container when the log says `build end` as image
  `farm-bridge-build:5176b1b2` (the build tree in `/tmp/builds`, sources in `/tmp/gaudi-pytorch-bridge`), log
  `~/farm/native-replay/bridge-build/commit.log`. Next builds: a long-lived dev container `nr-bridge-dev` from that image
  (same CPUs, `-v ~/farm/native-replay/bridge-build:/out`, `sleep infinity`) and `late/bridge-build/dev_build.sh BUILD_ID
  PATCH` (reset to 5176b1b2, apply `patches/PATCH`, incremental `build.py -r -l -j 4`, `package_out.py`).
- **Deployment interface, agreed with the collectives track (agent a3344386b2bc11c38), live on the node:** builds in
  `~/farm/native-replay/bridge-build/out/<build-id>/` (`wheels/*.whl`, `build.log`, `manifest.json` written last, the
  folder renamed into place); ids `stock-5176b1b2` and `<name>-<12 hex of the patch's sha256>`; `bin/farm job|tiny|serve
  ... --bridge <build-id>` installs a build's wheels in the job's container, checks the sha256 and that the loaded .so
  files are the wheel's, and keeps a per-build recipe cache (exit 7: install failed, 8: wrong bridge loaded).
- **Incident 00148 (00:46 UTC):** `perf record --call-graph dwarf` on the benchmark's replay loop, card 0 (module 3):
  `PCIE_ADDR_DEC_ERR ... read from unmapped reg, addr 0x1000007ffd9200a8` = Gaudi2 CFG_BASE (0x1000007FF8000000) +
  0x59200A8, inside `NIC10_UMR1_0_UNSECURE_DOORBELL1` (a write-only NIC doorbell page mapped into the process); the
  driver reset the card (done 00:47:00), the peers' links flapped (19 down, 19 up by 00:47:12). Likely cause
  (INFERRED): perf's DWARF user-stack copy (8 KB above the stack pointer, clamped to the address-space end, not to the
  stack mapping) read the mapped doorbell page. **BRIEF rule 12: no perf call graphs (dwarf or -g) on processes that
  have a card open; IP-only perf and gdb are fine.**
- **Profile of a replayed launch (00148's samples before the fault, 1 card, small recipes, about 22 us per launch):**
  ExecuteSynapseCache -> RecipeLauncher::Launch about 2/3 (the bridge's per-address event bookkeeping plus Synapse's
  launch), UpdatePatchingInformation about 1/4 (half of it allocating outputs and intermediates), per-launch object churn
  about 1/6 (HlExec's constructor builds two empty JIT graphs; lazy tensors; hashes; the launcher), the spec and the
  LRU lookup under 1/20. The AccThreadPool thread spins at 100 % of a CPU. Reports:
  `reports/native-replay/card/perf1/perf_bench.{self,dso}.txt`, inclusive `~/farm/native-replay/results/perf1/
  perf_bench.incl.txt`.
- **First patch: phase timers** (`late/bridge-build/timers.patch`, the bridge tree with it in
  `reports/native-replay/bridge-src/`): `PT_HPU_HPUGRAPH_REPLAY_TIMING=N` prints every N replayed launches per thread a line
  `HPUGRAPH_REPLAY_TIMING ... total_us ... graph_prep ... syn_launch ... | per launch inputs ... externals ...` (16
  phases of SingleHPUGraph::replayGraph, ExecuteCachedGraph, HlExec::Launch, HabanaLaunchOpPT::run,
  ExecuteSynapseCache, RecipeLauncher::Launch; only synchronously replayed launches; about 20 ns per stamp). Plan
  (coordinator, approved): build it as `timers-<hash>`, then one all-card job with `--bridge`: the tiny model at TP8 and
  the real model at TP8 with the timers, giving the real per-phase split per launch; then the lean path aims at what
  dominates.
- **Build state at 01:10 UTC.** `out/stock-5176b1b2` published (the unpatched wheels: `habana_torch_plugin-1.24.1+git5176b1b
  -cp310-cp310-linux_x86_64.whl` 44 MB, `habana_torch_dataloader-...whl`; nothing outside the wheels); the collectives
  track's job 00160 checks it (tiny TP8, image bridge against ours, token parity, our .so loaded). The snapshot of that
  build tree failed (the container ended within a second of the build), so `timers-c281d8b53d08` builds from scratch in
  the persistent dev container `nr-bridge-dev` (CPUs 18,37,56,75,94,113,132,151 for this one build, coordinator's word;
  log `~/farm/native-replay/bridge-build/out-logs/timers-c281d8b53d08.log`, about 01:47 UTC). Chained after it
  (`chain_lean.sh`, node pids 1361605/1361606, log `out-logs/chain.log`): the dev container back to 4 CPUs, then the
  incremental build of `lean-e1dbb4942fb8` (`tools/flash_next/bridge/patches/lean1.patch`: the timers plus
  `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`, each captured segment keeps its `HabanaLaunchOpPT` and recipe holder from its
  first replay in a side table in HPUGraph.cpp, later replays call `HabanaLaunchOpPT::RunPrepared`: no HlExec, no new
  launcher, no recipe spec, no recipe lookup; the patching and the launch unchanged; never compiled before this build).
  Proposed to the collectives track: one all-card job with four arms (tiny TP8 timers, tiny TP8 lean + flag, real TP8
  timers, real TP8 lean + flag), each only if the one before passed. Per-phase report:
  `python3 reports/native-replay/probes/replay_timing.py <server0.log>...`.
- **The four-arm job is armed on the collectives side** (agent a3344386b2bc11c38, 01:10 UTC): once 00160 ends with exit 0 and
  `out/timers-c281d8b53d08/manifest.json` exists, it waits up to 25 min for `out/lean-e1dbb4942fb8/` and submits
  `real8-bridge-timers-lean` (all cards, 50 min): tiny-timers (`PT_HPU_HPUGRAPH_REPLAY_TIMING=2000`), tiny-lean (+
  `PT_HPU_HPUGRAPH_PREPARED_REPLAY=1`, tokens equal to tiny-timers prompt by prompt), real-timers (`...=5000`, greedy 1 and
  32 streams), real-lean (+ the flag); each arm only if every arm before passed. Without the lean build: the timers job
  alone (`real8-bridge-timers`, 30 min). Logs: `<id>-<name>-<arm>/server0.log` (tiny: `hc/server0.log` in the same
  folder). The collectives agent sends the id to this track and the coordinator.
- **02:15 UTC.** **00160 PASSED** (collectives track, all cards): `stock-5176b1b2` loads at TP8 (all 14 mapped bridge
  .so files are our wheel's; 18 of its 23 .so differ from the image's; nothing needed the image's `.upstream` twins) and
  the tiny model's tokens equal the image bridge's prompt by prompt: we can build and load our own bridge on this stack.
  `timers-c281d8b53d08` published 01:39 UTC. The chain (`chain_lean.sh`) never started the lean build: it ran as the node
  user while `out-logs/` was root-owned (the dev container created it), every redirection failed, so no command ran.
  Fixed by `chown` of `out-logs/`, `out/`, `dev-wheels/` to the node user (always run host-side scripts' logs there after
  this); the dev container is back to CPUs 18,37,56,75; `lean-e1dbb4942fb8` started by hand 02:10 UTC (incremental, 52
  build steps; the prepared-replay sources compiled clean; **published 02:17:35 UTC**, 7 min). VERIFIED in the wheels
  (python zipfile): stock has none of the patch strings; timers has `HPUGRAPH_REPLAY_TIMING` in
  libhabana_pytorch_backend.so; lean has it plus `PT_HPU_HPUGRAPH_PREPARED_REPLAY` and `RunPrepared`. VERIFIED with `nm -DC`
  in the dev container: the thread-local state shared between libhabana_pytorch_backend.so (HPUGraph.cpp) and
  libhabana_pytorch_plugin.so (hpu_lazy_tensors.cpp, hlexec.cpp), i.e. `PreparedSlot()::slot`, `ThreadTotals()::totals`,
  `InReplay()::in_replay`, is exported STB_GNU_UNIQUE in both (one copy per thread at run time); default visibility. The
  job confirms it at run time: in the timers arms `exec_setup`, `launch_hashes`, `create_launcher` must be nonzero; in the
  lean arms `launch_hashes`, `create_launcher`, `run_spec`, `run_lookup` about 0 (the prepared path engaged). The bridge job is
  **00174** `real8-bridge-phase` (65 min, after integration's 00161, so after 02:48): tiny-timers, tiny-accsync, tiny-lean
  (last; it looks for the lean build when it starts), then real-timers, real-accsync, real-lean; each real arm only if
  its tiny arm passed; verdicts `<id>-real8-bridge-phase-verdicts.txt`, arm results `<id>-real8-bridge-phase-<arm>/`.
- The integration track's build 2 runs as **00156** (`real8-build2b`; 00142 was cancelled and resubmitted): arms tiny,
  nospec (A), nospec-pin (B: `VLLM_HPU_WORKER_CPU_PIN=1 PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`), mtp2. The read-only
  CPU watcher was moved to it (`late/watch_job.sh 00156 ... nospec nospec-pin`, node pids 1372497/1372500, attached to
  nospec's workers at 01:08:37 UTC; results `~/farm/native-replay/results/cpuwatch-00156/<arm>/`).

### Where the two host levers are tested now (coordinator, 00:28 UTC)

The owner put 8-card runs first (BRIEF.md 00:28 UTC): the TP4 job 00134 below waits behind the TP8 jobs and will
probably not run tonight; keep its plan for the next rental. The worker CPU pin and the hash switch are tested at TP8
instead, as **arm B of the integration track's build 2** (`VLLM_HPU_WORKER_CPU_PIN=1
PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` on top of its proven env, greedy 1 and 32 streams, text compared with arm
A). The integration track (agent a81792d74f2de0f60) has from this track (00:29 UTC): the commits (6884fc07 + the
fallback 0835754e, both in its 9b0e7655), how mode 1 splits an all-card container's 144 CPUs (8 shares of 18, rank r
on CPUs 19r to 19r+17, its own module's slice; the engine and API server unpinned; weight loading unchanged), the
8 log lines to check, the watcher (`~/farm/native-replay/late/cpu_watch.py ... --cpu 151 --window 2`, in arms A and
B) and the hash switch's caveats (set at server start; never on a card here, no token parity yet, so arm B's
1-stream text against A decides; it removes only the spec construction per launch; arm B combines two levers). It
was asked to send back per arm the steady ms per step at 1 and 32 streams, the text, the pin lines and the
watcher's per-rank main-thread share during the measured 1-stream round: write them into section 7.9. It runs arm B
in job 00142 (TP8, image 2de56804 with 9b0e7655; arm folders `tiny`, `nospec` = A, `nospec-pin` = B, `mtp2` = C under
`~/farm/integration/results/00142-real8-build2/`, fetched to `reports/integration/card/00142-real8-build2/`), without a
watcher inside its job. With its agreement (00:35 UTC: read-only, /proc and `docker ps` only, CPU 151, writing only
under this track's folder) this track runs `late/watch_job.sh 00142 ~/farm/integration/results/00142-real8-build2 nospec
nospec-pin` on the node (started 00:36:22 UTC, pids 817765 and 817768; it ends by itself after B's workers exit):
results in `~/farm/native-replay/results/cpuwatch-00142/<arm>/cpus_series.txt|.jsonl`, log `watch_job.log`.

### The TP4 job 00134 `real4-cut-pin-hash-ab`: cancelled 01:35 UTC (made moot by 00156 arm B and the bridge job's accsync arms; its plan below stays for reference)

`any:4`, 58 min, replaces 00122 (which replaced 00084) and keeps the coordinator's bump; starts by 01:55 UTC at the
latest, on cards 0,1,2,7 (modules 0-3). Stub: `reports/native-replay/late/stub_real4_cut_pin_hash.sh`. First the tiny
model at TP4 with `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` (`replay_job.sh real4-hashtiny tinyenv`, log
`results/real4-hashtiny/tiny-hash.log`; token parity), then `~/farm/native-replay/late/real_ab.sh real4 1,32 128` on the
track's synced commit (`~/farm/native-replay/current` = `0835754e8c3e`), arms in the coordinator's order (the last is
lost if the hold runs short): `base`, `cutpin` (`VLLM_HPU_DECODE_CUT_INTERVAL=1000 VLLM_HPU_WORKER_CPU_PIN=1`),
`cuthash` (`VLLM_HPU_DECODE_CUT_INTERVAL=1000 PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1`, only if the tiny check
passed), `cut` (`VLLM_HPU_DECODE_CUT_INTERVAL=1000`); each with `VLLM_GRAPH_RESERVED_MEM=0.5` (BRIEF.md 23:50 UTC), one
server start per arm. Per arm: `real4-<arm>/scaling.json`, `server0.log`, `probe.log`, `hostwatch/` (rank 0's threads
every 2 s, stacks after the second 32-stream round), `cpuwatch/cpus_series.txt` and `.jsonl` (`late/cpu_watch.py` on
CPU 151: per rank every 2 s the main thread and the two busiest other threads, their CPU share, the CPUs they ran on
and their CPU changes; asked for by the coordinator: whether a worker needs about 3 CPUs of its own), `hlsmi.txt`
(device memory every 15 s). When it ends: `bin/farm fetch native-replay 00134 real4-hashtiny real4-hashtiny-hash
real4-base real4-cutpin real4-cuthash real4-cut`, then `python3 reports/native-replay/probes/ab_report.py
reports/native-replay/card base cutpin cuthash cut` (per arm and level the steady ms per step, the 1-stream text against
`base`, the pin and cut log lines, per rank the main thread's CPU share and CPUs in the measured 1-stream round and
the 32-stream rounds), then report to the coordinator per arm (as asked): ms per step at
1 and 32 streams (the probe's median gap), the 1-stream text against `base`, the per-rank spread (`cpuwatch`: the
main thread's CPU share per rank at 1 stream; a fast rank waits blocked at the read). Read also: `cutpin/server0.log`
has one line per worker `VLLM_HPU_WORKER_CPU_PIN=1: rank r module m: share i of 4, CPUs a-b (18 of the 72 ...)`.
Decisions: the pin pays if the 1-stream step falls by more than about 2 ms against `base` and the ranks' shares even
out (then tell the coordinator `VLLM_HPU_WORKER_CPU_PIN=1` for the integration build, and try mode 2 or 3 CPUs per
worker from `cpuwatch`); the hash switch pays if `cuthash` beats `cut` (or `base`, if `cut` did not run) by more than
noise with the same text; `cut` against `base` is the cut flag's real-model gain (step-anatomy expects none: the
all-reduces already split every layer).

### Pending, in order

0. **lean2: published 03:24:56 UTC** (`out/lean2-81b9b47e3599/`, 7 min incremental; VERIFIED in the wheel:
   `ReleaseLaunchState` and `RunPrepared` in the backend and both plugin .so files, the flag and timer strings in the
   backend .so) and **in the collectives track's job 00180** `real8-bridge-headline` (all cards, 45 min, after
   integration's 00177, which ends by 03:56 UTC). The collectives track (agent a3344386b2bc11c38) submitted
   `~/farm/collectives/bridge/bridge_headline_job.sh LEAN IMAGE` (its branch 04f690be; all cards, about 45 min):
   integration build 3's image `farm-integration:d00c0a9a3e18` (also tagged `farm-collectives-int:d00c0a9a3e18`), the
   headline's no-speculation env (QSA fused + shard, GDN interleave + glue cache, HC copy mode 2,
   `VLLM_HPU_DECODE_CUT_INTERVAL=1000`, candidate sampling, the two prefix flags, `VLLM_GRAPH_RESERVED_MEM=0.4`), the
   spinner off in every arm, build 3's probe (levels 32, 8, 1; 128 tokens; two rounds per level; greedy and sampled).
   Arms: tiny-stock (the image's bridge), tiny-lean (lean2, the flag, timing 2000; tokens equal to tiny-stock), then
   real-stock and real-lean (timing 5000). Read in real-lean: run_prep back to about 2 us (if instead the unattributed
   time, total minus the phase sum, rises by about 12 us, the frees only moved: say so), the step at 1, 8 and 32
   streams against real-stock. Write it into section 7.10; tell the coordinator.
1. **lean3** (section 7.10, step 2): sub-timers inside syn_launch, run_patching and launcher_inputs first (one build,
   qualified by tiny tokens), then (a) to (d) in order, each measured, each a new patch on top
   (`tools/flash_next/bridge/patches/lean3*.patch`, ids `<name>-<12 hex of the patch's sha256>`). Qualification: tiny
   TP8 tokens and `bench_replay.py` at TP8, then a real A/B at 1, 8 and 32 streams. The stock
   `PT_HPU_DISABLE_ASYNC_COLLECTIVE=1` as one arm when it is cheap.
2. **Pull request (owner's rule).** The program head moved to `d9791183` (pull request #16); merged in as `75b1de32`
   (no conflict; my four test files pass on its snapshot: 31 + 13 + 4 + 26). The gate runs on that snapshot
   (`reports/native-replay/trees/gate-75b1de32`, log `reports/native-replay/logs/gate_75b1de32.log`; the gate on
   `c556e47b` was stopped while it still waited for a slot). Known red elsewhere: the rental files
   (`tests/unit_tests/tools/rental/test_sh_run_rental.py`, `test_sh_tiny_smoke.py`, `test_rental_scripts.py`,
   `test_tiny_smoke.py`, sometimes `test_sh_rent.py`) fail alone in other tracks' gates too (engine-step `425fb2ee`); if
   they are the only red files, check them on a snapshot of the program head before calling the gate green. On green:
   push, `gh pr create --base feat/flash-next-gaudi` (body `reports/native-replay/probes/pr_body_2.txt`; no absolute
   performance numbers), `gh pr merge --merge`, tell the coordinator the number.
3. **lean3t** (`lean3t-412153bbab3e`, `patches/lean3t.patch`: lean2 plus the sub-timers of section 7.10, printed after
   the tensor counts as `| sub in_registry .. in_refs .. in_waits .. in_future .. in_streamwait .. syn_mem ..
   syn_workspace .. syn_lock .. syn_trace .. syn_call .. syn_after .. patch_alloc .. patch_table .. patch_info ..
   post_holder .. post_events .. post_record .. post_coll ..`; nested, not in the phase sum) **published 03:34:04 UTC**
   (7 min incremental; VERIFIED in the wheel: the sub-timer strings in the backend .so, `ReleaseLaunchState` in all
   three .so files; the bridge clone's local branch `native-replay-lean` commit 6edcb0f = this patch). Sent to the
   collectives track at 03:35 UTC. **Now job 00181 `real8-bridge-headline2`** (65 min, replaces 00180 in its place; starts
   after integration's 00178, by 04:53 UTC at the latest): tiny-stock, tiny-lean, real-stock, real-lean, then
   tiny-lean-inline and real-lean-inline; lean3t in every lean arm. Earlier note: **00180 ran lean3t in its tiny-lean and real-lean arms** (the collectives
   track's `bridge_headline_job.sh` reads `~/farm/collectives/bridge/late/00180.env` at start: `LEAN=lean3t-412153bbab3e`;
   the job log prints it). `INLINE=1` in that file adds tiny-lean-inline and real-lean-inline (+
   `PT_HPU_DISABLE_ASYNC_COLLECTIVE=1`, the real one only if its tiny one passed); the collectives track asked the
   coordinator for a 65-minute resubmit that keeps the place. From real-lean (and real-lean-inline) the collectives
   track sends the medians of every `sub_*`, the phases, run_prep, unattributed and the step times at 32, 8 and 1
   streams, greedy and sampled; then choose among cuts (a)-(d) by the measured sub-phases. `probes/replay_timing.py` prints the sub-phases; the collectives
   track's parser reads them as `sub_*`. Arms to ask for (collectives track places the job): tiny-lean3t and
   tiny-lean3t-inline (tokens), real-lean3t and real-lean3t-inline (`PT_HPU_DISABLE_ASYNC_COLLECTIVE=1`, VERIFIED in
   `habana_kernels/hccl_kernels.cpp:354`: the collective runs inline instead of `JobThreadLazyHCCL::addJob` +
   `submit_future`), all with the spinner off, the flag and the headline env.
3b. **lean3a** (`lean3a-8f6f7faf88dc`, `patches/lean3a.patch`, published 03:47:54 UTC, VERIFIED in the wheel; the
   bridge clone's local commit after 6edcb0f): lean3t plus two cuts under the flag, written before 00181's data because
   they remove pure overhead. (1) `backend/synapse_helpers/event.cpp` `event::stream_wait_event`: the stock code
   releases and re-takes the GIL (`AutoNoGIL`) for every input that has a producer event, also when the event is done
   or on the same stream; the HPU-graph replay binding (`replayV3`, `python_packages/.../csrc/bindings.cpp:311`) holds
   the GIL during the whole replay, so a replayed launch does about one GIL release and re-take per such input (and
   any other Python thread can take the GIL in between: INFERRED, a possible source of the ranks' random multi-ms
   stalls). Now: `try_lock`; done or same stream returns at once; otherwise the stock path. (2) `graph::launch`: the
   logs and traces that are off are skipped before their arguments are built. Measured by the sub-timers
   (in_streamwait, in_waits, syn_workspace, syn_trace, syn_after) against lean3t in 00181. Offered for 00181 or the
   next bridge job.
4. **Background: the lean path's first plan (next-week item 1, the coordinator's choice of the program's best fixed-cost plan): a bridge-only lean replay
   path.** Read from the bridge source at `5176b1b2`. What a replayed launch does on the main thread today, per
   captured segment (step-anatomy: 148 per step at 1 stream, 209 at 32; about 150 us of bridge work each under the
   profiler, of which the Synapse enqueue `enqueueWithExternalEventsExt` is 37 us; the all-reduces go out on a
   separate Synapse thread):
   a) `habana_lazy/hpu_lazy_tensors.cpp:1504` `HbLazyTensor::ExecuteCachedGraph`: copies the input and output
      `HbLazyTensor` vectors by value, builds the stack, copies scalars;
   b) `habana_lazy/hlexec.cpp:132-188` `HlExec::Launch`: `ComputeSymSizeHashCode` + `ComputePermutationHashCode` over
      every input, `GetGraphIndex`, then `CreateLauncher`, a new `HabanaLaunchOpPT` for every launch;
   c) `backend/kernel/hpu_habana_launch_op_pt.cpp:5128` `HabanaLaunchOpPT::run`: `ProcessInputStack`, a new
      `RecipeArgumentSpec` that hashes every input and the graph (`:5199-5221`);
   d) `GetCachedRecipe` -> `backend/kernel/hpu_recipe_cache.cpp:101` `RecipeCacheLRU::get`: two hash-map lookups
      under a mutex, each with a deep `RecipeArgumentSpec::operator==` (no pointer short-circuit; the memcmp of the
      1-card sampler, section 7.5);
   e) `:5082` `UpdatePatchingInformation`: rebuilds the whole patching table (input refs, intermediates, DMA inputs,
      outputs, new output allocations), then `RecipeValueSpec::patch_launch_info` (also in the sampler);
   f) `:4879` `ExecuteSynapseCache`: an inference-mode scan of every constant input per launch, then the enqueue.
   The lean path: at the first replay of a captured graph keep each segment's `HabanaLaunchOpPT` (or its
   `RecipeLauncher` and `syn_launch_info_` as a patch template) with the entries that change per replay marked (the
   user inputs; the intermediates when the tensor cache is off); every later replay re-patches only those entries and
   enqueues: a) to e) and the constant scan of f) go. Files: `backend/habana_device/HPUGraph.cpp`
   (`SingleHPUGraph::replayGraph`), `habana_lazy/hpu_lazy_tensors.cpp`, `habana_lazy/hlexec.cpp`,
   `backend/kernel/hpu_habana_launch_op_pt.cpp` and `.h` (a "prepared replay" entry). No Synapse or HCL change. Estimate
   (INFERRED): a launch's main-thread cost from about 130 us to about 35-45 us (basis: TP8's fastest rank issues 148
   launches in about 20 ms; the enqueue is 37 us under the profiler); **at 1 stream 148 x about 90 us = about 13 ms
   less host time per step** (the 1-stream TP8 step from about 36 to about 23 ms, plus what the pin closes); **at 32
   streams 209 x about 90 us = about 19 ms less host issue**, and the step becomes device-bound (about 84 to about
   60 ms today, lower with the QSA change). Work: 2-3 days of bridge C++, our own bridge wheel in the farm image,
   qualification on the tiny model (token parity) and `bench_replay.py` (per launch, `replay_seg`), then the real
   model at 1 and 32 streams. Precondition: item 2 (we can build and load our own bridge on this stack).
   The existing switch `PT_HPU_DISABLE_HPUGRAPH_REPLAY_HASHCHECK=1` (`HPUGraph.cpp:535-561`) removes c) only; 00134's
   `cuthash` arm measures it.
5. Next rental, the fork's full native runtime (coordinator, 23:46 UTC; lower than item 3 since the bridge-only path
   needs no Synapse snapshot): (a) checked: farm-base has the SDK's public headers (`synapse_api.h`, `hccl.h`,
   `hlthunk.h`, ...; no `scal.h`), the runtime libraries (`libSynapse.so`, `libhcl.so`, `libscal.so`,
   `libSynapseMme.so`, ...), cmake, g++ 13, build-essential, protobuf-compiler (the distro's) and pybind11, no `-dev`
   package; the three pins are public (`git ls-remote`): the bridge pin is the head of `v1.24.1`, HCL's and the
   Synapse snapshot's are the heads of their `main` (the snapshot is the whole stack, 2024-06, older than 1.24.1;
   the fork's 347 KB patch carries the backports); (b) the unattended build recipe is drafted, never run:
   `tools/flash_next/native_runtime/build_native_runtime.sh` (stages: versions, sources at the pins, the fork's patches
   through `apply_native_runtime.py`, HCL, Synapse through the snapshot's own environment and build order, the bridge,
   the plugin's adapter, a tarball with a manifest); (c) drafted, never run: `tools/flash_next/replay/bench_native_tp4.py`
   (the same TP4 body, 2 plans of 24 layers x 2 sub-blocks `x = all_reduce((x + 1) * 0.25) - 1` on `[rows, 2560]` bf16,
   rows 1 and 32, value-checked, under stock lazy HPU-graph replay or under the fork's native joint replay, written on
   the pattern of `tools/communication/check_dsv4_native_allreduce.py`: the patched bridge's adapter,
   `set_prepared_communication(True)`, `register_tp2_prepared_group_pass()`, `torch.compile(..., backend="hpu_backend")`
   in eager mode, capture under `collect_prepared_group_replays` with `DecoderTopology("deepseek_v4", (24, 24), 2,
   False)`, then `replay_native_decoder` per step) and `tools/flash_next/replay/native_tp4_job.sh` (the 30-minute
   `any:4` job: stock and native, each with and without collectives).
6. What is left of the fixed cost (written before 00174; section 7.10 has the measured split) (INFERRED, TP8, 1 stream, about 36 ms per step): the replay's launches on the main
   thread (148 x about 130 us, about 20 ms on the fastest rank: item 3 takes about 13 ms of it), about 10 ms of
   per-rank spread (the pin, if 00134 shows it), about 3.5 ms of Python around the forward (engine-step), and the
   device time of the ops behind it. The collectives' count is a floor at TP8 + EP8 and they leave on a separate
   Synapse thread.
