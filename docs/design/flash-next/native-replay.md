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

## Handoff (if this agent stops)

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

### The card job in the queue: 00134 `real4-cut-pin-hash-ab`

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

1. 00134 as above; write its numbers into a section 7.9 and sections 0 and 5; tell the coordinator.
2. The coordinator's bridge build (23:43 UTC): the UNPATCHED bridge at `5176b1b2` (head of `v1.24.1`, the node's
   version) builds in container `nr-bridge-build` from `farm-base:latest` on the node, `--cpuset-cpus 18,37,56,75`, nice
   19, no card, without the C++ tests (`.devops/build.py -c -r -l -j 4 --no-iwyu --no-swig`); script
   `reports/native-replay/late/bridge-build/build_bridge.sh` (on the node `~/farm/native-replay/bridge-build/`, log
   `build.log`; at its end the wheels in `wheels/`, the libraries it wrote in `lib/`). 215 of 840 build steps at 23:47
   UTC. If it ends before about 03:00 UTC: one 1-card job that installs those wheels in the job's container
   (`pip install --no-deps --force-reinstall`) and serves the tiny model (`python /farm/farm_tiny.py --gdn hc`, TP1):
   PASSED with the same tokens as every tiny run proves we can build and load our own bridge on this stack (the
   precondition of item 3). Skip it if the build is not done by about 02:30 UTC. Tell the coordinator either way.
3. **Next week, item 1 (the coordinator's choice of the program's best fixed-cost plan): a bridge-only lean replay
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
4. Next rental, the fork's full native runtime (coordinator, 23:46 UTC; lower than item 3 since the bridge-only path
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
5. After the node is gone (after 04:00 UTC): a snapshot of HEAD (`git archive HEAD | tar -x -C
   reports/native-replay/trees/gate-<sha>`), `bin/gate <snapshot> --tools`, the three worker test files and
   `tests/unit_tests/tools/test_bench_replay.py`, then `gh pr create --base feat/flash-next-gaudi --title "Replay
   benchmark, decode graphs without per-layer cuts, worker CPU placement, and the native-replay design" --body-file
   <body>` (draft: `reports/native-replay/probes/pr_body_1.txt`; add the pin flag; no absolute performance numbers in
   pull request text, AGENTS.md), `gh pr merge --merge`.
6. What is left of the fixed cost (INFERRED, TP8, 1 stream, about 36 ms per step): the replay's launches on the main
   thread (148 x about 130 us, about 20 ms on the fastest rank: item 3 takes about 13 ms of it), about 10 ms of
   per-rank spread (the pin, if 00134 shows it), about 3.5 ms of Python around the forward (engine-step), and the
   device time of the ops behind it. The collectives' count is a floor at TP8 + EP8 and they leave on a separate
   Synapse thread.
