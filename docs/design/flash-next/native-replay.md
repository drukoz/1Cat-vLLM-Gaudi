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

## 8. Build steps

| Step | What | Proof off the card | Card | State |
|---|---|---|---|---|
| 1 | `tools/flash_next/replay/bench_replay.py`: the experiment of section 6 | gloo self-test of every regime, replay equals lazy; `tests/unit_tests/tools/test_bench_replay.py` | job 00012 | committed `bd813c29` |
| 2 | `VLLM_HPU_DECODE_CUT_INTERVAL` (default 0): decode and verification forwards cut after every N-th decoder layer instead of every `VLLM_CONFIG_HIDDEN_LAYERS`-th; prompt forwards unchanged. Hooks are only re-registered when the flag is set, so 0 is the old code path | `tests/unit_tests/worker/test_decode_cut_interval.py` (11 tests) | token parity on the tiny model (job 00027, `tiny_smoke.sh --gdn hc --env VLLM_HPU_DECODE_CUT_INTERVAL=1000`); the real-model decode effect of the same graph shape is step-anatomy's `VLLM_CONFIG_HIDDEN_LAYERS` 48/4 A/B (00016) | committed `021f9bf6` |
| 3 | Replay-entry regimes in the benchmark: `graph_raw` (direct `HPUGraph`, no input hash), `replay_async`, `replay_meta` (20 int32 tensors and 2 scalars as a namedtuple keyword, like the trimmed decode metadata) | gloo self-test; `test_bench_replay.py` (8 tests) | job 00027 step A | committed `aef003f4` |
| 4 | Only if 00012/00027 show the replay entry or the tail costs about 1 ms or more per step: a direct `HPUGraph` per decode bucket over persistent input buffers (with engine-step's D3) and the logits tail captured with the forward | CPU: same logits and tokens with and without, graph keys unchanged | real model at 1 and 32 streams | decided by the measurements |
| 5 | `VLLM_HPU_GRAPH_ASYNC_REPLAY` (default 0): the model's HPU graph wrapped with `asynchronous=True`, so the forward returns once the replay is queued | `tests/unit_tests/worker/test_hpu_graph_async_replay.py` (4 tests) | job 00027: bench `replay_tail` against `replay_async_tail`, the async-equals-sync token check, and tiny-model parity with the flag on | committed `6966226e` |
| 6 | Bench regimes for grouped collectives (`replay_par2`, `replay_fused2`, `replay_coalesce`), a sampler tail, a torch.profiler trace of a replay, the knob names of the bridge and HCL | `test_bench_replay.py` (11 tests); gloo self-test | job 00027 step A | committed `d85d9730`, `e2726c41` |

## Handoff (if this agent stops)

State at 2026-11-20 19:45 UTC (third agent of this track, started 19:00 UTC). Branch
`feat/flash-next-gaudi-native-replay` (worktree `/Volumes/Storage/Claude/gaudi-work/wt/native-replay`), pushed to
`origin`, the program head `b0cd0bb` merged in (`b0140c3f`); nothing of this track is merged into
`feat/flash-next-gaudi` yet. No uncommitted change in the repository. This harness refuses report files, so this
section is the handoff. Outside the repository (track folder
`/Volumes/Storage/Claude/gaudi-work/reports/native-replay/`): `late/replay_job.sh` and `late/real_ab.sh` (the
host-side scripts the queued jobs run, also in `~/farm/native-replay/late/` on the node; md5 logged by each job),
`probes/knobs/upper_names.txt` (every upper-case name in the bridge, Synapse, HCL and SCAL libraries of the image:
the settings that exist), `probes/const_ops.py` (section 7.3), `logs/` (CPU test and gate logs), `card/` (results).

### Done and verified

- Decode replays a stock HPU graph; the replay issues every recipe and collective from the calling thread. Measured
  at TP8 (job 00012, `card/replay-bench-r8/`): 130-195 us per all-reduce, 45-90 us per extra recipe, host held for
  73-93 % of a replay with collectives; compile mode no cheaper (section 7.2).
- The fork's native replay and prepared/pre-posted HCCL paths do not transfer (section 3, file:line).
- `VLLM_HPU_DECODE_CUT_INTERVAL` (commit `021f9bf6`) and `VLLM_HPU_GRAPH_ASYNC_REPLAY` (`6966226e`), both default 0,
  CPU tests green: `tests/unit_tests/worker/test_decode_cut_interval.py` (11),
  `tests/unit_tests/worker/test_hpu_graph_async_replay.py` (4).
- The benchmark separates a launch from a collective and looks at the host from outside (commits `649d32f6`,
  `eefaf058`, `22ec52cf`; `tests/unit_tests/tools/test_bench_replay.py`, 21 tests; gloo self-test of every new
  regime): `replay_seg`, `*_cache`, `*_busy` (the device's floor behind one long operation), `replay_ag`
  (all-gather plus local sum), host CPU per call, `--env-sets`, `--hold-file` with
  `tools/flash_next/replay/host_sampler.sh` (thread CPU and context switches from /proc, gdb stacks, strace of
  rank 0; its process matching tested in the Linux VM; gdb and strace themselves first run in job 00067),
  `--ranks 1`.
- Both flags log one line when they take effect (commit `6423a274`): look for
  `VLLM_HPU_DECODE_CUT_INTERVAL=1000: a decode forward cuts after every 1000-th decoder layer` in a server log.
- Device incident of job 00012 (module 0, "No progress error" in `replay_coll` at 128 rows, 8 ranks): card-farm's
  verdict is a workload trigger, not a flaky module. No job of this track runs `replay_coll`.

### Queued card jobs (submitted 19:22-19:24 UTC) and what to do with each result

Every job is a stub that runs `~/farm/native-replay/late/replay_job.sh RUN MODE 649d32f69cb2`; the commit that
really runs is the one in `~/farm/native-replay/late/SHA` (now `22ec52cf4480`, image built): write a newer
synced commit there to move the queued jobs to it. Fetch with
`bin/farm wait <id>; bin/farm fetch native-replay <id> <RUN>...` into `reports/native-replay/card/`.
`reports/native-replay/probes/knob_table.py <knobs_rN.json>` prints the per-setting table,
`probes/stacks.py <bench/sampler/stacks.txt>` the stack histogram.

1. **00065** `replay-host-path-r8-tiny-cut` (`all`, 20 min; the coordinator bumps it behind prefill's 00014).
   Run folders `host-r8` and `host-r8-cut`. Steps: (a) `bench/replay_r8.log|json`: 14 regimes at rows 1 and 32,
   a hold loop of 2,500 replays with the sampler (`bench/sampler/threads.txt`, `stacks.txt`, `stacks_full.txt`,
   `strace_c.txt`, `strace_main.txt`), a profiler trace; (b) `tiny-cut.log` and `../host-r8-cut/summary.json`:
   the tiny model at TP8 with `VLLM_HPU_DECODE_CUT_INTERVAL=1000` (PASSED = token parity = the merge condition of
   that flag); (c) `bench/busy_r8.log|json`: the device's floor (`*_busy`: valid where the printed host call is
   shorter than the long operation); (d) `bench/knobs_r8.log|json`: `replay` under 8 settings (`replay@<name>`).
   Readings: `replay_async_tail` against `replay_tail` (`vs_total_ms`): what an asynchronous replay overlaps
   (expected under 1 ms: only the logits and sampler issue can overlap). `replay_seg`: "one launch" and "one
   collective alone" (the two parts of 00012's 130-195 us). `replay_fused2` against `replay_par2`: one collective
   less per layer at the same bytes and launches, so the difference is the cost of a collective call alone.
   `replay_cache` against `replay`: whether keeping the captured tensors' memory makes a launch cheaper.
   `threads.txt`/`stacks.txt`: which thread and which library the host time is in. Tell the coordinator
   (`SendMessage` to `main`) at once: does asynchronous replay overlap host issue with the device, does a grouped
   or coalesced form cut the host cost per collective.
2. **00066** `tiny-tp4-cut-then-async` (`any:4`, 14 min; took 00053's place). Run folders `tiny4-cut`,
   `tiny4-cut-async`, logs `tiny4/tiny-cut.log`, `tiny4/tiny-cut-async.log`. PASSED on the first = parity of the cut
   flag at TP4; on the second = parity with the asynchronous replay as well.
3. **00067** `replay-launch-path-1card` (`any:1`, 10 min; bumped by the coordinator, expected 20:20 to 21:15
   UTC). Run folder `launch-r1`: the launch path on one card (no collectives): `replay_seg` and `replay_layer`
   (cost per launch), `*_cache`, the sampler on a loop of `replay_seg` replays, the device's floor per launch
   (`bench/busy_r1.log`), then 12 bridge and Synapse settings (`bench/knobs_r1.log`).
4. **00068** `replay-host-path-r4` (`any:4`, 14 min, end of the queue). Run folder `host-r4`: job 00065's bench
   steps at 4 ranks (the layout of E1).
The coordinator bumped 00067 and bumps 00065 (19:28 UTC) and asked for: the host cost per launch and per
collective under every setting, and which settings are safe for every server (output unchanged on the tiny
model). For the second: `replay_job.sh RUN tinyenv <sha12> name:KEY=VALUE[,KEY=VALUE]...` serves the tiny model
once per setting (submit it as a stub like the others, `any:4` or `any:1`, about 4 min per setting). `late/real_ab.sh` is ready for a real-model A/B
(not submitted yet): `printf 'bash $HOME/farm/native-replay/late/real_ab.sh real4 1,32 128 base cut:VLLM_HPU_DECODE_CUT_INTERVAL=1000\n' | bin/farm submit native-replay real4-cut-ab any:4 32m`
(two server starts on the same four cards; results in `real4-base/` and `real4-cut/`, `scaling.json`).

### Pending subtasks, in order

1. **Merge the cut-interval flag** once 00065 step (b) or 00066 passes. Protocol:
   `git fetch origin && git merge origin/feat/flash-next-gaudi`, rerun the three test files above and
   `/Volumes/Storage/Claude/gaudi-work/bin/gate <worktree> --tools` (its log for `b0140c3f`:
   `reports/native-replay/logs/gate_b0140c3f.log`), `git push`,
   `gh pr create --base feat/flash-next-gaudi --title "Replay benchmark, decode graphs without per-layer cuts, and the native-replay design" --body-file <body>`,
   `gh pr merge --merge`. A body without raw numbers is drafted in the session scratchpad
   (`pr_native_replay_1.md`; AGENTS.md forbids absolute performance numbers in PR text); if it is gone, write it
   from section 0 and section 8.
2. **Write the results of 00065-00068 into section 7** (7.4 the launch and the collective apart, 7.5 the host
   inside a replay, 7.6 settings) and bring sections 0 and 5 in line.
3. **Real-model A/B of the cut flag** (and of the asynchronous replay only if step (a) shows 0.3 ms or more):
   the command above; add `cut-async:VLLM_HPU_DECODE_CUT_INTERVAL=1000,VLLM_HPU_GRAPH_ASYNC_REPLAY=1` as a third
   arm with a 45 min hold.
4. **Whatever setting or form the sweeps show cheaper** (tensor cache for decode graphs, a Synapse or bridge
   setting, the primitive all-reduce): a flag in `vllm_gaudi/envs.py` default 0, a CPU test, tiny-model parity, a
   real-model arm.
