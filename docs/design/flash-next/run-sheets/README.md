# Run sheets — template

Run sheets are the only way this program touches hardware: no agent has Gaudi access and the node is rented by the hour (system map, assumption 3). The user runs one sheet in one rental session, fail-fast: stages ordered cheapest and most informative first, each with a stop rule and the lines to capture. When a stop rule fires, stop the session, run the capture commands and post what they printed — never debug on rented time (system map §14).

The current sheet is [qualification.md](qualification.md): stages 1–2 (build and native compile, tiny-model bring-up, real-checkpoint TP8 BF16 bring-up) are written by M0; stage 3 (flag-by-flag A/B: sparse QSA, GDN tactics at TP8 with a decode sweep, MoE expert slices, and the decode batch baseline) is written by M2; stage 3e (MTP speculative decoding A/B) is written by M3; stage 4 (admission variants A–D, the prefix-hit check, the recurrent state pool) is written by M1; stages 5–6 are filled by milestones M4 and M1/M4.

## Rules for every sheet

1. One section per stage, with the eight sections of the skeleton below, in this order and no others.
2. Every command in a fenced `bash` block, runnable as posted. Paths that exist only on the node are marked "(on the node)".
3. Every expected log line is quoted exactly and carries its source in a trailing comment, for example `# system map §11, Run 1`. No invented expectations.
4. The environment is set explicitly inside the launch block; a stage never depends on shell history or on a previous stage's exports.
5. The capture commands read `docker logs`, which only shows a server started as the container's command. If you start `vllm serve` from `docker exec` or a shell inside the container, append `2>&1 | tee /tmp/serve.log` to it and grep `/tmp/serve.log` instead.
6. Predictions (for example S4's `Usable num_blocks`) are pasted from the source command, labelled "predicted", and the stop rule names the tolerance.

## Skeleton

````markdown
## Stage N: <title>

### Purpose
<One sentence: what this stage proves and what failure it rules out.>

### Prerequisites
<Image, checkpoint paths, branch and commit to build; "(on the node)" where node-side.>

### Launch
```bash
<exact environment and command>
```

### Expected startup lines
| Expected line (quoted) | Source | Observed |
|---|---|---|
| `<the line>` | <document §, run> | |

### Capture commands
```bash
docker logs <container> 2>&1 | grep -E "Avg prompt throughput|Prefix cache hit rate|Running: |Waiting: "   # R7, RUK-2728
docker logs <container> 2>&1 | grep -i -E "recompil|graph compile|bucket|warmup finished|max_num_batched_tokens|GDN layers detected|Compact GDN|mamba"   # R7, RUK-2728
```

### Results table
| Check | Expected | Observed | Pass? |
|---|---|---|---|

### Stop rule
<The conditions that end the stage, and what to post when one fires.>

### What to post back
- the two `docker logs` greps above, complete;
- the results table, filled in;
- an `hl-smi` snapshot (on the node).
````
