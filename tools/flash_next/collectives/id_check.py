#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Card-against-CPU check of token ids that are made on the device (card rule 9).

Card rule 9 (kernel-fixes job 00038): ``torch.topk`` ids, int64 for torch and 4 bytes on the lazy bridge, narrowed
with ``.to(torch.int32)`` in the same lazy graph come back as every second id, the second half out of range. This
script runs every id-producing pattern of the token path on the card, in one lazy graph and in a replayed HPU
graph, at 1, 2, 5, 32 and 128 rows, and compares the ids with the CPU's:

* serving cases (the exit code is theirs): ``vllm_gaudi.distributed.vocab_parallel.vocab_parallel_argmax`` (ids as
  an fp32 table, max / where / min, an fp32 to int32 cast; with ``--ranks N`` the all-gather of the ``(value, id)``
  pairs through the plugin's communicator), and the same reduction over a whole row of logits;
* probes (reported, they do not change the exit code): ``argmax`` then ``.to(torch.int32)`` (the fused greedy
  region), ``argmax`` then ``.int()`` (the MTP draft loop), ``argmax`` read as int64, narrowed after a
  ``mark_step``, cast to fp32, ``max(dim).indices``, ``topk`` then int32 (the measured failing pattern: the
  control that this check sees rule 9), ``topk`` ids as they are, cast to fp32, read out of an fp32 id table with
  ``torch.gather``, and ``vocab_parallel_topk``;
* the same narrowed ids with their consumer inside the graph (``*_embed``): the ids go through the
  vocabulary-parallel embedding's mask arithmetic and a lookup in a table whose row ``i`` holds ``i``, and only the
  looked-up rows are read back. They say whether a wrong narrowing is seen by the graph itself or only by a read
  to the host (``argmax_int_embed``: the MTP draft loop; ``topk_int32_embed``: the QSA block selection's shape).

Every shard has its 64 largest logits at known columns, all different (and, in the tied data set, the largest
twice: the lowest id must win in the serving cases; what the card's own arg max returns for a tie is reported),
so every top-k up to 64 has one answer. The top-k cases run once per ``--top`` value (default 20 and 64: a
sampled request's top-k and the candidates per rank) and are named ``case@k``. ``stage2_topk_gather@k`` and
``first_maximum@k`` take what the all-gather of 8 ranks' candidates hands on, ``[rows, 8k]`` values and ids.

    python3 tools/flash_next/collectives/id_check.py                 # one card (the job's visible card)
    python3 tools/flash_next/collectives/id_check.py --ranks 8       # one process per card, hccl
    python3 tools/flash_next/collectives/id_check.py --selftest      # CPU (gloo with --ranks 2)

Writes ``idcheck.json`` (``--out``; default ``$FARM_OUT/idcheck[_rN].json``). Prints one ``IDCHECK`` line per case
and a table. Exit code 0: every serving case equals the CPU; 3: one differs; 2: a rank failed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from types import SimpleNamespace

TAG = "IDCHECK "
SHARD_WIDTH = 31040  # one rank's vocabulary shard at TP8 (248,320 / 8)
PADDING = 13  # zero-logit padding columns at the end of a shard (cut off by the vocabulary-parallel reduction)
KNOWN = 64  # the largest logits that are planted, distinct: every top-k up to 64 has one answer
PLANTED = tuple(163.0 - j for j in range(KNOWN))  # bf16-exact, above everything: the largest logits of every row
LOCAL = tuple(63.5 - 0.5 * j for j in range(KNOWN))  # the largest logits of a shard, below PLANTED, above the noise
SERVING = ("vp_argmax", "fp32_ids_row")


def make_logits(torch, rows, world, width, count, seed, tied):
    """The logits of every rank as one CPU tensor ``[rows, world * width]`` bf16 (rank ``r`` owns the columns
    ``[r * width, (r + 1) * width)``; its last ``width - count`` columns are padding, 0.0).

    Noise within +-30. In the upper half of the real columns, at KNOWN distinct global ids per row, PLANTED (the
    largest logits of the row; with ``tied`` a second PLANTED[0] at one more id). In the lower half of every shard
    LOCAL (and with ``tied`` a second LOCAL[0]), so that the KNOWN largest logits of every shard are known and
    distinct too.
    """
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, world * width, generator=gen) * 4.0).clamp_(-30.0, 30.0)
    row_index = torch.arange(rows)
    half = count // 2
    stride = (half - 1) // (KNOWN + 1)  # KNOWN + 1 columns, stride apart, never wrap onto each other
    assert stride >= 1, "a shard needs more than %d real columns" % (2 * (KNOWN + 2))
    steps = torch.arange(KNOWN + 1).view(1, -1)
    for r in range(world):  # padding, then the local values in the lower half of the shard
        x[:, r * width + count:(r + 1) * width] = 0.0
        first = torch.randint(0, half, (rows, 1), generator=gen)
        columns = (first + steps * stride) % half
        for j, value in enumerate(LOCAL):
            x[row_index, r * width + columns[:, j]] = value
        if tied:
            x[row_index, r * width + columns[:, KNOWN]] = LOCAL[0]
    # the global values: slot g of the world * (count - half) upper-half columns is rank g // n, column half + g % n
    n = count - half
    stride = (world * n - 1) // (KNOWN + 1)
    first = torch.randint(0, world * n, (rows, 1), generator=gen)
    slots = (first + steps * stride) % (world * n)
    for j, value in enumerate(PLANTED + ((PLANTED[0], ) if tied else ())):
        slot = slots[:, j]
        x[row_index, (slot // n) * width + half + slot % n] = value
    return x.to(torch.bfloat16)


def make_candidates(torch, rows, k, seed, tied):
    """``[rows, 2, 8 * k]`` fp32 on the CPU: what 8 ranks' gathered candidates look like. Row 0 of the middle
    dimension holds distinct values (with ``tied`` the largest twice), row 1 distinct global token ids."""
    gen = torch.Generator().manual_seed(seed)
    n = 8 * k
    values = torch.stack([torch.randperm(n, generator=gen).float() * 0.25 - 50.0 for _ in range(rows)])
    ids = torch.stack([torch.randperm(248320, generator=gen)[:n].float() for _ in range(rows)])
    if tied:
        best = values.argmax(dim=-1)
        other = (best + n // 2) % n
        values[torch.arange(rows), other] = values[torch.arange(rows), best]
    return torch.stack((values, ids), dim=1)


def child(args):
    import torch
    htorch = None
    if args.selftest:
        device = "cpu"
        try:
            import habana_frameworks  # noqa: F401
        except ImportError:  # the plugin's modules import habana_frameworks: stand-ins on a CPU-only machine
            from tests.unit_tests import hpu_cpu_stubs
            hpu_cpu_stubs.install()
    else:
        import habana_frameworks.torch as htorch  # noqa: F811
        import habana_frameworks.torch.core  # noqa: F401
        device = "hpu"
    world = args.ranks
    group = None
    if world > 1:
        import torch.distributed as dist
        backend = "gloo" if args.selftest else "hccl"
        if not args.selftest:
            try:
                import habana_frameworks.torch.distributed.hccl  # noqa: F401
            except Exception:  # noqa: BLE001
                pass
            torch.hpu.set_device(args.rank)
        dist.init_process_group(backend=backend, init_method=args.init, world_size=world, rank=args.rank,
                                timeout=timedelta(seconds=600))
        group = dist.new_group(list(range(world)), backend=backend)

    from vllm_gaudi.distributed import vocab_parallel as vp

    def mark():
        if htorch is not None:
            htorch.core.mark_step()

    def sync():
        if htorch is not None:
            torch.hpu.synchronize()

    if world > 1:  # the (value, id) pairs go through the plugin communicator's all_gather, as on a served rank
        from vllm_gaudi.distributed.device_communicators import hpu_communicator
        comm = SimpleNamespace(world_size=world, device_group=group)
        os.environ.setdefault("VLLM_HPU_COLLECTIVE_MARK_STEP", "0")
        vp.tensor_model_parallel_all_gather = lambda t, dim=-1: hpu_communicator.HpuCommunicator.all_gather(comm, t, dim)
        vp.get_tensor_model_parallel_world_size = lambda: world

    width = args.width
    count = width - PADDING  # real columns of a shard; the last columns are padding
    start = args.rank * count  # this rank's first global id (rank r owns [r * count, (r + 1) * count))
    vocab = world * count

    def owner_for(shard_start):
        head = SimpleNamespace(tp_size=world, org_vocab_size=vocab, num_added_embeddings=0,
                               shard_indices=SimpleNamespace(org_vocab_start_index=shard_start,
                                                             org_vocab_end_index=shard_start + count,
                                                             added_vocab_start_index=vocab, added_vocab_end_index=vocab))
        return SimpleNamespace(lm_head=head, logits_processor=lambda _head, hidden, skip_gather=True: hidden)

    owner = owner_for(start)

    def real(x):
        return x.narrow(-1, 0, count).to(torch.float32)

    def id_table(rows):
        return torch.arange(start, start + count, dtype=torch.float32, device=device).expand(rows, count)

    def fp32_ids_row(x):
        _, token = vp._lowest_id_of_max(real(x), id_table(x.shape[0]))
        return token.clamp(min=0.0, max=float(start + count - 1)).to(torch.int32)

    def argmax_two_graphs(x):
        ids = real(x).argmax(dim=-1)
        mark()
        return ids.to(torch.int32)

    # Consumers inside the graph: an embedding table whose row i is (i, i, i, i), so the looked-up row names the
    # id the lookup saw. Nothing but that row is read to the host.
    id_rows = torch.arange(count, dtype=torch.float32).unsqueeze(1).repeat(1, 4).to(device)

    def masked_lookup(ids):
        """The vocabulary-parallel embedding's ops on ``ids`` (its mask arithmetic, ``.long()``, the lookup), for
        a shard that owns every id: what the MTP draft loop's ``.int()`` ids go through."""
        from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask
        masked, _ = get_masked_input_and_mask(ids, 0, count, 0, count, count)
        return torch.nn.functional.embedding(masked.long(), id_rows)

    def argmax_int_embed(x):  # hpu_flash_next_mtp.py: argmax, .int(), .view(-1, 1), the embedding of the next pass
        return masked_lookup(x.narrow(-1, 0, count).argmax(dim=-1).int().view(-1, 1))[:, 0, 0]

    # name: (function of the device input, what it returns, needs a collective). The input is this rank's logits
    # [rows, width] bf16, except for the kinds "s" and "m" (gathered candidates [rows, 2, 8k] fp32). Kinds: "g"
    # the global arg max, "G" the global top k ids, "l" this shard's arg max as a local column, "L" the same as a
    # global id, "t" its top k as local columns, "T" as global ids, "s" the top k ids of the candidates, "m" the
    # lowest id among the candidates' largest values.
    cases = {
        "vp_argmax": (lambda x: vp.vocab_parallel_argmax(owner, x), "g", True),
        "fp32_ids_row": (fp32_ids_row, "L", False),
        "argmax_to_int32": (lambda x: real(x).argmax(dim=-1).to(torch.int32).unsqueeze(-1), "l", False),
        "argmax_int": (lambda x: x.narrow(-1, 0, count).argmax(dim=-1).int().view(-1, 1), "l", False),
        "argmax_int64": (lambda x: real(x).argmax(dim=-1), "l", False),
        "argmax_two_graphs": (argmax_two_graphs, "l", False),
        "argmax_to_fp32": (lambda x: real(x).argmax(dim=-1).to(torch.float32), "l", False),
        "max_dim_to_int32": (lambda x: real(x).max(dim=-1).indices.to(torch.int32), "l", False),
        "argmax_int_embed": (argmax_int_embed, "l", False),
        "argmax_embed": (lambda x: torch.nn.functional.embedding(real(x).argmax(dim=-1), id_rows)[:, 0], "l", False),
    }
    case_top = {}  # the k of a case with one

    def add_top_cases(k):
        def topk_gather_fp32(x):  # vocab_parallel_candidates' ids: an fp32 id table read with the indices as they are
            index = torch.topk(real(x), k, dim=-1).indices
            return torch.gather(id_table(x.shape[0]), -1, index).to(torch.int32)

        def topk_int32_embed(x):  # qsa_sparse_ops.select_blocks: topk, .to(int32), integer arithmetic, a lookup
            return masked_lookup(torch.topk(real(x), k, dim=-1).indices.to(torch.int32))[..., 0]

        def stage2_topk_gather(x):  # vocab_parallel_topk after the all-gather: [rows, 8k] candidates, their top k
            values, ids = x.select(1, 0), x.select(1, 1)
            chosen = torch.topk(values, k, dim=-1).indices
            return torch.gather(ids, -1, chosen).to(torch.int32)

        def first_maximum(x):  # the greedy pick of a sampled step's candidates, ids int32 as top_k returns them
            return vp.first_maximum(x.select(1, 0), x.select(1, 1).to(torch.int32))

        more = {
            "vp_topk": (lambda x: vp.vocab_parallel_topk(owner, x, k)[1], "G", True),
            "topk_to_int32": (lambda x: torch.topk(real(x), k, dim=-1).indices.to(torch.int32), "t", False),
            "topk_int64": (lambda x: torch.topk(real(x), k, dim=-1).indices, "t", False),
            "topk_to_fp32": (lambda x: torch.topk(real(x), k, dim=-1).indices.to(torch.float32) + float(start), "T",
                             False),
            "topk_gather_fp32": (topk_gather_fp32, "T", False),
            "topk_int32_embed": (topk_int32_embed, "t", False),
            "topk_embed": (lambda x: torch.nn.functional.embedding(torch.topk(real(x), k, dim=-1).indices,
                                                                    id_rows)[..., 0], "t", False),
            "stage2_topk_gather": (stage2_topk_gather, "s", False),
        }
        if hasattr(vp, "first_maximum"):
            more["first_maximum"] = (first_maximum, "m", False)
        for name, case in more.items():
            cases["%s@%d" % (name, k)] = case
            case_top["%s@%d" % (name, k)] = k

    for k in args.top:
        add_top_cases(k)
    if args.selftest:  # the detector itself: what rule 9 does to [rows] ids (element i holds the true element 2 * i)

        def rule9_emulated(x):
            ids = real(x).argmax(dim=-1)
            return torch.cat((ids[0::2], torch.full((ids.shape[0] // 2, ), 2**31 - 1, dtype=ids.dtype)))

        cases["selftest_rule9"] = (rule9_emulated, "l", False)
    # --cases names a case with or without its "@k". Order: the serving cases, the forms this track's top-k uses,
    # the other top-k forms, the arg max forms, the consumers: a time budget cuts from the end.
    order = ("vp_argmax", "fp32_ids_row", "vp_topk", "topk_gather_fp32", "stage2_topk_gather", "first_maximum",
             "topk_to_int32", "topk_to_fp32", "topk_int64", "argmax_to_int32", "argmax_int", "argmax_int64",
             "argmax_two_graphs", "argmax_to_fp32", "max_dim_to_int32", "argmax_int_embed", "argmax_embed",
             "topk_int32_embed", "topk_embed")
    wanted = [c for c in cases if not args.cases or c in args.cases or c.split("@")[0] in args.cases]
    wanted.sort(key=lambda c: order.index(c.split("@")[0]) if c.split("@")[0] in order else len(order))

    class Wrapped(torch.nn.Module):

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    def expected(kind, full_cpu, k):
        """The CPU's ids for ``kind``: from the whole logits ``[rows, world * width]`` (every rank builds them), or
        for the kinds "s" and "m" from the candidates ``[rows, 2, 8k]``."""
        if kind in ("s", "m"):
            values, ids = full_cpu[:, 0], full_cpu[:, 1]
            if kind == "s":
                return torch.gather(ids, -1, torch.topk(values, k, dim=-1).indices).long()
            best = values.amax(dim=-1, keepdim=True)  # the lowest id among the largest values
            return torch.where(values == best, ids, torch.full_like(ids, float(2**30))).amin(dim=-1).long()
        shards = [full_cpu[:, r * width:r * width + count].to(torch.float32) for r in range(world)]
        if kind in ("g", "G"):
            whole = torch.cat(shards, dim=-1)  # global id = rank * count + column
            return whole.argmax(dim=-1) if kind == "g" else torch.topk(whole, k, dim=-1).indices
        mine = shards[args.rank]
        if kind in ("l", "L"):
            ids = mine.argmax(dim=-1)  # the first of equal values: the lowest column
            return ids + (start if kind == "L" else 0)
        ids = torch.topk(mine, k, dim=-1).indices
        return ids + (start if kind == "T" else 0)

    def compare(got, want, tied, kind):
        got = got.detach().to("cpu").reshape(want.shape[0], -1).to(torch.float64)
        want = want.reshape(want.shape[0], -1).to(torch.float64)
        rec = {"shape": list(got.shape)}
        if got.shape != want.shape:
            rec.update(ok=False, note="shape %s, expected %s" % (list(got.shape), list(want.shape)))
            return rec
        if tied and kind in ("t", "T", "G", "s"):  # two equal largest values: their order in a top-k is not defined
            got, want = got.sort(dim=-1).values, want.sort(dim=-1).values
        equal_rows = int((got == want).all(dim=-1).sum())
        rec.update(ok=equal_rows == want.shape[0], equal_rows=equal_rows, rows=want.shape[0])
        if not rec["ok"]:
            flat_got, flat_want = got.reshape(-1), want.reshape(-1)
            half = flat_want.numel() // 2
            rec["got_head"] = [int(v) if abs(v) < 2**53 else None for v in flat_got[:8].tolist()]
            rec["want_head"] = [int(v) for v in flat_want[:8].tolist()]
            # rule 9's signature: element i holds the true element 2 * i, the second half is out of range
            rec["every_second_id"] = bool(half >= 1 and (flat_got[:half] == flat_want[0:2 * half:2]).all())
        return rec

    records = []
    started = time.monotonic()
    seed = 1000

    def run(name, regime, rows, tied):
        nonlocal seed
        fn, kind, collective = cases[name]
        rec = {"case": name, "regime": regime, "rows": rows, "tied": tied, "rank": args.rank, "width": width,
               "world": world, "serving": name in SERVING}
        if time.monotonic() - started > args.budget_s and not collective:
            rec.update(skipped="time budget")  # collective cases never skip: every rank must issue them
            return rec
        try:
            calls = 3 if regime == "replay" else 1
            module = fn
            if regime == "replay" and htorch is not None:
                module = htorch.hpu.wrap_in_hpu_graph(Wrapped(fn), disable_tensor_cache=True)
            results = []
            for _ in range(calls):  # a replay gets new inputs on every call
                seed += 1
                k = case_top.get(name, 1)
                if kind in ("s", "m"):  # the candidates of 8 ranks, made on the host
                    full = make_candidates(torch, rows, k, seed, tied)
                    x = full.to(device)
                else:
                    full = make_logits(torch, rows, world, width, count, seed, tied)
                    x = full[:, args.rank * width:(args.rank + 1) * width].contiguous().to(device)
                mark()
                sync()
                got = module(x)
                mark()
                results.append(compare(got, expected(kind, full, k), tied, kind))
            rec.update(results[-1])
            rec["ok"] = all(r.get("ok") for r in results)
            rec["calls_ok"] = [bool(r.get("ok")) for r in results]
            for r in results:
                if not r.get("ok"):
                    rec.update({k: v for k, v in r.items() if k != "ok"})
                    break
        except Exception as exc:  # noqa: BLE001 - a pattern the bridge refuses is a finding, not a crash
            rec.update(ok=False, error="%s: %s" % (type(exc).__name__, str(exc)[:300]))
            try:
                mark()
                sync()
            except Exception:  # noqa: BLE001
                pass
        return rec

    def emit(rec):
        records.append(rec)
        print(TAG + json.dumps(rec), flush=True)

    regimes = ["lazy", "replay"] if not args.selftest else ["lazy"]
    # 1. the cases with a collective, in the same order on every rank; 2. the local probes
    for collective_pass in (True, False):
        for name in wanted:
            if cases[name][2] != collective_pass:
                continue
            for tied in (False, True):
                for regime in regimes:
                    for rows in args.rows:
                        emit(run(name, regime, rows, tied))
    print(TAG + json.dumps({"done": args.rank, "seconds": round(time.monotonic() - started, 1)}), flush=True)
    sys.stdout.flush()
    os._exit(0)


def summarize(records):
    """One line per (case, regime, tied): the row counts that are equal to the CPU and those that are not."""
    by_key = {}
    for rec in records:
        if "case" in rec:
            by_key.setdefault((rec["case"], rec["regime"], rec["tied"]), []).append(rec)
    lines, serving_ok = [], True
    for (case, regime, tied), recs in by_key.items():
        ok_rows = sorted({r["rows"] for r in recs if r.get("ok")} - {r["rows"] for r in recs if r.get("ok") is False})
        bad = [r for r in recs if r.get("ok") is False]
        skipped = sorted({r["rows"] for r in recs if "skipped" in r})
        bad_rows = sorted({r["rows"] for r in bad})
        notes = []
        if any(r.get("every_second_id") for r in bad):
            notes.append("every second id (rule 9)")
        errors = sorted({r["error"][:120] for r in bad if "error" in r})
        if errors:
            notes.append("error: " + errors[0])
        heads = [(r.get("got_head"), r.get("want_head")) for r in bad if "got_head" in r][:1]
        if heads:
            notes.append("got %s want %s" % (heads[0][0][:4], heads[0][1][:4]))
        serving = case in SERVING
        if serving and bad:
            serving_ok = False
        lines.append({"case": case, "regime": regime, "tied": tied, "serving": serving, "equal_rows": ok_rows,
                      "wrong_rows": bad_rows, "skipped_rows": skipped, "notes": notes})
        print("idcheck: %-18s %-6s %-6s %s equal at rows %s%s%s%s" % (
            case, regime, "tied" if tied else "untied", "SERVING" if serving else "probe  ", ok_rows or "-",
            ("; WRONG at rows %s" % bad_rows) if bad_rows else "", ("; skipped %s" % skipped) if skipped else "",
            ("; " + "; ".join(notes)) if notes else ""), flush=True)
    return lines, serving_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--rows", default="1,2,5,32,128")
    ap.add_argument("--width", type=int, default=SHARD_WIDTH, help="columns of one rank's logits (with %d padding)" % PADDING)
    ap.add_argument("--cases", default="", help="comma-separated case names, with or without @k (default: all)")
    ap.add_argument("--top", default="20,64", help="the k of the top-k cases (at most %d each)" % KNOWN)
    ap.add_argument("--budget-s", type=float, default=360.0, help="local probes are skipped after this many seconds")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--cards", default="", help="HABANA_VISIBLE_MODULES of the ranks (default: inherited)")
    ap.add_argument("--logdir", default="")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args()
    args.rows = [int(r) for r in str(args.rows).split(",") if r]
    args.cases = [c for c in str(args.cases).split(",") if c]
    args.top = [int(k) for k in str(args.top).split(",") if k]
    if any(k < 1 or k > KNOWN for k in args.top):
        ap.error("--top takes values from 1 to %d" % KNOWN)
    if args.child:
        return child(args)
    if args.selftest:
        args.rows = [r for r in args.rows if r <= 32]
        args.width = min(args.width, 2048)
    out = args.out or os.path.join(os.environ.get("FARM_OUT", "."),
                                   "idcheck.json" if args.ranks == 1 else "idcheck_r%d.json" % args.ranks)
    workdir = tempfile.mkdtemp(prefix="idcheck-")
    logdir = args.logdir or workdir
    os.makedirs(logdir, exist_ok=True)
    env = dict(os.environ)
    if not args.selftest:  # as the plugin runs lazy tensor parallel
        env.setdefault("PT_HPU_LAZY_MODE", "1")
        env.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        env.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        if args.cards:
            env["HABANA_VISIBLE_MODULES"] = args.cards
    procs = []
    for rank in range(args.ranks):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--rank", str(rank), "--ranks", str(args.ranks),
               "--init", "file://" + os.path.join(workdir, "init"), "--rows", ",".join(map(str, args.rows)),
               "--width", str(args.width), "--budget-s", str(args.budget_s), "--cases", ",".join(args.cases),
               "--top", ",".join(map(str, args.top))]
        if args.selftest:
            cmd.append("--selftest")
        log = open(os.path.join(logdir, "idcheck_rank%d.log" % rank), "w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env), log))
    deadline = time.monotonic() + args.timeout_s
    records, failed = [], False
    for rank, (proc, log) in enumerate(procs):
        try:
            proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            failed = True
            print("idcheck: rank %d did not end in %d s: killed" % (rank, args.timeout_s), flush=True)
        log.close()
        with open(log.name) as f:
            text = f.read()
        done = False
        for line in text.splitlines():
            if line.startswith(TAG):
                rec = json.loads(line[len(TAG):])
                done = done or "done" in rec
                records.append(rec)
        if not done:
            failed = True
            print("idcheck: rank %d ended without its last record; the end of its log:" % rank, flush=True)
            print("\n".join("  " + ln[:300] for ln in text.splitlines()[-25:]), flush=True)
    lines, serving_ok = summarize(records)
    with open(out, "w") as f:
        json.dump({"ranks": args.ranks, "width": args.width, "rows": args.rows, "summary": lines, "records": records},
                  f, indent=1)
    verdict = "FAILED (a rank did not finish)" if failed else ("PASS" if serving_ok else "FAIL")
    print("idcheck: serving cases (%s) on %d rank(s): %s; results in %s" % (", ".join(SERVING), args.ranks, verdict,
                                                                            out), flush=True)
    return 2 if failed else (0 if serving_ok else 3)


if __name__ == "__main__":
    sys.exit(main())
