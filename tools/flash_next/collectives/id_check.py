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
  ``torch.gather``, and ``vocab_parallel_topk``.

Every row has one largest logit at a known column (and, in the tied data set, a second column with the same
value: the lowest id must win in the serving cases; what the card's own arg max returns for a tie is reported).

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
PLANTED = (100.0, 90.0, 80.0, 70.0)  # bf16-exact, far above the noise: the four largest logits of every row
LOCAL = (50.0, 49.0, 48.0, 47.0)  # the four largest logits of a shard that holds none of PLANTED
TOP = 4
SERVING = ("vp_argmax", "fp32_ids_row")


def make_logits(torch, rows, world, width, count, seed, tied):
    """The logits of every rank as one CPU tensor ``[rows, world * width]`` bf16 (rank ``r`` owns the columns
    ``[r * width, (r + 1) * width)``; its last ``width - count`` columns are padding, 0.0).

    Noise within +-30. In the upper half of the real columns, at four distinct global ids per row, PLANTED (the
    four largest logits of the row; with ``tied`` a second PLANTED[0] at a fifth id). In the lower half of every
    shard LOCAL (and with ``tied`` a second LOCAL[0]), so that the four largest logits of every shard are known
    and distinct too.
    """
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, world * width, generator=gen) * 4.0).clamp_(-30.0, 30.0)
    row_index = torch.arange(rows)
    half = count // 2
    stride = half // 7 + 1
    for r in range(world):  # padding, then the local values in the lower half of the shard
        x[:, r * width + count:(r + 1) * width] = 0.0
        first = torch.randint(0, half, (rows, 1), generator=gen)
        columns = (first + torch.arange(5).view(1, 5) * stride) % half
        for j, value in enumerate(LOCAL):
            x[row_index, r * width + columns[:, j]] = value
        if tied:
            x[row_index, r * width + columns[:, 4]] = LOCAL[0]
    # the global values: slot g of the world * (count - half) upper-half columns is rank g // n, column half + g % n
    n = count - half
    stride = (world * n) // 7 + 1
    first = torch.randint(0, world * n, (rows, 1), generator=gen)
    slots = (first + torch.arange(5).view(1, 5) * stride) % (world * n)
    for j, value in enumerate(PLANTED + ((PLANTED[0], ) if tied else ())):
        slot = slots[:, j]
        x[row_index, (slot // n) * width + half + slot % n] = value
    return x.to(torch.bfloat16)


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

    def topk_gather_fp32(x):
        index = torch.topk(real(x), TOP, dim=-1).indices
        return torch.gather(id_table(x.shape[0]), -1, index).to(torch.int32)

    # name: (function of the device logits, what it returns: "g" global arg max, "l" this shard's arg max as a
    # local column, "L" the same as a global id, "t" local top-4 columns, "T" global top-4 ids, needs a collective)
    cases = {
        "vp_argmax": (lambda x: vp.vocab_parallel_argmax(owner, x), "g", True),
        "vp_topk": (lambda x: vp.vocab_parallel_topk(owner, x, TOP)[1], "G", True),
        "fp32_ids_row": (fp32_ids_row, "L", False),
        "argmax_to_int32": (lambda x: real(x).argmax(dim=-1).to(torch.int32).unsqueeze(-1), "l", False),
        "argmax_int": (lambda x: x.narrow(-1, 0, count).argmax(dim=-1).int().view(-1, 1), "l", False),
        "argmax_int64": (lambda x: real(x).argmax(dim=-1), "l", False),
        "argmax_two_graphs": (argmax_two_graphs, "l", False),
        "argmax_to_fp32": (lambda x: real(x).argmax(dim=-1).to(torch.float32), "l", False),
        "max_dim_to_int32": (lambda x: real(x).max(dim=-1).indices.to(torch.int32), "l", False),
        "topk_to_int32": (lambda x: torch.topk(real(x), TOP, dim=-1).indices.to(torch.int32), "t", False),
        "topk_int64": (lambda x: torch.topk(real(x), TOP, dim=-1).indices, "t", False),
        "topk_to_fp32": (lambda x: torch.topk(real(x), TOP, dim=-1).indices.to(torch.float32) + float(start), "T",
                         False),
        "topk_gather_fp32": (topk_gather_fp32, "T", False),
    }
    if args.selftest:  # the detector itself: what rule 9 does to [rows] ids (element i holds the true element 2 * i)

        def rule9_emulated(x):
            ids = real(x).argmax(dim=-1)
            return torch.cat((ids[0::2], torch.full((ids.shape[0] // 2, ), 2**31 - 1, dtype=ids.dtype)))

        cases["selftest_rule9"] = (rule9_emulated, "l", False)
    wanted = [c for c in args.cases if c in cases] if args.cases else list(cases)

    class Wrapped(torch.nn.Module):

        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    def expected(kind, full_cpu, tied):
        """The CPU's ids for ``kind`` from the whole logits ``[rows, world * width]`` (every rank builds them)."""
        shards = [full_cpu[:, r * width:r * width + count].to(torch.float32) for r in range(world)]
        if kind in ("g", "G"):
            whole = torch.cat(shards, dim=-1)  # global id = rank * count + column
            return whole.argmax(dim=-1) if kind == "g" else torch.topk(whole, TOP, dim=-1).indices
        mine = shards[args.rank]
        if kind in ("l", "L"):
            ids = mine.argmax(dim=-1)  # the first of equal values: the lowest column
            return ids + (start if kind == "L" else 0)
        ids = torch.topk(mine, TOP, dim=-1).indices
        return ids + (start if kind == "T" else 0)

    def compare(got, want, tied, kind):
        got = got.detach().to("cpu").reshape(want.shape[0], -1).to(torch.float64)
        want = want.reshape(want.shape[0], -1).to(torch.float64)
        rec = {"shape": list(got.shape)}
        if got.shape != want.shape:
            rec.update(ok=False, note="shape %s, expected %s" % (list(got.shape), list(want.shape)))
            return rec
        if tied and kind in ("t", "T", "G"):  # two equal largest values: their order in a top-k is not defined
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
                full = make_logits(torch, rows, world, width, count, seed, tied)
                x = full[:, args.rank * width:(args.rank + 1) * width].contiguous().to(device)
                mark()
                sync()
                got = module(x)
                mark()
                results.append(compare(got, expected(kind, full, tied), tied, kind))
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
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
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
               "--width", str(args.width), "--budget-s", str(args.budget_s), "--cases", ",".join(args.cases)]
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
