#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""engine-step card probe, one card: the sampling stage of a decode step, op by op, at 1 to 32 rows.

Two questions, each answered against the CPU result of the same data:

1. Are the integer-id patterns of the in-graph greedy region (``VLLM_HPU_FUSED_GREEDY_LOGITS=1``) and of the
   block-table gather right on the card (card rule 9: ``torch.topk`` ids narrowed to int32 in the graph that
   made them came back wrong)? Cases: arg max ids narrowed in the same graph (what vLLM's sampler and the fused
   region do), read back as int64, narrowed in a second graph; top-k ids the same three ways (the control: the
   pattern rule 9 was measured on); the min_tokens mask with a host-made int64 index (what the plugin sends) and
   with an int32 index widened in the graph; a host-made int32 block table widened in the graph before a gather
   (``models/qwen4_exp/qsa.py`` ``_select_decode``) against a host-made int64 table.
2. What does the sampling stage cost per step, and what of it grows with the rows? Cases: the general sampler
   graph of a greedy step (fp32 cast of ``[rows, 248320]`` logits, arg max, int32 cast, the read of the ids);
   the same with vLLM's ``MinTokensLogitsProcessor`` mask (``logits.index_put_``; on HPU today no request ever
   reaches that processor, so this is what a fixed ``update_state`` would add); the fused region's mask; the LM
   head with the arg max as one graph and as two (what the fused region saves in launches); a host round trip
   of the fp32 logits (what a CPU fallback of one logits op would cost per row).

A second, short pass runs every case with ``LOG_LEVEL_PT_FALLBACK=1`` and lists the bridge's fallback lines.

Needs only ``torch`` and, on the card, ``habana_frameworks``; ``mask_stop_tokens`` is loaded from the tree by file
path (no plugin import). ``--selftest`` runs on the CPU with a small vocabulary: it checks this script only.

    python3 tools/flash_next/engine_step/hpu_sampling_ops.py --out "$FARM_OUT"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

VOCAB = 248320
HIDDEN = 2560
BLOCKS = 1024  # block-table width at 131,072 tokens and blocks of 128
STOPS = 2  # stop ids per row of the mask cases
TAG = "SAMPLING_OPS "


def load_fused_greedy():
    """``vllm_gaudi/v1/sample/fused_greedy.py`` by path: importing the package needs a card and vLLM."""
    path = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "sample" / "fused_greedy.py"
    spec = importlib.util.spec_from_file_location("engine_step_fused_greedy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Backend:

    def __init__(self, selftest: bool):
        import torch
        self.torch = torch
        self.htorch = None
        self.device = "cpu"
        if not selftest:
            import habana_frameworks.torch as htorch
            self.htorch = htorch
            self.device = "hpu"

    def mark_step(self) -> None:
        if self.htorch is not None:
            self.htorch.core.mark_step()

    def synchronize(self) -> None:
        if self.htorch is not None:
            self.torch.hpu.synchronize()


def ms(seconds):
    if not seconds:
        return None
    return round(1e3 * statistics.median(seconds), 4)


class Data:
    """Logits with one clear maximum per row and a clear runner-up, the stop ids that hide the maximum of every
    second row, the LM-head operands that give the same maxima, and a block table."""

    def __init__(self, be: Backend, rows: int, vocab: int, hidden: int, with_head: bool):
        torch = be.torch
        gen = torch.Generator().manual_seed(1000 + rows)
        self.rows, self.vocab = rows, vocab
        step = max(1, (vocab - 7) // max(rows, 1))
        self.best = torch.tensor([(vocab - 1 - r * step) for r in range(rows)], dtype=torch.int64)  # incl. vocab-1
        self.second = torch.tensor([(3 + r * step + step // 2) % vocab for r in range(rows)], dtype=torch.int64)
        logits = torch.randn(rows, vocab, generator=gen)
        logits[torch.arange(rows), self.best] = 40.0
        logits[torch.arange(rows), self.second] = 30.0
        self.logits_cpu = logits.to(torch.bfloat16)
        self.logits = self.logits_cpu.to(be.device)
        # Every second row masks its maximum and one more id; the other rows mask nothing.
        self.masking = torch.tensor([r % 2 == 0 for r in range(rows)])
        self.stop_lists = [[int(self.best[r]), int((self.best[r] + 11) % vocab)] if self.masking[r] else None
                           for r in range(rows)]
        self.masked_best = torch.where(self.masking, self.second, self.best)
        want = self.logits_cpu.float()
        for r, stops in enumerate(self.stop_lists):
            for token in stops or []:
                want[r, token] = float("-inf")
        assert torch.equal(want.argmax(-1), self.masked_best)
        self.topk_cpu = self.logits_cpu.float().topk(4, dim=-1).indices
        # vLLM's MinTokensLogitsProcessor slices: (request index, token id), int32, one entry per stop id.
        reqs = [r for r, stops in enumerate(self.stop_lists) for _ in (stops or [])]
        toks = [t for stops in self.stop_lists for t in (stops or [])]
        self.put_rows = torch.tensor(reqs, dtype=torch.int32).to(be.device)
        self.put_toks = torch.tensor(toks, dtype=torch.int32).to(be.device)
        self.put_empty = (torch.tensor([], dtype=torch.int32).to(be.device),
                          torch.tensor([], dtype=torch.int32).to(be.device))
        self.neg_inf = torch.tensor(-float("inf"), dtype=torch.float32, device=be.device)
        # A block table like the runner's: int32 [rows, BLOCKS], distinct ids well above 65,535.
        table = (torch.arange(rows * BLOCKS, dtype=torch.int64).view(rows, BLOCKS) * 37 + 70001) % 2000003
        self.table_cpu = table
        self.table_i32 = table.to(torch.int32).to(be.device)
        self.table_i64 = table.to(be.device)
        self.cols_cpu = (torch.arange(64, dtype=torch.int64) * 13 % BLOCKS)[None, :].expand(rows, -1).contiguous()
        self.cols = self.cols_cpu.to(be.device)
        self.hidden = self.weight = None
        if with_head:
            hid = torch.randn(rows, hidden, generator=gen)
            chunks = []
            for start in range(0, vocab, 16384):
                chunks.append((0.02 * torch.randn(min(16384, vocab - start), hidden, generator=gen)).to(
                    torch.bfloat16))
            weight = torch.cat(chunks, dim=0)
            norm = (hid * hid).sum(-1, keepdim=True)
            weight[self.best] = (hid * (40.0 / norm)).to(torch.bfloat16)
            weight[self.second] = (hid * (30.0 / norm)).to(torch.bfloat16)
            self.hidden = hid.to(torch.bfloat16).to(be.device)
            self.weight = weight.to(be.device)
        be.mark_step()
        be.synchronize()


def build_cases(be: Backend, data: Data, fused):
    """name -> (body, expected CPU tensor or None, graphs). ``body()`` issues the step and returns the device
    tensor the runner would read; a body that cuts its work in two graphs calls ``be.mark_step()`` itself."""
    torch = be.torch
    rows = data.rows
    stop = fused.min_tokens_stop_ids(data.stop_lists, rows)
    assert stop is not None
    ids_cpu, active_cpu = stop
    stop_i64 = ids_cpu.to(torch.int64).to(be.device)
    stop_i32 = ids_cpu.to(torch.int32).to(be.device)
    stop_rows = active_cpu.to(torch.int32).to(be.device)
    be.mark_step()
    be.synchronize()
    best32 = data.best.to(torch.int32).view(-1, 1)
    masked32 = data.masked_best.to(torch.int32).view(-1, 1)
    gathered = data.table_cpu.gather(1, data.cols_cpu)

    def sampler():  # vLLM Sampler.forward of an all-greedy batch, as it runs on the card today
        return data.logits.to(torch.float32).argmax(dim=-1).view(-1).long().to(torch.int32).unsqueeze(-1)

    def sampler_int64():
        return data.logits.to(torch.float32).argmax(dim=-1).view(-1).unsqueeze(-1)

    def sampler_cast_second_graph():
        ids = data.logits.to(torch.float32).argmax(dim=-1).view(-1)
        be.mark_step()
        return ids.to(torch.int32).unsqueeze(-1)

    def topk_cast_same_graph():  # card rule 9: the pattern measured wrong in job 00038
        return data.logits.to(torch.float32).topk(4, dim=-1).indices.to(torch.int32)

    def topk_int64():
        return data.logits.to(torch.float32).topk(4, dim=-1).indices

    def topk_cast_second_graph():
        ids = data.logits.to(torch.float32).topk(4, dim=-1).indices
        be.mark_step()
        return ids.to(torch.int32)

    def min_tokens_index_put():  # vLLM MinTokensLogitsProcessor.apply, then the sampler
        x = data.logits.to(torch.float32)
        x.index_put_((data.put_rows, data.put_toks), data.neg_inf)
        return x.argmax(dim=-1).view(-1).to(torch.int32).unsqueeze(-1)

    def min_tokens_index_put_empty():  # the same for requests without stop ids (ignore_eos)
        x = data.logits.to(torch.float32)
        x.index_put_(data.put_empty, data.neg_inf)
        return x.argmax(dim=-1).view(-1).to(torch.int32).unsqueeze(-1)

    def fused_mask_int64_index():  # the fused region's mask as the plugin sends it
        x = fused.mask_stop_tokens(data.logits.to(torch.float32), stop_i64, stop_rows)
        return x.argmax(dim=-1).to(torch.int32).unsqueeze(-1)

    def fused_mask_int32_widened():  # an int32 index widened to int64 in the graph
        x = fused.mask_stop_tokens(data.logits.to(torch.float32), stop_i32.to(torch.int64), stop_rows)
        return x.argmax(dim=-1).to(torch.int32).unsqueeze(-1)

    def table_int32_widened():  # qsa.py _select_decode: md.block_table.long().gather(...)
        return data.table_i32.long().gather(1, data.cols)

    def table_int64():
        return data.table_i64.gather(1, data.cols)

    def table_gather_narrowed():  # gathered ids narrowed to int32 in the same graph
        return data.table_i64.gather(1, data.cols).to(torch.int32)

    def logits_round_trip():  # what a CPU fallback of one logits op would move: fp32 logits out and back
        host = data.logits.to(torch.float32).cpu()
        back = host.to(be.device)
        return back[:, :1]

    cases = {
        "sampler": (sampler, best32, 1),
        "sampler_int64": (sampler_int64, data.best.view(-1, 1), 1),
        "sampler_cast_second_graph": (sampler_cast_second_graph, best32, 2),
        "topk_cast_same_graph": (topk_cast_same_graph, data.topk_cpu.to(torch.int32), 1),
        "topk_int64": (topk_int64, data.topk_cpu, 1),
        "topk_cast_second_graph": (topk_cast_second_graph, data.topk_cpu.to(torch.int32), 2),
        "min_tokens_index_put": (min_tokens_index_put, masked32, 1),
        "min_tokens_index_put_empty": (min_tokens_index_put_empty, best32, 1),
        "fused_mask_int64_index": (fused_mask_int64_index, masked32, 1),
        "fused_mask_int32_widened": (fused_mask_int32_widened, masked32, 1),
        "table_int32_widened": (table_int32_widened, gathered, 1),
        "table_int64": (table_int64, gathered, 1),
        "table_gather_narrowed": (table_gather_narrowed, gathered.to(torch.int32), 1),
        "logits_round_trip": (logits_round_trip, None, 2),
    }
    if data.hidden is not None:
        linear = torch.nn.functional.linear

        def head_two_graphs():  # today: the decode graph ends with the LM head, the sampler is its own graph
            logits = linear(data.hidden, data.weight)
            be.mark_step()
            return logits.to(torch.float32).argmax(dim=-1).view(-1).long().to(torch.int32).unsqueeze(-1)

        def head_one_graph():  # the fused region
            logits = linear(data.hidden, data.weight).to(torch.float32)
            return logits.argmax(dim=-1).to(torch.int32).unsqueeze(-1)

        def head_one_graph_mask():  # the fused region with the min_tokens mask
            logits = linear(data.hidden, data.weight).to(torch.float32)
            logits = fused.mask_stop_tokens(logits, stop_i64, stop_rows)
            return logits.argmax(dim=-1).to(torch.int32).unsqueeze(-1)

        cases["head_two_graphs"] = (head_two_graphs, best32, 2)
        cases["head_one_graph"] = (head_one_graph, best32, 1)
        cases["head_one_graph_mask"] = (head_one_graph_mask, masked32, 1)
    return cases


def run_case(be: Backend, name, body, expected, graphs, warm: int, repeats: int) -> dict:
    record = {"case": name, "graphs": graphs}
    issue, mark, read, total = [], [], [], []
    try:
        first = None
        for index in range(warm + repeats):
            t0 = time.perf_counter()
            out = body()
            t1 = time.perf_counter()
            be.mark_step()
            t2 = time.perf_counter()
            host = out.cpu()
            t3 = time.perf_counter()
            if index == 0:
                first = t3 - t0
                record["first_s"] = round(first, 3)
                if expected is not None:
                    wrong = int((host.to(expected.dtype) != expected).sum()) if host.shape == expected.shape else -1
                    record["dtype"] = str(host.dtype)
                    record["wrong"] = wrong
                    record["ok"] = wrong == 0
                    if wrong:
                        record["got"] = host.flatten()[:8].tolist()
                        record["want"] = expected.flatten()[:8].tolist()
            elif index >= warm:
                issue.append(t1 - t0)
                mark.append(t2 - t1)
                read.append(t3 - t2)
                total.append(t3 - t0)
            if expected is not None and index == warm + repeats - 1:
                last_wrong = int((host.to(expected.dtype) != expected).sum()) if host.shape == expected.shape else -1
                record["wrong_last"] = last_wrong
                record["ok"] = bool(record.get("ok")) and last_wrong == 0
            del out, host
        record.update(issue_ms=ms(issue), mark_ms=ms(mark), read_ms=ms(read), total_ms=ms(total))
    except Exception as error:  # noqa: BLE001 - a probe records every failure and goes on
        record["error"] = "%s: %s" % (type(error).__name__, str(error).replace("\n", " ")[:600])
        record["ok"] = False
        try:
            be.mark_step()
        except Exception:  # noqa: BLE001
            record["error_after"] = traceback.format_exc(limit=1)[-300:]
    return record


def child(args) -> int:
    be = Backend(args.selftest)
    torch = be.torch
    fused = load_fused_greedy()
    vocab = 4096 if args.selftest else VOCAB
    hidden = 64 if args.selftest else HIDDEN
    print(TAG + json.dumps({"case": "env", "device": be.device, "torch": torch.__version__, "vocab": vocab,
                             "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
                             "acc_par": os.environ.get("PT_HPU_LAZY_ACC_PAR_MODE"),
                             "fallback_log": os.environ.get("LOG_LEVEL_PT_FALLBACK")}), flush=True)
    for rows in args.rows:
        data = Data(be, rows, vocab, hidden, with_head=not args.no_head)
        for name, (body, expected, graphs) in build_cases(be, data, fused).items():
            if (args.only and name not in args.only) or name in args.skip:
                continue
            record = run_case(be, name, body, expected, graphs, args.warm, args.repeats)
            record["rows"] = rows
            print(TAG + json.dumps(record), flush=True)
        del data
    return 0


def fmt(value):
    return "-" if value is None else ("%.3f" % value)


# The id checks, in the order their verdicts are printed (card rule 9 first).
VERDICTS = (
    ("sampler", "argmax ids narrowed to int32 in the same graph (vLLM's sampler, the fused greedy region)"),
    ("sampler_int64", "argmax ids read back as int64"),
    ("sampler_cast_second_graph", "argmax ids narrowed to int32 in a second graph"),
    ("topk_cast_same_graph", "topk ids narrowed to int32 in the same graph (rule 9's pattern)"),
    ("topk_int64", "topk ids read back as int64"),
    ("topk_cast_second_graph", "topk ids narrowed to int32 in a second graph"),
    ("fused_mask_int64_index", "min_tokens mask (gather, where, scatter) with a host-made int64 index"),
    ("fused_mask_int32_widened", "min_tokens mask with a host-made int32 index widened to int64 in the graph"),
    ("table_int32_widened", "host-made int32 block table widened to int64 in the graph, then gather"),
    ("table_int64", "host-made int64 block table, gather"),
    ("table_gather_narrowed", "gathered int64 ids narrowed to int32 in the same graph"),
)


def print_verdicts(records) -> None:
    """One line per id check: OK at which rows, WRONG at which (with the first ids got and wanted)."""
    by_case = {}
    for record in records:
        if record.get("case") != "env":
            by_case.setdefault(record["case"], []).append(record)
    print("")
    for name, what in VERDICTS:
        rows = by_case.get(name, [])
        good = [str(r["rows"]) for r in rows if r.get("ok") is True]
        bad = [r for r in rows if r.get("ok") is not True]
        if not rows:
            print("VERDICT %s: NOT RUN" % what)
        elif not bad:
            print("VERDICT %s: OK against the CPU at rows %s" % (what, ",".join(good)))
        else:
            first = bad[0]
            detail = first.get("error") or "%s of the ids differ, got %s want %s" % (
                first.get("wrong"), first.get("got"), first.get("want"))
            print("VERDICT %s: WRONG at rows %s%s; at rows %d: %s" % (
                what, ",".join(str(r["rows"]) for r in bad), (" (OK at rows %s)" % ",".join(good)) if good else "",
                first["rows"], detail[:300]))
    print("", flush=True)


def parent(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).resolve()), "--child", "--rows", ",".join(map(str, args.rows)),
            "--warm", str(args.warm)]
    if args.selftest:
        base.append("--selftest")
    if args.no_head:
        base.append("--no-head")
    # One process per pass (the bridge reads its variables when it loads, and a case that kills its process
    # must not take the others with it): the id checks first, a few iterations each, and their verdicts printed
    # at once; then the timed cases, then vLLM's index_put_ mask, then every case once more with the bridge's
    # fallback log.
    risky = "min_tokens_index_put,min_tokens_index_put_empty"
    passes = [("ids", ["--repeats", "2", "--warm", "1", "--only", ",".join(name for name, _ in VERDICTS),
                       "--no-head"], {}),
              ("timing", ["--repeats", str(args.repeats), "--skip", risky], {}),
              ("index_put", ["--repeats", str(args.repeats), "--only", risky, "--no-head"], {}),
              ("fallback", ["--repeats", "2", "--rows", "8", "--no-head"],
               {"LOG_LEVEL_PT_FALLBACK": "1", "ENABLE_CONSOLE": "true"})]
    records, rc = [], 0
    for label, extra, env_add in passes:
        log_path = out_dir / ("sampling_ops_%s.log" % label)
        env = dict(os.environ, **env_add)
        started = time.time()
        with open(log_path, "w") as log:
            try:
                code = subprocess.run(base + extra, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        text = log_path.read_text(errors="replace")
        print("pass %s: exit %d, %.0f s, log %s" % (label, code, time.time() - started, log_path), flush=True)
        if label == "ids":
            id_records = [json.loads(line[len(TAG):]) for line in text.splitlines() if line.startswith(TAG)]
            (out_dir / "sampling_ops_ids.json").write_text(json.dumps(id_records, indent=1))
            print_verdicts(id_records)
            if code != 0:
                print(text[-3000:])
                rc = code
        elif label != "fallback":
            if label == "timing":
                rc = rc or code
            records += [json.loads(line[len(TAG):]) for line in text.splitlines() if line.startswith(TAG)
                        and not (records and '"case": "env"' in line)]
            if code != 0:
                print(text[-3000:])
        else:
            lines = [line for line in text.splitlines()
                     if "fallback" in line.lower() and not line.startswith(TAG)]
            counts = {}
            for line in lines:
                key = line.split("]")[-1].strip()[:160]
                counts[key] = counts.get(key, 0) + 1
            print("fallback pass: %d line(s) mention a fallback" % len(lines))
            for key, count in sorted(counts.items(), key=lambda kv: -kv[1])[:40]:
                print("  %5d x %s" % (count, key))
            failed = [json.loads(line[len(TAG):]) for line in text.splitlines() if line.startswith(TAG)]
            (out_dir / "sampling_ops_fallback.json").write_text(json.dumps(failed, indent=1))
    (out_dir / "sampling_ops.json").write_text(json.dumps(records, indent=1))
    print("\n%-28s %4s %-5s %-6s %8s %8s %8s %8s  %s" % ("case", "rows", "ok", "graphs", "issue", "mark", "read",
                                                           "total ms", "note"))
    bad = 0
    for record in records:
        if record.get("case") == "env":
            print("env: %s" % json.dumps(record))
            continue
        note = record.get("error") or ("wrong %s of ids, got %s want %s" %
                                       (record.get("wrong"), record.get("got"), record.get("want"))
                                       if record.get("wrong") else "")
        ok = record.get("ok")
        if ok is False:
            bad += 1
        print("%-28s %4d %-5s %-6d %8s %8s %8s %8s  %s" % (
            record["case"], record["rows"], {True: "ok", False: "WRONG", None: "-"}[ok], record["graphs"],
            fmt(record.get("issue_ms")), fmt(record.get("mark_ms")), fmt(record.get("read_ms")),
            fmt(record.get("total_ms")), note[:220]))
    print("\ncases with a wrong result or an error: %d (they are findings, not failures of this probe)" % bad)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("FARM_OUT", "."))
    ap.add_argument("--rows", default="1,2,8,16,32", type=lambda s: [int(x) for x in s.split(",") if x])
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--only", default="", type=lambda s: [x for x in s.split(",") if x])
    ap.add_argument("--skip", default="", type=lambda s: [x for x in s.split(",") if x])
    ap.add_argument("--no-head", action="store_true", help="leave out the LM-head cases (1.3 GB of weight)")
    ap.add_argument("--selftest", action="store_true", help="CPU, small vocabulary: checks this script only")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    sys.exit(main())
