#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay lookup (n-gram) drafters over token streams the model really produced: acceptance per draft position.

Greedy verification accepts a draft token exactly when it equals the token the target emits. For a KNOWN greedy
stream (prompt ids, output ids) the outcome of any lookup drafter is therefore a matter of arithmetic: play the
stream step by step, draft from the context so far, accept the draft's common prefix with the stream, commit the
accepted tokens plus one. No model runs, so every depth and policy can be scored from one recorded run, also a
run without speculation (``tools/flash_next/rental/lookup_probe.py`` keeps the ids: ``probe.json``). What a
replay cannot give: the acceptance of the MTP layer (that needs the layer), and time.

Drafters (``--policies``), all over the request's own context (prompt + output so far):
  earliest  the longest suffix n-gram (``--max-ngram`` down to ``--min-ngram`` tokens) that occurs earlier, at its
            EARLIEST occurrence, up to k following tokens: vLLM's ``NgramProposer`` search, which the ``ngram``
            method and the lookup beside MTP use today (``test_lookup_replay.py`` checks it against vLLM's kernel)
  recent    the same n-gram at its MOST RECENT earlier occurrence (an edited file's latest copy)
  follow    ``recent``, and after a draft that was accepted in full and whose bonus token also continued the
            source, the next draft reads on from the same source position without a search

Per (kind of prompt, policy, depth k): tokens per step (``1 + accepted per step``), the share of steps with a
draft, and for every draft position j the share of all steps that accepted at least j tokens (the server's own
"acceptance rate per position" counts the same way: every step of an n-gram server carries a draft, the empty
placeholder included), and the same among the steps that had a draft. Also the rows a verification step
carries at 1, 8 and 32 streams: ``streams x (1 + k)`` in the MTP shapes of the lookup + MTP hybrid, and for
the n-gram server, whose step is as wide as its longest draft (an empty draft is the 1-token placeholder),
the expected width if the streams draft independently of each other (an assumption: streams of one workload
reach their copy spans together).

Usage:
  lookup_replay.py PROBE.json [PROBE.json ...] [--kinds edit,editf,editl,prose] [--depths 3,4,5,8,12,16]
                   [--policies earliest,recent,follow] [--min-ngram 3] [--max-ngram 8] [--out REPLAY.json]
Each prompt is replayed once: its longest recorded stream over all rounds and files. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys

POLICIES = ("earliest", "recent", "follow")


class IndexDrafter:
    """An n-gram index of the context, extended as tokens are committed: for each n in ``min_n..max_n`` the end
    positions of every n-gram. ``propose`` is O(max_n) per step (the search of vLLM's proposer is O(context))."""

    def __init__(self, min_n: int, max_n: int, k: int, policy: str = "earliest") -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r} (one of {', '.join(POLICIES)})")
        if not 1 <= min_n <= max_n or k < 1:
            raise ValueError(f"need 1 <= min_n <= max_n and k >= 1, got {min_n}, {max_n}, {k}")
        self.min_n, self.max_n, self.k, self.policy = min_n, max_n, k, policy
        # n -> {n-gram: [first, last but one, last end position]} (an n-gram's continuation starts at its end
        # position; the last but one is None while the n-gram occurred once)
        self.tables: dict[int, dict[tuple, list]] = {n: {} for n in range(min_n, max_n + 1)}
        self.tokens: list[int] = []
        self.cursor: int | None = None  # follow: the source position of the next token of a running copy

    def extend(self, tokens) -> None:
        for token in tokens:
            self.tokens.append(int(token))
            end = len(self.tokens)
            for n, table in self.tables.items():
                if end >= n:
                    key = tuple(self.tokens[end - n:end])
                    entry = table.get(key)
                    if entry is None:
                        table[key] = [end, None, end]
                    else:
                        entry[1], entry[2] = entry[2], end

    def propose(self) -> tuple[list[int], int | None]:
        """``(draft, source position)``: up to ``k`` tokens, or ``([], None)`` without a match."""
        tokens, end = self.tokens, len(self.tokens)
        if self.policy == "follow" and self.cursor is not None and self.cursor < end:
            return tokens[self.cursor:self.cursor + self.k], self.cursor
        for n in range(self.max_n, self.min_n - 1, -1):
            if end < n + 1:
                continue
            entry = self.tables[n].get(tuple(tokens[end - n:end]))
            if entry is None:
                continue
            # the suffix itself is the entry's last occurrence (it ends at ``end``): an EARLIER occurrence is the
            # first one when the n-gram occurred before, and the last but one is the most recent of those
            first, previous, _last = entry
            source = first if self.policy == "earliest" else previous
            if source is None or source >= end:
                continue
            return tokens[source:source + self.k], source
        return [], None

    def advance(self, accepted: int, drafted: int, source: int | None, committed) -> None:
        """Commit ``committed`` (the accepted drafts and the target's own token) after a step."""
        nxt = None
        if self.policy == "follow" and source is not None and drafted and accepted == drafted:
            bonus_at = source + accepted
            if bonus_at < len(self.tokens) and self.tokens[bonus_at] == committed[-1]:
                nxt = bonus_at + 1
        self.extend(committed)
        self.cursor = nxt


def replay(prompt, output, min_n: int, max_n: int, k: int, policy: str) -> list[tuple[int, int]]:
    """``[(tokens drafted, tokens accepted)]`` per decode step of ``output`` after ``prompt`` under one drafter.
    A step commits its accepted drafts plus one token; no draft reaches past the last output token."""
    drafter = IndexDrafter(min_n, max_n, k, policy)
    drafter.extend(prompt)
    steps, position, total = [], 0, len(output)
    while position < total:
        draft, source = drafter.propose()
        draft = draft[:max(0, total - position - 1)]
        accepted = 0
        while accepted < len(draft) and draft[accepted] == output[position + accepted]:
            accepted += 1
        drafter.advance(accepted, len(draft), source, output[position:position + accepted + 1])
        position += accepted + 1
        steps.append((len(draft), accepted))
    return steps


ROW_STREAMS = (1, 8, 32)


def expected_ngram_width(steps: list[tuple[int, int]], k: int, streams: int) -> float | None:
    """The expected token width of an n-gram server's verification step with ``streams`` requests: one plus the
    longest draft among them, an empty draft counting as the 1-token placeholder; the draft lengths are drawn
    independently from the lengths of ``steps``."""
    total = len(steps)
    if not total:
        return None
    counts = [0] * (k + 1)  # by draft length 1..k (0 and 1 both run as one draft row)
    for drafted, _accepted in steps:
        counts[max(1, min(drafted, k))] += 1
    expected, below = 0.0, 0.0
    for length in range(1, k + 1):
        upto = below + counts[length] / total
        expected += (1 + length) * (upto**streams - below**streams)
        below = upto
    return expected


def summarize(steps: list[tuple[int, int]], k: int) -> dict:
    """The figures of one drafter over the steps of all its streams."""
    total = len(steps)
    hits = [(drafted, accepted) for drafted, accepted in steps if drafted > 0]
    accepted_all = sum(accepted for _drafted, accepted in steps)
    out = {"steps": total, "tokens": total + accepted_all, "hit_steps": len(hits),
           "tokens_per_step": (total + accepted_all) / total if total else None,
           "hit_share": len(hits) / total if total else None,
           "accepted_per_hit": (sum(a for _d, a in hits) / len(hits)) if hits else None,
           # position j (1-based): the share of ALL steps that accepted at least j tokens, and among the hits
           "position_rates": [sum(1 for _d, a in steps if a >= j) / total if total else None for j in range(1, k + 1)],
           "position_rates_on_hits": [sum(1 for _d, a in hits if a >= j) / len(hits) if hits else None
                                      for j in range(1, k + 1)],
           "accepted_histogram": [sum(1 for _d, a in steps if a == j) for j in range(0, k + 1)],
           "drafted_histogram": [sum(1 for d, _a in steps if d == j) for j in range(0, k + 1)],
           # rows of one verification step: the MTP shapes (every row 1 + k wide), and the n-gram server
           "rows_per_step_mtp_shapes": {str(n): n * (1 + k) for n in ROW_STREAMS},
           "rows_per_step_ngram": {str(n): (n * expected_ngram_width(steps, k, n)) if total else None
                                   for n in ROW_STREAMS}}
    return out


def streams_of(probes: list[dict], kinds: list[str]) -> dict[str, list[tuple[list[int], list[int]]]]:
    """``{kind: [(prompt ids, output ids)]}``: every prompt once, with its longest recorded stream over all rounds
    of all probe records (the rounds of one prompt are the same greedy answer at different lengths or modes)."""
    best: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    for probe in probes:
        prompts = probe.get("prompts") or {}
        for rnd in probe.get("rounds", []):
            kind = rnd.get("kind")
            if kind not in kinds:
                continue
            for sha, ids in zip(rnd.get("prompt_sha") or [], rnd.get("token_ids") or []):
                prompt = prompts.get(sha)
                if not prompt or not ids:
                    continue
                key = (kind, sha)
                if key not in best or len(ids) > len(best[key][1]):
                    best[key] = (prompt, ids)
    out: dict[str, list] = {kind: [] for kind in kinds}
    for (kind, _sha), pair in sorted(best.items()):
        out[kind].append(pair)
    return out


def run(probes: list[dict], kinds: list[str], depths: list[int], policies: list[str], min_n: int,
        max_n: int) -> dict:
    record = {"min_ngram": min_n, "max_ngram": max_n, "kinds": {}}
    for kind, streams in streams_of(probes, kinds).items():
        if not streams:
            continue
        entry = {"streams": len(streams), "prompt_tokens": sorted(len(p) for p, _o in streams),
                 "output_tokens": sorted(len(o) for _p, o in streams), "drafters": {}}
        for policy in policies:
            for k in depths:
                steps = []
                for prompt, output in streams:
                    steps += replay(prompt, output, min_n, max_n, k, policy)
                entry["drafters"]["%s-k%d" % (policy, k)] = dict(summarize(steps, k), policy=policy, k=k)
        record["kinds"][kind] = entry
    return record


def table(record: dict) -> list[str]:
    lines = []
    for kind, entry in record["kinds"].items():
        outs = entry["output_tokens"]
        lines.append("%s: %d streams, %d output tokens (%d to %d per stream), prompts %d to %d tokens; n-gram %d..%d" % (
            kind, entry["streams"], sum(outs), outs[0], outs[-1], entry["prompt_tokens"][0],
            entry["prompt_tokens"][-1], record["min_ngram"], record["max_ngram"]))
        for name, d in entry["drafters"].items():
            rates = " ".join("%.2f" % r for r in d["position_rates"])
            lines.append("  %-14s tokens/step %5.2f  draft on %3.0f%% of steps, %.2f accepted per draft  "
                         "rows/step at 32 streams %d (n-gram server %.0f)  accepted>=j over all steps, j=1..%d: %s" % (
                             name, d["tokens_per_step"], 100.0 * d["hit_share"], d["accepted_per_hit"] or 0.0,
                             d["rows_per_step_mtp_shapes"]["32"], d["rows_per_step_ngram"]["32"], d["k"], rates))
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probes", nargs="+", metavar="PROBE.json")
    ap.add_argument("--kinds", default="edit,editf,editl,prose")
    ap.add_argument("--depths", default="3,4,5,8,12,16")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--min-ngram", type=int, default=3)
    ap.add_argument("--max-ngram", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    probes = []
    for path in args.probes:
        with open(path, encoding="utf-8") as f:
            probes.append(json.load(f))
    policies = [p for p in args.policies.split(",") if p]
    for policy in policies:
        if policy not in POLICIES:
            ap.error("unknown policy %r (of: %s)" % (policy, ", ".join(POLICIES)))
    record = run(probes, [k for k in args.kinds.split(",") if k], [int(d) for d in args.depths.split(",") if d],
                 policies, args.min_ngram, args.max_ngram)
    if not record["kinds"]:
        print("lookup_replay: no stream with token ids in %s (probes of lookup_probe.py with return_token_ids)" %
              ", ".join(args.probes), file=sys.stderr)
        return 1
    print("\n".join(table(record)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
