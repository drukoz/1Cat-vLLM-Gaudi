# SPDX-License-Identifier: Apache-2.0
"""Choose the MTP draft count k from measured step cost and acceptance (idea-sweep I43).

    ms per token(k) = ms per step(k) / E(k),   E(k) = 1 + r_1 + ... + r_k

``r_j`` is vLLM's acceptance rate of draft position j (accepted at j over drafts, the spec_decode counters'
``position_rates``), so E(k) is vLLM's acceptance length: tokens per verification step per request, the bonus
token included. ``ms per step(k)`` is one engine step at that k: the verification forward of ``1 + k`` rows per
request plus k draft passes (the probe's median gap between streamed chunks).

From the runs the card measured (one or more k at one stream count) this module

* fits ``ms per step(k) = base + k * per_k`` by least squares (one point: no slope, the point is used as is
  and other k are not predicted), the per-draft cost ``per_k`` being one draft pass plus the verification
  rows it adds;
* extends the measured position rates to larger k with the last measured conditional rate
  ``r_j / r_{j-1}`` (capped at 1), so ``E`` of an unmeasured k is an extrapolation, labelled as such;
* prints the table and the k with the lowest predicted ms per token (measured k win over predicted ones only
  by their own numbers: the table says which is which).

It also gives the closed form of the target model (reports/idea-sweep/target-model.md, section 6) for a fixed
per-position acceptance a: ``E(k) = (1 - a^(k+1)) / (1 - a)``.

Standard library only. ``python -m tools.flash_next.mtp_k_choice --run 3=scaling_k3.json --run 4=scaling_k4.json
--streams 32 [--k-max 6]``: each scaling.json is what reports/mtp-card-ready/job/spec_scaling.py writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def closed_form_acceptance_length(a: float, k: int) -> float:
    """E(k) for a fixed per-position (conditional) acceptance a: 1 + a + a^2 + ... + a^k."""
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"acceptance must be in [0, 1], got {a}")
    return float(k + 1) if a == 1.0 else (1.0 - a**(k + 1)) / (1.0 - a)


def extend_rates(rates: list[float], k: int) -> tuple[list[float], int]:
    """Position rates r_1..r_k: the measured ones, then the last conditional rate r_j / r_{j-1} (capped at 1)
    applied again; returns the rates and how many of them were measured."""
    measured = [float(r) for r in rates if r is not None]
    if not measured:
        raise ValueError("no measured position rate")
    out = list(measured[:k])
    if len(out) < k:
        if len(measured) >= 2 and measured[-2] > 0:
            ratio = min(1.0, measured[-1] / measured[-2])
        else:
            ratio = min(1.0, measured[-1])  # one rate: take it as the conditional rate as well
        while len(out) < k:
            out.append(out[-1] * ratio)
    return out, min(len(measured), k)


def acceptance_length(rates: list[float], k: int) -> float:
    """E(k) = 1 + r_1 + ... + r_k (vLLM's acceptance length), extrapolated beyond the measured positions."""
    extended, _ = extend_rates(rates, k)
    return 1.0 + sum(extended)


def fit_step_cost(points: dict[int, float]) -> tuple[float, float | None]:
    """(base, per_k) of ms per step(k) = base + k * per_k, least squares over the measured k; per_k is None
    with a single point (no slope can be measured from one k)."""
    if not points:
        raise ValueError("no measured step cost")
    ks = sorted(points)
    if len(ks) == 1:
        return float(points[ks[0]]), None
    n = len(ks)
    mean_k = sum(ks) / n
    mean_t = sum(points[k] for k in ks) / n
    var = sum((k - mean_k)**2 for k in ks)
    per_k = sum((k - mean_k) * (points[k] - mean_t) for k in ks) / var
    return mean_t - per_k * mean_k, per_k


def choose_k(step_ms: dict[int, float], rates_by_k: dict[int, list[float]], k_max: int = 5,
             baseline_ms: float | None = None) -> dict:
    """The table of ms per token over k = 1..k_max (and k = 0 when ``baseline_ms``, the server without
    speculation, is given) and the best k. ``step_ms[k]`` and ``rates_by_k[k]`` are measured at the same
    stream count; the position rates of the run with the largest k are used for E (the most positions)."""
    base, per_k = fit_step_cost(step_ms)
    rates = rates_by_k[max(rates_by_k)]
    rows = []
    if baseline_ms is not None:
        rows.append({"k": 0, "step_ms": baseline_ms, "step_measured": True, "E": 1.0, "E_measured_positions": 0,
                     "ms_per_token": baseline_ms})
    for k in range(1, k_max + 1):
        if k in step_ms:
            step, measured = float(step_ms[k]), True
        elif per_k is not None:
            step, measured = base + k * per_k, False
        else:
            continue
        _, measured_positions = extend_rates(rates, k)
        e = acceptance_length(rates, k)
        rows.append({"k": k, "step_ms": step, "step_measured": measured, "E": e,
                     "E_measured_positions": measured_positions, "ms_per_token": step / e})
    best = min(rows, key=lambda row: row["ms_per_token"]) if rows else None
    return {"base_ms": base, "per_k_ms": per_k, "rows": rows, "best": best}


def read_run(path: Path, streams: int) -> tuple[float, list[float], float]:
    """(ms per step, position rates, effective ms per token) of the measured round at ``streams`` streams in a
    spec_scaling.py scaling.json (the last repeat). ms per step is the median chunk gap (steady ms per token
    times tokens per chunk)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rounds = [r for r in data.get("levels", []) if r.get("streams") == streams and r.get("ok")]
    if not rounds:
        raise ValueError(f"{path}: no measured round at {streams} streams")
    r = rounds[-1]
    step = float(r["steady_ms_per_token_p50"]) * float(r["tokens_per_chunk_p50"])
    spec = r.get("spec") or {}
    rates = [x for x in (spec.get("position_rates") or []) if x is not None]
    return step, rates, float(r["ms_per_token_p50"])


def format_table(result: dict) -> str:
    lines = ["k  ms/step        E (measured positions)  ms/token"]
    for row in result["rows"]:
        lines.append("%-2d %7.2f %-6s %5.2f (%d of %d)            %7.2f" % (
            row["k"], row["step_ms"], "meas." if row["step_measured"] else "pred.", row["E"],
            row["E_measured_positions"], row["k"], row["ms_per_token"]))
    per_k = result["per_k_ms"]
    lines.append("step cost fit: %.2f ms + k x %s" % (result["base_ms"], "%.2f ms" % per_k if per_k is not None
                                                                              else "(one k measured: no slope)"))
    best = result["best"]
    if best is not None:
        lines.append("best k = %d: %.2f ms per token" % (best["k"], best["ms_per_token"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.mtp_k_choice", description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="K=SCALING_JSON",
                        help="a spec_scaling.py result measured at k drafted tokens (repeatable)")
    parser.add_argument("--streams", type=int, default=32)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--baseline-ms", type=float, default=None,
                        help="ms per token of the same server without speculation at the same stream count")
    args = parser.parse_args(argv)
    step_ms: dict[int, float] = {}
    rates_by_k: dict[int, list[float]] = {}
    for item in args.run:
        k_text, path = item.split("=", 1)
        k = int(k_text)
        step, rates, _ = read_run(Path(path), args.streams)
        step_ms[k] = step
        rates_by_k[k] = rates
    result = choose_k(step_ms, rates_by_k, args.k_max, args.baseline_ms)
    print(format_table(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
