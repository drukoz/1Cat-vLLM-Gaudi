#!/usr/bin/env python3
"""G4-1 / I34 / C6 readings from bench_c4.py results (c4_r8.json, c4_r4.json of bench_job.sh). Stdlib only.

    python3 -I decide_g41.py <results dir> [--json out.json]

Per ranks and regime: alpha (one reduction of one [1, 2560] bf16 row) and the cost per MB of payload, for the
all-reduce and for the all-gather + local sum; the row counts at which the gather form is cheaper; the collective
term of a step F_c = n x cost(rows) for a decode step (98 at 32 rows) and a k = 3 MTP step (98 at 128 rows plus
9 at 32 rows); the G4-1 band; the C6 ceiling (alpha8 - 16 us) x collectives; the E1 rule (alpha8 - alpha4 >= 34 us).
"""
import json
import sys
from pathlib import Path

HIDDEN = 2560


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"{path}: {exc!r}")
        return None


def costs(data):
    """{(regime, form): {rows: per_reduction_us}}"""
    out = {}
    for line in (data or {}).get("summary", []):
        if "per_reduction_us" in line:
            out.setdefault((line["regime"], line["form"]), {})[line["rows"]] = line["per_reduction_us"]
    return out


def at(table, rows):
    """Linear interpolation of a cost table at a row count (clamped to the measured range)."""
    pts = sorted(table.items())
    if not pts:
        return None
    if rows <= pts[0][0]:
        return pts[0][1]
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if r0 <= rows <= r1:
            return c0 + (c1 - c0) * (rows - r0) / (r1 - r0)
    return pts[-1][1]


def main(argv):
    root = Path(argv[1])
    result = {}
    tables = {n: costs(load(root / f"c4_r{n}.json")) for n in (8, 4)}
    for n, table in tables.items():
        for (regime, form), pts in sorted(table.items()):
            rows = sorted(pts)
            alpha = pts[rows[0]]
            per_mb = (pts[rows[-1]] - pts[rows[0]]) / ((rows[-1] - rows[0]) * HIDDEN * 2 / 1e6) if len(rows) > 1 else None
            result.setdefault(str(n), {}).setdefault(regime, {})[form] = {
                "alpha_us": alpha, "us_per_mb": per_mb, "points": {str(r): pts[r] for r in rows}}
            print(f"ranks {n} {regime:6s} {form:10s} alpha {alpha:7.1f} us, "
                  f"{per_mb if per_mb is not None else float('nan'):7.1f} us/MB, points "
                  + ", ".join(f"{r}: {pts[r]:.1f}" for r in rows))
    for n, table in tables.items():
        for regime in ("lazy", "replay", "replay_async"):
            ar, ag = table.get((regime, "all_reduce")), table.get((regime, "gather_sum"))
            if not ar or not ag:
                continue
            cheaper = [r for r in sorted(ar) if r in ag and ag[r] < ar[r]]
            decode = 98 * at(ar, 32) / 1e3
            mtp = (98 * at(ar, 128) + 9 * at(ar, 32)) / 1e3
            best_decode = 98 * min(at(ar, 32), at(ag, 32)) / 1e3
            print(f"ranks {n} {regime}: gather cheaper at rows {cheaper or 'none'}; collective term of a 32-row decode "
                  f"step {decode:.2f} ms (best form {best_decode:.2f} ms); of a k=3 MTP step {mtp:.2f} ms")
            result[str(n)][regime]["gather_cheaper_rows"] = cheaper
            result[str(n)][regime]["decode32_ms"] = decode
            result[str(n)][regime]["mtp3_ms"] = mtp
            alone = table.get((regime, "gather_first"))  # the all-gather without the sum: the fused-kernel form
            if alone:
                margin = {r: ar[r] - alone[r] for r in sorted(ar) if r in alone}
                print(f"ranks {n} {regime}: the all-gather alone against the all-reduce, us saved per reduction: "
                      + ", ".join(f"{r} rows: {m:.1f}" for r, m in margin.items()))
                result[str(n)][regime]["gather_alone_saves_us"] = {str(r): m for r, m in margin.items()}
    t8, t4 = tables.get(8, {}), tables.get(4, {})
    for regime in ("replay", "replay_async", "lazy"):
        a8 = t8.get((regime, "all_reduce"))
        a4 = t4.get((regime, "all_reduce"))
        if not a8:
            continue
        alpha8 = at(a8, 32)
        band = "A (<= 20 us)" if alpha8 <= 20 else ("B (20-45 us)" if alpha8 <= 45 else "C (> 45 us)")
        c6 = max(0.0, alpha8 - 16.0) * 111 / 1e3
        line = f"{regime}: alpha8 at 32 rows {alpha8:.1f} us -> band {band}; C6 ceiling (alpha8 - 16 us) x 111 = {c6:.2f} ms"
        if a4:
            alpha4 = at(a4, 32)
            line += f"; alpha4 {alpha4:.1f} us, alpha8 - alpha4 = {alpha8 - alpha4:.1f} us (E1 pays on collectives " \
                    f"only if >= 34 us)"
            result.setdefault("decision", {})[regime] = {"alpha8_32": alpha8, "alpha4_32": alpha4, "band": band,
                                                         "c6_ceiling_ms": c6}
        print(line)
    if len(argv) > 3 and argv[2] == "--json":
        Path(argv[3]).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
