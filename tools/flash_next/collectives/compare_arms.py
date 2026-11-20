#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the arms of a real-model A/B (ab_job.sh): ms per token per level, and the texts stream by stream.

    python3 -I compare_arms.py <base arm folder> <arm folder>... [--json out.json]

Each folder holds the ``scaling.json`` of probe_ab.py. Per arm and level: the median ms per token (the stream's
decode span, compile pauses included), the steady ms per token (the median gap between streamed chunks) and the
total tokens per second. Per arm against the base, per level and round: how many streams produced the same text,
and the share of the text before the first difference (a wrong token id parts at once; an exact bf16 tie broken
another way parts late and rarely). Stdlib only.
"""
import json
import sys
from pathlib import Path


def load(folder):
    path = Path(folder) / "scaling.json"
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - an arm that died has no file
        print(f"{path}: {exc!r}")
        return None


def rounds(data):
    """``{(streams, round index at that level): the round}`` of the measured rounds."""
    out, seen = {}, {}
    for level in (data or {}).get("levels", []):
        n = level.get("streams")
        seen[n] = seen.get(n, 0) + 1
        out[(n, seen[n] - 1)] = level
    return out


def text_report(base, other):
    same, shares, total = 0, [], 0
    for a, b in zip(base.get("texts") or [], other.get("texts") or []):
        if a is None or b is None:
            continue
        total += 1
        same += int(a == b)
        first = next((i for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb), min(len(a), len(b)))
        shares.append(first / max(1, max(len(a), len(b))))
    shares.sort()
    return {"streams": total, "identical": same, "common_prefix_min": shares[0] if shares else None,
            "common_prefix_median": shares[len(shares) // 2] if shares else None}


def main(argv):
    args = [a for a in argv[1:] if a != "--json"]
    out_path = argv[argv.index("--json") + 1] if "--json" in argv else None
    if out_path:
        args.remove(out_path)
    if len(args) < 1:
        print(__doc__)
        return 2
    arms = {Path(a).name: rounds(load(a)) for a in args}
    names = list(arms)
    result = {"arms": {}, "texts": {}}
    for name in names:
        for (n, rep), level in sorted(arms[name].items()):
            if not level.get("ok"):
                print(f"{name:40s} N={n:<3d} round {rep}: no stream finished ({level.get('errors')} errors)")
                continue
            line = {k: level.get(k) for k in ("ms_per_token_p50", "steady_ms_per_token_p50", "aggregate_tok_s",
                                              "ms_per_token_min", "ms_per_token_max", "pause_s_max", "ok", "errors")}
            result["arms"].setdefault(name, {})[f"{n}:{rep}"] = line
            print(f"{name:40s} N={n:<3d} round {rep}: ms/token p50 {line['ms_per_token_p50']:.2f}, steady "
                  f"{line['steady_ms_per_token_p50']:.2f}, total {line['aggregate_tok_s'] or 0:.1f} tok/s, "
                  f"longest pause {line['pause_s_max']:.2f} s, {line['ok']} streams ok")
    base = names[0]
    for name in names[1:]:
        for key in sorted(set(arms[base]) & set(arms[name])):
            rep = text_report(arms[base][key], arms[name][key])
            result["texts"].setdefault(name, {})[f"{key[0]}:{key[1]}"] = rep
            print(f"texts {name} against {base} N={key[0]} round {key[1]}: {rep['identical']}/{rep['streams']} identical; "
                  f"common prefix min {rep['common_prefix_min']}, median {rep['common_prefix_median']}")
    # the same arm twice (two measured rounds of a level) must agree with itself: the noise floor of the comparison
    for name in names:
        for (n, rep), level in sorted(arms[name].items()):
            if rep > 0 and (n, 0) in arms[name]:
                own = text_report(arms[name][(n, 0)], level)
                print(f"texts {name} round {rep} against its round 0, N={n}: {own['identical']}/{own['streams']} identical")
                result["texts"].setdefault(name + ":self", {})[f"{n}:{rep}"] = own
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
