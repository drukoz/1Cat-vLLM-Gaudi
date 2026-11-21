# SPDX-License-Identifier: Apache-2.0
"""Simulator cost of the fused hyper-connection kernels at the real geometry, per row count.

Runs inside the tpcsim container (docs/design/flash-next/tpcsim.md), from the worktree root:

    tools/flash_next/tpcsim.sh 'python3 -m tools.flash_next.hc_kernels.sim_cost --build /tmp/hcb \
        --kinds gate_mix,combine_norm --rows 4,32,128'

It builds csrc/hc_fused with its own CMake project (the shipped flags), links the simulator host program of
tests/standalone/hc_fused, runs one case per (kind, rows) at H 2560 / 4 streams / low rank 320 on ONE simulated
TPC, checks the outputs against the oracle (tests/unit_tests/hc_fused/hc_case.compare) and prints the
simulator's instruction count, the index-space size, and instructions per index-space point. The simulator counts
instruction words including the nop words tpc-clang inserts to wait out latency, so the count is a cycle count
of the kernel on one TPC without memory stalls (tpcsim.md). A card estimate: points are split over the TPCs the
graph compiler gives the node (24 on Gaudi2), each runs ceil(points / tpcs) points at about 1.7 GHz (INFERRED from
the GDN decode kernel: 71.6K simulator instructions per point = 42 us per point in a card trace).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from tests.unit_tests.hc_fused import hc_case, sim_build

CLOCK_GHZ = 1.7
TPCS = 24


def build(root: Path, csrc: Path | None = None) -> Path:
    """libhc_fused_gaudi2.so and sim_host under ``root`` (reused when both exist)."""
    csrc = csrc or sim_build.CSRC
    lib = root / "lib"
    host = root / "sim_host"
    if (lib / sim_build.LIBRARY).is_file() and host.is_file():
        return root
    root.mkdir(parents=True, exist_ok=True)
    sim_build.run(["cmake", "-S", csrc, "-B", lib, f"-DTPC_INCLUDE_DIR={sim_build.TOOLS_INCLUDE}",
                   f"-DHABANA_INCLUDE_DIR={sim_build.host_include()}"], timeout=600)
    proc = sim_build.run(["cmake", "--build", lib, "--target", "hc_fused_gaudi2", "--", "VERBOSE=1"], timeout=3600)
    (root / "build.log").write_text(proc.stdout + proc.stderr)
    common = ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", f"-I{csrc}", f"-I{sim_build.TOOLS_INCLUDE}",
              f"-I{sim_build.host_include()}"]
    sim_build.run(common + [sim_build.STANDALONE / "sim_host.cpp", f"-I{sim_build.REPO_ROOT / 'tools'}", f"-L{lib}",
                            f"-Wl,-rpath,{lib}", "-lhc_fused_gaudi2", "-L/usr/lib/habanatools",
                            "-Wl,-rpath,/usr/lib/habanatools", "-ltpc_tests_core_ext", "-o", host], timeout=1800)
    return root


def run_case(root: Path, case: hc_case.HcCase, work: Path, shuffle: bool, guid: str | None = None,
             reference_root: Path | None = None) -> dict:
    case_dir = work / case.name
    inputs = hc_case.make_inputs(case)
    hc_case.write_inputs(case, inputs, case_dir)
    if guid:
        manifest = case_dir / "case.txt"
        lines = manifest.read_text().splitlines()
        manifest.write_text("\n".join(f"guid {guid}" if line.startswith("guid ") else line for line in lines) + "\n")
    args = [root / "sim_host", case_dir]
    if not shuffle:
        args += ["--no-mapping-test", "--no-shuffle"]
    t0 = time.time()
    proc = sim_build.run(args, timeout=7200)
    wall = time.time() - t0
    count = int(re.search(r"instructions executed: (\d+)", proc.stdout).group(1))
    geometry = re.search(r"index space: rank \d+ \{([^}]*)\}", proc.stdout).group(1)
    points = math.prod(int(v) for v in geometry.split(",") if v.strip())
    got = hc_case.read_outputs(case, case_dir)
    problems = hc_case.compare(case, inputs, got)
    per_point = count / points
    per_tpc = math.ceil(points / TPCS)
    result = dict(case=case.name, kind=case.kind, rows=case.rows, index_space=geometry.strip(), points=points,
                  instructions=count, per_point=per_point, est_us_24tpc=per_tpc * per_point / (CLOCK_GHZ * 1e3),
                  ok=not problems, problems=problems[:3], sim_wall_s=round(wall, 1))
    if reference_root is not None:
        # The same inputs through the shipped kernel of another build: the fraction of bit-equal outputs.
        ref_dir = work / (case.name + "_ref")
        hc_case.write_inputs(case, inputs, ref_dir)
        sim_build.run([reference_root / "sim_host", ref_dir, "--no-mapping-test", "--no-shuffle"], timeout=7200)
        ref = hc_case.read_outputs(case, ref_dir)
        exact = {}
        for name in got:
            a = got[name].view(torch.int16)
            b = ref[name].view(torch.int16)
            exact[name] = float((a == b).float().mean())
        result["exact_vs_reference"] = exact
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=Path("/tmp/hc_sim_cost"))
    ap.add_argument("--csrc", type=Path, default=None, help="kernel source tree (default: csrc/hc_fused)")
    ap.add_argument("--kinds", default="gate_mix,combine_norm,silu_split")
    ap.add_argument("--rows", default="4,32,128")
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--lora", type=int, default=320)
    ap.add_argument("--shuffle", action="store_true", help="index-space mapping test and shuffle on (slower)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--guid", default=None, help="run this GUID instead of the case kind's shipped GUID")
    ap.add_argument("--reference-build", type=Path, default=None,
                    help="a build folder whose shipped kernel gets the same inputs: report bit-equal fractions")
    ap.add_argument("--sim-cases", action="store_true",
                    help="run hc_case.SIM_CASES of the kinds (tiny, stress, edge, rounding) instead of --rows")
    a = ap.parse_args()
    root = build(a.build, a.csrc)
    reference = build(a.reference_build) if a.reference_build else None
    kinds = a.kinds.split(",")
    if a.sim_cases:
        cases = [case for case in hc_case.SIM_CASES if case.kind in kinds]
    else:
        cases = [
            hc_case.HcCase(name=f"cost_{kind}_r{rows}", kind=kind, hidden=a.hidden, rows=rows, lora=a.lora,
                           seed=900 + rows) for kind in kinds for rows in (int(v) for v in a.rows.split(","))
        ]
    results = []
    for case in cases:
        r = run_case(root, case, a.build / "cases", a.shuffle, a.guid, reference)
        results.append(r)
        exact = r.get("exact_vs_reference")
        exact_text = (" bit-equal to the reference build: " +
                      ", ".join(f"{k} {100 * v:.3f} %" for k, v in exact.items())) if exact else ""
        print(f"{case.name:34s} rows {case.rows:4d}: index space {{{r['index_space']}}} = {r['points']} points, "
              f"{r['instructions']} instructions, {r['per_point']:.0f} per point, est. "
              f"{r['est_us_24tpc']:.1f} us on 24 TPCs at {CLOCK_GHZ} GHz; oracle "
              f"{'ok' if r['ok'] else 'FAIL ' + str(r['problems'])} (sim {r['sim_wall_s']} s){exact_text}",
              flush=True)
    if a.json:
        a.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
