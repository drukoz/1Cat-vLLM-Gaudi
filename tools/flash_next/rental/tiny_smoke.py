# SPDX-License-Identifier: Apache-2.0
"""The tiny smoke test: can a Flash-Next server start and decode on this node at all (RUK-2709).

The first thing to run on a new rental node, before the checkpoint download and the gates. It builds
the tiny checkpoint and its CPU reference (`make_tiny_checkpoint.py`, with the GDN layers at head dim
128, the only one the HPU fused kernel accepts), then, once per GDN mode, starts the tiny gate's
`vllm serve` command with the environment every rental server gets (`serve.server_environment`), waits
for `/health`, compares the served tokens with the reference (`gates.tiny_compare`) and stops the
server, so the cards are free again.

Modes: `pytorch` turns the fused GDN kernel off (the flags gate G1b sets when it fails); `fused` turns
the decode kernel on and keeps the verify op off, and fails when the server log says a layer fell back
to the torch decode path; `hc` is `fused` plus the fused hyper-connection kernels (`VLLM_HPU_HC_FUSED=1`),
and also fails when the log says a hyper-connection ran the torch path ("HC fused ... is off for", or the
startup line "HC fused ... has no HPU kernel"). `hc` is not in the default list: `--gdn pytorch,fused,hc`
or `--gdn hc` selects it. Each mode writes `<out>/<mode>/server0.log` and `details.json`;
`<out>/summary.json` holds one entry per mode. The exit code is 0 only when every requested mode passed.
A mode whose tokens match also records `decode_ms_per_token` (printed on a line of its own after the
PASSED line): the shortest reference prompt generated with 16 tokens and with 1 token, best of three
each, the difference divided by 15. It is recorded only, for comparing two runs of one mode (for
example with and without an `--env` pair); it never changes a result.
A server that exits before it is healthy is a failure with its exit code and the last error lines of
its log (`serve.unhealthy_reason`, `serve.error_lines`: the run's gates report it the same way).

By default the servers skip the bucket warmup (`VLLM_SKIP_WARMUP=true`): the warmup alone takes about
15 minutes per mode on the tiny model, while the four short prompts need only a handful of graphs,
which then compile on first use. `--full-warmup` keeps the warmup (it measures the real one), and
`--env KEY=VALUE` (repeatable) adds a server environment variable applied last, after the mode's own
flags, so it can override anything. The health wait defaults to 1800 s.

`--recipe-cache DIR` gives the compiled-recipe cache to the servers that may carry it
(`serve.recipe_cache_env`): the `pytorch` mode's, and every server once the combined kernel database
is installed and not switched off (`FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0`). The `fused` and `hc` modes
run without it otherwise: their kernel databases must be registered in `GC_KERNEL_PATH`, and the Habana
bridge accepts only one kernel library there next to a recipe cache. The cache lives in the per-bundle
subfolder the run uses, `DIR/<first 12 characters of the plugin commit>` (`unknown` without a commit).

Runs on Python 3.10 with the standard library only, from the plugin root of the bundle:
`python -m tools.flash_next.rental.tiny_smoke --out DIR [--ckpt DIR] [--gdn pytorch,fused[,hc]] [--port N]
[--health-timeout S] [--full-warmup] [--recipe-cache DIR] [--recipe-cache-mb N] [--env KEY=VALUE]`.
`--ckpt` names a folder the checkpoint is built in once and reused from then on (default: `<out>/ckpt`,
built every time); a folder built with other settings (another GDN head dim, generator or fixture: the
`build` block of its reference.json) is rebuilt. tiny_smoke.sh starts this inside the container.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tools.flash_next.rental import gates, serve
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental.schema import GateStatus

GDN_MODES = {
    "pytorch": dict(gates.GDN_FALLBACK_ENV),
    "fused": {gates.GDN_MODE_ENV: str(gates.GDN_MODE_FUSED), gates.GDN_VERIFY_ENV: str(gates.GDN_VERIFY_OFF)},
}
# `hc`: the fused mode plus the fused hyper-connection kernels. Not a default mode: the kernels have not
# run in a server yet, and the two default modes are the ones the run's gates depend on.
GDN_MODES["hc"] = dict(GDN_MODES["fused"], **{gates.HC_FUSED_ENV: "1"})
DEFAULT_MODES = "pytorch,fused"
DECODE_KERNEL_MODES = ("fused", "hc")  # fail when a GDN layer ran the torch decode path
HC_KERNEL_MODES = ("hc", )  # fail when a hyper-connection ran the torch path
HEALTH_TIMEOUT_S = 1800.0
DECODE_TIMING_REPEATS = 3
DECODE_TIMING_TIMEOUT_S = 120.0  # per request; the graphs are compiled by the comparison that ran before
MAX_LINES = serve.MAX_ERROR_LINES
DECODE_FALLBACK = "GDN core decode"

# The two commands are the tiny gate's own, so the smoke test starts what the run will start.
checkpoint_command = orchestrator.tiny_checkpoint_command
serve_command = orchestrator.tiny_serve_command


# One definition for the smoke test, the tiny gate and the variants (the orchestrator reports a server
# that exited the same way).
error_lines = serve.error_lines


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_checkpoint(ckpt: Path) -> str | None:
    """Build the tiny checkpoint in `ckpt` unless it is there with the build settings of this bundle
    (the GDN head dim, the generator, the fixture: `orchestrator.ensure_tiny_checkpoint`); the failure
    reason or None."""
    return orchestrator.ensure_tiny_checkpoint(ckpt, checkpoint_command(ckpt),
                                               say=lambda text: print("tiny smoke: %s" % text, flush=True))


def decode_ms_per_token(base_url: str, reference_path: Path, post=None, clock=time.monotonic):
    """The decode time per token of the served tiny model in ms, or None when it cannot be measured.

    The shortest reference prompt is generated with gates.TINY_MAX_TOKENS tokens and with 1 token,
    DECODE_TIMING_REPEATS times each; the best long time minus the best short time, divided by the
    extra tokens. The prefill and the HTTP round trip are in both and cancel. The comparison just ran
    this prompt, so no graph compiles here. Recorded only: nothing depends on the value. Never raises."""
    post = post or gates._post_json
    try:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        prompt = min((p["prompt_ids"] for p in reference["prompts"]), key=len)
        best = {}
        tokens = 0
        for max_tokens in (gates.TINY_MAX_TOKENS, 1):
            for _ in range(DECODE_TIMING_REPEATS):
                started = clock()
                resp = post(base_url, "/v1/completions", {"model": "tiny", "prompt": prompt,
                                                          "max_tokens": max_tokens, "temperature": 0,
                                                          "return_token_ids": True}, DECODE_TIMING_TIMEOUT_S)
                seconds = clock() - started
                best[max_tokens] = min(best.get(max_tokens, seconds), seconds)
                if max_tokens != 1:
                    tokens = len((resp.get("choices") or [{}])[0].get("token_ids") or [])
        if tokens < 2:
            return None
        return max(0.0, (best[gates.TINY_MAX_TOKENS] - best[1]) / (tokens - 1) * 1000.0)
    except Exception:  # noqa: BLE001 - a measurement, not a check
        return None


def run_mode(mode: str, ckpt: Path, out_dir: Path, port: int, health_timeout_s: float, skip_warmup: bool = True,
             extra_env: dict | None = None, recipe_cache: str | None = None, recipe_cache_mb: int = 8192,
             combined_db: bool = False) -> dict:
    """Serve the tiny checkpoint in one GDN mode, compare it with the reference, stop the server.

    The overrides are the mode's GDN flags, then `VLLM_SKIP_WARMUP=true` when `skip_warmup`, then the
    recipe cache where `serve.recipe_cache_env` allows it for this mode (the fused mode's kernel
    database must stay alone in `GC_KERNEL_PATH`), then `extra_env` applied last so it may override
    any of them. Returns the mode's summary entry; the comparison's own record goes to
    `<out_dir>/details.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server0.log"
    entry: dict = {"mode": mode, "status": None, "reason": None, "startup_seconds": None, "errors": [],
                   "fallback_lines": [], "log": str(log_path),
                   "warmup": "warmup skipped" if skip_warmup else "full warmup", "decode_ms_per_token": None}
    details: dict = {}
    status = GateStatus.FAILED
    if serve.port_in_use(port):
        entry["reason"] = "port %d already in use" % port
    else:
        overrides = dict(GDN_MODES[mode])
        if skip_warmup:
            overrides["VLLM_SKIP_WARMUP"] = "true"
        cache_env, note = serve.recipe_cache_env(recipe_cache, recipe_cache_mb, overrides, combined_db)
        if cache_env:
            try:
                Path(recipe_cache).mkdir(parents=True, exist_ok=True)
            except OSError as e:  # the bridge reports an unusable folder itself
                print("tiny smoke [%s]: cannot create the recipe cache folder %s: %s" % (mode, recipe_cache, e),
                      flush=True)
        overrides.update(cache_env)
        overrides.update(extra_env or {})
        env = serve.server_environment(overrides)
        entry["recipe_cache"] = note
        cache_word = "on" if cache_env else "off"
        print("tiny smoke [%s]: starting the server (recipe cache %s; log: %s)" % (mode, cache_word, log_path),
              flush=True)
        popen = serve.start_server(serve_command(ckpt, port), env, log_path, out_dir)
        try:
            started = time.monotonic()
            healthy = serve.wait_healthy(port, health_timeout_s, popen)
            entry["startup_seconds"] = round(time.monotonic() - started, 1)
            if healthy:
                status, details = gates.tiny_compare("http://127.0.0.1:%d" % port, ckpt / "reference.json")
                entry["reason"] = details.get("reason") or details.get("error")
                # Recorded only, while the server is still up. Also after a failed comparison: a server
                # that decodes the wrong tokens still shows what a decode step costs (None when it died).
                entry["decode_ms_per_token"] = decode_ms_per_token("http://127.0.0.1:%d" % port,
                                                                   ckpt / "reference.json")
            else:
                status, entry["reason"] = serve.unhealthy_reason(popen, entry["startup_seconds"], health_timeout_s)
        finally:
            serve.stop_server(popen)  # also on Ctrl-C: the workers must not keep the cards
        log_text = _read(log_path)
        if status is GateStatus.PASSED and mode in DECODE_KERNEL_MODES:
            fallback = [ln for ln in gates.gdn_fallback_lines(log_text) if DECODE_FALLBACK in ln]
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the fused kernel did not run: %d log line(s) say a layer used the torch "
                                   "decode path" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
        if status is GateStatus.PASSED and mode in HC_KERNEL_MODES:
            fallback = gates.hc_fallback_lines(log_text)
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the fused hyper-connection kernels did not run: %d log line(s) say a layer "
                                   "used the torch path" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
        if status is not GateStatus.PASSED:
            entry["errors"] = error_lines(log_text)
    entry["status"] = status.name.lower()
    _write_json(out_dir / "details.json", dict(details, mode=mode, status=int(status), reason=entry["reason"]))
    return entry


def report_mode(entry: dict) -> None:
    """One result line for the console, and on a failure the error lines and the log to read."""
    mode = entry["mode"]
    if entry["status"] == "passed":
        note = entry.get("recipe_cache", serve.RECIPE_CACHE_NONE)
        cache_word = "off" if note in (serve.RECIPE_CACHE_NONE, serve.RECIPE_CACHE_OFF_REASON) else "on"
        print("tiny smoke [%s]: PASSED (server up in %s s, tokens match the reference, %s, recipe cache %s)" %
              (mode, entry["startup_seconds"], entry.get("warmup", "warmup skipped"), cache_word), flush=True)
        decode = entry.get("decode_ms_per_token")
        if decode is not None:  # its own line: the PASSED line keeps its form
            print("tiny smoke [%s]: decode %.1f ms/token (the shortest prompt, %d tokens against 1, best of %d)" %
                  (mode, decode, gates.TINY_MAX_TOKENS, DECODE_TIMING_REPEATS), flush=True)
        return
    word = "TIMED OUT" if entry["status"] == "timed_out" else "FAILED"
    print("tiny smoke [%s]: %s (%s)" % (mode, word, entry["reason"]), flush=True)
    decode = entry.get("decode_ms_per_token")
    if decode is not None:  # the server answered: its decode step is worth knowing even with wrong tokens
        print("tiny smoke [%s]: decode %.1f ms/token (the shortest prompt, %d tokens against 1, best of %d)" %
              (mode, decode, gates.TINY_MAX_TOKENS, DECODE_TIMING_REPEATS), flush=True)
    for line in entry["errors"] + entry["fallback_lines"]:
        print("    %s" % line, flush=True)
    print("    log: %s" % entry["log"], flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.tiny_smoke",
                                     description="Serve the tiny Flash-Next checkpoint and compare it with its "
                                     "CPU reference, once per GDN mode.")
    parser.add_argument("--out", required=True, help="results folder of this call")
    parser.add_argument("--ckpt", default=None, help="checkpoint folder, built once and reused (default: OUT/ckpt)")
    parser.add_argument("--gdn", default=DEFAULT_MODES,
                        help="comma-separated modes: pytorch, fused, hc (default: %(default)s; hc is fused plus "
                        "the fused hyper-connection kernels, VLLM_HPU_HC_FUSED=1)")
    parser.add_argument("--port", type=int, default=orchestrator.TINY_PORT)
    parser.add_argument("--health-timeout", type=float, default=HEALTH_TIMEOUT_S, metavar="S")
    parser.add_argument("--full-warmup", action="store_true",
                        help="run the real bucket warmup (about 15 minutes per mode): by default the servers "
                        "skip it (VLLM_SKIP_WARMUP=true) and the graphs compile on first use")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE",
                        help="extra server environment variable, repeatable; applied after the mode's own flags")
    parser.add_argument("--recipe-cache", default=None, metavar="DIR",
                        help="compiled-recipe cache folder: given to the servers that may carry it (the pytorch "
                        "mode's, and every server once the combined kernel database is installed), in the "
                        "per-bundle subfolder DIR/<plugin commit, 12 characters>")
    parser.add_argument("--recipe-cache-mb", type=int, default=8192, metavar="N",
                        help="recipe cache size in MB (default: 8192)")
    args = parser.parse_args(argv)
    modes = [m for m in args.gdn.split(",") if m]
    if not modes or any(m not in GDN_MODES for m in modes):
        parser.error("--gdn takes a comma-separated list of: %s" % ", ".join(GDN_MODES))
    extra_env: dict = {}
    for pair in args.env or []:
        if "=" not in pair:
            parser.error("--env takes KEY=VALUE: %r" % pair)
        key, value = pair.split("=", 1)
        extra_env[key] = value

    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt).absolute() if args.ckpt else out / "ckpt"
    try:
        combined_db = orchestrator.combined_kernel_db_installed()
    except Exception:  # noqa: BLE001 - no lookup: the plan of record is the fallback
        combined_db = False
    # the per-bundle subfolder, keyed like the run's cache: a new bundle starts a fresh folder
    recipe_cache = orchestrator.keyed_recipe_cache_dir(args.recipe_cache, orchestrator.bundle_commit())
    summary: dict = {"passed": False, "reason": None, "checkpoint": str(ckpt), "modes": []}
    summary["reason"] = build_checkpoint(ckpt)
    if summary["reason"] is None:
        for mode in modes:
            entry = run_mode(mode, ckpt, out / mode, args.port, args.health_timeout,
                             skip_warmup=not args.full_warmup, extra_env=extra_env,
                             recipe_cache=recipe_cache, recipe_cache_mb=args.recipe_cache_mb,
                             combined_db=combined_db)
            summary["modes"].append(entry)
            report_mode(entry)
        summary["passed"] = all(entry["status"] == "passed" for entry in summary["modes"])
    _write_json(out / "summary.json", summary)
    if summary["passed"]:
        print("tiny smoke: PASSED (%s)" % ", ".join(modes), flush=True)
        return 0
    failed = [entry["mode"] for entry in summary["modes"] if entry["status"] != "passed"]
    print("tiny smoke: FAILED (%s)" % (summary["reason"] or ", ".join(failed)), flush=True)
    print("summary: %s" % (out / "summary.json"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
