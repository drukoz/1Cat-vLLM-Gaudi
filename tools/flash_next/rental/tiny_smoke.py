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
or `--gdn hc` selects it. `moe` is `fused` plus the MoE glue kernels (`VLLM_HPU_MOE_GLUE=1`), and also
fails when the log says a MoE block ran the torch path ("MoE glue ... is off for", or the startup line
"MoE glue ... has no HPU kernel"). It is not in the default list either: `--gdn moe` or
`--gdn pytorch,fused,hc,moe` selects it. Each mode writes `<out>/<mode>/server0.log` and `details.json`;
`<out>/summary.json` holds one entry per mode. The exit code is 0 only when every requested mode passed.
A mode whose tokens match also records `decode_ms_per_token` (printed on a line of its own after the
PASSED line): the shortest reference prompt generated with 16 tokens and with 1 token, best of three
each, the difference divided by 15. It is recorded only, for comparing two runs of one mode (for
example with and without an `--env` pair); it never changes a result.
A server that exits before it is healthy is a failure with its exit code and the last error lines of
its log (`serve.unhealthy_reason`, `serve.error_lines`: the run's gates report it the same way).

`--spec K` adds after each mode its speculative mode `<mode>+mtpK`: the same server with the MTP layer drafting
K tokens (`--speculative-config {"method": "mtp", "num_speculative_tokens": K}`), its tokens compared with the
reference and token for token with the mode's own server without speculation (`spec_vs_plain`), the acceptance
length printed from the speculative counters of `/metrics` (`spec_probe`). Its rungs (`SPEC_RUNGS`) are tried in
order until one passes: the fused GDN verification op (`VLLM_HPU_GDN_CORE_VERIFY=1`; a log line saying a layer
ran the torch verification path fails the rung), the torch verification, no speculation. A pass on the last
rung only is the status "fallback", a failure of the speculative mode.

By default the servers skip the bucket warmup (`VLLM_SKIP_WARMUP=true`): the warmup alone takes about
15 minutes per mode on the tiny model, while the four short prompts need only a handful of graphs,
which then compile on first use. `--full-warmup` keeps the warmup (it measures the real one), and
`--env KEY=VALUE` (repeatable) adds a server environment variable applied last, after the mode's own
flags, so it can override anything. The health wait defaults to 1800 s.

`--recipe-cache DIR` gives the compiled-recipe cache to the servers that may carry it
(`serve.recipe_cache_env`): the `pytorch` mode's, and every server once the combined kernel database
is installed and not switched off (`FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0`). The `fused`, `hc` and `moe`
modes run without it otherwise: their kernel databases must be registered in `GC_KERNEL_PATH`, and the Habana
bridge accepts only one kernel library there next to a recipe cache. The cache lives in the per-bundle
subfolder the run uses, `DIR/<first 12 characters of the plugin commit>` (`unknown` without a commit).

Runs on Python 3.10 with the standard library only, from the plugin root of the bundle:
`python -m tools.flash_next.rental.tiny_smoke --out DIR [--ckpt DIR] [--gdn pytorch,fused[,hc][,moe]] [--port N]
[--health-timeout S] [--full-warmup] [--recipe-cache DIR] [--recipe-cache-mb N] [--env KEY=VALUE]`.
`--ckpt` names a folder the checkpoint is built in once and reused from then on (default: `<out>/ckpt`,
built every time); a folder built with other settings (another GDN head dim, generator or fixture: the
`build` block of its reference.json) is rebuilt. tiny_smoke.sh starts this inside the container.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from tools.flash_next.rental import gates, serve, spec_probe
from tools.flash_next.rental import main as orchestrator
from tools.flash_next.rental.schema import GateStatus

GDN_MODES = {
    "pytorch": dict(gates.GDN_FALLBACK_ENV),
    "fused": {gates.GDN_MODE_ENV: str(gates.GDN_MODE_FUSED), gates.GDN_VERIFY_ENV: str(gates.GDN_VERIFY_OFF)},
}
# `hc`: the fused mode plus the fused hyper-connection kernels. Not a default mode: the kernels have not
# run in a server yet, and the two default modes are the ones the run's gates depend on.
GDN_MODES["hc"] = dict(GDN_MODES["fused"], **{gates.HC_FUSED_ENV: "1"})
# `moe`: the fused mode plus the MoE glue kernels (the same library as the hyper-connection kernels, its
# own flag). Not a default mode either: the two default modes are the ones the run's gates depend on.
GDN_MODES["moe"] = dict(GDN_MODES["fused"], **{gates.MOE_GLUE_ENV: "1"})
# `qsa`: the fused mode plus the fused QSA decode path with its kernels and the request-sharded indexer
# (VLLM_HPU_QSA_FUSED=2, VLLM_HPU_QSA_INDEXER_SHARD=1). Not a default mode.
GDN_MODES["qsa"] = dict(GDN_MODES["fused"], **{gates.QSA_FUSED_ENV: "2", gates.QSA_SHARD_ENV: "1"})
DEFAULT_MODES = "pytorch,fused"
QSA_KERNEL_MODES = ("qsa", )  # fail when a QSA layer ran without the kernels
QSA_FALLBACK_MARKERS = ("QSA fused (VLLM_HPU_QSA_FUSED=2) is off for",
                        "QSA fused (VLLM_HPU_QSA_FUSED=2): the kernels are not loaded")
DECODE_KERNEL_MODES = ("fused", "hc", "moe", "qsa")  # fail when a GDN layer ran the torch decode path
HC_KERNEL_MODES = ("hc", )  # fail when a hyper-connection ran the torch path
MOE_KERNEL_MODES = ("moe", )  # fail when a MoE block ran the torch path
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


def served_streams(base_url: str, reference_path: Path, post=None) -> dict | None:
    """``{prompt length: the token ids served}`` for every reference prompt (greedy, gates.TINY_MAX_TOKENS
    tokens), or None when a request fails. The comparison ran these prompts before, so nothing compiles."""
    post = post or gates._post_json
    try:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        streams = {}
        for p in reference["prompts"]:
            resp = post(base_url, "/v1/completions", {"model": "tiny", "prompt": p["prompt_ids"],
                                                      "max_tokens": gates.TINY_MAX_TOKENS, "temperature": 0,
                                                      "return_token_ids": True}, gates.HTTP_TIMEOUT_S)
            streams[len(p["prompt_ids"])] = list((resp.get("choices") or [{}])[0].get("token_ids") or [])
        return streams
    except Exception:  # noqa: BLE001 - a record, the comparison decides
        return None


def compare_streams(spec: dict | None, plain: dict | None) -> dict | None:
    """The speculative server's streams against the same mode's server without speculation: per prompt length
    the first differing token (None: equal). None when either side is missing."""
    if spec is None or plain is None:
        return None
    result = {"equal": True, "first_diff": {}}
    for length, want in sorted(plain.items()):
        got = spec.get(length)
        diff = None
        if got != want:
            diff = next((j for j, (a, b) in enumerate(zip(got or [], want)) if a != b), min(len(got or []), len(want)))
            result["equal"] = False
        result["first_diff"][str(length)] = diff
    return result


def spec_serve_args(k: int) -> list[str]:
    """The serve arguments of an MTP server drafting ``k`` tokens (the variant B recipe at k = 3)."""
    return ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": int(k)})]


# The speculative mode's rungs, tried in order until one passes (rule 8: an automatic fallback at every gate
# that loads a component on its first card run): the fused GDN verification op on, then off, then no
# speculation at all (which says whether the mode itself still serves).
SPEC_RUNG_VERIFY = "verify"
SPEC_RUNG_TORCH = "torch-verify"
SPEC_RUNG_OFF = "no-spec"
SPEC_RUNGS = (SPEC_RUNG_VERIFY, SPEC_RUNG_TORCH, SPEC_RUNG_OFF)


def spec_mode_name(mode: str, k: int) -> str:
    return "%s+mtp%d" % (mode, k)


def _spec_counters(base_url: str) -> dict | None:
    try:
        return spec_probe.read_spec_counters(base_url)
    except Exception:  # noqa: BLE001 - a record: a server without /metrics gives no acceptance
        return None


def run_spec_mode(mode: str, k: int, ckpt: Path, out_dir: Path, port: int, health_timeout_s: float,
                  plain_streams: dict | None = None, **kwargs) -> dict:
    """The speculative mode `<mode>+mtp<k>`: the rungs of SPEC_RUNGS in order until one passes.

    Each rung is a `run_mode` call in `<out_dir>/<rung>/`. The entry passes on a speculative rung (`verify` or
    `torch-verify`); a pass only on `no-spec` is a failure of the speculative mode (`status` "fallback", the
    mode itself still serves). `spec_vs_plain` compares the passing speculative rung's tokens with the same
    mode's server without speculation (`plain_streams`, from the run of that mode before), token for token.
    """
    name = spec_mode_name(mode, k)
    entry: dict = {"mode": name, "status": "failed", "reason": None, "rung": None, "rungs": [], "spec_k": k,
                   "acceptance": None, "spec_vs_plain": None, "decode_ms_per_token": None, "startup_seconds": None,
                   "errors": [], "fallback_lines": [], "log": None, "warmup": None}
    reasons = []
    for rung in SPEC_RUNGS:
        print("tiny smoke [%s]: rung %s" % (name, rung), flush=True)
        attempt = run_mode(mode, ckpt, out_dir / rung, port, health_timeout_s, spec_k=k, rung=rung,
                           keep_streams=True, **kwargs)
        entry["rungs"].append({key: attempt.get(key) for key in ("rung", "status", "reason", "startup_seconds",
                                                                   "decode_ms_per_token", "acceptance", "log")})
        if attempt["status"] != "passed":
            reasons.append("%s: %s" % (rung, attempt["reason"]))
            entry["errors"], entry["fallback_lines"] = attempt["errors"], attempt["fallback_lines"]
            continue
        for key in ("rung", "acceptance", "decode_ms_per_token", "startup_seconds", "log", "warmup",
                    "recipe_cache"):
            entry[key] = attempt.get(key)
        if rung == SPEC_RUNG_OFF:
            # the errors of the last speculative rung stay: they say why speculation did not serve
            entry["status"] = "fallback"
            entry["reason"] = "speculation fell back to off (%s)" % "; ".join(reasons)
        else:
            entry["errors"], entry["fallback_lines"] = [], []
            entry["status"] = "passed"
            entry["reason"] = "; ".join(reasons) or None  # why an earlier rung did not pass
            entry["spec_vs_plain"] = compare_streams(attempt.get("streams"), plain_streams)
        break
    else:
        entry["reason"] = "; ".join(reasons)
    _write_json(out_dir / "details.json", entry)
    return entry


def run_mode(mode: str, ckpt: Path, out_dir: Path, port: int, health_timeout_s: float, skip_warmup: bool = True,
             extra_env: dict | None = None, recipe_cache: str | None = None, recipe_cache_mb: int = 8192,
             combined_db: bool = False, spec_k: int = 0, rung: str | None = None, keep_streams: bool = False) -> dict:
    """Serve the tiny checkpoint in one GDN mode, compare it with the reference, stop the server.

    The overrides are the mode's GDN flags, then `VLLM_SKIP_WARMUP=true` when `skip_warmup`, then the
    recipe cache where `serve.recipe_cache_env` allows it for this mode (the fused mode's kernel
    database must stay alone in `GC_KERNEL_PATH`), then `extra_env` applied last so it may override
    any of them. Returns the mode's summary entry; the comparison's own record goes to
    `<out_dir>/details.json`.

    With `spec_k` > 0 the server drafts `spec_k` tokens with the MTP layer (`spec_serve_args`) on the given
    `rung` (SPEC_RUNGS): `verify` sets `VLLM_HPU_GDN_CORE_VERIFY=1` and fails when the log says a layer ran the
    torch verification path, `torch-verify` sets it to 0, `no-spec` serves without speculation. A speculative
    rung also records the acceptance (`spec_probe` counters of `/metrics` around the comparison).
    `keep_streams` records the tokens served for every reference prompt (`served_streams`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server0.log"
    speculative = bool(spec_k) and rung != SPEC_RUNG_OFF
    entry: dict = {"mode": mode, "status": None, "reason": None, "startup_seconds": None, "errors": [],
                   "fallback_lines": [], "log": str(log_path),
                   "warmup": "warmup skipped" if skip_warmup else "full warmup", "decode_ms_per_token": None}
    if spec_k:
        entry.update(rung=rung, spec_k=int(spec_k) if speculative else 0, acceptance=None)
    details: dict = {}
    status = GateStatus.FAILED
    if serve.port_in_use(port):
        entry["reason"] = "port %d already in use" % port
    else:
        overrides = dict(GDN_MODES[mode])
        if speculative:
            overrides[gates.GDN_VERIFY_ENV] = "1" if rung == SPEC_RUNG_VERIFY else "0"
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
        command = serve_command(ckpt, port) + (spec_serve_args(spec_k) if speculative else [])
        popen = serve.start_server(command, env, log_path, out_dir)
        base_url = "http://127.0.0.1:%d" % port
        try:
            started = time.monotonic()
            healthy = serve.wait_healthy(port, health_timeout_s, popen)
            entry["startup_seconds"] = round(time.monotonic() - started, 1)
            if healthy:
                before = _spec_counters(base_url) if speculative else None
                status, details = gates.tiny_compare(base_url, ckpt / "reference.json")
                entry["reason"] = details.get("reason") or details.get("error")
                if speculative:
                    entry["acceptance"] = spec_probe.spec_summary(
                        spec_probe.counters_delta(before, _spec_counters(base_url)))
                if keep_streams:
                    entry["streams"] = served_streams(base_url, ckpt / "reference.json")
                # Recorded only, while the server is still up. Also after a failed comparison: a server
                # that decodes the wrong tokens still shows what a decode step costs (None when it died).
                entry["decode_ms_per_token"] = decode_ms_per_token(base_url, ckpt / "reference.json")
            else:
                status, entry["reason"] = serve.unhealthy_reason(popen, entry["startup_seconds"], health_timeout_s)
        finally:
            serve.stop_server(popen)  # also on Ctrl-C: the workers must not keep the cards
        log_text = _read(log_path)
        if status is GateStatus.PASSED and speculative and rung == SPEC_RUNG_VERIFY:
            fallback = [ln for ln in gates.gdn_fallback_lines(log_text) if "GDN core verify" in ln]
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the fused verification op did not run: %d log line(s) say a layer used the "
                                   "torch verification path" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
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
        if status is GateStatus.PASSED and mode in MOE_KERNEL_MODES:
            fallback = gates.moe_fallback_lines(log_text)
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the MoE glue kernels did not run: %d log line(s) say a layer used the torch "
                                   "path" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
        if status is GateStatus.PASSED and mode in QSA_KERNEL_MODES:
            fallback = [ln for ln in log_text.splitlines() if any(m in ln for m in QSA_FALLBACK_MARKERS)]
            if fallback:
                status = GateStatus.FAILED
                entry["reason"] = ("the fused QSA kernels did not run: %d log line(s) say a layer used the torch "
                                   "data flow" % len(fallback))
                entry["fallback_lines"] = fallback[:MAX_LINES]
        if status is not GateStatus.PASSED:
            entry["errors"] = error_lines(log_text)
    entry["status"] = status.name.lower()
    _write_json(out_dir / "details.json", dict(details, mode=mode, status=int(status), reason=entry["reason"]))
    return entry


def report_spec(entry: dict) -> None:
    """The speculative mode's lines: which rung passed, the acceptance, the tokens against the plain mode."""
    mode = entry["mode"]
    acceptance = entry.get("acceptance") or {}
    length = acceptance.get("acceptance_length")
    rates = ", ".join("%.2f" % r for r in acceptance.get("position_rates") or [] if r is not None)
    if entry["status"] == "passed":
        print("tiny smoke [%s]: PASSED on rung %s (server up in %s s, tokens match the reference)%s" %
              (mode, entry["rung"], entry["startup_seconds"],
               "; earlier rungs: %s" % entry["reason"] if entry.get("reason") else ""), flush=True)
    else:
        print("tiny smoke [%s]: %s (%s)" % (mode, "FALLBACK" if entry["status"] == "fallback" else "FAILED",
                                            entry["reason"]), flush=True)
    if length is not None:
        print("tiny smoke [%s]: acceptance length %.2f (%d drafts, %d of %d draft tokens accepted; per position "
              "%s)" % (mode, length, acceptance["drafts"], acceptance["accepted"], acceptance["draft_tokens"],
                       rates or "-"), flush=True)
    elif entry["status"] == "passed":
        print("tiny smoke [%s]: acceptance not measured (no speculative counters on /metrics)" % mode, flush=True)
    versus = entry.get("spec_vs_plain")
    if versus is not None:
        diffs = {n: d for n, d in versus["first_diff"].items() if d is not None}
        print("tiny smoke [%s]: tokens %s the same mode without speculation%s" %
              (mode, "equal to" if versus["equal"] else "DIFFER from",
               "" if versus["equal"] else " (prompt length: first differing token %s)" % diffs), flush=True)
    decode = entry.get("decode_ms_per_token")
    if decode is not None:
        print("tiny smoke [%s]: decode %.1f ms/token (the shortest prompt, %d tokens against 1, best of %d)" %
              (mode, decode, gates.TINY_MAX_TOKENS, DECODE_TIMING_REPEATS), flush=True)
    if entry["status"] != "passed":
        for line in entry["errors"] + entry["fallback_lines"]:
            print("    %s" % line, flush=True)
        for rung in entry.get("rungs") or []:
            print("    rung %s: %s, log: %s" % (rung["rung"], rung["status"], rung["log"]), flush=True)


def report_mode(entry: dict) -> None:
    """One result line for the console, and on a failure the error lines and the log to read."""
    mode = entry["mode"]
    if "rungs" in entry:
        report_spec(entry)
        return
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


def run_argmax_check(out: Path) -> dict:
    """``argmax_precision_check.py --report-only`` on the card before any server (its lines into ``out``): the fused
    precision of an arg max in the LM head's lazy graph is card-only behaviour the CPU reference cannot show."""
    import importlib.util
    if importlib.util.find_spec("habana_frameworks") is None:  # no card here (the CPU tests): nothing to check
        return {"skipped": "no card"}
    script = Path(__file__).with_name("argmax_precision_check.py")
    log = out / "argmax_check.log"
    try:
        done = subprocess.run([sys.executable, str(script), "--report-only"], capture_output=True, text=True,
                              timeout=600)
        text = done.stdout + done.stderr
    except Exception as error:  # noqa: BLE001 - the check never stops the smoke
        text = "ARGMAX-CHECK not run: %r\n" % (error, )
    log.write_text(text, encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("ARGMAX-CHECK")]
    for line in lines:
        print("tiny smoke: " + line, flush=True)
    return {"lines": lines, "log": str(log)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.tiny_smoke",
                                     description="Serve the tiny Flash-Next checkpoint and compare it with its "
                                     "CPU reference, once per GDN mode.")
    parser.add_argument("--out", required=True, help="results folder of this call")
    parser.add_argument("--ckpt", default=None, help="checkpoint folder, built once and reused (default: OUT/ckpt)")
    parser.add_argument("--gdn", default=DEFAULT_MODES,
                        help="comma-separated modes: pytorch, fused, hc, moe (default: %(default)s; hc is fused "
                        "plus the fused hyper-connection kernels, VLLM_HPU_HC_FUSED=1; moe is fused plus the MoE "
                        "glue kernels, VLLM_HPU_MOE_GLUE=1; qsa is fused plus the fused QSA decode kernels, "
                        "VLLM_HPU_QSA_FUSED=2 with the request-sharded indexer)")
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
    parser.add_argument("--spec", type=int, default=0, metavar="K",
                        help="after each mode, its speculative mode <mode>+mtpK: the MTP layer drafts K tokens, "
                        "the tokens are compared with the reference and with the mode's own tokens, the "
                        "acceptance is printed; rungs: the fused verification op, then the torch "
                        "verification, then no speculation (default 0: no speculative mode)")
    parser.add_argument("--no-argmax-check", action="store_true",
                        help="skip argmax_precision_check.py (by default it runs once on the card before the "
                        "servers, report only: every in-graph arg max against the gathered path's)")
    args = parser.parse_args(argv)
    modes = [m for m in args.gdn.split(",") if m]
    if not modes or any(m not in GDN_MODES for m in modes):
        parser.error("--gdn takes a comma-separated list of: %s" % ", ".join(GDN_MODES))
    if args.spec < 0:
        parser.error("--spec takes the number of drafted tokens, 1 or more (0: off)")
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
    if summary["reason"] is None and not args.no_argmax_check:
        summary["argmax_check"] = run_argmax_check(out)
    if summary["reason"] is None:
        common = dict(skip_warmup=not args.full_warmup, extra_env=extra_env, recipe_cache=recipe_cache,
                      recipe_cache_mb=args.recipe_cache_mb, combined_db=combined_db)
        for mode in modes:
            entry = run_mode(mode, ckpt, out / mode, args.port, args.health_timeout, keep_streams=bool(args.spec),
                             **common)
            plain_streams = entry.pop("streams", None)
            summary["modes"].append(entry)
            report_mode(entry)
            if args.spec:
                spec_entry = run_spec_mode(mode, args.spec, ckpt, out / spec_mode_name(mode, args.spec), args.port,
                                           args.health_timeout, plain_streams=plain_streams, **common)
                summary["modes"].append(spec_entry)
                report_mode(spec_entry)
        summary["passed"] = all(entry["status"] == "passed" for entry in summary["modes"])
    _write_json(out / "summary.json", summary)
    if summary["passed"]:
        print("tiny smoke: PASSED (%s)" % ", ".join(entry["mode"] for entry in summary["modes"]), flush=True)
        return 0
    failed = [entry["mode"] for entry in summary["modes"] if entry["status"] != "passed"]
    print("tiny smoke: FAILED (%s)" % (summary["reason"] or ", ".join(failed)), flush=True)
    print("summary: %s" % (out / "summary.json"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
