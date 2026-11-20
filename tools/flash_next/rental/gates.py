# SPDX-License-Identifier: Apache-2.0
"""Fail-fast gates of the Flash-Next rental (plan M4, RUK-2709).

One function per gate: preflight (cards, checkpoint, FP8 layout, disk),
build (installed vLLM / vllm_gaudi / flashinfer_gaudi), the fused GDN
decode kernel and the GDN verify op on the card (G1b: the kernel's own
checks, then the verify op's checks, and the GDN decode and verify
modes every later server must use, with one rerun without the combined
kernel database when the decode checks fail through it), the fused
hyper-connection kernels' card test that the orchestrator runs right
after it (`hc_fused`, recorded in G1b's details) and the MoE glue
kernels' card test after that one (`moe_glue`, the same library, recorded
the same way), tiny TP8 token agreement, capacity from the startup line,
the first token of a started server, and coherence with a 120K-token
needle.
Every gate returns ``(GateStatus, details)`` and never raises for an
expected failure: the exception text goes into ``details["error"]``.
Python 3.10 standard library only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from tools.flash_next.rental import lazy_graph, serve
from tools.flash_next.rental.schema import MAX_MODEL_LEN, GateStatus

try:  # M4-H0 (written in parallel): the FP8 layout checker
    from tools.flash_next.check_fp8_layout import check_layout
except ImportError:  # pragma: no cover - covered by the monkeypatched tests
    check_layout = None

EXPECTED_VLLM_VERSION = "0.28.1rc1.dev392+g8f816a3f6"

DEFAULT_HLSMI_CMD = ["hl-smi", "-Q", "index,name", "-f", "csv,noheader"]
DEFAULT_MIN_CHECKPOINT_BYTES = 160 * 2**30  # the published FP8 checkpoint is about 185.5 GB
DEFAULT_MIN_FREE_BYTES = 50 * 2**30

HLSMI_TIMEOUT_S = 10.0
BUILD_TIMEOUT_S = 300.0
HTTP_TIMEOUT_S = 600.0

TINY_MAX_TOKENS = 16
FIRST_TOKEN_PROMPT = "The capital of France is"
FIRST_TOKEN_MAX_TOKENS = 8
COHERENCE_MAX_TOKENS = 64
NEEDLE_MAX_TOKENS = 32

NEEDLE_SENTENCE = "Note for the operator: the secret deployment code for the Gaudi rental is 7319-ALPHA."
NEEDLE_CODE = "7319-ALPHA"
NEEDLE_QUESTION = "\n\nQuestion: what is the secret deployment code for the Gaudi rental? Answer with the code only."

# The 8 short prompts of the qualification run sheet, stage 2b, "Coherence check"
# (docs/design/flash-next/run-sheets/qualification.md): prompt 7 asks for a JSON object.
COHERENCE_PROMPTS = (
    "Explain in two sentences why the sky is blue during the day.",
    "Write a haiku about the sea.",
    "List the three primary colors and one use of each.",
    "What is the capital of France? Answer in one sentence.",
    "Summarize in one sentence what a compiler does.",
    "Continue this sentence: The best way to learn programming is",
    "Reply with a JSON object only, keys name, age, city, for a fictional person.",
    "Write a Python function fib(n) that returns the n-th Fibonacci number.",
)
COHERENCE_JSON_PROMPT_INDEX = 6  # the run sheet's "prompt 7"


def _post_json(base_url: str, path: str, payload: dict, timeout_s: float) -> dict:
    """POST a JSON body and parse the JSON answer; a socket timeout raises TimeoutError."""
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:500]
        except Exception:  # nosec B110 - the body is best-effort context
            pass
        raise RuntimeError("HTTP %s from %s: %s" % (e.code, path, body)) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("request to %s timed out after %s s" % (path, timeout_s)) from e
        raise


def _is_repetition_loop(text: str) -> bool:
    """True when the most frequent word is more than 40% of the words and there are >= 10 words."""
    words = text.split()
    if len(words) < 10:
        return False
    _, most_count = Counter(words).most_common(1)[0]
    return most_count > 0.4 * len(words)


def _has_json_object(text: str) -> bool:
    """True when the slice from the first '{' to the last '}' parses as JSON."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        return False
    try:
        json.loads(text[start:end + 1])
        return True
    except ValueError:
        return False


def _insert_needle(content: str) -> str:
    """Insert the needle sentence at the next newline after 30% of the length (else at the end)."""
    pos = int(len(content) * 0.30)
    idx = content.find("\n", pos)
    if idx == -1:
        return content + "\n" + NEEDLE_SENTENCE
    return content[:idx + 1] + NEEDLE_SENTENCE + "\n" + content[idx + 1:]


def capacity(startup: dict, agents: int) -> tuple[GateStatus, dict]:
    """The startup-line gate: vLLM's own "Maximum concurrency" must cover the run."""
    details: dict = {"agents": agents}
    try:
        conc = startup.get("max_concurrency")
        conc_len = startup.get("max_concurrency_len")
        details["max_concurrency"] = conc
        details["max_concurrency_len"] = conc_len
        if conc is None or conc_len is None:
            details["reason"] = "startup line missing"
            return GateStatus.FAILED, details
        if conc >= agents and conc_len == MAX_MODEL_LEN:
            return GateStatus.PASSED, details
        details["reason"] = ("max_concurrency %s < agents %d" % (conc, agents) if conc < agents else
                             "max_concurrency_len %s != %d" % (conc_len, MAX_MODEL_LEN))
        return GateStatus.FAILED, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details


def preflight(model_dir: Path,
              layouts: list,
              results_dir: Path,
              expected_cards: int = 8,
              hlsmi_cmd: list = None,
              min_checkpoint_bytes: int = DEFAULT_MIN_CHECKPOINT_BYTES,
              min_free_bytes: int = DEFAULT_MIN_FREE_BYTES) -> tuple[GateStatus, dict]:
    """The G0 gate: cards, block-FP8 checkpoint, FP8 layouts and free disk."""
    details: dict = {
        "expected_cards": expected_cards,
        "cards": None,
        "config_ok": False,
        "quant_method": None,
        "weight_block_size": None,
        "index_ok": False,
        "checkpoint_bytes": 0,
        "min_checkpoint_bytes": min_checkpoint_bytes,
        "layout_rows": [],
        "free_bytes": None,
        "min_free_bytes": min_free_bytes,
        "checks": {},
    }
    failed = []
    try:
        # cards
        if expected_cards == 0:
            details["checks"]["cards"] = "skipped"
        else:
            cmd = list(hlsmi_cmd) if hlsmi_cmd else list(DEFAULT_HLSMI_CMD)
            out = run_step(cmd, tempfile.gettempdir(), HLSMI_TIMEOUT_S)
            cards = len([ln for ln in out.stdout.splitlines() if ln.strip()])
            details["cards"] = cards
            if cards == expected_cards:
                details["checks"]["cards"] = "passed"
            else:
                details["checks"]["cards"] = "failed"
                failed.append("cards: %d seen, %d expected" % (cards, expected_cards))

        # checkpoint config: block-FP8 [128, 128]
        config = None
        config_path = Path(model_dir) / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                details["config_ok"] = True
            except Exception as e:  # noqa: BLE001
                details["checks"]["config"] = "failed"
                failed.append("config.json unreadable: %s: %s" % (type(e).__name__, e))
        if details["config_ok"]:
            quant = config.get("quantization_config") or {}
            details["quant_method"] = quant.get("quant_method")
            details["weight_block_size"] = quant.get("weight_block_size")
            fp8_ok = (quant.get("quant_method") == "fp8" and quant.get("weight_block_size") == [128, 128])
            if fp8_ok:
                details["checks"]["config"] = "passed"
            else:
                details["checks"]["config"] = "failed"
                failed.append("checkpoint is not block-FP8 [128, 128]: quant_method=%r weight_block_size=%r" %
                              (quant.get("quant_method"), quant.get("weight_block_size")))
        elif "config" not in details["checks"]:
            details["checks"]["config"] = "failed"
            failed.append("config.json missing in %s" % model_dir)

        # safetensors index and total size
        details["index_ok"] = (Path(model_dir) / "model.safetensors.index.json").is_file()
        details["checkpoint_bytes"] = sum(f.stat().st_size for f in Path(model_dir).glob("*.safetensors"))
        if details["index_ok"] and details["checkpoint_bytes"] >= min_checkpoint_bytes:
            details["checks"]["checkpoint"] = "passed"
        else:
            details["checks"]["checkpoint"] = "failed"
            if not details["index_ok"]:
                failed.append("model.safetensors.index.json missing")
            if details["checkpoint_bytes"] < min_checkpoint_bytes:
                failed.append("checkpoint is %.3f GiB < %.3f GiB (incomplete download?)" %
                              (details["checkpoint_bytes"] / 2**30, min_checkpoint_bytes / 2**30))

        # FP8 layout per (tp, ep)
        if not layouts:
            details["checks"]["layout"] = "skipped"
        elif config is None:
            details["checks"]["layout"] = "skipped"
            failed.append("layout check skipped: config unreadable")
        elif check_layout is None:
            details["checks"]["layout"] = "failed"
            failed.append("check_fp8_layout.check_layout is not importable (M4-H0 not merged)")
        else:
            layout_bad = []
            for tp, ep in layouts:
                try:
                    rows = check_layout(config, tp, ep)
                except (KeyError, ValueError, TypeError) as e:  # not a block-FP8 Flash-Next config
                    rows = [{"module": "config", "dim": str(e), "size": 0, "per_rank": 0, "block": 0, "ok": False}]
                details["layout_rows"].extend(dict(row, tp=tp, ep=ep) for row in rows)
                layout_bad.extend(dict(row, tp=tp, ep=ep) for row in rows if not row.get("ok"))
            details["layout_failed"] = layout_bad
            if layout_bad:
                details["checks"]["layout"] = "failed"
                failed.append("FP8 layout: %d rows not shardable at the run layouts" % len(layout_bad))
            else:
                details["checks"]["layout"] = "passed"

        # free disk for the results
        details["free_bytes"] = shutil.disk_usage(Path(results_dir)).free
        if details["free_bytes"] >= min_free_bytes:
            details["checks"]["disk"] = "passed"
        else:
            details["checks"]["disk"] = "failed"
            failed.append("results disk has %.3f GiB free < %.3f GiB required" %
                          (details["free_bytes"] / 2**30, min_free_bytes / 2**30))
    except subprocess.TimeoutExpired:
        details["error"] = "hl-smi did not answer in %s s" % HLSMI_TIMEOUT_S
        return GateStatus.TIMED_OUT, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details

    if failed:
        details["reason"] = "; ".join(failed)
        return GateStatus.FAILED, details
    return GateStatus.PASSED, details


def _last_line(text: str) -> str:
    """The last non-empty line of a probe's stdout: importing vllm logs to stdout before the probe's own print."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


STEP_KILL_WAIT_S = 30.0  # after SIGKILL, how long to wait for the pipes to close


def run_step(cmd: list, cwd, timeout_s: float, env: dict = None) -> subprocess.CompletedProcess:
    """Run one gate step in its own process group, stdout and stderr captured as text.

    On timeout, SIGKILL the whole group (pytest's workers and anything else the step started),
    then raise subprocess.TimeoutExpired. After a normal exit, SIGKILL whatever is left in the
    group, so no step can keep holding a card.
    """
    args = [str(arg) for arg in cmd]
    proc = subprocess.Popen(args, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        try:
            out, err = proc.communicate(timeout=STEP_KILL_WAIT_S)
        except subprocess.TimeoutExpired:
            out, err = "", ""  # the pipes did not close after the SIGKILL: no partial output to keep
        raise subprocess.TimeoutExpired(args, timeout_s, output=out, stderr=err) from None
    _kill_group(proc.pid)
    return subprocess.CompletedProcess(args, proc.returncode, out, err)


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def build_check(python: str = sys.executable) -> tuple[GateStatus, dict]:
    """The G1 gate: the installed vLLM version, vllm_gaudi and flashinfer_gaudi import."""
    details: dict = {
        "python": str(python),
        "vllm_version": None,
        "vllm_gaudi_ok": False,
        "flashinfer_gaudi_ok": False,
        "flashinfer_capabilities": None,
    }
    checks = (("vllm", "import vllm; print(vllm.__version__)"),
              ("vllm_gaudi", "import vllm_gaudi"),
              ("flashinfer_gaudi", "import flashinfer_gaudi, json; "
               "print(json.dumps(flashinfer_gaudi.get_capabilities(), default=str))"))
    try:
        for name, code in checks:
            proc = run_step([str(python), "-c", code], tempfile.gettempdir(),  # never the repository root
                            BUILD_TIMEOUT_S)
            if proc.returncode != 0:
                details["reason"] = "%s: exit %d: %s" % (name, proc.returncode, (proc.stderr or "").strip()[:500])
                return GateStatus.FAILED, details
            out = _last_line(proc.stdout)
            if name == "vllm":
                details["vllm_version"] = out
                if out != EXPECTED_VLLM_VERSION:
                    details["reason"] = "vllm version %r != %r" % (out, EXPECTED_VLLM_VERSION)
                    return GateStatus.FAILED, details
            elif name == "vllm_gaudi":
                details["vllm_gaudi_ok"] = True
            else:
                details["flashinfer_gaudi_ok"] = True
                details["flashinfer_capabilities"] = out  # an empty mapping is not a failure
        return GateStatus.PASSED, details
    except subprocess.TimeoutExpired:
        details["error"] = "a build check did not finish in %s s" % BUILD_TIMEOUT_S
        return GateStatus.TIMED_OUT, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details


GDN_MODE_ENV = "VLLM_HPU_GDN_CORE_DECODE"
GDN_VERIFY_ENV = "VLLM_HPU_GDN_CORE_VERIFY"
GDN_LAZY_ENV = "PT_HPU_LAZY_MODE"  # the Habana bridge mode: "1" lazy (the servers), "0" eager (torch.compile)
GDN_VERIFY_OP_TEST = "tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py"
# The lines a server writes when a fused kernel did not run: the fused GDN ops', the fused
# hyper-connection kernels' ("HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel", "... is off for <layer>")
# and the MoE glue kernels' ("MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel", "... is off for <layer>").
GDN_FALLBACK_RE = re.compile(r"has no HPU kernel|GDN core (decode|verify) .* is off for|HC fused .* is off for|"
                             r"MoE glue .* is off for")
HC_FALLBACK_RE = re.compile(r"HC fused .* (is off for|has no HPU kernel)")
MOE_FALLBACK_RE = re.compile(r"MoE glue .* (is off for|has no HPU kernel)")
FALLBACK_GDN = "gdn"
FALLBACK_HC = "hc"
FALLBACK_MOE = "moe"
GDN_OP_TIMEOUT_S = 600.0       # test_hpu_op.py and test_hpu_verify_op.py (about 16x the card's 37 s)
GDN_CRASH_PROBE = "tools/flash_next/gdn_verify_crash_probe.sh"
GDN_CRASH_PROBE_TIMEOUT_S = 4500.0  # the probe's worst case: apt 600 s + cmake 1800 s + six runs of 310 s = 4260 s
GDN_REPLAY_TIMEOUT_S = 900.0   # each replay run
GDN_BENCH_TIMEOUT_S = 1200.0

GDN_LIB_LOOKUP_CODE = ("import flashinfer_gaudi, pathlib; "
                       "print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')")
GDN_KERNEL_LIB = "libgdn_fused_gaudi2.so"
GDN_MODE_FUSED = 1
GDN_MODE_PYTORCH = 0
GDN_VERIFY_ON = 1
GDN_VERIFY_OFF = 0
GDN_FALLBACK_ENV = {GDN_MODE_ENV: "0", GDN_VERIFY_ENV: "0"}

# The steps of G1b whose result is recorded only (it never changes the gate's status, the GDN mode or the
# verify mode): `skip_recorded_only` leaves all of them out, `skip_verify_probe` only the crash probe.
GDN_STEP_EAGER_REPLAYS = "replay_eager"
GDN_STEP_BENCH = "bench"
GDN_STEP_CRASH_PROBE = "verify_crash_probe"
GDN_RECORDED_ONLY_STEPS = (GDN_STEP_EAGER_REPLAYS, GDN_STEP_BENCH, GDN_STEP_CRASH_PROBE)

# details["combined_kernel_db"] of G1b: the combined kernel database (RUK-3243) in the gate's steps.
COMBINED_DB_NOT_INSTALLED = "not_installed"  # no such file next to the fused kernel library
COMBINED_DB_SWITCHED_OFF = "switched_off"    # the operator's FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0
COMBINED_DB_ACTIVE = "active"                # the steps ran through it and the decode checks passed
COMBINED_DB_DISABLED_BY_G1B = "disabled_by_g1b"  # the decode checks failed with it and passed without
COMBINED_DB_FAILED_BOTH = "failed_with_and_without"  # they failed either way: not the database's fault


def _gdn_pytest_summary(output: str) -> dict:
    """The counts of pytest's last summary line ("2 failed, 5 passed in 3.4s"); zeros when absent."""
    counts = {"passed": 0, "skipped": 0, "failed": 0, "error": 0}
    summary = None
    for line in output.splitlines():
        if re.search(r"\b\d+ (passed|failed|skipped|error)", line):
            summary = line
    for name in counts:
        m = re.search(r"(\d+) %s" % name, summary) if summary else None
        counts[name] = int(m.group(1)) if m else 0
    return counts


_PYTEST_FAILURE_SECTION = re.compile(r"^=+ (FAILURES|ERRORS) =+$")
_PYTEST_FAILURE_HEADER = re.compile(r"^_{3,} (\S.*?) _{3,}$")


def _pytest_failed_tests(output: str) -> list[str]:
    """The failed tests pytest names in `output`, each as a line starting with "FAILED ".

    With `-rf` pytest prints "FAILED <node id> - <message>" lines and those are returned. The card tests run
    with `-q -rs`, which prints none: the names are then taken from the headers of the FAILURES and ERRORS
    sections ("____ test_name[case] ____"), in order and once each.
    """
    lines = output.splitlines()
    named = [ln for ln in lines if ln.startswith("FAILED ")]
    if named:
        return named
    found: list[str] = []
    in_failures = False
    for ln in lines:
        if _PYTEST_FAILURE_SECTION.match(ln):
            in_failures = True
            continue
        header = _PYTEST_FAILURE_HEADER.match(ln) if in_failures else None
        if header and "FAILED " + header.group(1) not in found:
            found.append("FAILED " + header.group(1))
    return found


def _gdn_verify_not_run(reason: str) -> dict:
    """The "verify" details of a run in which the verify checks did not run at all."""
    return {"status": "not_run", "reason": reason, "hpu_op": None, "replay": []}


def _gdn_crash_probe(plugin_root: Path, out_dir: Path, env: dict, python: str) -> dict:
    """Run gdn_verify_crash_probe.sh after V1 ended by a signal or a timeout; recorded only.

    The probe reads PYTHON, PLUGIN_ROOT and PROBE_TIMEOUT_S from its environment and writes
    OUT_DIR/summary.txt. Its stdout+stderr go to out_dir/verify_crash_probe.log and the summary lines
    are returned; the function never raises and its result never changes a status or a mode.
    """
    probe = Path(plugin_root) / GDN_CRASH_PROBE
    if not probe.is_file():
        return {"exit": None, "reason": "not in the bundle", "summary": []}
    crash_dir = Path(out_dir) / "verify_crash"
    try:
        try:
            proc = run_step(["bash", GDN_CRASH_PROBE, str(crash_dir)], Path(plugin_root), GDN_CRASH_PROBE_TIMEOUT_S,
                            dict(env, PYTHON=str(python), PLUGIN_ROOT=str(plugin_root)))
            text, exit_code, timed_out = (proc.stdout or "") + (proc.stderr or ""), proc.returncode, False
        except subprocess.TimeoutExpired as e:
            text, exit_code, timed_out = (e.output or "") + (e.stderr or ""), None, True
        with open(Path(out_dir) / "verify_crash_probe.log", "w", encoding="utf-8") as f:
            f.write(text)
        try:
            summary = (crash_dir / "summary.txt").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            summary = []
        if timed_out:
            return {"exit": None, "timed_out": True, "summary": summary}
        return {"exit": exit_code, "summary": summary}
    except Exception as e:  # noqa: BLE001 - the probe is recorded only, it never changes a status
        return {"exit": None, "reason": "%s: %s" % (type(e).__name__, e), "summary": []}


def fused_gdn(plugin_root: Path, out_dir: Path, python: str = sys.executable, skip_recorded_only: str = None,
              skip_verify_probe: str = None) -> tuple[GateStatus, dict]:
    """G1b. details always has "mode" (1 = the fused kernel stays on, 0 = later servers run the PyTorch path),
    "verify_mode" (1 = the GDN verify op's card checks all passed, 0 = the op stays off) and "env" (the GDN
    flags of every later server, added by the orchestrator: the decode entry as before, and
    "VLLM_HPU_GDN_CORE_VERIFY" = "1" only when every verify check passed, else an explicit "0").

    Only the lazy checks decide the kernel: the op tests, the lazy replays, the verify checks and the bench
    all run with PT_HPU_LAZY_MODE=1, the bridge mode every server uses. The two decode replays run once
    more with PT_HPU_LAZY_MODE=0 for torch.compile and are recorded only ("replay_eager", one entry per
    case with the integer "lazy_mode" 0, and "eager_compile_ok" = True only when both eager replays exit
    0): an eager failure or timeout never turns the kernel off and never changes the gate's status,
    because no server runs eager mode or torch.compile. Every step runs in its own process group and is
    SIGKILLed with all of its children at its timeout (run_step), so no step can outlive it and hold a card.

    The combined kernel database (RUK-3248). The loader finds it next to GDN_FUSED_GAUDI_KERNEL_LIBRARY, so
    with the file installed and the operator's switch not at 0 the steps run through it
    (details["combined_kernel_db"] = "active"). When the decode checks (the op test or a lazy decode
    replay) then fail or time out, they run ONCE more with FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 in the
    step environment. If they pass, the gate goes on with the switch set and passes with the fused kernel
    on: "combined_kernel_db" is "disabled_by_g1b", "env" carries the switch for every later server, and
    the first attempt is kept in "combined_attempt" with its logs renamed to hpu_op.combined.log and
    replay.combined.log. If they fail again the kernel goes off as without the database
    ("failed_with_and_without").

    `skip_recorded_only` (a reason text, the orchestrator's "--quick-gates") leaves out the steps that
    are recorded only (GDN_RECORDED_ONLY_STEPS: the eager replays, the bench, the verify crash probe);
    `skip_verify_probe` leaves out only the probe. details["skipped_steps"] maps each such step to the
    reason; the checks that decide the kernel and the verify op always run."""
    started = time.monotonic()
    details: dict = {"mode": GDN_MODE_PYTORCH, "env": dict(GDN_FALLBACK_ENV), "verify_mode": GDN_VERIFY_OFF}
    probe_skip = skip_recorded_only or skip_verify_probe
    if skip_recorded_only:
        details["skipped_steps"] = {name: skip_recorded_only for name in GDN_RECORDED_ONLY_STEPS}
    else:
        details["skipped_steps"] = {GDN_STEP_CRASH_PROBE: skip_verify_probe} if skip_verify_probe else {}

    def finish(status: GateStatus, reason: str = None) -> tuple[GateStatus, dict]:
        if reason is not None:
            details["reason"] = reason
        details["seconds"] = time.monotonic() - started
        return status, details

    step = "the fused GDN library lookup"
    timeout_s = BUILD_TIMEOUT_S
    try:
        plugin_root = Path(plugin_root)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. the installed library folder (the bundle's source tree has an empty flashinfer_gaudi/lib/)
        lookup = run_step([str(python), "-c", GDN_LIB_LOOKUP_CODE], tempfile.gettempdir(),  # never the repo root
                          timeout_s)
        if lookup.returncode != 0:
            return finish(GateStatus.SKIPPED, "the fused GDN library is not installed: the interpreter exited %d: %s" %
                          (lookup.returncode, (lookup.stderr or "").strip()[:200]))
        lib_dir = Path(_last_line(lookup.stdout))
        kernel = lib_dir / GDN_KERNEL_LIB
        ops_files = sorted(lib_dir.glob("gdn_fused_ops*.so"))
        if not kernel.is_file():
            return finish(GateStatus.SKIPPED,
                          "the fused GDN library is not installed: %s missing in %s" % (GDN_KERNEL_LIB, lib_dir))
        if len(ops_files) != 1:
            return finish(GateStatus.SKIPPED,
                          "the fused GDN library is not installed: %d gdn_fused_ops*.so files in %s, expected 1" %
                          (len(ops_files), lib_dir))

        # 2. the environment of steps 3-5: the plugin tree first on PYTHONPATH, the installed library
        #    by path, and the fused kernel ON for the checks themselves
        env = dict(os.environ)
        env["PYTHONPATH"] = str(plugin_root)
        env["GDN_FUSED_GAUDI_KERNEL_LIBRARY"] = str(kernel)
        env["GDN_FUSED_GAUDI_NATIVE_LIBRARY"] = str(ops_files[0])
        env[GDN_LAZY_ENV] = "1"  # the bridge mode of every server; the container leaves it unset
        env.pop(GDN_MODE_ENV, None)
        env.pop(GDN_VERIFY_ENV, None)
        if not (lib_dir / serve.COMBINED_KERNEL_DB).is_file():
            details["combined_kernel_db"] = COMBINED_DB_NOT_INSTALLED
        elif env.get(serve.COMBINED_KERNEL_DB_ENV, "1").strip() == "0":
            details["combined_kernel_db"] = COMBINED_DB_SWITCHED_OFF
        else:
            details["combined_kernel_db"] = COMBINED_DB_ACTIVE

        def run(cmd: list, log_name: str, limit: float, append: bool = False, lazy_mode: int = 1):
            nonlocal step, timeout_s
            step = " ".join(str(arg) for arg in cmd[1:])
            timeout_s = limit
            try:
                proc = run_step(cmd, plugin_root, limit, dict(env, **{GDN_LAZY_ENV: str(lazy_mode)}))
            except subprocess.TimeoutExpired as e:  # the step's partial output reaches its log before the raise
                with open(out_dir / log_name, "a" if append else "w", encoding="utf-8") as f:
                    f.write((e.output or "") + (e.stderr or ""))
                raise
            text = (proc.stdout or "") + (proc.stderr or "")
            with open(out_dir / log_name, "a" if append else "w", encoding="utf-8") as f:
                f.write(text)
            return proc, text

        def decode_checks() -> tuple:
            """Steps 3 and 4, the checks that decide the kernel; (None, None) when they all passed, else
            the (status, reason) of the first one that did not. A verify run is only attempted after them."""
            try:
                # 3. the op numerics on the card; a module skip counts as a failure
                proc, text = run(
                    [python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                     "tests/unit_tests/gdn_core_decode/test_hpu_op.py"], "hpu_op.log", GDN_OP_TIMEOUT_S)
                counts = _gdn_pytest_summary(text)
                details["hpu_op"] = {
                    "exit": proc.returncode,
                    "passed": counts["passed"],
                    "skipped": counts["skipped"],
                    "failed": counts["failed"],
                }
                details["failed_tests"] = _pytest_failed_tests(text)
                op_ok = (proc.returncode == 0 and counts["passed"] > 0 and counts["skipped"] == 0 and
                         counts["failed"] == 0 and counts["error"] == 0)
                if not op_ok:
                    if counts["skipped"] > 0:
                        return GateStatus.FAILED, "test_hpu_op.py skipped"
                    return GateStatus.FAILED, "test_hpu_op.py failed (exit %d)" % proc.returncode

                # 4. the 1,000-step graph replay against eager, two cases, in one appended log
                details["replay"] = []
                for case in ("tp8_b16", "tp8_b64"):
                    cmd = [python, "tools/flash_next/gdn_core_decode_replay.py", "--steps", "1000"]
                    if case != "tp8_b16":
                        cmd += ["--case", case]
                    proc, _ = run(cmd, "replay.log", GDN_REPLAY_TIMEOUT_S, append=True)
                    details["replay"].append({"case": case, "lazy_mode": 1, "exit": proc.returncode})
                    if proc.returncode != 0:
                        return GateStatus.FAILED, "gdn_core_decode_replay.py (%s) failed (exit %d)" % (
                            case, proc.returncode)
            except subprocess.TimeoutExpired:
                return GateStatus.TIMED_OUT, "%s did not finish in %s s" % (step, timeout_s)
            return None, None

        # 3-4. the decode checks; once more without the combined kernel database when they fail through it
        failed_status, failed_reason = decode_checks()
        if failed_status is not None and details["combined_kernel_db"] == COMBINED_DB_ACTIVE:
            first = {
                "status": failed_status.name.lower(),
                "reason": failed_reason,
                "hpu_op": details.pop("hpu_op", None),
                "failed_tests": details.pop("failed_tests", []),
                "replay": details.pop("replay", None),
                "logs": [],
            }
            for log_name in ("hpu_op.log", "replay.log"):  # the rerun writes these names again
                if (out_dir / log_name).is_file():
                    kept = log_name[:-len(".log")] + ".combined.log"
                    os.replace(out_dir / log_name, out_dir / kept)
                    first["logs"].append(kept)
            details["combined_attempt"] = first
            env[serve.COMBINED_KERNEL_DB_ENV] = "0"  # for the rerun and, when it passes, the rest of the gate
            failed_status, failed_reason = decode_checks()
            details["combined_kernel_db"] = (COMBINED_DB_DISABLED_BY_G1B if failed_status is None else
                                             COMBINED_DB_FAILED_BOTH)
        if failed_status is not None:
            details["verify"] = _gdn_verify_not_run("decode checks failed")
            return finish(failed_status, failed_reason)

        # 4c. the same two decode replays in eager mode, where the replay tool exercises torch.compile's
        #     hpu_backend. Recorded only: an eager failure or timeout never turns the kernel off and
        #     never changes the gate's status, because no server runs eager mode.
        details["replay_eager"] = []
        if skip_recorded_only:
            details["eager_compile_ok"] = None
        else:
            for case in ("tp8_b16", "tp8_b64"):
                cmd = [python, "tools/flash_next/gdn_core_decode_replay.py", "--steps", "1000"]
                if case != "tp8_b16":
                    cmd += ["--case", case]
                try:
                    proc, _ = run(cmd, "replay.log", GDN_REPLAY_TIMEOUT_S, append=True, lazy_mode=0)
                    details["replay_eager"].append({"case": case, "lazy_mode": 0, "exit": proc.returncode})
                except subprocess.TimeoutExpired:
                    details["replay_eager"].append({"case": case, "lazy_mode": 0, "exit": None, "timed_out": True})
            details["eager_compile_ok"] = all(entry["exit"] == 0 for entry in details["replay_eager"])

        def crash_probe() -> dict:
            """Ask gdn_verify_crash_probe.sh why V1 crashed or hung (up to GDN_CRASH_PROBE_TIMEOUT_S), unless
            the caller left the probe out."""
            if probe_skip:
                return {"exit": None, "skipped": probe_skip, "summary": []}
            return _gdn_crash_probe(plugin_root, out_dir, env, python)

        # 4b. the GDN verify op's own card checks, only after the decode checks passed and only when its
        #     test file is in the bundle; a verify failure never changes the gate's status
        verify: dict = {"status": "passed", "reason": "", "hpu_op": None, "replay": []}
        verify_ok = False
        if not (plugin_root / GDN_VERIFY_OP_TEST).is_file():
            verify = _gdn_verify_not_run("%s is not in the bundle" % GDN_VERIFY_OP_TEST)
        else:
            verify_ok = True
            try:
                # V1. the verify op numerics on the card; the same pass rule as step 3
                proc, text = run(
                    [python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", GDN_VERIFY_OP_TEST],
                    "hpu_verify_op.log", GDN_OP_TIMEOUT_S)
                counts = _gdn_pytest_summary(text)
                verify["hpu_op"] = {
                    "exit": proc.returncode,
                    "passed": counts["passed"],
                    "skipped": counts["skipped"],
                    "failed": counts["failed"],
                }
                if not (proc.returncode == 0 and counts["passed"] > 0 and counts["skipped"] == 0 and
                        counts["failed"] == 0 and counts["error"] == 0):
                    verify_ok = False
                    verify["status"] = "failed"
                    verify["reason"] = ("test_hpu_verify_op.py skipped" if counts["skipped"] > 0 else
                                        "test_hpu_verify_op.py failed (exit %d)" % proc.returncode)
                if proc.returncode < 0 or proc.returncode in (134, 139):  # SIGSEGV/SIGABRT: ask the probe why
                    verify["crash_probe"] = crash_probe()
            except subprocess.TimeoutExpired:
                verify_ok = False
                verify["status"] = "timed_out"
                verify["reason"] = "test_hpu_verify_op.py did not finish in %s s" % GDN_OP_TIMEOUT_S
                verify["crash_probe"] = crash_probe()
            if verify_ok:
                # V2. the 1,000-step verify replay against eager, two cases, appended to the replay log
                for case in ("ver_tp8_b16", "ver_tp8_b64"):
                    cmd = [python, "tools/flash_next/gdn_core_decode_replay.py", "--steps", "1000", "--ops", "verify"]
                    if case != "ver_tp8_b16":
                        cmd += ["--case", case]
                    try:
                        proc, _ = run(cmd, "replay.log", GDN_REPLAY_TIMEOUT_S, append=True)
                    except subprocess.TimeoutExpired:
                        verify_ok = False
                        verify["status"] = "timed_out"
                        verify["reason"] = ("gdn_core_decode_replay.py (verify %s) did not finish in %s s" %
                                            (case, GDN_REPLAY_TIMEOUT_S))
                        break
                    verify["replay"].append({"case": case, "lazy_mode": 1, "exit": proc.returncode})
                    if proc.returncode != 0:
                        verify_ok = False
                        verify["status"] = "failed"
                        verify["reason"] = ("gdn_core_decode_replay.py (verify %s) failed (exit %d)" %
                                            (case, proc.returncode))
                        break
        details["verify"] = verify
        details["verify_mode"] = GDN_VERIFY_ON if verify_ok else GDN_VERIFY_OFF

        # 5. the speed measurement: recorded only, its result never changes the status
        if skip_recorded_only:
            details["bench"] = {"exit": None, "skipped": skip_recorded_only, "lines": []}
        else:
            try:
                proc, text = run(
                    [python, "tools/flash_next/gdn_core_decode_bench.py", "--tp", "2,8", "--batches", "1,16,64"],
                    "bench.log", GDN_BENCH_TIMEOUT_S)
                details["bench"] = {
                    "exit": proc.returncode,
                    "lines": [ln.strip() for ln in text.splitlines() if "slope" in ln or "B=1" in ln],
                }
            except subprocess.TimeoutExpired:  # a slow measurement never turns the verified kernel off
                details["bench"] = {"exit": None, "timed_out": True, "lines": []}

        # 6. the fused kernel stays on; the verify op only when its own checks passed; and the combined
        #    kernel database stays out of every later server when the decode checks only passed without it
        details["mode"] = GDN_MODE_FUSED
        details["env"] = {GDN_VERIFY_ENV: "1" if verify_ok else "0"}
        if details["combined_kernel_db"] == COMBINED_DB_DISABLED_BY_G1B:
            details["env"][serve.COMBINED_KERNEL_DB_ENV] = "0"
        return finish(GateStatus.PASSED)
    except subprocess.TimeoutExpired:
        return finish(GateStatus.TIMED_OUT, "%s did not finish in %s s" % (step, timeout_s))
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return finish(GateStatus.FAILED)


def gdn_fallback_lines(log_text: str) -> list[str]:
    """The lines of a server log that match GDN_FALLBACK_RE, stripped, in order: every line that says a
    fused kernel (GDN decode, GDN verify, hyper-connection, MoE glue) did not run."""
    return [ln.strip() for ln in (log_text or "").splitlines() if GDN_FALLBACK_RE.search(ln)]


def fallback_family(line: str) -> str:
    """Whose line a fallback line is: FALLBACK_HC (the fused hyper-connection kernels), FALLBACK_MOE (the
    MoE glue kernels) or FALLBACK_GDN (anything else). Every line has exactly one family: a line that
    names two (a reason that quotes the other kernels' line) belongs to the one whose message starts
    first."""
    hc, moe = HC_FALLBACK_RE.search(line), MOE_FALLBACK_RE.search(line)
    if hc and moe:
        return FALLBACK_HC if hc.start() < moe.start() else FALLBACK_MOE
    return FALLBACK_HC if hc else FALLBACK_MOE if moe else FALLBACK_GDN


def hc_fallback_lines(log_text: str) -> list[str]:
    """The lines of `gdn_fallback_lines` that are about the fused hyper-connection kernels."""
    return [ln for ln in gdn_fallback_lines(log_text) if fallback_family(ln) == FALLBACK_HC]


def moe_fallback_lines(log_text: str) -> list[str]:
    """The lines of `gdn_fallback_lines` that are about the MoE glue kernels."""
    return [ln for ln in gdn_fallback_lines(log_text) if fallback_family(ln) == FALLBACK_MOE]


HC_FUSED_ENV = "VLLM_HPU_HC_FUSED"
LAZY_ACC_PAR_ENV = "PT_HPU_LAZY_ACC_PAR_MODE"  # the lazy bridge's accumulation thread: 0 off, what custom ops need
HC_OP_TEST = "tests/unit_tests/hc_fused/test_hpu_op.py"
HC_OP_LOG = "hc_hpu_op.log"
HC_OP_TIMEOUT_S = 600.0  # 95 tests, each at least one graph compile; the limit of the GDN op tests
HC_KERNEL_LIB = "libhc_fused_gaudi2.so"
HC_OPS_GLOB = "hc_fused_ops*.so"
HC_KERNEL_LIBRARY_ENV = "HC_FUSED_GAUDI_KERNEL_LIBRARY"
HC_NATIVE_LIBRARY_ENV = "HC_FUSED_GAUDI_NATIVE_LIBRARY"
HC_NOT_INSTALLED = "the fused hyper-connection library is not installed: "
# The statuses of a kernel card test (the hyper-connection kernels' and the MoE glue kernels').
HC_PASSED = "passed"
HC_FAILED = "failed"
HC_SKIPPED = "skipped"
HC_NOT_RUN = "not_run"

# The MoE glue kernels live in the library and the op extension of the fused hyper-connection kernels
# (HC_KERNEL_LIB, HC_OPS_GLOB, found through the same two variables): only the flag, the test file, its
# log and its limit are their own.
MOE_GLUE_ENV = "VLLM_HPU_MOE_GLUE"
MOE_OP_TEST = "tests/unit_tests/hc_fused/test_moe_hpu_op.py"
MOE_OP_LOG = "moe_hpu_op.log"
MOE_OP_TIMEOUT_S = 900.0  # about 90 cases, two of them large
MOE_NOT_INSTALLED = "the library of the MoE glue kernels (the fused hyper-connection library) is not installed: "


def hc_fused(plugin_root: Path, out_dir: Path, python: str = sys.executable, skip: str = None,
             combined_db_off: bool = False) -> dict:
    """The fused hyper-connection kernels' card test, run by the orchestrator right after G1b's own steps
    and recorded as G1b's details["hc_fused"].

    It runs HC_OP_TEST with pytest on the card in lazy mode (PT_HPU_LAZY_MODE=1), from the plugin tree
    with the installed libraries by path (HC_FUSED_GAUDI_KERNEL_LIBRARY, HC_FUSED_GAUDI_NATIVE_LIBRARY,
    and GDN_FUSED_GAUDI_KERNEL_LIBRARY so the loader finds the combined kernel database next to it), in
    its own process group (run_step) with its own limit HC_OP_TIMEOUT_S and its own log out_dir/HC_OP_LOG.
    `combined_db_off` puts FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 into the step when the servers of this
    run will not load the combined database either. The step runs with PT_HPU_LAZY_ACC_PAR_MODE=0 unless
    the operator's environment sets the variable ("lazy_acc_par_mode" records the value).

    Returns {"status", "reason", "hpu_op", ...}: "passed" (exit 0, at least one test passed, none
    skipped, failed or in error), "failed" (anything else the test did, a skip and a timeout included),
    "skipped" (`skip` names why it was left out, or the library is not installed) or "not_run" (the test
    file is not in the bundle, or the step could not start). The result is recorded only: it is not a
    gate status, and nothing here reads or changes what G1b decided about the fused GDN kernel. The
    orchestrator turns the kernels on for the servers only when the operator asked and this passed.
    Never raises."""
    return _kernel_card_test(plugin_root, out_dir, python, skip, combined_db_off, HC_OP_TEST, HC_OP_LOG,
                             HC_OP_TIMEOUT_S, HC_NOT_INSTALLED)


def moe_glue(plugin_root: Path, out_dir: Path, python: str = sys.executable, skip: str = None,
             combined_db_off: bool = False) -> dict:
    """The MoE glue kernels' card test, run by the orchestrator right after the hyper-connection kernels'
    (`hc_fused`) and recorded as G1b's details["moe_glue"].

    The same step with MOE_OP_TEST, its own limit MOE_OP_TIMEOUT_S and its own log out_dir/MOE_OP_LOG,
    next to HC_OP_LOG: the kernels are in the hyper-connection library and op extension, so the installed
    files, the variables that name them and every rule of `hc_fused` are the same, and so are the
    returned keys and statuses ("passed", "failed", "skipped", "not_run"; a skip inside the test file
    is a failure, a test file that is not in the bundle is "not_run"). Recorded only: the orchestrator
    turns the kernels on for the servers only when the operator asked and this passed. Never raises."""
    return _kernel_card_test(plugin_root, out_dir, python, skip, combined_db_off, MOE_OP_TEST, MOE_OP_LOG,
                             MOE_OP_TIMEOUT_S, MOE_NOT_INSTALLED)


def _kernel_card_test(plugin_root: Path, out_dir: Path, python: str, skip: str, combined_db_off: bool, op_test: str,
                      op_log: str, timeout_s: float, not_installed: str) -> dict:
    """One card test of the kernels in the fused hyper-connection library: what `hc_fused` and `moe_glue`
    do, for the test file `op_test`, the log `op_log`, the limit `timeout_s` and the words `not_installed`
    that open the reason when the library is missing. Neither plugin flag reaches the test. Never raises."""
    started = time.monotonic()
    test_name = Path(op_test).name
    result: dict = {"status": HC_NOT_RUN, "reason": "", "hpu_op": None}
    if skip:
        result.update(status=HC_SKIPPED, reason=skip)
        return result
    try:
        plugin_root = Path(plugin_root)
        out_dir = Path(out_dir)
        if not (plugin_root / op_test).is_file():
            result["reason"] = "%s is not in the bundle" % op_test
            return result
        out_dir.mkdir(parents=True, exist_ok=True)

        # the installed library folder, as G1b finds it (the bundle's source tree has an empty lib/)
        lookup = run_step([str(python), "-c", GDN_LIB_LOOKUP_CODE], tempfile.gettempdir(), BUILD_TIMEOUT_S)
        if lookup.returncode != 0:
            result.update(status=HC_SKIPPED, reason=not_installed + "the interpreter exited %d: %s" %
                          (lookup.returncode, (lookup.stderr or "").strip()[:200]))
            return result
        lib_dir = Path(_last_line(lookup.stdout))
        kernel = lib_dir / HC_KERNEL_LIB
        ops_files = sorted(lib_dir.glob(HC_OPS_GLOB))
        if not kernel.is_file():
            result.update(status=HC_SKIPPED, reason=not_installed + "%s missing in %s" % (HC_KERNEL_LIB, lib_dir))
            return result
        if len(ops_files) != 1:
            result.update(status=HC_SKIPPED, reason=not_installed + "%d %s files in %s, expected 1" %
                          (len(ops_files), HC_OPS_GLOB, lib_dir))
            return result

        env = dict(os.environ)
        env["PYTHONPATH"] = str(plugin_root)
        env[HC_KERNEL_LIBRARY_ENV] = str(kernel)
        env[HC_NATIVE_LIBRARY_ENV] = str(ops_files[0])
        if (lib_dir / GDN_KERNEL_LIB).is_file():
            env["GDN_FUSED_GAUDI_KERNEL_LIBRARY"] = str(lib_dir / GDN_KERNEL_LIB)
        env[GDN_LAZY_ENV] = "1"  # the bridge mode of every server
        # A custom op is not safe next to the lazy bridge's accumulation thread, and the loader refuses the
        # kernels without this at 0 (flashinfer_gaudi._native.lazy_accumulation_reason). The plugin sets it for
        # a server that asks for the kernels; the test has no flag, so the step sets it. An operator's own
        # value stays, and reaches the servers the same way.
        env.setdefault(LAZY_ACC_PAR_ENV, "0")
        result["lazy_acc_par_mode"] = env[LAZY_ACC_PAR_ENV]
        for flag in (HC_FUSED_ENV, MOE_GLUE_ENV):  # the test loads the kernels itself
            env.pop(flag, None)
        env.pop(serve.RECIPE_CACHE_ENV, None)  # a recipe cache would block the colon-list registration
        if combined_db_off:
            env[serve.COMBINED_KERNEL_DB_ENV] = "0"
        if not (lib_dir / serve.COMBINED_KERNEL_DB).is_file():
            result["combined_kernel_db"] = COMBINED_DB_NOT_INSTALLED
        elif env.get(serve.COMBINED_KERNEL_DB_ENV, "1").strip() == "0":
            result["combined_kernel_db"] = COMBINED_DB_SWITCHED_OFF
        else:
            result["combined_kernel_db"] = COMBINED_DB_ACTIVE
        result["log"] = op_log

        cmd = [str(python), "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", op_test]
        timed_out = False
        try:
            proc = run_step(cmd, plugin_root, timeout_s, env)
            exit_code, text = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired as e:
            timed_out, exit_code, text = True, None, (e.output or "") + (e.stderr or "")
        with open(out_dir / op_log, "w", encoding="utf-8") as f:
            f.write(text)
        counts = _gdn_pytest_summary(text)
        result["hpu_op"] = {"exit": exit_code, "passed": counts["passed"], "skipped": counts["skipped"],
                            "failed": counts["failed"], "error": counts["error"], "timed_out": timed_out}
        result["failed_tests"] = _pytest_failed_tests(text)
        # which of the failures are the lazy bridge's empty graph input, and whether its producer was queued
        result["empty_inputs"] = lazy_graph.summary(text)
        if timed_out:
            result.update(status=HC_FAILED, reason="%s did not finish in %s s" % (test_name, timeout_s))
        elif (exit_code == 0 and counts["passed"] > 0 and counts["skipped"] == 0 and counts["failed"] == 0
              and counts["error"] == 0):
            result["status"] = HC_PASSED
        else:  # a module skip means the extension or its kernel database did not load: a failure
            skipped_only = counts["skipped"] > 0 and counts["failed"] == 0 and counts["error"] == 0
            result.update(status=HC_FAILED, reason="%s %s" % (test_name, "skipped" if skipped_only else "failed"))
    except Exception as e:  # noqa: BLE001 - recorded only: it never raises
        result.update(status=HC_NOT_RUN, reason="%s: %s" % (type(e).__name__, e))
    result["seconds"] = time.monotonic() - started
    return result


TINY_ERROR_TEXT_LIMIT = 300
TINY_NOT_SENT = "not sent: an earlier prompt timed out"


TINY_NEAR_TIE_MARGIN = 0.002  # logits; one bfloat16 step for values in [0.25, 0.5)


def _tiny_near_tie(prompt: dict, got: list, exp: list, first_diff) -> float | None:
    """The reference's logit margin when a full-length answer differs from the first token on and that
    first token is the reference's second choice within TINY_NEAR_TIE_MARGIN of its first; else None."""
    top5 = prompt.get("last_prefill_top5") or {}
    ids, logits = top5.get("ids") or [], top5.get("logits") or []
    if first_diff != 0 or not got or len(got) != len(exp) or len(ids) < 2 or len(logits) < 2 or got[0] != ids[1]:
        return None
    margin = float(logits[0]) - float(logits[1])
    return round(margin, 6) if 0.0 <= margin <= TINY_NEAR_TIE_MARGIN else None


def tiny_compare(base_url: str, reference_path: Path) -> tuple[GateStatus, dict]:
    """Stage 2a: the served tiny checkpoint must reproduce reference.json 16/16 on every prompt.

    The prompts go out shortest first. Every prompt gets one entry in details["prompts"] (in the order
    sent; "index" is its place in reference.json): a compared answer has "match", "expected",
    "first_diff" and "full_match"; a request that failed has "error" (the exception text) and
    "full_match" False, and the loop goes on, so the record always shows which lengths passed. Only a
    timeout stops the sending: the remaining prompts are recorded as not sent. The gate fails when any
    prompt fails (TIMED_OUT when a request timed out) and details["reason"] names the shortest failing
    prompt length."""
    details: dict = {"prompts": [], "n_prompts": 0}
    try:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        prompts = sorted(enumerate(reference["prompts"]), key=lambda item: len(item[1]["prompt_ids"]))
        details["n_prompts"] = len(prompts)
        failed = []  # (prompt_tokens, what went wrong), in the order sent
        near_ties = 0  # prompts where the card took the reference's other first token of a tie
        timed_out = False
        for i, p in prompts:
            entry: dict = {"index": i, "prompt_tokens": len(p["prompt_ids"])}
            details["prompts"].append(entry)
            if timed_out:
                entry.update(full_match=False, error=TINY_NOT_SENT)
                failed.append((entry["prompt_tokens"], TINY_NOT_SENT))
                continue
            try:
                resp = _post_json(
                    base_url, "/v1/completions", {
                        "model": "tiny",
                        "prompt": p["prompt_ids"],
                        "max_tokens": TINY_MAX_TOKENS,
                        "temperature": 0,
                        "return_token_ids": True,
                    }, HTTP_TIMEOUT_S)
                choice = (resp.get("choices") or [{}])[0]
                got = choice.get("token_ids") or []
            except Exception as e:  # noqa: BLE001 - one prompt's failure is recorded, the others still run
                timed_out = isinstance(e, TimeoutError)
                # _post_json's own texts ("HTTP 400 from ...", "request to ... timed out") need no type name
                text = str(e) if type(e) in (TimeoutError, RuntimeError) else "%s: %s" % (type(e).__name__, e)
                entry.update(full_match=False, error=text[:TINY_ERROR_TEXT_LIMIT])
                failed.append((entry["prompt_tokens"], entry["error"]))
                continue
            exp = p["generated_ids"]
            match = sum(1 for a, b in zip(got, exp) if a == b)
            first_diff = next((j for j, (a, b) in enumerate(zip(got, exp)) if a != b), None)
            if first_diff is None and len(got) != len(exp):
                first_diff = min(len(got), len(exp))
            full = got == exp
            entry.update(match=match, expected=len(exp), first_diff=first_diff, full_match=full)
            near_tie = _tiny_near_tie(p, got, exp, first_diff)
            if near_tie is not None:
                # The reference itself is undecided at the first token: its best two logits are closer
                # than one bfloat16 step, so a card that picks the other one is not wrong (on 2026-11-20
                # two of six reference prompts had margins of 0.0 and 0.0009). Recorded, not failed.
                entry.update(near_tie=True, near_tie_margin=near_tie, got=got[:TINY_MAX_TOKENS],
                             want=exp[:TINY_MAX_TOKENS])
                near_ties += 1
            elif not full:
                # What the server really answered: an empty answer, a near-tie (the first token is another
                # of the reference's top-5 ids for that position) and a wrong model all read "0 of 16
                # tokens match, first difference at token 0" otherwise.
                top5 = ((p.get("last_prefill_top5") or {}).get("ids") or [])
                entry.update(got=got[:TINY_MAX_TOKENS], got_tokens=len(got), want=exp[:TINY_MAX_TOKENS],
                             finish_reason=choice.get("finish_reason"),
                             first_token_reference_rank=(top5.index(got[0]) if got and got[0] in top5 else None))
                failed.append((entry["prompt_tokens"],
                               "%d of %d tokens match, first difference at token %s (the server sent %d tokens)" %
                               (match, len(exp), first_diff, len(got))))
        details["near_ties"] = near_ties
        if failed:
            details["reason"] = "%d of %d prompts failed; the shortest failing prompt has %d tokens: %s" % (
                len(failed), len(prompts), failed[0][0], failed[0][1])
            return (GateStatus.TIMED_OUT if timed_out else GateStatus.FAILED), details
        return GateStatus.PASSED, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details


def first_token(base_url: str, model: str, timeout_s: float = HTTP_TIMEOUT_S) -> tuple[GateStatus, dict]:
    """The proof of life of a server that just started: one short completion (FIRST_TOKEN_PROMPT,
    FIRST_TOKEN_MAX_TOKENS tokens, temperature 0), not streamed.

    details has "prompt", "max_tokens", "timeout_s", "text", "tokens" (the answer's completion_tokens,
    None when the server reports no usage) and "seconds" (the wall time of the whole request, which
    includes the first graph compile when the warmup was skipped). PASSED when the answer has text or
    tokens; FAILED with "reason" when it has neither, or with "error" when the request failed;
    TIMED_OUT when it ran into `timeout_s`. Never raises."""
    details: dict = {"prompt": FIRST_TOKEN_PROMPT, "max_tokens": FIRST_TOKEN_MAX_TOKENS, "timeout_s": timeout_s,
                     "text": None, "tokens": None, "seconds": None}
    started = time.monotonic()
    try:
        resp = _post_json(base_url, "/v1/completions", {
            "model": model,
            "prompt": FIRST_TOKEN_PROMPT,
            "max_tokens": FIRST_TOKEN_MAX_TOKENS,
            "temperature": 0,
        }, timeout_s)
        details["seconds"] = round(time.monotonic() - started, 3)
        text = (resp.get("choices") or [{}])[0].get("text")
        details["text"] = "" if text is None else text
        usage = resp.get("usage")
        details["tokens"] = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if not details["text"] and not details["tokens"]:
            details["reason"] = "the server answered without a token"
            return GateStatus.FAILED, details
        return GateStatus.PASSED, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["seconds"] = round(time.monotonic() - started, 3)
        details["error"] = str(e) if type(e) in (TimeoutError, RuntimeError) else "%s: %s" % (type(e).__name__, e)
        return (GateStatus.TIMED_OUT if isinstance(e, TimeoutError) else GateStatus.FAILED), details


def coherence(base_url: str,
              model: str,
              needle_messages: list,
              request_timeout_s: float = HTTP_TIMEOUT_S) -> tuple[GateStatus, dict]:
    """Stage 2b: the 8 short-prompt coherence check plus the 120K-token needle.

    request_timeout_s is the per-request HTTP timeout; the orchestrator raises it above HTTP_TIMEOUT_S
    when the servers skip the bucket warmup and the first prompts pay the graph compilation."""
    details: dict = {"answers": [], "needle": None}
    try:
        failed = False
        for i, prompt in enumerate(COHERENCE_PROMPTS):
            resp = _post_json(base_url,
                              "/v1/completions", {
                                  "model": model,
                                  "prompt": prompt,
                                  "max_tokens": COHERENCE_MAX_TOKENS,
                                  "temperature": 0,
                              }, request_timeout_s)
            text = (resp.get("choices") or [{}])[0].get("text")
            text = "" if text is None else text
            words = text.split()
            most_count = Counter(words).most_common(1)[0][1] if words else 0
            entry = {
                "index": i,
                "prompt": prompt,
                "answer": text[:500],
                "empty": not text.strip(),
                "n_words": len(words),
                "most_common_word_share": (most_count / len(words)) if words else 0.0,
                "loop": _is_repetition_loop(text),
            }
            if i == COHERENCE_JSON_PROMPT_INDEX:
                entry["json_ok"] = _has_json_object(text)
            entry["ok"] = not (entry["empty"] or entry["loop"] or not entry.get("json_ok", True))
            details["answers"].append(entry)
            if not entry["ok"]:
                failed = True
        if failed:
            details["reason"] = "coherence prompts failed (empty answer, repetition loop or unparsable JSON)"
            return GateStatus.FAILED, details

        messages = [dict(m) for m in (needle_messages or [])]
        user_idx = next((j for j in range(len(messages) - 1, -1, -1) if messages[j].get("role") == "user"), None)
        if user_idx is None:
            details["reason"] = "needle messages have no user message"
            return GateStatus.FAILED, details
        content = messages[user_idx].get("content") or ""
        messages[user_idx]["content"] = _insert_needle(content) + NEEDLE_QUESTION
        resp = _post_json(
            base_url, "/v1/chat/completions", {
                "model": model,
                "messages": messages,
                "max_tokens": NEEDLE_MAX_TOKENS,
                "temperature": 0,
                "chat_template_kwargs": {
                    "enable_thinking": False
                },
            }, request_timeout_s)
        answer = (resp.get("choices") or [{}])[0].get("message", {}).get("content")
        answer = "" if answer is None else answer
        hit = NEEDLE_CODE in answer
        details["needle"] = {"answer": answer[:500], "hit": hit}
        if hit:
            return GateStatus.PASSED, details
        details["reason"] = "needle %s not in the answer" % NEEDLE_CODE
        return GateStatus.FAILED, details
    except TimeoutError as e:
        details["error"] = str(e)
        return GateStatus.TIMED_OUT, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details


CARD_TEST_FILES = ("tests/unit_tests/test_defragmentation.py", "tests/unit_tests/ops/test_hpu_fused_moe.py")
CARD_TESTS_TIMEOUT_S = 600.0


def card_tests(plugin_root: Path, out_dir: Path, python: str = sys.executable) -> tuple[GateStatus, dict]:
    """The HPU-only unit tests (RUK-3106): the files of CARD_TEST_FILES run with pytest on the card in
    lazy mode (PT_HPU_LAZY_MODE=1), CARD_TESTS_TIMEOUT_S each, through run_step.

    Recorded only: the orchestrator keeps the run going whatever this returns. Each file's output goes
    to out_dir/<file stem>.log (on a timeout, the partial output if the interpreter left one) and one
    entry {"file", "exit", "timed_out", "passed", "failed", "skipped", "error"} per file lands in
    details["files"]. PASSED when every file exited 0 with failed == error == 0, FAILED otherwise.
    The gate never raises."""
    details: dict = {"files": []}
    try:
        plugin_root = Path(plugin_root)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(plugin_root), PT_HPU_LAZY_MODE="1")
        all_ok = True
        for file in CARD_TEST_FILES:
            entry = {
                "file": file,
                "exit": None,
                "timed_out": False,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "error": 0,
            }
            text = ""
            try:
                proc = run_step([python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", file], plugin_root,
                                CARD_TESTS_TIMEOUT_S, env)
                entry["exit"] = proc.returncode
                text = (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired as e:
                entry["timed_out"] = True
                text = (e.output or "") + (e.stderr or "")
            with open(out_dir / (Path(file).stem + ".log"), "w", encoding="utf-8") as f:
                f.write(text)
            entry.update(_gdn_pytest_summary(text))
            details["files"].append(entry)
            if not (entry["exit"] == 0 and entry["failed"] == 0 and entry["error"] == 0):
                all_ok = False
        if all_ok:
            return GateStatus.PASSED, details
        details["reason"] = "a card test file failed or timed out"
        return GateStatus.FAILED, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details
