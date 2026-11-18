# SPDX-License-Identifier: Apache-2.0
"""Fail-fast gates of the Flash-Next rental (plan M4, RUK-2709).

One function per gate: preflight (cards, checkpoint, FP8 layout, disk),
build (installed vLLM / vllm_gaudi / flashinfer_gaudi), the fused GDN
decode kernel and the GDN verify op on the card (G1b: the kernel's own
checks, then the verify op's checks, and the GDN decode and verify
modes every later server must use), tiny TP8 token agreement,
capacity from the startup line, and coherence with a 120K-token needle.
Every gate returns ``(GateStatus, details)`` and never raises for an
expected failure: the exception text goes into ``details["error"]``.
Python 3.10 standard library only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

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
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=HLSMI_TIMEOUT_S)
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
            proc = subprocess.run([str(python), "-c", code],
                                  cwd=tempfile.gettempdir(),  # never the repository root
                                  capture_output=True,
                                  text=True,
                                  timeout=BUILD_TIMEOUT_S)
            if proc.returncode != 0:
                details["reason"] = "%s: exit %d: %s" % (name, proc.returncode, (proc.stderr or "").strip()[:500])
                return GateStatus.FAILED, details
            out = proc.stdout.strip()
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
GDN_VERIFY_OP_TEST = "tests/unit_tests/gdn_core_decode/test_hpu_verify_op.py"
GDN_FALLBACK_RE = re.compile(r"has no HPU kernel|GDN core (decode|verify) .* is off for")
GDN_OP_TIMEOUT_S = 1800.0      # test_hpu_op.py and test_hpu_verify_op.py
GDN_REPLAY_TIMEOUT_S = 1800.0  # each replay run
GDN_BENCH_TIMEOUT_S = 1200.0

GDN_LIB_LOOKUP_CODE = ("import flashinfer_gaudi, pathlib; "
                       "print(pathlib.Path(flashinfer_gaudi.__file__).parent / 'lib')")
GDN_KERNEL_LIB = "libgdn_fused_gaudi2.so"
GDN_MODE_FUSED = 1
GDN_MODE_PYTORCH = 0
GDN_VERIFY_ON = 1
GDN_VERIFY_OFF = 0
GDN_FALLBACK_ENV = {GDN_MODE_ENV: "0", GDN_VERIFY_ENV: "0"}


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


def _gdn_verify_not_run(reason: str) -> dict:
    """The "verify" details of a run in which the verify checks did not run at all."""
    return {"status": "not_run", "reason": reason, "hpu_op": None, "replay": []}


def fused_gdn(plugin_root: Path, out_dir: Path, python: str = sys.executable) -> tuple[GateStatus, dict]:
    """G1b. details always has "mode" (1 = the fused kernel stays on, 0 = later servers run the PyTorch path),
    "verify_mode" (1 = the GDN verify op's card checks all passed, 0 = the op stays off) and "env" (the GDN
    flags of every later server, added by the orchestrator: the decode entry as before, and
    "VLLM_HPU_GDN_CORE_VERIFY" = "1" only when every verify check passed, else an explicit "0")."""
    started = time.monotonic()
    details: dict = {"mode": GDN_MODE_PYTORCH, "env": dict(GDN_FALLBACK_ENV), "verify_mode": GDN_VERIFY_OFF}

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
        lookup = subprocess.run([str(python), "-c", GDN_LIB_LOOKUP_CODE],
                                cwd=tempfile.gettempdir(),  # never the repository root
                                capture_output=True,
                                text=True,
                                timeout=timeout_s)
        if lookup.returncode != 0:
            return finish(GateStatus.SKIPPED, "the fused GDN library is not installed: the interpreter exited %d: %s" %
                          (lookup.returncode, (lookup.stderr or "").strip()[:200]))
        lib_dir = Path(lookup.stdout.strip())
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
        env.pop(GDN_MODE_ENV, None)
        env.pop(GDN_VERIFY_ENV, None)

        def run(cmd: list, log_name: str, limit: float, append: bool = False):
            nonlocal step, timeout_s
            step = " ".join(str(arg) for arg in cmd[1:])
            timeout_s = limit
            proc = subprocess.run([str(arg) for arg in cmd],
                                  cwd=str(plugin_root),
                                  env=env,
                                  capture_output=True,
                                  text=True,
                                  timeout=limit)
            text = (proc.stdout or "") + (proc.stderr or "")
            with open(out_dir / log_name, "a" if append else "w", encoding="utf-8") as f:
                f.write(text)
            return proc, text

        # 3-4. the decode checks: a verify run is only attempted when they all pass
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
            details["failed_tests"] = [ln for ln in text.splitlines() if ln.startswith("FAILED ")]
            op_ok = (proc.returncode == 0 and counts["passed"] > 0 and counts["skipped"] == 0 and
                     counts["failed"] == 0 and counts["error"] == 0)
            if not op_ok:
                details["verify"] = _gdn_verify_not_run("decode checks failed")
                if counts["skipped"] > 0:
                    return finish(GateStatus.FAILED, "test_hpu_op.py skipped")
                return finish(GateStatus.FAILED, "test_hpu_op.py failed (exit %d)" % proc.returncode)

            # 4. the 1,000-step graph replay against eager, two cases, in one appended log
            details["replay"] = []
            for case in ("tp8_b16", "tp8_b64"):
                cmd = [python, "tools/flash_next/gdn_core_decode_replay.py", "--steps", "1000"]
                if case != "tp8_b16":
                    cmd += ["--case", case]
                proc, _ = run(cmd, "replay.log", GDN_REPLAY_TIMEOUT_S, append=True)
                details["replay"].append({"case": case, "exit": proc.returncode})
                if proc.returncode != 0:
                    details["verify"] = _gdn_verify_not_run("decode checks failed")
                    return finish(GateStatus.FAILED,
                                  "gdn_core_decode_replay.py (%s) failed (exit %d)" % (case, proc.returncode))
        except subprocess.TimeoutExpired:
            details["verify"] = _gdn_verify_not_run("decode checks failed")
            raise

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
            except subprocess.TimeoutExpired:
                verify_ok = False
                verify["status"] = "timed_out"
                verify["reason"] = "test_hpu_verify_op.py did not finish in %s s" % GDN_OP_TIMEOUT_S
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
                    verify["replay"].append({"case": case, "exit": proc.returncode})
                    if proc.returncode != 0:
                        verify_ok = False
                        verify["status"] = "failed"
                        verify["reason"] = ("gdn_core_decode_replay.py (verify %s) failed (exit %d)" %
                                            (case, proc.returncode))
                        break
        details["verify"] = verify
        details["verify_mode"] = GDN_VERIFY_ON if verify_ok else GDN_VERIFY_OFF

        # 5. the speed measurement: recorded only, its result never changes the status
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

        # 6. the fused kernel stays on; the verify op only when its own checks passed
        details["mode"] = GDN_MODE_FUSED
        details["env"] = {GDN_VERIFY_ENV: "1" if verify_ok else "0"}
        return finish(GateStatus.PASSED)
    except subprocess.TimeoutExpired:
        return finish(GateStatus.TIMED_OUT, "%s did not finish in %s s" % (step, timeout_s))
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return finish(GateStatus.FAILED)


def gdn_fallback_lines(log_text: str) -> list[str]:
    """The lines of a server log that match GDN_FALLBACK_RE, stripped, in order."""
    return [ln.strip() for ln in (log_text or "").splitlines() if GDN_FALLBACK_RE.search(ln)]


def tiny_compare(base_url: str, reference_path: Path) -> tuple[GateStatus, dict]:
    """Stage 2a: the served tiny checkpoint must reproduce reference.json 16/16 on every prompt."""
    details: dict = {"prompts": [], "n_prompts": 0}
    try:
        reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
        prompts = reference["prompts"]
        details["n_prompts"] = len(prompts)
        bad = 0
        for i, p in enumerate(prompts):
            resp = _post_json(
                base_url, "/v1/completions", {
                    "model": "tiny",
                    "prompt": p["prompt_ids"],
                    "max_tokens": TINY_MAX_TOKENS,
                    "temperature": 0,
                    "return_token_ids": True,
                }, HTTP_TIMEOUT_S)
            got = (resp.get("choices") or [{}])[0].get("token_ids") or []
            exp = p["generated_ids"]
            match = sum(1 for a, b in zip(got, exp) if a == b)
            first_diff = next((j for j, (a, b) in enumerate(zip(got, exp)) if a != b), None)
            if first_diff is None and len(got) != len(exp):
                first_diff = min(len(got), len(exp))
            full = got == exp
            details["prompts"].append({
                "index": i,
                "prompt_tokens": len(p["prompt_ids"]),
                "match": match,
                "expected": len(exp),
                "first_diff": first_diff,
                "full_match": full,
            })
            if not full:
                bad += 1
        if bad:
            details["reason"] = "%d of %d prompts do not match %d/%d" % (bad, len(prompts), TINY_MAX_TOKENS,
                                                                          TINY_MAX_TOKENS)
            return GateStatus.FAILED, details
        return GateStatus.PASSED, details
    except TimeoutError as e:
        details["error"] = str(e)
        return GateStatus.TIMED_OUT, details
    except Exception as e:  # noqa: BLE001 - the gate never raises
        details["error"] = "%s: %s" % (type(e).__name__, e)
        return GateStatus.FAILED, details


def coherence(base_url: str, model: str, needle_messages: list) -> tuple[GateStatus, dict]:
    """Stage 2b: the 8 short-prompt coherence check plus the 120K-token needle."""
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
                              }, HTTP_TIMEOUT_S)
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
            }, HTTP_TIMEOUT_S)
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
