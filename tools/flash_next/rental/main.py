# SPDX-License-Identifier: Apache-2.0
"""The rental test orchestrator: gates, variants A/B/C(/P), the report.

What the operator sees on the console: one line per finished gate and variant (a failure with the last
error lines of its server log and the log to read), one `first token` line per variant server, and the
final `RUN <id>: PASSED|FAILED at <phase>`. A server that exits before it is healthy is FAILED with its
exit code, never a timeout. A failed gate or variant leaves pip-freeze.txt and hl-smi.txt in the run
folder. `--from PHASE` resumes a run that has a manifest.json; `--quick-gates` and `--no-verify-probe`
leave out steps that are recorded only; `--load-read-timeout` and `--warm-request-timeout` replace two
request timeouts. The combined kernel database has an automatic fallback at gate G1b and at the tiny
gate (`gate_tiny`), and the operator's FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 switches it off.

`--hc-fused` asks for the fused hyper-connection kernels (VLLM_HPU_HC_FUSED=1, off by default). Their card
test runs right after G1b's own steps and is recorded in G1b's details; it never changes G1b's status or
the GDN flags. The servers get the kernels only when the operator asked and the card test passed, every
other server gets an explicit VLLM_HPU_HC_FUSED=0, and a tiny gate that fails with the kernels runs once
more without them. The manifest's `hc_fused` and the summary's `HC fused:` line record the outcome.

`--moe-glue` asks for the MoE glue kernels (VLLM_HPU_MOE_GLUE=1, off by default; they live in the
hyper-connection library) in the same way: their card test runs right after the hyper-connection one and
is recorded in G1b's details, the servers get them only when the operator asked and that test passed, and
every other server gets an explicit VLLM_HPU_MOE_GLUE=0. The tiny gate's retry is one for both: a gate
that fails with either kind of kernels runs once more without both. The manifest's `moe_glue` and the
summary's `MoE glue:` line record the outcome.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from tools.flash_next.rental import agent_load, gates, report, schema, serve, workload
from tools.flash_next.rental.schema import (
    FIRST_TOKEN_FILE,
    METRICS_FILE,
    REQUESTS_FILE,
    STARTUP_FILE,
    STATUS_FILE,
    GateStatus,
    OnFail,
)

VARIANTS_FILE = Path(__file__).resolve().parent / "variants.json"
GATE_PHASES = ("preflight", "build", "fused_gdn", "tiny", "card_tests")
# What a manifest and a status file keep of an environment: the plugin, bridge, kernel-loader and Hub
# variables, plus the names of ENV_KEYS; a value whose name looks like a secret is never written.
ENV_PREFIXES = ("VLLM_", "HABANA_", "PT_", "HF_", "FLASHINFER_", "GC_", "GDN_", "HC_")
ENV_KEYS = ("PYTHONFAULTHANDLER",)
ENV_SECRET_SUFFIXES = ("_TOKEN", "_SECRET", "_PASSWORD", "_API_KEY")  # HF_TOKEN, VLLM_API_KEY
HLSMI_Q_LIMIT = 200 * 1024
SERVER_HEALTH_TIMEOUT_S = 1800.0
COHERENCE_REQUEST_TIMEOUT_SKIP_WARMUP_S = 3600.0
TINY_HEALTH_TIMEOUT_S = 1200.0
TINY_CHECKPOINT_TIMEOUT_S = 900.0
TINY_PORT = 8010
TINY_FIXTURE = "tests/unit_tests/models/flash_next/tiny_tp8_config.json"
TINY_GENERATOR = "tools/flash_next/make_tiny_checkpoint.py"
TINY_BUILD_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
TINY_GDN_HEAD_DIM = 128  # the head dim of the real model's GDN layers and the only one the HPU kernel accepts
TINY_MIN_MODEL_LEN = 512
TINY_DEFAULT_MODEL_LEN = 4096
TINY_ATTEMPT_COMBINED = "attempt-combined"
TINY_ATTEMPT_HC_FUSED = "attempt-hc-fused"
TINY_ATTEMPT_GDN_FUSED = "attempt-gdn-fused"
TINY_ATTEMPT_MOE_GLUE = "attempt-moe-glue"
HC_QUICK_SKIP_REASON = "--quick-gates without --hc-fused"
MOE_QUICK_SKIP_REASON = "--quick-gates without --moe-glue"
LOAD_TIMEOUT_S = 5400.0
LOAD_REQUEST_TIMEOUT_S = 1800.0
LOAD_STAGGER_S = 60.0
WORKLOAD_TURNS = 4
WARM_PASS_MAX_TOKENS = 64
WARM_PASS_FILE_OFFSET = 1
WARM_PASS_REQUEST_TIMEOUT_S = 3600.0
WARM_WORKLOAD_FILE = "warm-workload.json"
WARM_REQUESTS_FILE = "warm-requests.jsonl"
WARM_PASS_FILE = "warm-pass.json"
MONITOR_TERM_WAIT_S = 15.0
CARD_FREE_WAIT_S = 20.0
STUB_STAGGER_S = 0.0
STUB_MONITOR_INTERVAL_S = 0.05
STUB_TOKENS_PER_SECOND = "400"
STUB_PORT_TIMEOUT_S = 30.0
QUICK_GATES_REASON = "--quick-gates"
NO_VERIFY_PROBE_REASON = "--no-verify-probe"
BUNDLE_SHA_RE = re.compile(r"^plugin sha:\s*([0-9a-f]{40})\s*$", re.MULTILINE)
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
# What a failed gate or variant leaves in the run folder: (file, command, timeout in s, open mode).
# pip freeze is the same at every failure and is rewritten; every failure adds its own hl-smi snapshot.
FAILURE_EVIDENCE = (
    ("pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"], 120.0, "w"),
    ("hl-smi.txt", ["hl-smi"], 60.0, "a"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_variants(path: Path = VARIANTS_FILE) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def tiny_checkpoint_command(ckpt: Path) -> list:
    """The command that builds the tiny checkpoint and its CPU reference in `ckpt` (run from the repo root).

    The GDN layers are built at head dim TINY_GDN_HEAD_DIM, the only one the HPU fused kernel accepts:
    with the fixture's own 16 every layer logs "GDN core decode ... is off" and runs the torch path."""
    return [sys.executable, TINY_GENERATOR, "--out", str(ckpt), "--fixture", TINY_FIXTURE,
            "--gdn-head-dim", str(TINY_GDN_HEAD_DIM)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tiny_build_settings() -> dict:
    """What the `build` block of a tiny checkpoint's reference.json must say for the folder to be reused:
    the settings of `tiny_checkpoint_command` and the generator and fixture of this bundle."""
    root = repo_root()
    return {
        "seed": 0,
        "gdn_head_dim": TINY_GDN_HEAD_DIM,
        "fixture_sha256": _sha256(root / TINY_FIXTURE),
        "generator_sha256": _sha256(root / TINY_GENERATOR),
    }


def tiny_checkpoint_stale_reason(ckpt: Path):
    """None when the tiny checkpoint in `ckpt` was built with `tiny_build_settings`; else why it is rebuilt.

    The prompt lengths and the schema version are pinned by the generator's sha256; the reference must
    also agree with itself on them (a file edited by hand or cut short is not reused)."""
    reference = read_json(Path(ckpt) / "reference.json")
    if not isinstance(reference, dict):
        return "no reference.json"
    build = reference.get("build")
    if not isinstance(build, dict):
        return "no build settings in reference.json (an older generator)"
    for key, wanted in tiny_build_settings().items():
        if build.get(key) != wanted:
            return "%s is %r, wanted %r" % (key, build.get(key), wanted)
    try:
        lengths = [len(p["prompt_ids"]) for p in reference["prompts"]]
    except (KeyError, TypeError):
        return "no prompts in reference.json"
    if build.get("prompt_lengths") != lengths:
        return "prompt_lengths %r do not match the prompts in the file %r" % (build.get("prompt_lengths"), lengths)
    if build.get("schema") != reference.get("schema"):
        return "schema %r does not match the file's %r" % (build.get("schema"), reference.get("schema"))
    return None


def tiny_build_env() -> dict:
    """The environment of the tiny checkpoint build: one CPU thread unless the operator set a count.
    The tiny model's tensors are too small for threads to pay off, and with the GDN layers at head dim
    128 the default count made the build 13 times slower on the CPU venv (172 s against 13 s, the
    same weights and the same reference: generated ids, top-5 logits, draft ids)."""
    env = dict(os.environ)
    for name in TINY_BUILD_THREAD_ENV:
        env.setdefault(name, "1")
    return env


def ensure_tiny_checkpoint(ckpt: Path, command: list = None, say=print):
    """Build the tiny checkpoint in `ckpt` unless it is there with the wanted build settings
    (`tiny_checkpoint_stale_reason`); the failure reason, or None. The tiny gate and the smoke test share it."""
    ckpt = Path(ckpt)
    stale = tiny_checkpoint_stale_reason(ckpt)
    if stale is None:
        say("reusing the tiny checkpoint in %s" % ckpt)
        return None
    had_reference = (ckpt / "reference.json").is_file()
    say("building the tiny checkpoint in %s (about a minute)%s" %
        (ckpt, ": the one there is not reused, %s" % stale if had_reference else ""))
    try:
        (ckpt / "reference.json").unlink(missing_ok=True)  # a failed build must not leave the old reference
        proc = gates.run_step(command or tiny_checkpoint_command(ckpt), repo_root(), TINY_CHECKPOINT_TIMEOUT_S,
                              tiny_build_env())
    except subprocess.TimeoutExpired:
        return "tiny checkpoint did not finish in %d s" % int(TINY_CHECKPOINT_TIMEOUT_S)
    except OSError as e:
        return "tiny checkpoint could not start: %s" % e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return "tiny checkpoint failed (exit %d): %s" % (proc.returncode, tail.strip())
    return None


def tiny_model_len(ckpt: Path) -> int:
    """`--max-model-len` of the tiny server: the longest prompt of `ckpt`/reference.json plus the tokens
    the gate generates, rounded up to a power of two and at least TINY_MIN_MODEL_LEN. With no readable
    reference it is TINY_DEFAULT_MODEL_LEN, the tiny fixtures' max_position_embeddings."""
    reference = read_json(Path(ckpt) / "reference.json")
    try:
        longest = max(len(p["prompt_ids"]) for p in reference["prompts"])
        new_tokens = max(int(reference.get("max_new_tokens") or 0), gates.TINY_MAX_TOKENS)
    except (KeyError, TypeError, ValueError):
        return TINY_DEFAULT_MODEL_LEN
    length = TINY_MIN_MODEL_LEN
    while length < longest + new_tokens:
        length *= 2
    return length


def tiny_serve_command(ckpt: Path, port: int = TINY_PORT) -> list:
    """The `vllm serve` command of the tiny gate; tiny_smoke.py starts the same one. Build it after the
    checkpoint: the model length comes from its reference (`tiny_model_len`)."""
    return [
        "vllm",
        "serve",
        str(ckpt),
        "--served-model-name",
        "tiny",
        "--tensor-parallel-size",
        "8",
        "--max-model-len",
        str(tiny_model_len(ckpt)),
        "--max-num-seqs",
        "4",
        "--language-model-only",
        "--trust-remote-code",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]


def server_env(variant_env: dict, gdn_env: dict) -> dict:
    env = dict(variant_env)
    env.update(gdn_env)
    return env


def combined_kernel_db_switched_off() -> bool:
    """True when the operator's kill switch is set: FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0."""
    return os.environ.get(serve.COMBINED_KERNEL_DB_ENV, "1").strip() == "0"


def combined_kernel_db_installed(python: str = sys.executable) -> bool:
    """True when the installed flashinfer_gaudi `lib` folder holds the combined kernel database (RUK-3242):
    one file with Intel's kernels and ours, the only kernel library the bridge accepts next to a recipe cache.
    False when the operator switched it off (FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 in this process's
    environment, which every server inherits): the loader then registers the libraries as a list, and a
    fused server must not get a recipe cache. Never import it here: the lookup is a subprocess, as in the
    G1b gate."""
    if combined_kernel_db_switched_off():
        return False
    try:
        lookup = gates.run_step([python, "-c", gates.GDN_LIB_LOOKUP_CODE], tempfile.gettempdir(), 120)
        if lookup.returncode != 0:
            return False
        lib_dir = Path(gates._last_line(lookup.stdout))
        return (lib_dir / serve.COMBINED_KERNEL_DB).is_file()
    except Exception:  # noqa: BLE001 - no lookup, no combined database: the plan of record is the fallback
        return False


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_status(phase_dir: Path,
                 phase_id: str,
                 status: GateStatus,
                 reason,
                 seconds: float,
                 extra: dict = None) -> None:
    obj = {"id": phase_id, "status": int(status), "reason": reason, "seconds": round(float(seconds), 3)}
    if extra:
        obj.update(extra)
    write_json(phase_dir / STATUS_FILE, obj)


def recorded_env(env: dict) -> dict:
    """The variables of `env` that go into the manifest and the status files (ENV_PREFIXES, ENV_KEYS)."""
    out = {}
    for key, value in env.items():
        if key.startswith(ENV_PREFIXES) or key in ENV_KEYS:
            out[key] = "<redacted>" if key.upper().endswith(ENV_SECRET_SUFFIXES) else value
    return out


def git_commit(root: Path = None):
    """HEAD of the git checkout whose top level is `root` (default: the plugin root), 40 hex characters;
    None when `root` is not one (a bundle has no .git) or git is missing."""
    root = Path(root) if root is not None else repo_root()
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel", "HEAD"], cwd=str(root), capture_output=True,
                             text=True, timeout=30)
        lines = out.stdout.split("\n")
        if out.returncode != 0 or len(lines) < 2 or not GIT_SHA_RE.fullmatch(lines[1].strip()):
            return None
        if os.path.realpath(lines[0]) != os.path.realpath(root):
            return None  # some other checkout around the folder: not this code's commit
        return lines[1].strip()
    except (OSError, subprocess.SubprocessError):
        return None


def bundle_text(root: Path = None):
    """BUNDLE.txt of the rental bundle, which sits next to the plugin folder (/bundle/BUNDLE.txt on the
    node); None outside a bundle."""
    root = Path(root) if root is not None else repo_root()
    try:
        return (root.parent / "BUNDLE.txt").read_text(encoding="utf-8")
    except OSError:
        return None


def bundle_commit():
    """The plugin commit this code is: git's HEAD in a checkout, else the `plugin sha:` line of the
    bundle's BUNDLE.txt, else None."""
    commit = git_commit()
    if commit:
        return commit
    match = BUNDLE_SHA_RE.search(bundle_text() or "")
    return match.group(1) if match else None


def keyed_recipe_cache_dir(base, commit):
    """The recipe cache folder of one bundle: `base`/<first 12 characters of the plugin commit> (`unknown`
    when the commit is not known), so a new bundle gets a fresh folder: the bridge keys cached recipes
    by the GC_KERNEL_PATH file, not by the kernels that file forwards to (RUK-3242). None without `base`."""
    if not base:
        return None
    return "%s/%s" % (str(base).rstrip("/"), (commit or "unknown")[:12])


def write_failure_evidence(run_dir: Path, phase: str, commands=None) -> None:
    """After a failed gate or variant: the installed packages and the cards as they are now, into the run
    folder (FAILURE_EVIDENCE). Best effort: a command that cannot run leaves its error in its file, and
    nothing here ever raises."""
    for name, cmd, timeout_s, mode in (FAILURE_EVIDENCE if commands is None else commands):
        try:
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
                text = out.stdout or ""
                if out.returncode != 0 or (out.stderr or "").strip():
                    text += "[exit %d] %s\n" % (out.returncode, (out.stderr or "").strip())
            except (OSError, subprocess.SubprocessError) as e:
                text = "%s: %s\n" % (type(e).__name__, e)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(Path(run_dir) / name, mode, encoding="utf-8") as fh:
                fh.write("# %s at the failure of %s, %s\n%s%s" % (" ".join(str(c) for c in cmd), phase, stamp, text,
                                                                 "" if text.endswith("\n") else "\n"))
        except Exception:  # noqa: BLE001 - evidence only
            pass


def max_running_sum(path: Path) -> float:
    sums = {}
    if not path.is_file():
        return 0.0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            running = row.get("running")
            if running is None:
                continue
            sums[row.get("t")] = sums.get(row.get("t"), 0.0) + float(running)
    return max(sums.values()) if sums else 0.0


def gdn_summary_lines(run_dir, variant_ids) -> list:
    run_dir = Path(run_dir)
    details = read_json(run_dir / "fused_gdn" / "details.json")
    mode = details.get("mode") if isinstance(details, dict) else None
    reason = details.get("reason") if isinstance(details, dict) else None
    if mode == 1:
        lines = ["GDN decode: fused kernel (G1b passed)"]
    elif mode == 0:
        lines = ["GDN decode: PyTorch path, VLLM_HPU_GDN_CORE_DECODE=0 (%s)" % reason]
    else:
        lines = ["GDN decode: not checked (stub mode)"]
    for vid in variant_ids:
        path = run_dir / vid / "gdn_fallback.txt"
        if not path.is_file():
            continue
        found = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        families = [gates.fallback_family(ln) for ln in found]  # each line counts once, for one family
        hc_count = families.count(gates.FALLBACK_HC)
        moe_count = families.count(gates.FALLBACK_MOE)
        count = len(found) - hc_count - moe_count
        if count:
            lines.append("GDN fallback lines in %s: %d (the kernel did not run on every layer)" % (vid, count))
        if hc_count:
            lines.append("HC fused fallback lines in %s: %d (the hyper-connection kernels did not run on every "
                         "layer)" % (vid, hc_count))
        if moe_count:
            lines.append("MoE glue fallback lines in %s: %d (the MoE glue kernels did not run on every layer)" %
                         (vid, moe_count))
    toolchain = read_json(run_dir / "toolchain.json")  # written by container_entry.sh before the build
    if isinstance(toolchain, dict):
        lines.append("Native kernels: %s (tpc-clang: %s; habanalabs-tools: %s)" % (
            "compiled" if toolchain.get("native_build") == 1 else "NOT compiled, PyTorch GDN path",
            toolchain.get("tpc_clang_version") or "absent",
            toolchain.get("habanalabs_tools_version") or "absent"))
    return lines


class Runner:

    def __init__(self, args, variants: list):
        self.args = args
        self.mode = args.mode
        self.stub = args.mode == "stub"
        self.model = args.model
        self.served_name = args.served_name
        self.results_dir = Path(args.results)
        self.run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        self.run_dir = self.results_dir / self.run_id
        self.variants = variants
        by_id = {v["id"]: v for v in variants}
        ids = [tok.strip() for tok in args.variants.split(",") if tok.strip()]
        if args.with_profile and "P" not in ids:
            ids.append("P")
        unknown = [t for t in ids if t not in by_id]
        if unknown:
            raise ValueError("unknown variant ids: %s" % ", ".join(unknown))
        self.selected = [by_id[t] for t in ids]
        self.from_phase = args.from_phase
        if self.from_phase:  # before anything else: a wrong name or run must not cost a lookup or a folder
            names = [n for _, n in self.phases()]
            if self.from_phase not in names:
                raise ValueError("--from %s is not a phase of this run (phases: %s)" %
                                 (self.from_phase, ", ".join(names)))
            if not args.run_id:
                raise ValueError("--from requires --run-id")
            if not args.dry_run and not isinstance(read_json(self.run_dir / "manifest.json"), dict):
                raise ValueError(
                    "--from %s resumes a run that already has results, and %s has no manifest.json from an "
                    "earlier start (the gates would be skipped and the GDN flags unknown): start this run "
                    "without --from" % (self.from_phase, self.run_dir))
        self.quick_gates = bool(args.quick_gates)
        self.verify_probe = not (args.quick_gates or args.no_verify_probe)
        self.load_read_timeout_s = float(args.load_read_timeout)
        self.warm_request_timeout_s = float(args.warm_request_timeout)
        self.gdn_env = {}
        self.gdn_mode = None
        self.gdn_reason = None
        # The fused hyper-connection kernels: the operator's request (--hc-fused), the card test's result
        # in G1b (None before G1b ran), and who took the kernels out of this run's servers ("tiny").
        self.hc_fused_requested = bool(args.hc_fused)
        self.hc_fused_status = None
        self.hc_fused_reason = None
        self.hc_fused_disabled_by = None
        self.gdn_fused_disabled_by = None  # "tiny" when the tiny gate only passed on the PyTorch GDN path
        # The MoE glue kernels, the same three things: --moe-glue, their card test in G1b, and "tiny".
        self.moe_glue_requested = bool(args.moe_glue)
        self.moe_glue_status = None
        self.moe_glue_reason = None
        self.moe_glue_disabled_by = None
        self.skip_warmup = not args.full_warmup
        self.warm_pass = not args.no_warm_pass
        self.server_timeout_s = float(args.server_timeout)
        self.request_timeout_s = (COHERENCE_REQUEST_TIMEOUT_SKIP_WARMUP_S if self.skip_warmup else
                                  gates.HTTP_TIMEOUT_S)
        self.active = []
        self.stopped_at = None
        self.current_phase = GATE_PHASES[0]
        self.ran_variant_ids = []
        self.start_index = 0
        self.hl_smi_q = ""
        self.bundle_commit = None
        self.workload_info = None
        self.recipe_cache_mb = int(args.recipe_cache_mb)
        self.recipe_cache_dir = keyed_recipe_cache_dir(args.recipe_cache, self.bundle_commit)
        self.recipe_cache_notes = {}
        self.recipe_cache_note = serve.RECIPE_CACHE_NONE
        # The combined kernel database: installed and not switched off by the operator; and who took it
        # out of this run's servers, if anyone ("operator", "g1b" or "tiny").
        self.combined_db_installed = False if self.stub else combined_kernel_db_installed()
        self.combined_db_disabled_by = "operator" if not self.stub and combined_kernel_db_switched_off() else None

    def variant_by_id(self, vid: str):
        for v in self.variants:
            if v["id"] == vid:
                return v
        return None

    def phases(self) -> list:
        phases = [("gate", g) for g in GATE_PHASES]
        phases += [("variant", v["id"]) for v in self.selected]
        phases.append(("report", "report"))
        return phases

    def run(self) -> int:
        names = [n for _, n in self.phases()]
        if self.from_phase:  # the name and the earlier manifest were checked in __init__
            self.start_index = names.index(self.from_phase)
            manifest = read_json(self.run_dir / "manifest.json") or {}
            # the commit of the bundle that runs now (a newer one may have been extracted over the old):
            # the recipe cache key must follow the code, not the run
            self.bundle_commit = bundle_commit() or manifest.get("bundle_commit")
            self.hl_smi_q = manifest.get("hl_smi_q", "")
            self.restore_gdn()
            self.restore_combined_db(manifest)
            self.restore_hc_fused(manifest)
            self.restore_moe_glue(manifest)
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.bundle_commit = bundle_commit()
            self.hl_smi_q = self._hl_smi_q()
        self.recipe_cache_dir = keyed_recipe_cache_dir(self.args.recipe_cache, self.bundle_commit)
        self.write_manifest()
        for kind, name in self.phases()[self.start_index:]:
            self.current_phase = name
            if kind == "gate":
                keep = self.run_gate(name)
            elif kind == "variant":
                keep = self.run_variant(self.variant_by_id(name))
            else:
                keep = True
            if not keep:
                self.stopped_at = name
                break
        return self.finish()

    def finish(self) -> int:
        try:
            self.write_manifest()
        except OSError as exc:
            print("manifest write failed: %s" % exc)
        try:
            self.run_report()
        except Exception as exc:  # noqa: BLE001
            print("report failed: %s: %s" % (type(exc).__name__, exc))
        if self.stopped_at is not None:
            print("RUN %s: FAILED at %s" % (self.run_id, self.stopped_at))
            return 1
        for v in self.selected:
            if v.get("optional"):
                continue
            if self.variant_status(v["id"]) != int(GateStatus.PASSED):
                self.stopped_at = v["id"]
                print("RUN %s: FAILED at %s" % (self.run_id, v["id"]))
                return 1
        print("RUN %s: PASSED" % self.run_id)
        return 0

    def dry_run(self) -> int:
        print("run id: %s" % self.run_id)
        for kind, name in self.phases():
            print("phase: %s" % name)
            if kind != "variant":
                continue
            variant = self.variant_by_id(name)
            eff, note = self.effective_variant(variant)
            eff["env"] = server_env(eff["env"], self.gdn_env)
            if note:
                print("  inheritance: %s" % note)
            for k, srv in enumerate(variant["servers"]):
                cmd, _env = serve.build_command(eff, srv, self.model, self.served_name)
                print("  server %d: %s" % (k, shlex.join(cmd)))
        return 0

    def run_gate(self, name: str) -> bool:
        phase_dir = self.run_dir / name
        t0 = time.time()
        try:
            if name == "preflight":
                status, details = self.gate_preflight()
            elif name == "build":
                if self.stub:
                    status, details = GateStatus.SKIPPED, {"reason": "stub mode"}
                else:
                    status, details = gates.build_check()
            elif name == "fused_gdn":
                status, details = self.gate_fused_gdn(phase_dir)
            elif name == "tiny":
                status, details = self.gate_tiny(phase_dir)
            else:
                status, details = self.gate_card_tests(phase_dir)
        except Exception as exc:  # noqa: BLE001
            status = GateStatus.FAILED
            details = {"reason": "%s: %s" % (type(exc).__name__, exc)}
        seconds = time.time() - t0
        reason = details.get("reason") or details.get("error")
        errors = list(details.get("errors") or [])
        write_status(phase_dir, name, status, reason, seconds, extra={"errors": errors} if errors else None)
        write_json(phase_dir / "details.json", details)
        self.announce(name, status, reason, seconds, errors, details.get("log") or [])
        if name == "fused_gdn" and not self.stub:
            self.announce_hc_fused(phase_dir, details.get("hc_fused"))
            self.announce_moe_glue(phase_dir, details.get("moe_glue"))
        if status in (GateStatus.FAILED, GateStatus.TIMED_OUT):
            self.failure_evidence(name)
        if name in ("fused_gdn", "card_tests"):
            return True  # recorded-only phases: their result never stops the run
        return status in (GateStatus.PASSED, GateStatus.SKIPPED)

    def announce(self, label: str, status, reason, seconds: float, errors=(), logs=()) -> None:
        """One console line per finished gate or variant; a failure also gets the last error lines of
        its server log and the log to read."""
        line = "%s: %s after %.0f s" % (label, GateStatus(status).name, seconds)
        print(line + (" (%s)" % reason if reason else ""), flush=True)
        if status in (GateStatus.FAILED, GateStatus.TIMED_OUT):
            for text in errors:
                print("    %s" % text, flush=True)
            for log in logs:
                print("    log: %s" % log, flush=True)

    def announce_hc_fused(self, phase_dir: Path, hc_fused) -> None:
        """One console line after G1b's own: the hyper-connection kernels' card test and what the servers get."""
        hc_fused = hc_fused if isinstance(hc_fused, dict) else {}
        status = hc_fused.get("status") or gates.HC_NOT_RUN
        reason = hc_fused.get("reason")
        line = "fused_gdn: hyper-connection kernels' card test %s%s; HC fused: %s" % (
            status, " (%s)" % reason if reason else "", self.hc_fused_outcome())
        print(line, flush=True)
        if status == gates.HC_FAILED and hc_fused.get("log"):
            print("    log: %s" % (phase_dir / hc_fused["log"]), flush=True)

    def announce_moe_glue(self, phase_dir: Path, moe_glue) -> None:
        """One more console line after the hyper-connection one: the MoE glue kernels' card test and what
        the servers get."""
        moe_glue = moe_glue if isinstance(moe_glue, dict) else {}
        status = moe_glue.get("status") or gates.HC_NOT_RUN
        reason = moe_glue.get("reason")
        line = "fused_gdn: MoE glue kernels' card test %s%s; MoE glue: %s" % (
            status, " (%s)" % reason if reason else "", self.moe_glue_outcome())
        print(line, flush=True)
        if status == gates.HC_FAILED and moe_glue.get("log"):
            print("    log: %s" % (phase_dir / moe_glue["log"]), flush=True)

    def failure_evidence(self, phase: str) -> None:
        """pip-freeze.txt and a fresh hl-smi.txt in the run folder (not in stub mode: no card, no stack)."""
        if not self.stub:
            write_failure_evidence(self.run_dir, phase)

    def skipped_steps(self) -> dict:
        """The recorded-only steps this run leaves out, by flag: {step: reason} for the manifest."""
        if self.quick_gates:
            steps = {"fused_gdn.%s" % step: QUICK_GATES_REASON for step in gates.GDN_RECORDED_ONLY_STEPS}
            steps["card_tests"] = QUICK_GATES_REASON
            return steps
        if not self.verify_probe:
            return {"fused_gdn.%s" % gates.GDN_STEP_CRASH_PROBE: NO_VERIFY_PROBE_REASON}
        return {}

    def gate_card_tests(self, phase_dir: Path):
        if self.stub:
            return GateStatus.SKIPPED, {"reason": "stub mode"}
        if self.quick_gates:  # recorded only: it never decides anything a server runs with
            return GateStatus.SKIPPED, {"reason": QUICK_GATES_REASON}
        return gates.card_tests(repo_root(), phase_dir, sys.executable)

    def gate_preflight(self):
        layouts = sorted({(int(s["tp"]), bool(s.get("ep"))) for v in self.selected for s in v["servers"]})
        model_dir = Path(self.model)
        if self.stub:
            return gates.preflight(model_dir,
                                   layouts,
                                   self.results_dir,
                                   expected_cards=0,
                                   min_checkpoint_bytes=0,
                                   min_free_bytes=0)
        return gates.preflight(model_dir, layouts, self.results_dir)

    def gate_fused_gdn(self, phase_dir: Path):
        self.hc_fused_disabled_by = None  # a rerun of the gates decides again
        self.moe_glue_disabled_by = None
        if self.stub:
            status = GateStatus.SKIPPED
            details = {"mode": None, "env": {}, "reason": "stub mode"}
            hc_fused = {"status": gates.HC_NOT_RUN, "reason": "stub mode", "hpu_op": None}
            moe_glue = {"status": gates.HC_NOT_RUN, "reason": "stub mode", "hpu_op": None}
        else:
            if self.combined_db_disabled_by in ("g1b", "tiny"):
                self.combined_db_disabled_by = None  # a rerun of the gates decides again
            skip_reason = QUICK_GATES_REASON if self.quick_gates else None
            probe_reason = None if self.verify_probe else (skip_reason or NO_VERIFY_PROBE_REASON)
            try:
                status, details = gates.fused_gdn(repo_root(), phase_dir, sys.executable,
                                                  skip_recorded_only=skip_reason, skip_verify_probe=probe_reason)
            except Exception as exc:  # noqa: BLE001
                status = GateStatus.FAILED
                details = {
                    "mode": 0,
                    "env": {"VLLM_HPU_GDN_CORE_DECODE": "0"},
                    "reason": "%s: %s" % (type(exc).__name__, exc),
                }
            hc_fused = self.check_hc_fused(phase_dir, details)
            moe_glue = self.check_moe_glue(phase_dir, details)
        # The hyper-connection kernels' card test and the MoE glue kernels' are recorded next to G1b's own
        # result; `status`, the GDN mode and the GDN flags above are final and nothing below changes them.
        hc_fused["requested"] = self.hc_fused_requested
        details["hc_fused"] = hc_fused
        self.hc_fused_status = hc_fused.get("status")
        self.hc_fused_reason = hc_fused.get("reason")
        moe_glue["requested"] = self.moe_glue_requested
        details["moe_glue"] = moe_glue
        self.moe_glue_status = moe_glue.get("status")
        self.moe_glue_reason = moe_glue.get("reason")
        if not self.stub:
            details["env"] = dict(details.get("env") or {})
            details["env"][gates.HC_FUSED_ENV] = "1" if self.hc_fused_on() else "0"
            details["env"][gates.MOE_GLUE_ENV] = "1" if self.moe_glue_on() else "0"
        self.gdn_env = dict(details.get("env") or {})
        self.gdn_mode = details.get("mode")
        self.gdn_reason = details.get("reason")
        if details.get("combined_kernel_db") == gates.COMBINED_DB_DISABLED_BY_G1B and \
                self.combined_db_disabled_by is None:
            self.combined_db_disabled_by = "g1b"  # the kill switch itself is in details["env"]
        return status, details

    def check_hc_fused(self, phase_dir: Path, gdn_details: dict) -> dict:
        """The fused hyper-connection kernels' card test, right after G1b's own steps (gates.hc_fused).

        It runs unless --quick-gates is set and the operator did not ask for the kernels: without the
        request its result is recorded only, with it the result decides a kernel. It runs without the
        combined kernel database when this run's servers will not load that database either. Never raises."""
        skip = HC_QUICK_SKIP_REASON if self.quick_gates and not self.hc_fused_requested else None
        return self.kernel_card_test(gates.hc_fused, phase_dir, gdn_details, skip)

    def check_moe_glue(self, phase_dir: Path, gdn_details: dict) -> dict:
        """The MoE glue kernels' card test, right after the hyper-connection one (gates.moe_glue), under
        the same rules: skipped only with --quick-gates and without --moe-glue. Never raises."""
        skip = MOE_QUICK_SKIP_REASON if self.quick_gates and not self.moe_glue_requested else None
        return self.kernel_card_test(gates.moe_glue, phase_dir, gdn_details, skip)

    def kernel_card_test(self, card_test, phase_dir: Path, gdn_details: dict, skip) -> dict:
        """One card test of G1b's recorded-only kernels (`card_test` is gates.hc_fused or gates.moe_glue):
        without the combined kernel database when this run's servers will not load it. Never raises."""
        combined_db_off = not (gdn_details.get("combined_kernel_db") == gates.COMBINED_DB_ACTIVE and
                               gdn_details.get("mode") == gates.GDN_MODE_FUSED)
        try:
            return dict(card_test(repo_root(), phase_dir, sys.executable, skip=skip,
                                  combined_db_off=combined_db_off))
        except Exception as exc:  # noqa: BLE001 - recorded only
            return {"status": gates.HC_NOT_RUN, "reason": "%s: %s" % (type(exc).__name__, exc), "hpu_op": None}

    def hc_fused_on(self) -> bool:
        """True when this run's servers get VLLM_HPU_HC_FUSED=1: the operator asked (--hc-fused), the card
        test in G1b passed, and the tiny gate did not fail with the kernels."""
        return (self.hc_fused_requested and self.hc_fused_status == gates.HC_PASSED and
                self.hc_fused_disabled_by is None)

    def moe_glue_on(self) -> bool:
        """True when this run's servers get VLLM_HPU_MOE_GLUE=1: the operator asked (--moe-glue), the card
        test in G1b passed, and the tiny gate did not fail with the kernels."""
        return (self.moe_glue_requested and self.moe_glue_status == gates.HC_PASSED and
                self.moe_glue_disabled_by is None)

    def hc_fused_outcome(self) -> str:
        """"on", or "off (<why>)": the manifest's hc_fused.outcome and the summary's `HC fused:` line."""
        return self.kernel_outcome(self.hc_fused_requested, self.hc_fused_status, self.hc_fused_reason,
                                   self.hc_fused_disabled_by)

    def moe_glue_outcome(self) -> str:
        """"on", or "off (<why>)": the manifest's moe_glue.outcome and the summary's `MoE glue:` line."""
        return self.kernel_outcome(self.moe_glue_requested, self.moe_glue_status, self.moe_glue_reason,
                                   self.moe_glue_disabled_by)

    def kernel_outcome(self, requested: bool, status, reason, disabled_by) -> str:
        """The outcome words of one kind of kernels, from its request, its card test and who took it out."""
        if not requested:
            return "off (not requested)"
        if disabled_by == "tiny":
            passed = self.variant_status("tiny") == int(GateStatus.PASSED)
            return "off (the tiny gate failed with it%s)" % (" and passed without" if passed else "")
        if status == gates.HC_PASSED:
            return "on"
        if status is None:
            return "off (the card test has not run)"
        return "off (%s)" % (reason or status)

    def gate_tiny(self, phase_dir: Path):
        """G2 with every fallback. After the newest pieces (``gate_tiny_kernels``), the fused GDN decode
        kernel: gate G1b qualifies it with op tests, and a server is still its first run inside the full
        model (on 2026-11-20 the op tests passed on the card and the tiny server died with
        "ValidateSyncInputTensors tensor_data is empty"). When the gate still fails or times out with
        the kernel on, the attempt is kept under tiny/attempt-gdn-fused/, every later server runs the
        PyTorch GDN path (VLLM_HPU_GDN_CORE_DECODE=0) and the gate runs once more."""
        status, details = self.gate_tiny_kernels(phase_dir)
        failed = (GateStatus.FAILED, GateStatus.TIMED_OUT)
        if (self.stub or status not in failed or self.gdn_mode != gates.GDN_MODE_FUSED or
                str(details.get("reason") or "").startswith("tiny port")):
            return status, details
        first = self.keep_tiny_attempt(phase_dir, status, details, 0.0, TINY_ATTEMPT_GDN_FUSED)
        self.gdn_env = dict(gates.GDN_FALLBACK_ENV)
        self.gdn_mode = gates.GDN_MODE_PYTORCH
        self.gdn_fused_disabled_by = "tiny"
        self.write_manifest()
        print("tiny: %s with the fused GDN decode kernel (%s); kept under %s, running the gate once more on "
              "the PyTorch GDN path (%s=0), which every later server then uses" %
              (GateStatus(status).name, first["reason"], phase_dir / TINY_ATTEMPT_GDN_FUSED, gates.GDN_MODE_ENV),
              flush=True)
        for text in first["errors"]:
            print("    %s" % text, flush=True)
        earlier = details
        status, details, _ = self.tiny_attempt(phase_dir)
        for key in ("combined_kernel_db", "combined_kernel_db_disabled_by", "attempt_combined", "hc_fused",
                    "hc_fused_disabled_by", "attempt_hc_fused", "moe_glue", "moe_glue_disabled_by",
                    "attempt_moe_glue"):  # what the earlier fallbacks decided
            if key in earlier:
                details.setdefault(key, earlier[key])
        details["gdn_fused_disabled_by"] = "tiny"
        details["attempt_gdn_fused"] = first
        return status, details

    def gate_tiny_kernels(self, phase_dir: Path):
        """G2. When the gate fails or times out with a server that ran the fused hyper-connection
        kernels (--hc-fused), the MoE glue kernels (--moe-glue) or both, the attempt is kept (under
        tiny/attempt-hc-fused/ when only the hyper-connection kernels were on, under
        tiny/attempt-moe-glue/ when the MoE glue kernels were on, alone or not), every kind that was on
        is switched off for the rest of the run and the gate runs once more: one retry without both
        kinds, never one retry for each. When it (still) fails or times out with
        a server that loaded the combined kernel database (first hardware run of the adapter in a full
        server), the attempt is kept under tiny/attempt-combined/, the database is switched off for the
        rest of the run and the gate runs once more (RUK-3248). The newest piece goes first."""
        if self.stub:
            return GateStatus.SKIPPED, {"reason": "stub mode"}
        if self.combined_db_disabled_by == "tiny":
            self.combined_db_disabled_by = None  # a rerun of the gate decides again
        if self.hc_fused_disabled_by == "tiny":
            self.hc_fused_disabled_by = None
        if self.moe_glue_disabled_by == "tiny":
            self.moe_glue_disabled_by = None
        failed = (GateStatus.FAILED, GateStatus.TIMED_OUT)
        kept = {}  # the kinds of kernels this gate switched off ("hc_fused", "moe_glue"), see note_kernel_attempt
        combined = self.combined_db_active()
        t0 = time.time()
        status, details, server_ran = self.tiny_attempt(phase_dir)
        details["combined_kernel_db"] = combined
        hc_on, moe_on = self.hc_fused_on(), self.moe_glue_on()
        details["hc_fused"] = hc_on
        details["moe_glue"] = moe_on
        if (hc_on or moe_on) and server_ran and status in failed:
            name = TINY_ATTEMPT_MOE_GLUE if moe_on else TINY_ATTEMPT_HC_FUSED
            attempt = self.keep_tiny_attempt(phase_dir, status, details, time.time() - t0, name)
            what, flags = [], []
            if hc_on:
                self.hc_fused_disabled_by = "tiny"
                kept["hc_fused"] = None if moe_on else attempt  # one attempt, recorded once: under its folder's kind
                what.append("the fused hyper-connection kernels")
                flags.append("%s=0" % gates.HC_FUSED_ENV)
            if moe_on:
                self.moe_glue_disabled_by = "tiny"
                kept["moe_glue"] = attempt
                what.append("the MoE glue kernels")
                flags.append("%s=0" % gates.MOE_GLUE_ENV)
            self.write_manifest()
            print("tiny: %s with %s (%s); kept under %s, running the gate once more with %s" %
                  (GateStatus(status).name, " and ".join(what), attempt["reason"], phase_dir / name,
                   " and ".join(flags)), flush=True)
            for text in attempt["errors"]:
                print("    %s" % text, flush=True)
            t0 = time.time()
            status, details, server_ran = self.tiny_attempt(phase_dir)
            details["combined_kernel_db"] = combined
            self.note_kernel_attempt(details, kept)
        if not (combined and server_ran and status in failed):
            return status, details
        first = self.keep_tiny_attempt(phase_dir, status, details, time.time() - t0)
        self.combined_db_disabled_by = "tiny"
        self.write_manifest()
        print("tiny: %s with the combined kernel database (%s); kept under %s, running the gate once more "
              "with %s=0" % (GateStatus(status).name, first["reason"], phase_dir / TINY_ATTEMPT_COMBINED,
                             serve.COMBINED_KERNEL_DB_ENV), flush=True)
        for text in first["errors"]:
            print("    %s" % text, flush=True)
        status, details, _ = self.tiny_attempt(phase_dir)
        details["combined_kernel_db"] = False
        details["combined_kernel_db_disabled_by"] = "tiny"
        details["attempt_combined"] = first
        self.note_kernel_attempt(details, kept)
        return status, details

    def note_kernel_attempt(self, details: dict, kept: dict) -> None:
        """What a tiny attempt's details say about the hyper-connection and the MoE glue kernels: whether
        its server ran them, and for each kind the tiny gate switched off (a key of `kept`: "hc_fused",
        "moe_glue") that it did. The kept attempt that failed is recorded once, under the kind of its
        folder: `attempt_moe_glue` when the MoE glue kernels were on (`kept["hc_fused"]` is then None),
        `attempt_hc_fused` when only the hyper-connection kernels were."""
        details["hc_fused"] = self.hc_fused_on()
        details["moe_glue"] = self.moe_glue_on()
        for kind, attempt in kept.items():
            details["%s_disabled_by" % kind] = "tiny"
            if attempt is not None:
                details["attempt_%s" % kind] = attempt

    def tiny_attempt(self, phase_dir: Path) -> tuple:
        """One run of the tiny gate: (status, details, whether its server was started)."""
        if serve.port_in_use(TINY_PORT):
            return GateStatus.FAILED, {"reason": "tiny port %d already in use" % TINY_PORT}, False
        phase_dir.mkdir(parents=True, exist_ok=True)
        ckpt = phase_dir / "ckpt"
        reason = ensure_tiny_checkpoint(ckpt, say=lambda text: print("tiny: %s" % text, flush=True))
        if reason is not None:
            return GateStatus.FAILED, {"reason": reason}, False
        serve_cmd = tiny_serve_command(ckpt, TINY_PORT)
        env = self.tiny_env()
        log_path = phase_dir / "server0.log"
        started = time.monotonic()
        popen = serve.start_server(serve_cmd, env, log_path, phase_dir)
        self.active.append(("server", popen))
        try:
            if serve.wait_healthy(TINY_PORT, TINY_HEALTH_TIMEOUT_S, popen):
                status, details = gates.tiny_compare("http://127.0.0.1:%d" % TINY_PORT, ckpt / "reference.json")
            else:
                status, reason = serve.unhealthy_reason(popen, time.monotonic() - started, TINY_HEALTH_TIMEOUT_S,
                                                        "tiny server")
                details = {"reason": reason}
        finally:
            self.active = [a for a in self.active if a[1] is not popen]
            serve.stop_server(popen)
        log_text = self._read_log(log_path)
        fallback = gates.gdn_fallback_lines(log_text)  # recorded: did the fused kernel run in this server
        details["gdn_fallback_lines"] = {"count": len(fallback), "first": fallback[:serve.MAX_ERROR_LINES]}
        if status != GateStatus.PASSED:
            details["errors"] = serve.error_lines(log_text)
            details["log"] = [str(log_path)]
        return status, details, True

    def keep_tiny_attempt(self, phase_dir: Path, status, details: dict, seconds: float,
                          name: str = TINY_ATTEMPT_COMBINED) -> dict:
        """Move the failed attempt's server log, status and details to tiny/<name>/ (attempt-combined,
        attempt-hc-fused or attempt-moe-glue); its summary."""
        keep = phase_dir / name
        keep.mkdir(parents=True, exist_ok=True)
        log_path = phase_dir / "server0.log"
        if log_path.is_file():
            os.replace(log_path, keep / "server0.log")
        details = dict(details, log=[str(keep / "server0.log")])
        reason = details.get("reason") or details.get("error")
        errors = list(details.get("errors") or [])
        write_status(keep, "tiny", status, reason, seconds, extra={"errors": errors} if errors else None)
        write_json(keep / "details.json", details)
        return {"dir": name, "status": int(status), "reason": reason, "errors": errors}

    def run_variant(self, variant) -> bool:
        vid = variant["id"]
        vdir = self.run_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        self.ran_variant_ids.append(vid)
        t0 = time.time()
        failures = []
        eff, note = self.effective_variant(variant)
        eff["env"] = self.variant_server_env(eff["env"])
        self.recipe_cache_notes[vid] = self.recipe_cache_note
        servers = variant["servers"]
        urls = ["http://127.0.0.1:%d" % int(s["port"]) for s in servers]
        planned = []
        envs = []
        for srv in servers:
            cmd, env = serve.build_command(eff, srv, self.model, self.served_name)
            planned.append(cmd)
            envs.append(env)
        procs = []
        started = []
        try:
            if not self.stub:
                for k, srv in enumerate(servers):
                    if serve.port_in_use(int(srv["port"])):
                        failures.append((GateStatus.FAILED,
                                         "server %d port %d already in use" % (k, int(srv["port"]))))
            if not failures:
                for k, srv in enumerate(servers):
                    port_file = vdir / ("server%d.port" % k)
                    if self.stub:
                        port_file.unlink(missing_ok=True)
                    cmd = self._stub_command(port_file) if self.stub else planned[k]
                    env = self._child_env() if self.stub else envs[k]
                    started.append(time.monotonic())
                    proc = serve.start_server(cmd, env, vdir / ("server%d.log" % k), vdir)
                    procs.append(proc)
                    self.active.append(("server", proc))
            try:
                if not failures:
                    for k, srv in enumerate(servers):
                        port = int(srv["port"])
                        if self.stub:
                            port = serve.wait_port_file(vdir / ("server%d.port" % k), STUB_PORT_TIMEOUT_S, procs[k])
                            if port is None:
                                failures.append((GateStatus.FAILED, "server %d (stub) did not report its port" % k))
                                break
                            urls[k] = "http://127.0.0.1:%d" % port
                        if not serve.wait_healthy(port, self.server_timeout_s, procs[k]):
                            # a server that exited is FAILED with its exit code, not a timeout
                            failures.append(serve.unhealthy_reason(procs[k], time.monotonic() - started[k],
                                                                   self.server_timeout_s, "server %d" % k))
                            break
                if not failures:
                    startup = self._startup_records(servers, vdir, time.time() - t0)
                    write_json(vdir / STARTUP_FILE, startup)
                    for k, srv in enumerate(servers):
                        cap, cap_det = gates.capacity(startup[k], int(srv["agents"]))
                        if cap != GateStatus.PASSED:
                            reason = cap_det.get("reason") or cap_det.get("error") or "capacity failed"
                            failures.append((GateStatus.FAILED, "server %d capacity: %s" % (k, reason)))
                if not failures:
                    # the proof of life, before anything expensive: one short completion per server
                    failure = self.run_first_token(vid, urls, vdir)
                    if failure is not None:
                        failures.append(failure)
                if not failures:
                    w = self.ensure_workload(urls[0])
                if not failures and self.warm_pass:
                    non_ok = self.run_warm_pass(urls, vdir)
                    if non_ok:
                        failures.append((GateStatus.FAILED, "warm pass: %d non-OK rows" % non_ok))
                if not failures and vid == "A":
                    coh_status, coh_reason = self.run_coherence(urls[0], w)
                    if coh_status == GateStatus.FAILED:
                        failures.append((GateStatus.FAILED, "coherence: %s" % coh_reason))
                if not failures:
                    monitor = self.start_monitor(urls, vdir)
                    try:
                        stagger = STUB_STAGGER_S if self.stub else LOAD_STAGGER_S
                        non_ok = asyncio.run(
                            agent_load.run_load(w,
                                                urls,
                                                self.served_name,
                                                vdir / REQUESTS_FILE,
                                                stagger_s=stagger,
                                                timeout_s=LOAD_TIMEOUT_S,
                                                request_timeout_s=self.load_read_timeout_s))
                    finally:
                        self.stop_monitor(monitor)
                    if non_ok:
                        failures.append((GateStatus.FAILED, "load: %d non-OK rows" % non_ok))
                    required = min(sum(int(s["agents"]) for s in servers), len(w.get("agents") or []))
                    observed = max_running_sum(vdir / METRICS_FILE)
                    if observed + 1e-9 < required:
                        failures.append((GateStatus.FAILED, "running never reached %d (max %g)" % (required, observed)))
            except Exception as exc:  # noqa: BLE001
                failures.append((GateStatus.FAILED, "%s: %s" % (type(exc).__name__, exc)))
            self.write_gdn_fallback(vdir, servers)
            status = GateStatus.PASSED if not failures else failures[0][0]
            reason = failures[0][1] if failures else None
            if note:
                reason = note if reason is None else "%s; %s" % (reason, note)
            extra = {"servers": self._server_records(servers, urls, planned, envs)}
            errors, logs = [], []
            if status != GateStatus.PASSED:
                for k in range(len(procs)):
                    log_path = vdir / ("server%d.log" % k)
                    logs.append(str(log_path))
                    errors += ["server%d: %s" % (k, text) for text in serve.error_lines(self._read_log(log_path))]
                if errors:
                    extra["errors"] = errors
            write_status(vdir, vid, status, reason, time.time() - t0, extra=extra)
            self.announce("variant %s" % vid, status, reason, time.time() - t0, errors, logs)
            if status != GateStatus.PASSED:
                self.failure_evidence(vid)  # before the teardown: the cards as the failure left them
            if status != GateStatus.PASSED and int(variant.get("on_fail", 1)) == int(OnFail.STOP):
                return False
            return True
        finally:
            self.teardown_variant(procs)

    def effective_variant(self, variant) -> tuple:
        eff = dict(variant)
        eff["env"] = dict(variant.get("env") or {})
        eff["serve_args"] = list(variant.get("serve_args") or [])
        base_id = variant.get("inherit_spec_from")
        if not base_id:
            return eff, None
        base = self.variant_by_id(base_id)
        status = self.variant_status(base_id)
        if status == int(GateStatus.PASSED) and base is not None:
            for key, value in (base.get("env") or {}).items():
                eff["env"].setdefault(key, value)
            eff["serve_args"] += list(base.get("serve_args") or [])
            return eff, "inherits speculation from %s (%s PASSED)" % (base_id, base_id)
        name = GateStatus(status).name if status is not None else "NOT_RUN"
        return eff, "runs without speculation (%s %s)" % (base_id, name)

    def run_first_token(self, vid: str, urls: list, vdir: Path):
        """The proof of life of a started variant: one short completion from every server, before the
        workload is built and anything is warmed or measured. Each answer goes to first-token.json and
        to one console line. Returns None, or the (status, reason) of the first server that gave no token;
        the timeout is the coherence gate's (the first request pays the first graph compile)."""
        records = []
        failure = None
        for k, url in enumerate(urls):
            status, details = gates.first_token(url, self.served_name, self.request_timeout_s)
            records.append(dict(details, server=k, url=url, status=int(status)))
            seconds = float(details.get("seconds") or 0.0)
            if status == GateStatus.PASSED:
                print("variant %s server%d: first token after %.1f s: %r" % (vid, k, seconds, details.get("text")),
                      flush=True)
                continue
            reason = details.get("reason") or details.get("error") or "no answer"
            print("variant %s server%d: no first token after %.1f s: %s" % (vid, k, seconds, reason), flush=True)
            failure = (status, "server %d first token: %s" % (k, reason))
            break
        write_json(vdir / FIRST_TOKEN_FILE, records)
        return failure

    def run_coherence(self, url: str, w: dict):
        cdir = self.run_dir / "coherence"
        t0 = time.time()
        if self.stub:
            write_status(cdir, "coherence", GateStatus.SKIPPED, "stub mode", 0.0)
            return GateStatus.SKIPPED, "stub mode"
        agents = w.get("agents") or []
        messages = agents[0].get("messages") if agents else []
        status, details = gates.coherence(url, self.served_name, messages or [],
                                          request_timeout_s=self.request_timeout_s)
        reason = details.get("reason") or details.get("error")
        write_status(cdir, "coherence", status, reason, time.time() - t0)
        write_json(cdir / "details.json", details)
        return status, reason

    def ensure_workload(self, first_url: str) -> dict:
        path = self.run_dir / "workload.json"
        if path.is_file():
            w = read_json(path)
            if isinstance(w, dict):
                if self.workload_info is None:
                    self.set_workload_info(path, w)
                return w
        if self.stub:
            counter = workload.estimate_counter
        else:
            counter = workload.tokenize_counter(first_url, self.served_name)
        w = workload.build_workload(repo_root(),
                                    int(self.args.agents),
                                    int(self.args.shared_tokens),
                                    int(self.args.initial_tokens),
                                    WORKLOAD_TURNS,
                                    int(self.args.max_tokens),
                                    counter)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False, sort_keys=True)
        self.set_workload_info(path, w)
        return w

    def ensure_warm_workload(self, first_url: str) -> dict:
        """The warm pass's one-turn, 64-token copy of the measured workload (file offset 1).

        Same agents, same shared system block, each agent's file window shifted by one, so the
        prompts differ from the measured ones right after the shared prefix. Built once per run.
        """
        path = self.run_dir / WARM_WORKLOAD_FILE
        if path.is_file():
            w = read_json(path)
            if isinstance(w, dict):
                return w
        if self.stub:
            counter = workload.estimate_counter
        else:
            counter = workload.tokenize_counter(first_url, self.served_name)
        w = workload.build_workload(repo_root(),
                                    int(self.args.agents),
                                    int(self.args.shared_tokens),
                                    int(self.args.initial_tokens),
                                    1,
                                    WARM_PASS_MAX_TOKENS,
                                    counter,
                                    file_offset=WARM_PASS_FILE_OFFSET)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False, sort_keys=True)
        return w

    def run_warm_pass(self, urls: list, vdir: Path) -> int:
        """One turn of 64 tokens per agent over the warm workload before the gates measure anything.

        The first server start after a (recipe-cached) graph compile takes the prompt shapes of the
        run; this pass pays that compile cost on purpose, with prompts that share the measured
        workload's system block, so the coherence gate and the measured load meet warm graphs and a
        cached shared block. Returns the number of non-OK rows.
        """
        warm = self.ensure_warm_workload(urls[0])
        stagger = STUB_STAGGER_S if self.stub else LOAD_STAGGER_S
        t0 = time.time()
        non_ok = asyncio.run(
            agent_load.run_load(warm,
                                urls,
                                self.served_name,
                                vdir / WARM_REQUESTS_FILE,
                                stagger_s=stagger,
                                timeout_s=LOAD_TIMEOUT_S,
                                request_timeout_s=self.warm_request_timeout_s))
        write_json(vdir / WARM_PASS_FILE, {
            "requests": len(warm.get("agents") or []),
            "non_ok": int(non_ok),
            "seconds": round(time.time() - t0, 3),
            "max_tokens": WARM_PASS_MAX_TOKENS,
            "file_offset": WARM_PASS_FILE_OFFSET,
        })
        return int(non_ok)

    def set_workload_info(self, path: Path, w: dict) -> None:
        self.workload_info = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "agents": len(w.get("agents") or []),
            "turns": w.get("turns"),
            "version": w.get("version"),
            "shared_prefix_tokens": w.get("shared_prefix_tokens"),
            "target_initial_tokens": w.get("target_initial_tokens"),
            "max_tokens": w.get("max_tokens"),
        }

    def start_monitor(self, urls: list, out_dir: Path):
        cmd = [
            sys.executable,
            "-m",
            "tools.flash_next.rental.monitor",
            "--servers",
            ",".join(urls),
            "--out-dir",
            str(out_dir),
        ]
        if self.stub:
            cmd += ["--no-hlsmi", "--interval", str(STUB_MONITOR_INTERVAL_S)]
        proc = subprocess.Popen(cmd,
                                cwd=str(repo_root()),
                                env=self._child_env(),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.active.append(("monitor", proc))
        return proc

    def stop_monitor(self, proc) -> None:
        self.active = [a for a in self.active if a[1] is not proc]
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            return
        deadline = time.time() + MONITOR_TERM_WAIT_S
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:
            proc.kill()
        except OSError:
            pass

    def teardown_variant(self, procs: list) -> None:
        for proc in procs:
            self.active = [a for a in self.active if a[1] is not proc]
            try:
                serve.stop_server(proc)
            except Exception:  # noqa: BLE001
                pass
        if procs and not self.stub:
            time.sleep(CARD_FREE_WAIT_S)

    def teardown(self) -> None:
        remaining = list(self.active)
        self.active = []
        for kind, proc in remaining:
            try:
                if kind == "monitor":
                    self.stop_monitor(proc)
                elif proc.poll() is None:
                    serve.stop_server(proc, 10.0)
            except Exception:  # noqa: BLE001
                pass

    def write_gdn_fallback(self, vdir: Path, servers: list) -> None:
        lines = []
        for k in range(len(servers)):
            text = self._read_log(vdir / ("server%d.log" % k))
            for ln in gates.gdn_fallback_lines(text):
                lines.append("server%d: %s" % (k, ln))
        (vdir / "gdn_fallback.txt").write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")

    def run_report(self) -> None:
        report.main([str(self.run_dir)])
        lines = gdn_summary_lines(self.run_dir, [v["id"] for v in self.selected])
        summary = self.run_dir / "summary.md"
        with open(summary, "a", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")

    def variant_status(self, vid: str):
        obj = read_json(self.run_dir / vid / STATUS_FILE)
        return obj.get("status") if isinstance(obj, dict) else None

    def restore_gdn(self) -> None:
        details = read_json(self.run_dir / "fused_gdn" / "details.json")
        if isinstance(details, dict):
            self.gdn_env = dict(details.get("env") or {})
            self.gdn_mode = details.get("mode")
            self.gdn_reason = details.get("reason")
            if details.get("combined_kernel_db") == gates.COMBINED_DB_DISABLED_BY_G1B and \
                    self.combined_db_disabled_by is None:
                self.combined_db_disabled_by = "g1b"
            hc_fused = details.get("hc_fused")  # absent in a run made before the kernels were wired in
            if isinstance(hc_fused, dict):
                self.hc_fused_status = hc_fused.get("status")
                self.hc_fused_reason = hc_fused.get("reason")
            moe_glue = details.get("moe_glue")  # absent in a run made before --moe-glue existed
            if isinstance(moe_glue, dict):
                self.moe_glue_status = moe_glue.get("status")
                self.moe_glue_reason = moe_glue.get("reason")

    def restore_hc_fused(self, manifest: dict) -> None:
        """On a resume: a tiny gate of the earlier start that took the fused hyper-connection kernels out
        of the run keeps its decision (a gate that runs again resets its own). The request itself is this
        start's --hc-fused, and the card test's result comes back with G1b's details (restore_gdn)."""
        hc_fused = manifest.get("hc_fused")
        by = hc_fused.get("disabled_by") if isinstance(hc_fused, dict) else None
        if self.hc_fused_disabled_by is None and by == "tiny":
            self.hc_fused_disabled_by = by

    def restore_moe_glue(self, manifest: dict) -> None:
        """On a resume, the same for the MoE glue kernels: the tiny gate's decision of the earlier start
        stays, the request is this start's --moe-glue, the card test's result comes with G1b's details."""
        moe_glue = manifest.get("moe_glue")
        by = moe_glue.get("disabled_by") if isinstance(moe_glue, dict) else None
        if self.moe_glue_disabled_by is None and by == "tiny":
            self.moe_glue_disabled_by = by

    def restore_combined_db(self, manifest: dict) -> None:
        """On a resume: a gate of the earlier start that took the combined kernel database out of the
        run keeps its decision (a gate that runs again resets its own)."""
        recipe = manifest.get("recipe_cache")
        by = recipe.get("combined_kernel_db_disabled_by") if isinstance(recipe, dict) else None
        if self.combined_db_disabled_by is None and by in ("g1b", "tiny"):
            self.combined_db_disabled_by = by

    def write_manifest(self) -> None:
        manifest = read_json(self.run_dir / "manifest.json")
        if not isinstance(manifest, dict):
            manifest = {"created": time.time()}
        bundle = bundle_text()
        manifest.update({
            "run_id": self.run_id,
            "mode": self.mode,
            "args": dict(vars(self.args)),
            "bundle": bundle if bundle is not None else self.bundle_commit,
            "bundle_commit": self.bundle_commit,
            "python": sys.version,
            "env": recorded_env(os.environ),
            "hl_smi_q": self.hl_smi_q,
            "variants": self.selected,
            "phases": [n for _, n in self.phases()[self.start_index:]],
        })
        manifest["gdn_core_decode_mode"] = self.gdn_mode
        manifest["gdn_core_decode_reason"] = self.gdn_reason
        manifest["skip_warmup"] = self.skip_warmup
        manifest["warm_pass"] = self.warm_pass
        manifest["server_timeout_s"] = self.server_timeout_s
        manifest["load_read_timeout_s"] = self.load_read_timeout_s
        manifest["warm_request_timeout_s"] = self.warm_request_timeout_s
        manifest["quick_gates"] = self.quick_gates
        manifest["skipped_steps"] = self.skipped_steps()
        manifest["recipe_cache"] = {
            "dir": self.recipe_cache_dir,
            "size_mb": self.recipe_cache_mb,
            "combined_kernel_db": self.combined_db_installed,
            "combined_kernel_db_disabled_by": self.combined_db_disabled_by,
            "variants": dict(self.recipe_cache_notes),
        }
        manifest["hc_fused"] = {
            "requested": self.hc_fused_requested,
            "status": self.hc_fused_status,
            "reason": self.hc_fused_reason,
            "on": self.hc_fused_on(),
            "disabled_by": self.hc_fused_disabled_by,
            "outcome": self.hc_fused_outcome(),
        }
        manifest["moe_glue"] = {
            "requested": self.moe_glue_requested,
            "status": self.moe_glue_status,
            "reason": self.moe_glue_reason,
            "on": self.moe_glue_on(),
            "disabled_by": self.moe_glue_disabled_by,
            "outcome": self.moe_glue_outcome(),
        }
        if self.workload_info is not None:
            manifest["workload"] = self.workload_info
            manifest["workload_sha"] = self.workload_info.get("sha256")
        write_json(self.run_dir / "manifest.json", manifest)

    def _server_records(self, servers: list, urls: list, planned: list, envs: list) -> list:
        records = []
        for k, srv in enumerate(servers):
            records.append({
                "server": k,
                "url": urls[k],
                "agents": srv["agents"],
                "planned_command": planned[k],
                "planned_env": recorded_env(envs[k]),
            })
        return records

    def _startup_records(self, servers: list, vdir: Path, elapsed: float) -> list:
        records = []
        for k, srv in enumerate(servers):
            if self.stub:
                rec = {key: None for key in schema.STARTUP_KEYS}
                rec["server"] = k
                rec["max_concurrency"] = 64.0
                rec["max_concurrency_len"] = schema.MAX_MODEL_LEN
                rec["startup_seconds"] = round(elapsed, 3)
            else:
                rec = serve.parse_startup(self._read_log(vdir / ("server%d.log" % k)), k, elapsed)
            records.append(rec)
        return records

    def _stub_command(self, port_file: Path) -> list:
        return [
            sys.executable,
            "-m",
            "tools.flash_next.rental.stub_server",
            "--port",
            "0",
            "--port-file",
            str(port_file),
            "--model",
            self.served_name,
            "--tokens-per-second",
            STUB_TOKENS_PER_SECOND,
        ]

    def variant_server_env(self, variant_env: dict) -> dict:
        """The environment of a variant's servers: the variant's flags, the GDN flags, the fused
        hyper-connection flag and the MoE glue flag (always explicit: "1" only when the operator asked
        and the card test passed, `hc_fused_on` and `moe_glue_on`), the recipe cache for the servers that
        register no kernel database of their own (`serve.recipe_cache_env`) and, by default, the skipped
        bucket warmup (the compiled graphs land in the recipe cache)."""
        env = server_env(variant_env, self.gdn_env)
        env[gates.HC_FUSED_ENV] = "1" if self.hc_fused_on() else "0"
        env[gates.MOE_GLUE_ENV] = "1" if self.moe_glue_on() else "0"
        if self.skip_warmup:
            env["VLLM_SKIP_WARMUP"] = "true"
        combined = self.combined_db_active()
        if self.combined_db_installed and not combined:
            env[serve.COMBINED_KERNEL_DB_ENV] = "0"  # a database G1b did not pass stays out of every server
        cache_env, note = serve.recipe_cache_env(self.recipe_cache_dir, self.recipe_cache_mb, env, combined)
        if cache_env and not self.stub:
            Path(self.recipe_cache_dir).mkdir(parents=True, exist_ok=True)
        env.update(cache_env)
        self.recipe_cache_note = note
        return env

    def combined_db_active(self) -> bool:
        """True when this run's servers load the combined kernel database: it is installed, nobody
        (the operator, G1b, the tiny gate) took it out, and G1b passed with the fused kernel."""
        return (self.combined_db_installed and self.combined_db_disabled_by is None and
                self.gdn_mode == gates.GDN_MODE_FUSED)

    def tiny_env(self) -> dict:
        """The tiny gate's server environment: a variant's environment without a variant's own flags."""
        return serve.server_environment(self.variant_server_env({}))

    def _child_env(self) -> dict:
        env = dict(os.environ)
        root = str(repo_root())
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
        return env

    def _read_log(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _hl_smi_q(self) -> str:
        if self.stub:
            return ""
        try:
            out = subprocess.run(["hl-smi", "-q"], capture_output=True, text=True, timeout=60)
            return out.stdout[:HLSMI_Q_LIMIT]
        except (OSError, subprocess.SubprocessError):
            return ""


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m tools.flash_next.rental.main",
                                     description="the Flash-Next rental test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--served-name", default="flash-next")
    parser.add_argument("--variants", default="A,B,C")
    parser.add_argument("--with-profile", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--from", dest="from_phase", default=None)
    parser.add_argument("--mode", choices=("node", "stub"), default="node")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download", action="store_true", help="accepted for entry-script compatibility")
    parser.add_argument("--full-warmup", action="store_true", help="run the bucket warmup (default: skipped)")
    parser.add_argument("--server-timeout", type=float, default=SERVER_HEALTH_TIMEOUT_S,
                        help="seconds to wait for a server to become healthy")
    parser.add_argument("--no-warm-pass", action="store_true",
                        help="skip the one-turn warm pass that compiles the graphs before the gates")
    parser.add_argument("--load-read-timeout", type=float, default=LOAD_REQUEST_TIMEOUT_S, metavar="S",
                        help="measured load: seconds a request may go without receiving data "
                        "(default %(default)s)")
    parser.add_argument("--warm-request-timeout", type=float, default=WARM_PASS_REQUEST_TIMEOUT_S, metavar="S",
                        help="warm pass: seconds a request may go without receiving data, first graph compiles "
                        "included (default %(default)s)")
    parser.add_argument("--quick-gates", action="store_true",
                        help="skip the recorded-only steps before the first measurement: G1b's eager replays "
                        "and bench, the verify crash probe and the card_tests phase; the checks that decide "
                        "the GDN kernel and the verify op still run")
    parser.add_argument("--no-verify-probe", action="store_true",
                        help="skip only the automatic crash probe that follows a crashed or hung verify op test")
    parser.add_argument("--hc-fused", action="store_true",
                        help="ask for the fused hyper-connection kernels: the servers get VLLM_HPU_HC_FUSED=1 "
                        "when the kernels' card test in G1b passed (default: off, every server gets "
                        "VLLM_HPU_HC_FUSED=0); a tiny gate that fails with them runs once more without")
    parser.add_argument("--moe-glue", action="store_true",
                        help="ask for the MoE glue kernels: the servers get VLLM_HPU_MOE_GLUE=1 when the "
                        "kernels' card test in G1b passed (default: off, every server gets "
                        "VLLM_HPU_MOE_GLUE=0); a tiny gate that fails with them runs once more without")
    parser.add_argument("--recipe-cache", metavar="DIR", default=None,
                        help="folder for the HPU recipe cache; a server gets it only when it runs without the "
                        "fused GDN kernel, or when the combined kernel database is installed, G1b passed "
                        "through it and nobody (the operator, G1b, the tiny gate) switched it off")
    parser.add_argument("--recipe-cache-mb", type=int, default=16384,
                        help="recipe cache size in MB (default 16384)")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--shared-tokens", type=int, default=16384)
    parser.add_argument("--initial-tokens", type=int, default=124000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        runner = Runner(args, load_variants())
    except ValueError as exc:
        print("error: %s" % exc)
        return 2
    if args.dry_run:
        return runner.dry_run()
    rc = 1
    try:
        rc = runner.run()
    except ValueError as exc:
        print("error: %s" % exc)
        rc = 2
    except KeyboardInterrupt:
        if runner.stopped_at is None:
            runner.stopped_at = runner.current_phase
        rc = runner.finish()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        if runner.stopped_at is None:
            runner.stopped_at = runner.current_phase
        rc = runner.finish()
        if rc == 0:
            rc = 1
    finally:
        runner.teardown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
