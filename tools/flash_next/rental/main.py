# SPDX-License-Identifier: Apache-2.0
"""The rental test orchestrator: gates, variants A/B/C(/P), the report."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tools.flash_next.rental import agent_load, gates, report, schema, serve, workload
from tools.flash_next.rental.schema import (
    METRICS_FILE,
    REQUESTS_FILE,
    STARTUP_FILE,
    STATUS_FILE,
    GateStatus,
    OnFail,
)

VARIANTS_FILE = Path(__file__).resolve().parent / "variants.json"
GATE_PHASES = ("preflight", "build", "fused_gdn", "tiny", "card_tests")
ENV_PREFIXES = ("VLLM_", "HABANA_", "PT_", "HF_")
HLSMI_Q_LIMIT = 200 * 1024
SERVER_HEALTH_TIMEOUT_S = 1800.0
TINY_HEALTH_TIMEOUT_S = 1200.0
TINY_PORT = 8010
TINY_FIXTURE = "tests/unit_tests/models/flash_next/tiny_tp8_config.json"
LOAD_TIMEOUT_S = 5400.0
LOAD_REQUEST_TIMEOUT_S = 1800.0
LOAD_STAGGER_S = 60.0
WORKLOAD_TURNS = 4
MONITOR_TERM_WAIT_S = 15.0
CARD_FREE_WAIT_S = 20.0
STUB_STAGGER_S = 0.0
STUB_MONITOR_INTERVAL_S = 0.05
STUB_TOKENS_PER_SECOND = "400"
STUB_PORT_TIMEOUT_S = 30.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_variants(path: Path = VARIANTS_FILE) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def server_env(variant_env: dict, gdn_env: dict) -> dict:
    env = dict(variant_env)
    env.update(gdn_env)
    return env


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
        count = sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        if count:
            lines.append("GDN fallback lines in %s: %d (the kernel did not run on every layer)" % (vid, count))
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
        if self.from_phase and not args.run_id:
            raise ValueError("--from requires --run-id")
        self.gdn_env = {}
        self.gdn_mode = None
        self.gdn_reason = None
        self.active = []
        self.stopped_at = None
        self.current_phase = GATE_PHASES[0]
        self.ran_variant_ids = []
        self.start_index = 0
        self.hl_smi_q = ""
        self.bundle_commit = None
        self.workload_info = None

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
        if self.from_phase:
            if self.from_phase not in names:
                raise ValueError("--from %s is not a phase of this run" % self.from_phase)
            self.start_index = names.index(self.from_phase)
            if not self.run_dir.is_dir():
                raise ValueError("run folder %s does not exist" % self.run_dir)
            manifest = read_json(self.run_dir / "manifest.json") or {}
            self.bundle_commit = manifest.get("bundle_commit")
            self.hl_smi_q = manifest.get("hl_smi_q", "")
            self.restore_gdn()
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.bundle_commit = self._git_commit()
            self.hl_smi_q = self._hl_smi_q()
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
        write_status(phase_dir, name, status, reason, seconds)
        write_json(phase_dir / "details.json", details)
        if name in ("fused_gdn", "card_tests"):
            return True  # recorded-only phases: their result never stops the run
        return status in (GateStatus.PASSED, GateStatus.SKIPPED)

    def gate_card_tests(self, phase_dir: Path):
        if self.stub:
            return GateStatus.SKIPPED, {"reason": "stub mode"}
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
        if self.stub:
            status = GateStatus.SKIPPED
            details = {"mode": None, "env": {}, "reason": "stub mode"}
        else:
            try:
                status, details = gates.fused_gdn(repo_root(), phase_dir, sys.executable)
            except Exception as exc:  # noqa: BLE001
                status = GateStatus.FAILED
                details = {
                    "mode": 0,
                    "env": {"VLLM_HPU_GDN_CORE_DECODE": "0"},
                    "reason": "%s: %s" % (type(exc).__name__, exc),
                }
        self.gdn_env = dict(details.get("env") or {})
        self.gdn_mode = details.get("mode")
        self.gdn_reason = details.get("reason")
        return status, details

    def gate_tiny(self, phase_dir: Path):
        if self.stub:
            return GateStatus.SKIPPED, {"reason": "stub mode"}
        ckpt = phase_dir / "ckpt"
        cmd = [
            sys.executable,
            "tools/flash_next/make_tiny_checkpoint.py",
            "--out",
            str(ckpt),
            "--fixture",
            TINY_FIXTURE,
        ]
        proc = gates.run_step(cmd, repo_root(), 900)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            reason = "tiny checkpoint failed (exit %d): %s" % (proc.returncode, tail.strip())
            return GateStatus.FAILED, {"reason": reason}
        serve_cmd = [
            "vllm",
            "serve",
            str(ckpt),
            "--served-model-name",
            "tiny",
            "--tensor-parallel-size",
            "8",
            "--max-model-len",
            "512",
            "--max-num-seqs",
            "4",
            "--language-model-only",
            "--trust-remote-code",
            "--port",
            str(TINY_PORT),
            "--host",
            "127.0.0.1",
        ]
        env = self.tiny_env()
        popen = serve.start_server(serve_cmd, env, phase_dir / "server0.log", phase_dir)
        self.active.append(("server", popen))
        try:
            if not serve.wait_healthy(TINY_PORT, TINY_HEALTH_TIMEOUT_S, popen):
                reason = "tiny server not healthy within %d s" % int(TINY_HEALTH_TIMEOUT_S)
                return GateStatus.TIMED_OUT, {"reason": reason}
            return gates.tiny_compare("http://127.0.0.1:%d" % TINY_PORT, ckpt / "reference.json")
        finally:
            self.active = [a for a in self.active if a[1] is not popen]
            serve.stop_server(popen)

    def run_variant(self, variant) -> bool:
        vid = variant["id"]
        vdir = self.run_dir / vid
        vdir.mkdir(parents=True, exist_ok=True)
        self.ran_variant_ids.append(vid)
        t0 = time.time()
        failures = []
        eff, note = self.effective_variant(variant)
        eff["env"] = server_env(eff["env"], self.gdn_env)
        servers = variant["servers"]
        urls = ["http://127.0.0.1:%d" % int(s["port"]) for s in servers]
        planned = []
        envs = []
        for srv in servers:
            cmd, env = serve.build_command(eff, srv, self.model, self.served_name)
            planned.append(cmd)
            envs.append(env)
        procs = []
        try:
            for k, srv in enumerate(servers):
                port_file = vdir / ("server%d.port" % k)
                if self.stub:
                    port_file.unlink(missing_ok=True)
                cmd = self._stub_command(port_file) if self.stub else planned[k]
                env = self._child_env() if self.stub else envs[k]
                proc = serve.start_server(cmd, env, vdir / ("server%d.log" % k), vdir)
                procs.append(proc)
                self.active.append(("server", proc))
            try:
                for k, srv in enumerate(servers):
                    port = int(srv["port"])
                    if self.stub:
                        port = serve.wait_port_file(vdir / ("server%d.port" % k), STUB_PORT_TIMEOUT_S, procs[k])
                        if port is None:
                            failures.append((GateStatus.FAILED, "server %d (stub) did not report its port" % k))
                            break
                        urls[k] = "http://127.0.0.1:%d" % port
                    if not serve.wait_healthy(port, SERVER_HEALTH_TIMEOUT_S, procs[k]):
                        failures.append((GateStatus.TIMED_OUT,
                                         "server %d not healthy within %d s" % (k, int(SERVER_HEALTH_TIMEOUT_S))))
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
                    w = self.ensure_workload(urls[0])
                    if vid == "A":
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
                                                request_timeout_s=LOAD_REQUEST_TIMEOUT_S))
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
            write_status(vdir, vid, status, reason, time.time() - t0, extra=extra)
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

    def run_coherence(self, url: str, w: dict):
        cdir = self.run_dir / "coherence"
        t0 = time.time()
        if self.stub:
            write_status(cdir, "coherence", GateStatus.SKIPPED, "stub mode", 0.0)
            return GateStatus.SKIPPED, "stub mode"
        agents = w.get("agents") or []
        messages = agents[0].get("messages") if agents else []
        status, details = gates.coherence(url, self.served_name, messages or [])
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

    def write_manifest(self) -> None:
        manifest = read_json(self.run_dir / "manifest.json")
        if not isinstance(manifest, dict):
            manifest = {"created": time.time()}
        bundle = self._bundle_text()
        manifest.update({
            "run_id": self.run_id,
            "mode": self.mode,
            "args": dict(vars(self.args)),
            "bundle": bundle if bundle is not None else self.bundle_commit,
            "bundle_commit": self.bundle_commit,
            "python": sys.version,
            "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)},
            "hl_smi_q": self.hl_smi_q,
            "variants": self.selected,
            "phases": [n for _, n in self.phases()[self.start_index:]],
        })
        manifest["gdn_core_decode_mode"] = self.gdn_mode
        manifest["gdn_core_decode_reason"] = self.gdn_reason
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
                "planned_env": {key: value for key, value in envs[k].items() if key.startswith(ENV_PREFIXES)},
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

    def tiny_env(self) -> dict:
        """The tiny gate's server environment: a variant's environment without a variant's own flags."""
        return serve.server_environment(server_env({}, self.gdn_env))

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

    def _bundle_text(self):
        try:
            return (repo_root().parent / "BUNDLE.txt").read_text(encoding="utf-8")
        except OSError:
            return None

    def _git_commit(self):
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"],
                                 cwd=str(repo_root()),
                                 capture_output=True,
                                 text=True,
                                 timeout=30)
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

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
