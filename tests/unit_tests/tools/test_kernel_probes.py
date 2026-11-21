# SPDX-License-Identifier: Apache-2.0
"""The card probes of ``tools/flash_next/kernel_probes`` run their own logic on a CPU before they get a card.

* ``probe_moe_route.py`` (the open ``route_real_r16`` card failure of the MoE glue): its self-test injects every
  fault the card could have (a strided view read as contiguous, an 8-byte read of 4-byte ids, a wrong softmax
  or top-k, a kernel that moves in a shared graph, ...) and the verdict must name each one.
* ``probe_moe_block.py`` (one Flash-Next FP8 MoE block, torch path against the glue and the glue with the merged
  gate_up GEMM): with the tests' CPU stand-ins every variant returns the torch path's bits.
* ``probe_moe_expert_bytes.py`` (idea G2-1: does the vendor MoE op read every local expert): every routing arm
  runs and the verdict line is printed (times on a CPU mean nothing).
* ``probe_moe_layers.py`` (G2-1 again: a distinct weight set per call, TP4 and TP8 geometry, chunk_size): every
  arm runs and the three verdict lines are printed.
* ``probe_view_inputs.py`` (can a custom op take a view at accumulation mode 0): its own self-test.
* ``probe_index_narrowing.py`` (ids narrowed to int32 in the graph that makes them: the sampler's top logprobs,
  the sparse-QSA selection, argmax, replacements): on a CPU every case is RIGHT.
* ``probe_hc_copies.py`` (the owned copies in front of the fused hyper-connection ops): every copy mode returns
  the bits of mode 0.
* ``decode_ab_client.py --compare``: the 1-stream rounds are the test of two arms' texts.
* ``probe_moe_forms.py`` (the coordinator's MoE plan: the 2-D gate / up GEMM, the sum as a matmul, W2 with a shared
  scale; the small-row forms with a per-tensor scale): every form runs; the exact forms equal the dense form.
* ``probe_moe_grouped.py`` (only the routed pairs, at most C per expert, against the dense form and the vendor op
  with chunk 64): with no pair beyond C the grouped result is the dense form's; the overflow counts and the op
  counts are printed.
* ``moe_rt_driver.py`` (moe_rt_ab.sh's probe: the MoE dense range switched at run time on one server): against a
  stand-in server it switches every setting, measures every level and prompt set, compares the prompt logprobs of
  the settings, and stops (exit 9) when a switch reaches no worker's graphs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBES = ROOT / "tools" / "flash_next" / "kernel_probes"


def _run(args, tmp_path, **env):
    full = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_PLUGINS="", PROBE_DEVICE="cpu", **env)
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=tmp_path, env=full, capture_output=True, text=True,
                          timeout=1800)
    return proc, proc.stdout + proc.stderr


def test_route_probe_names_every_injected_fault(tmp_path):
    proc, out = _run([PROBES / "selftest_probe_moe_route.py"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "12 of 12 faults are named by the verdict" in out


def test_block_probe_variants_are_the_torch_path_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_moe_block.py", "--out", tmp_path / "block", "--iters", "1", "--blocks", "1"],
                     tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "merged gate_up (moe_silu_mul first_column): True" in out
    parity = [line for line in out.splitlines() if line.startswith("PROBE moe-block parity")]
    assert len(parity) == 6, parity  # glue and merged at 1, 32 and 128 rows
    assert all(" max abs diff 0.000e+00" in line for line in parity), parity
    assert "VERDICT moe-block:" in out


def test_expert_bytes_probe_runs_every_arm(tmp_path):
    proc, out = _run([PROBES / "probe_moe_expert_bytes.py", "--out", tmp_path / "eb", "--iters", "1", "--calls", "1",
                      "--calls-large", "1"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    arms = [line for line in out.splitlines() if line.startswith("PROBE expert-bytes rows")]
    assert len(arms) == 15, arms  # 3 arms at 1 row, 4 at 4, 32 and 128 rows
    assert "routing one  ( 1 local experts hit)" in out and "(64 local experts hit)" in out
    assert "VERDICT expert-bytes:" in out


@pytest.mark.parametrize("timing", [[], ["--replay"]], ids=["lazy", "replay"])
def test_moe_layers_probe_runs_every_arm(tmp_path, timing):
    proc, out = _run([PROBES / "probe_moe_layers.py", "--out", tmp_path / "ml", "--layers", "2", "--iters", "1",
                      "--rows", "1,32", "--chunks", "64", "--chunk-rows", "32", *timing], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    arms = [line for line in out.splitlines() if line.startswith("PROBE moe-layers local")]
    # TP4 and TP8 at 1 and 32 rows, none and dense at 32, one chunk_size, 2 and 4 slices at 32 rows
    assert len(arms) == 9, arms
    assert not any(" ERROR " in line for line in arms), arms
    assert "routing none    chunk  None" in out and "routing dense   chunk  None" in out and "chunk    64:" in out
    assert "chunk  None slices 2:" in out and "chunk  None slices 4:" in out
    assert "  0 local experts hit" in out and "128 local experts hit" in out
    for name in ("served", "routing", "chunk", "slices"):
        assert f"VERDICT moe-layers {name}:" in out
    result = json.loads((tmp_path / "ml" / "moe_layers.json").read_text())
    assert len(result["verdicts"]) == 4 and len([k for k in result if k.startswith("local")]) == 9


def test_moe_dense_probe_matches_the_vendor_stand_in_and_times_every_form(tmp_path):
    proc, out = _run([PROBES / "probe_moe_dense.py", "--out", tmp_path / "md", "--layers", "1", "--iters", "1",
                      "--rows", "32", "--locals", "8"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    forms = [line for line in out.splitlines() if line.startswith("PROBE moe-dense local") and " form " in line]
    assert len(forms) == 5 and not any("ERROR" in line for line in forms), forms
    parity = [line for line in out.splitlines() if " parity dense" in line and " vs vendor: " in line]
    assert len(parity) == 3 and all(float(line.split("relative ")[1].split()[0]) < 0.05 for line in parity), parity
    assert "VERDICT moe-dense: local8_rows32:" in out


def test_moe_grouped_probe_matches_the_dense_form_and_counts_overflow(tmp_path):
    proc, out = _run([PROBES / "probe_moe_grouped.py", "--out", tmp_path / "mg", "--layers", "1", "--iters", "1",
                      "--rows", "16", "--caps", "2,16", "--local", "8"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    forms = [line for line in out.splitlines() if line.startswith("PROBE moe-grouped rows") and " form " in line]
    assert len(forms) == 4 and not any("ERROR" in line for line in forms), forms  # chunk 64, dense, C = 2, C = 16
    parity = {line.split(" parity ")[1].split(" vs ")[0]: float(line.split("relative ")[1].split(",")[0].split()[0])
              for line in out.splitlines() if " parity " in line and " vs vendor: " in line}
    assert set(parity) == {"dense", "grouped C=2", "grouped C=16"} and max(parity.values()) < 0.05, parity
    assert parity["grouped C=16"] == pytest.approx(parity["dense"], rel=1e-3)  # no pair beyond C: the dense result
    assert " overflow pairs 0" in out and "layers with a pair beyond C: C=2" in out
    ops = {line.split(" form ")[1].split(":")[0]: int(line.split(": ")[1].split()[0])
           for line in out.splitlines() if " aten ops per layer" in line}
    assert set(ops) == {"dense", "grouped"}, ops
    assert "VERDICT moe-grouped: rows 16: vendor-chunk64" in out


def test_moe_forms_probe_runs_every_form_and_the_exact_forms_match_dense(tmp_path):
    proc, out = _run([PROBES / "probe_moe_forms.py", "--out", tmp_path / "mf", "--layers", "2", "--iters", "1",
                      "--big-rows", "64", "--small-rows", "3", "--local", "32"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    timed = [line for line in out.splitlines() if "us per layer wall" in line]
    assert len(timed) == 8 and not any("ERROR" in line for line in out.splitlines()), out[-3000:]
    against = {line.split(" form ")[1].split()[0]: float(line.split("mean ")[1].split()[0])
               for line in out.splitlines() if " x even " in line and " against dense:" in line}
    assert against["flat13"] < 1e-6 and against["flat13-mmsum"] < 1e-3  # (a) and the ones-matmul sum are exact
    assert against["flatdown"] > 0  # (c2) rounds W2 twice
    assert "form vendor-t  " in out and "VERDICT moe-forms" in out


def test_view_probe_selftest(tmp_path):
    proc, out = _run([PROBES / "probe_view_inputs.py", "--selftest"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    assert "VERDICT: VIEWS WORK" in out
    assert "IDENTITY VIEWS: 5 of 5 cases give the op the right rows" in out
    assert "OP OUTPUTS: 4 of 4 cases need no copy" in out


def test_index_narrowing_probe_is_right_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_index_narrowing.py", "--out", tmp_path / "narrow"], tmp_path)
    assert proc.returncode == 0, out[-3000:]
    states = ("RIGHT", "WRONG", "ERROR")
    cases = [line for line in out.splitlines()
             if line.startswith("PROBE narrow") and any(f": {state} (" in line for state in states)]
    # 66 logprobs, 8 served, 12 pattern, 18 fix; a boundary case prints only when it is not RIGHT
    assert len(cases) == 66 + 38, cases
    assert all(": RIGHT (" in line for line in cases), [line for line in cases if ": RIGHT (" not in line]
    assert "VERDICT logprobs: 0 of 66 top-logprob cases are not the CPU's: none" in out
    assert out.index("VERDICT logprobs:") < out.index("PROBE narrow") + out[out.index("PROBE narrow"):].index(
        "served select_blocks")  # the logprobs part runs first: a job cut at its hold keeps it
    logprobs = [line for line in out.splitlines() if "] LOGPROBS " in line]
    assert len(logprobs) == 22 and all(line.endswith(": RRR") for line in logprobs), logprobs
    assert any("columns 248320 k  5" in line for line in logprobs)
    assert "VERDICT select_blocks: select_blocks on the card: RIGHT in every case" in out
    assert "VERDICT boundary: 0 of 188 boundary cases are not the CPU's" in out
    matrix = [line for line in out.splitlines() if "] BOUNDARY " in line and " | " in line]
    assert len(matrix) == 47 and all(line.endswith(": RRRR") for line in matrix), matrix
    result = json.loads((tmp_path / "narrow" / "index_narrowing.json").read_text())
    assert len(result["results"]) == 66 + 38 + 188 and len(result["verdicts"]) == 5
    assert len(result["matrix"]) == 47 and len(result["logprobs_matrix"]) == 22


def test_index_narrowing_probe_finds_the_plugin_file_from_a_copy_outside_the_tree(tmp_path):
    """Job 00106 ran a copy of the probe from /results/probe-src/: `HERE.parents[3]` raised IndexError at import."""
    code = ("import importlib.util, sys; from pathlib import Path; "
            "spec = importlib.util.spec_from_file_location('p', sys.argv[1]); "
            "m = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(m); "
            "print('PATH', m.sparse_ops_path(Path('/results/probe-src/probe_index_narrowing.py'), cwd=sys.argv[2])); "
            "print('OPS', hasattr(m.sparse_ops(), 'select_blocks'))")
    copy = tmp_path / "probe_index_narrowing.py"
    copy.write_text((PROBES / "probe_index_narrowing.py").read_text())
    proc = subprocess.run([sys.executable, "-c", code, str(copy), str(ROOT)], cwd=ROOT, capture_output=True, text=True,
                          timeout=300, env=dict(os.environ, PROBE_DEVICE="cpu"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"PATH {ROOT / 'vllm_gaudi' / 'models' / 'qwen4_exp' / 'qsa_sparse_ops.py'}" in proc.stdout
    assert "OPS True" in proc.stdout


def test_hc_copies_probe_modes_are_bit_identical_on_cpu(tmp_path):
    proc, out = _run([PROBES / "probe_hc_copies.py", "--out", tmp_path / "hc", "--calls", "2", "--iters", "1"],
                     tmp_path)
    assert proc.returncode == 0, out[-3000:]
    rows = [line for line in out.splitlines() if line.startswith("PROBE hc-copies rows")]
    assert len(rows) == 9, rows  # copy modes 0, 1, 2 at 1, 32 and 128 rows
    assert all("bit-identical to mode 0: True" in line for line in rows), rows
    assert "VERDICT hc-copies:" in out and "RAISED" not in out


def test_decode_ab_client_compares_the_one_stream_rounds(tmp_path):
    a = [{"round": "meas N=1 rep=0", "tokens": 4, "ctx_tokens": 0, "texts": ["one two three four"]},
         {"round": "meas N=2 rep=0", "tokens": 4, "ctx_tokens": 0, "texts": ["a b c d", "e f g h"]}]
    b = [dict(a[0]), dict(a[1], texts=["a b X d", "e f g h"])]
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    client = PROBES / "decode_ab_client.py"
    proc, out = _run([client, "--compare", tmp_path / "a.json", tmp_path / "b.json"], tmp_path)
    assert proc.returncode == 0, out
    assert "COMPARE meas N=1 rep=0         1 of 1 streams returned the same text" in out
    assert "COMPARE meas N=2 rep=0         1 of 2 streams returned the same text; shortest common prefix 4" in out
    assert "1-stream rounds (the test): the same text in 1 of 1" in out
    source = client.read_text()
    assert '"min_tokens"] =' not in source and "min_tokens\": ntok" not in source  # no request carries it


def _fake_moe_rt_server(tmp_path, drop_lines=True):
    """A stand-in of the served model for moe_rt_driver.py: streamed completions (two tokens per chunk after the
    first), prompt logprobs that depend on the switch file's setting, /tokenize (one token per word), and the
    workers' drop lines in server0.log whenever a request sees a changed switch file."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    switch = tmp_path / "switches"
    log = tmp_path / "server0.log"
    log.write_text("(Worker_TP0_EP0 pid=1) WARNING VLLM_HPU_MOE_DENSE=1: 48 MoE layers keep their stacked experts\n")
    state = {"stamp": None, "low": 513}

    def poll():
        try:
            status = os.stat(switch)
        except OSError:
            return
        stamp = (status.st_mtime_ns, status.st_size, status.st_ino)
        if stamp != state["stamp"]:
            state["stamp"] = stamp
            for line in switch.read_text().splitlines():
                if line.startswith("VLLM_HPU_MOE_DENSE_MIN_TOKENS="):
                    state["low"] = int(line.split("=", 1)[1])
            if drop_lines:
                with open(log, "a") as handle:
                    for rank in range(8):
                        handle.write("(Worker_TP%d_EP%d pid=1) INFO Runtime switches: flags read inside the forward "
                                     "changed; captured HPU graphs dropped: model, drafter\n" % (rank, rank))
                    if state["low"] <= 512:
                        handle.write("(Worker_TP0_EP0 pid=1) WARNING VLLM_HPU_MOE_DENSE=1: a step of 4 tokens runs the "
                                     "dense MoE form\n")

    def ids_of(text):
        return [1] + [sum(map(ord, word)) % 50000 + 2 for word in text.split()]

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *args):
            pass

        def _json(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            poll()
            if self.path == "/tokenize":
                return self._json({"tokens": ids_of(body["prompt"]), "count": len(ids_of(body["prompt"]))})
            if body.get("prompt_logprobs"):
                ids = ids_of(body["prompt"])
                shift = 0.001 if state["low"] <= 512 else 0.0  # the dense form moves the logprobs a little
                entries = [None]
                for position, token in enumerate(ids[1:], 1):
                    entry = {str(token): {"logprob": -1.0 - (position % 7) * 0.1 + shift, "rank": 1 + position % 3}}
                    if position % 3:
                        entry["7"] = {"logprob": -0.5, "rank": 1}
                    entries.append(entry)
                return self._json({"choices": [{"text": "x", "prompt_logprobs": entries}]})
            if not body.get("stream"):
                return self._json({"choices": [{"text": "ok"}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            left = body["max_tokens"]
            first = True
            while left > 0:
                n = 1 if first else min(2, left)
                first = False
                left -= n
                chunk = {"choices": [{"text": " w%d" % left * n}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
                import time
                time.sleep(0.002)
            usage = {"choices": [], "usage": {"completion_tokens": body["max_tokens"], "prompt_tokens": 9}}
            self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, switch


@pytest.mark.parametrize("drop_lines", [True, False])
def test_moe_rt_driver_switches_measures_and_compares(tmp_path, drop_lines):
    server, switch = _fake_moe_rt_server(tmp_path, drop_lines)
    try:
        proc, out = _run([PROBES / "moe_rt_driver.py", "--port", server.server_address[1], "--out",
                          tmp_path / "scaling.json", "--levels", "1,2", "--sampled-levels=1", "--interleave-sampled"],
                         tmp_path, MOE_RT_SWITCH_FILE=str(switch), MOE_RT_SETTINGS="off,1", MOE_RT_LEVELS="1,3",
                         MOE_RT_ROUNDS="1", MOE_RT_TOKENS="9", MOE_RT_WARM="3", MOE_RT_ACCURACY="1",
                         MOE_RT_ACC_PROMPTS="2", MOE_RT_ACC_ORDER="1,off,1")
    finally:
        server.shutdown()
    if not drop_lines:  # a switch that reached no worker's graphs: no verdict
        assert proc.returncode == 9, out
        assert "moe_rt: FAIL: the switch to off reached no worker's graphs" in out
        return
    assert proc.returncode == 0, out
    result = json.loads((tmp_path / "moe_rt.json").read_text())
    assert json.loads((tmp_path / "scaling.json").read_text())["levels"] == result["levels"]
    cells = {(m["setting"], m["streams"], m["prompt_set"]) for m in result["levels"] if m["ok"] and not m["errors"]}
    assert cells == {(s, n, p) for s in ("off", "1") for n in (1, 3) for p in ("story", "code")}
    for m in result["levels"]:
        assert m["tokens_per_step_p50"] == pytest.approx(2.0)  # (9 - 1) / 4 chunks after the first
    assert [s["workers"] for s in result["switches"]] == [8, 8]
    assert "VLLM_HPU_MOE_DENSE_MIN_TOKENS=1\n" in switch.read_text()  # the accuracy runs ended on the dense form
    pairs = {(p["a"], p["b"]): p for p in result["accuracy"]["pairs"]}
    assert pairs[("1#0", "off#1")]["mean_abs"] == pytest.approx(0.001)
    assert pairs[("1#0", "1#2")]["max_abs"] == 0.0
    assert pairs[("1#0", "off#1")]["top1_agree"] == 1.0
    assert len(result["accuracy"]["prompts"]) == 2 and all(300 <= p["tokens"] <= 480 for p in result["accuracy"]["prompts"])
    assert "ACC  1#0" in out and "MEAS  off  N=3" in out and "TEXT  N=1" in out
