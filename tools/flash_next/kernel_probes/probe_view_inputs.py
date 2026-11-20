#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: UP031, SIM105
"""probe_view_inputs.py: with the accumulation thread off (PT_HPU_LAZY_ACC_PAR_MODE=0), can a custom HPU op take a
view as its input, and does it then read the right rows?

Why it matters: commit 22cb014 clones every activation before a custom op (`_owned` in
flashinfer_gaudi/hc_fused.py and in vllm_gaudi/models/qwen3_5.py) because the first servers failed on views
("tensor_data is empty ... Strided Params Has Value: 1"). Those servers ran at mode 1, where the same error
comes from the accumulation race, so the clones may be copies nobody needs: one per activation, per layer,
per decode step.

Run on a Gaudi2 node, inside the container that has the plugin and habana_frameworks, with a free card:

    python probe_view_inputs.py            # under a minute; needs one free card
    python probe_view_inputs.py --selftest # no card: checks this script only, its verdict is not a result

What it does: one process at mode 0. It calls `flashinfer_gaudi::hc_silu` directly (no clone) on a fresh tensor
(the control), on a clone of a slice (what the plugin does today), and on seven views: a row slice at offset 0, a
row slice at a non-zero offset, a reshape of a 1-D tensor, a column slice of one row (what the GDN layer passes at
decode batch 1), and a row slice, a reshape and a one-row column slice of the result of an op that has not run
yet. Each case five times, each result compared with torch's silu of the same elements.

The last line is the verdict:
  VIEWS WORK   every view case returns the right rows: the clones can be removed at mode 0
  VIEWS FAIL   at least one view case raises or returns other rows' values (named): keep the clones
  NOT RUN      the control failed or the child could not run
"""
# ---------------------------------------------------------------------------------------------
# Common part (the same text in every probe_*.py of this folder; each script stands alone).
# ---------------------------------------------------------------------------------------------
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

RESULT_TAG = "PROBE_RESULT "
ACC_FLAG = "PT_HPU_LAZY_ACC_PAR_MODE"
SELFTEST = "--selftest" in sys.argv  # no card: CPU tensors and torch stand-ins; checks this script, not the bridge
BANNER_RE = re.compile(r"PT_HPU_LAZY_ACC_PAR_MODE\s*=\s*(\S+)")
EMPTY_INPUT_ERRORS = ("Empty tensor optional", "tensor_data is empty")
HPU_TORCH_ATOL, HPU_TORCH_RTOL = 5e-2, 1e-2  # bf16 against the torch path, as tests/unit_tests/hc_fused/test_hpu_op.py


def child_environment(acc_par):
    """The environment of one child: the lazy bridge, the accumulation flag as asked (None = unset)."""
    env = dict(os.environ)
    env["PT_HPU_LAZY_MODE"] = "1"
    env.pop("PT_HPU_RECIPE_CACHE_CONFIG", None)  # a recipe cache blocks the kernel library's registration
    env.setdefault("FLASHINFER_GAUDI_COMBINED_KERNEL_DB", "0")  # as the card test of 2026-11-20 ran
    for name in ("VLLM_HPU_HC_FUSED", "VLLM_HPU_MOE_GLUE"):  # the probe loads the kernels itself
        env.pop(name, None)
    # A plugin build with the loader's refusal (branch fix-hc) loads no custom op in a lazy process whose
    # accumulation thread is on; the probes run there on purpose. Older builds ignore the variable.
    env["FLASHINFER_GAUDI_ALLOW_LAZY_ACC_THREAD"] = "1"
    if acc_par is None:
        env.pop(ACC_FLAG, None)
    else:
        env[ACC_FLAG] = str(acc_par)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _as_text(value):
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def run_child(mode, acc_par, timeout_s, extra=()):
    """Run this script again as `--child mode` and return what it reported."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", mode, *extra]
    if SELFTEST:
        cmd.append("--selftest")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, env=child_environment(acc_par), cwd=tempfile.gettempdir(), capture_output=True,
                              text=True, timeout=timeout_s)
        out, err, code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, err, code, timed_out = _as_text(exc.stdout), _as_text(exc.stderr), None, True
    result = None
    for line in out.splitlines():
        if line.startswith(RESULT_TAG):
            try:
                result = json.loads(line[len(RESULT_TAG):])
            except ValueError:
                result = None
    banner = BANNER_RE.search(err + "\n" + out)
    return {
        "mode": mode,
        "acc_par_env": acc_par,
        "exit": code,
        "timed_out": timed_out,
        "seconds": round(time.monotonic() - started, 1),
        "result": result,
        "banner_acc_par": banner.group(1) if banner else None,
        "stderr_tail": err[-1200:],
    }


def child_failure(run):
    """One sentence on why a child gave no usable result, or None when it did."""
    result = run["result"]
    if run["timed_out"]:
        return "the child did not finish in time"
    if result is None:
        lines = [ln for ln in run["stderr_tail"].splitlines() if ln.strip()]
        return "the child exited %s without a result (%s)" % (run["exit"], lines[-1][:200] if lines else "no stderr")
    if result.get("error"):
        return "the child stopped: %s" % result["error"]
    return None


def emit(result):
    print(RESULT_TAG + json.dumps(result), flush=True)


def child_main(children):
    """Run the child named on the command line; whatever it raises becomes its result."""
    mode = sys.argv[sys.argv.index("--child") + 1]
    try:
        result = children[mode]()
    except BaseException as exc:  # noqa: BLE001 - the parent reports it
        result = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:600]), "traceback": traceback.format_exc()[-1500:]}
    emit(result)


def _find_installed_libraries():
    """Point the plugin's loader at the installed kernel library and op extension.

    Run from a source checkout, `import flashinfer_gaudi` finds a package whose lib/ folder is empty; the
    built files are in the installed copy. The loader honours these two variables.
    """
    if os.environ.get("HC_FUSED_GAUDI_KERNEL_LIBRARY") and os.environ.get("HC_FUSED_GAUDI_NATIVE_LIBRARY"):
        return
    import site
    roots = []
    try:
        import flashinfer_gaudi
        roots.append(Path(flashinfer_gaudi.__file__).resolve().parent)
    except Exception:  # noqa: BLE001 - reported by the loader below
        pass
    candidates = list(sys.path)
    try:
        candidates += site.getsitepackages() + [site.getusersitepackages()]
    except Exception:  # noqa: BLE001
        pass
    roots += [Path(entry) / "flashinfer_gaudi" for entry in candidates if entry]
    for root in roots:
        lib = root / "lib"
        kernel = lib / "libhc_fused_gaudi2.so"
        ops = sorted(lib.glob("hc_fused_ops*.so"))
        if kernel.is_file() and ops:
            os.environ.setdefault("HC_FUSED_GAUDI_KERNEL_LIBRARY", str(kernel))
            os.environ.setdefault("HC_FUSED_GAUDI_NATIVE_LIBRARY", str(ops[0]))
            return


# The torch functions the model runs today (vllm_gaudi/models/qwen4_exp/hyperconnection.py): the reference.
def ref_norm(x, weight, eps, hc):
    import torch
    group = x.shape[-1] // hc
    xg = x.float().unflatten(-1, (hc, group))
    y = xg * torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + eps)
    w = weight.float()
    w = w.view(group) if weight.numel() == group else w.view(hc, group)
    return (y + y * w).flatten(-2).to(x.dtype)


def ref_silu(x, hc):
    import torch
    xf = x.float() / hc
    return (xf * torch.sigmoid(xf)).to(x.dtype)


def child_setup(need_ops=True):
    """(torch, htorch or None, device): the bridge imported, and with `need_ops` the fused ops loaded."""
    import torch
    if SELFTEST:
        if need_ops:
            import flashinfer_gaudi.hc_fused  # noqa: F401 - the schemas
            library = torch.library.Library("flashinfer_gaudi", "IMPL")
            library.impl("hc_norm", ref_norm, "CPU")
            library.impl("hc_silu", ref_silu, "CPU")
            child_setup.keep = library
        return torch, None, "cpu"
    import habana_frameworks.torch as htorch  # before the op extension: it binds to the plugin mapped here
    if need_ops:
        _find_installed_libraries()
        from flashinfer_gaudi import _native
        if not _native.load_hc_fused_extension():
            raise RuntimeError("the fused hyper-connection extension did not load: " +
                               "; ".join(_native.hc_fused_extension_errors()))
    return torch, htorch, "hpu"


def flush(htorch):
    """Drop whatever a failed graph left queued, so the next trial starts clean."""
    if htorch is None:
        return
    try:
        htorch.core.mark_step()
    except Exception:  # noqa: BLE001 - the leftover graph is the one that just failed
        pass


def empty_input_signature(text):
    """What the failing graph in an error text shows about the empty tensor, or None when it has no dump.

    {"uid", "placeholder": True when the tensor is wired as hpu::input, "consumer": the custom op that takes
    it, "producer": the op of the same graph that produces the same Unique ID, "race": all three found}.
    """
    head = re.search(r"Uniqueid: (\d+)[^\n]*Failing Graph", text)
    if not head:
        return None
    uid = head.group(1)
    placeholder = consumer = producer = None
    for line in text[head.end():].splitlines():
        node = re.search(r"%(.*?) = n\d+_([A-Za-z_0-9]+::[A-Za-z_0-9]+)\{([^}]*)\}", line)
        if not node:
            continue
        outputs, kind, args = node.groups()
        if re.search(r"Unique ID:%s\b" % uid, outputs):
            if kind == "hpu::input":
                placeholder = outputs.split(" ")[0]
            else:
                producer = kind
        elif kind.startswith("flashinfer_gaudi::") and re.search(r"Uniqueid: %s\b" % uid, args):
            consumer = kind
    return {"uid": uid, "placeholder": placeholder is not None, "consumer": consumer, "producer": producer,
            "race": bool(placeholder and consumer and producer)}


def failing_tensor_lines(text, limit=12):
    """The lines of a failing-graph dump that name the empty tensor: its graph input, its consumers, its producer."""
    head = re.search(r"Uniqueid: (\d+)[^\n]*Failing Graph", text)
    if not head:
        return text[:600].splitlines()
    uid = head.group(1)
    first = text[:head.end()].splitlines()[-1][:300]
    named = [line.strip()[:400] for line in text[head.end():].splitlines()
             if re.search(r"(Unique ID:|Uniqueid: )%s\b" % uid, line)]
    return [first] + named[:limit]

CHILD_TIMEOUT_S = 100.0
REPEATS = 5
VIEW_CASES = ("row_slice_offset_0", "row_slice_offset_2", "reshape_of_1d", "one_row_column_slice",
              "row_slice_of_pending_op", "reshape_of_pending_op", "one_row_column_slice_of_pending_op")
# Views that change nothing (the same shape, strides and offset): what a model block's
# hidden_states.view(-1, hidden_dim) and a quant method's output.view(*input_shape) are for a [tokens, H] tensor.
# Reported in a line of their own (IDENTITY VIEWS ...); the verdict of the seven real views does not count them.
IDENTITY_CASES = ("identity_view", "identity_view_of_pending_op", "identity_reshape_of_pending_op",
                  "contiguous_of_identity_view", "unsqueeze_squeeze_of_pending_op")
CONTROL_CASES = ("fresh_tensor", "clone_of_row_slice")
# Tensors that are no views but have not run yet: what the fused ops get from the layers without a copy if the
# copies of flashinfer_gaudi.hc_fused._owned are kept for views only (line OP OUTPUTS ...).
OWNED_CASES = ("pending_op_output", "gemm_output", "custom_op_output", "clone_of_identity_view")


def views_child():
    torch, htorch, device = child_setup()
    hc, rows, columns = 4, 8, 320
    g = torch.Generator().manual_seed(23)
    host = (torch.randn(rows, columns, generator=g) * 6.0).to(torch.bfloat16)
    host[2] = host[2].abs() + 1.0  # row 2 differs from row 0 in every element: a wrong offset shows
    host[0] = -host[0].abs() - 1.0
    matrix = host.to(device)
    flat = host.reshape(-1).contiguous().to(device)
    eye = torch.eye(columns, dtype=torch.bfloat16).to(device)  # a GEMM that returns its input's values

    def build(case):
        """(the op's input on the device, the same rows on the host)."""
        if case == "fresh_tensor":
            return matrix, host
        if case == "clone_of_row_slice":
            return matrix[2:3].clone(), host[2:3]
        if case == "row_slice_offset_0":
            return matrix[0:1], host[0:1]
        if case == "row_slice_offset_2":
            return matrix[2:3], host[2:3]
        if case == "reshape_of_1d":
            return flat.view(rows, columns), host
        if case == "one_row_column_slice":  # contiguous because it is one row: what the GDN layer passes at batch 1
            return matrix[2:3, 64:192], host[2:3, 64:192]
        if case == "row_slice_of_pending_op":
            return (matrix * 1.0)[2:3], host[2:3]
        if case == "one_row_column_slice_of_pending_op":
            return (matrix * 1.0)[2:3, 64:192], host[2:3, 64:192]
        if case == "reshape_of_pending_op":
            return (flat * 1.0).view(rows, columns), host
        if case == "pending_op_output":  # an element-wise op's output: no view, not yet run
            return matrix * 1.0, host
        if case == "gemm_output":  # what a layer's GEMM hands a fused op
            return torch.nn.functional.linear(matrix, eye), host
        if case == "custom_op_output":  # a fused op's output as the next fused op's input
            return torch.ops.flashinfer_gaudi.hc_silu(matrix, hc), ref_silu(host, hc)
        if case == "clone_of_identity_view":
            return (matrix * 1.0).view(-1, columns).clone(), host
        if case == "identity_view":
            return matrix.view(-1, columns), host
        if case == "identity_view_of_pending_op":
            return (matrix * 1.0).view(-1, columns), host
        if case == "identity_reshape_of_pending_op":
            return (matrix * 1.0).reshape(rows, columns), host
        if case == "contiguous_of_identity_view":  # .contiguous() of a contiguous view returns the view itself
            return (matrix * 1.0).view(-1, columns).contiguous(), host
        if case == "unsqueeze_squeeze_of_pending_op":
            return (matrix * 1.0).unsqueeze(0).squeeze(0), host
        raise KeyError(case)

    results = {}
    for case in CONTROL_CASES + VIEW_CASES + IDENTITY_CASES + OWNED_CASES:
        tally = {"ok": 0, "error": 0, "wrong_values": 0, "first_error": None, "strided_params": None}
        for _ in range(REPEATS):
            try:
                value, rows_on_host = build(case)
                got = torch.ops.flashinfer_gaudi.hc_silu(value, hc).cpu().float()
                want = ref_silu(rows_on_host, hc).float()
                same = got.shape == want.shape and torch.allclose(got, want, atol=HPU_TORCH_ATOL, rtol=HPU_TORCH_RTOL)
                tally["ok" if same else "wrong_values"] += 1
            except Exception as exc:  # noqa: BLE001 - a refusal is an answer too
                tally["error"] += 1
                text = str(exc)
                if tally["first_error"] is None:
                    tally["first_error"] = "%s: %s" % (type(exc).__name__, text[:300])
                    strided = re.search(r"Strided Params Has Value: (\d)", text)
                    tally["strided_params"] = strided.group(1) if strided else None
                value = None
                flush(htorch)
        results[case] = tally
    return {"acc_par_env": os.environ.get(ACC_FLAG), "cases": results}


def verdict(run):
    why = child_failure(run)
    if why:
        return "NOT RUN", why
    cases = run["result"]["cases"]
    broken_controls = [name for name in CONTROL_CASES if cases[name]["ok"] != REPEATS]
    if broken_controls:
        return "NOT RUN", ("the control %s failed at mode 0 (%s), so the view cases say nothing" %
                           (", ".join(broken_controls), cases[broken_controls[0]]["first_error"] or "wrong values"))
    wrong = [name for name in VIEW_CASES if cases[name]["wrong_values"]]
    raised = [name for name in VIEW_CASES if cases[name]["error"]]
    if wrong:
        return "VIEWS FAIL", ("%s return other rows' values WITHOUT an error at mode 0: a view must never reach a "
                              "custom op; keep the clones%s" %
                              (", ".join(wrong), "" if not raised else " (%s raise)" % ", ".join(raised)))
    if raised:
        return "VIEWS FAIL", ("%d of %d view cases raise at mode 0 (%s; first: %s); keep the clones for those" %
                              (len(raised), len(VIEW_CASES), ", ".join(raised), cases[raised[0]]["first_error"]))
    return "VIEWS WORK", ("all %d view cases return the right rows at mode 0, %d calls each: the clones of 22cb014 "
                          "are not needed once accumulation is serial" % (len(VIEW_CASES), REPEATS))


def main():
    run = run_child("views", "0", CHILD_TIMEOUT_S)
    print("mode 0: %.0f s, bridge banner says %s=%s" % (run["seconds"], ACC_FLAG, run["banner_acc_par"]))
    if child_failure(run):
        print("    stderr tail: " + run["stderr_tail"][-600:].replace("\n", "\n    "))
    else:
        for name, tally in run["result"]["cases"].items():
            print("    %-36s ok %d, error %d, wrong values %d%s" %
                  (name, tally["ok"], tally["error"], tally["wrong_values"],
                   "" if not tally["first_error"] else "  [Strided Params %s] %s" %
                   (tally["strided_params"], tally["first_error"][:160])))
    word, sentence = verdict(run)
    if SELFTEST:
        print("SELFTEST ONLY (CPU, no bridge): the line below checks this script, it is not a result.")
    if not child_failure(run):
        cases = run["result"]["cases"]
        fine = [name for name in IDENTITY_CASES if cases[name]["ok"] == REPEATS]
        bad = ["%s (%s)" % (name, "raises" if cases[name]["error"] else "other values")
               for name in IDENTITY_CASES if cases[name]["ok"] != REPEATS]
        print("IDENTITY VIEWS: %d of %d cases give the op the right rows at mode 0 (%s); not: %s" %
              (len(fine), len(IDENTITY_CASES), ", ".join(fine) or "none", ", ".join(bad) or "none"))
        fine = [name for name in OWNED_CASES if cases[name]["ok"] == REPEATS]
        bad = ["%s (%s)" % (name, "raises" if cases[name]["error"] else "other values")
               for name in OWNED_CASES if cases[name]["ok"] != REPEATS]
        print("OP OUTPUTS: %d of %d cases need no copy at mode 0 (%s); not: %s" %
              (len(fine), len(OWNED_CASES), ", ".join(fine) or "none", ", ".join(bad) or "none"))
    print("VERDICT: %s - %s" % (word, sentence))
    return 2 if word == "NOT RUN" else 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        child_main({"views": views_child})
    else:
        sys.exit(main())
