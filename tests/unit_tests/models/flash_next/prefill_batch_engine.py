# SPDX-License-Identifier: Apache-2.0
"""A step-driven CPU rehearsal of the engine for batched prefill (``VLLM_HPU_PREFILL_BATCH``).

``tests/unit_tests/hpu_cpu_engine.py`` sends prompts through ``LLM.generate`` (all at once or one by one). Batched
prefill needs arrivals at chosen engine steps: a burst of turns, a prompt that arrives while other requests decode,
an agent's next turn on its own answer (a prefix-cache hit). This driver runs the same in-process engine (TP1, the
HPU platform, worker and model runner on CPU, the plugin's ops; ``hpu_cpu_engine.install``) with
``LLMEngine.add_request`` / ``step`` and records every prompt forward the runner forms (its requests, query and
context lengths). Run it in a process of its own (``run``).

Plan (JSON): ``{"max_num_seqs", "max_num_batched_tokens", "max_model_len", "k", "logprobs", "requests": [...]}``;
a request is ``{"id", "max_tokens", "prompt_ids"}`` or ``{"id", "max_tokens", "turn_of": ID, "extra_ids"}`` (the
prompt of ID, its generated tokens and ``extra_ids``: the next turn of an agent) and arrives at
``"arrive_step": N`` (the first engine step at or after N) or ``"after": ID`` (the step after ID finished;
``turn_of`` implies it).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tests.unit_tests import hpu_cpu_engine

REPO_ROOT = hpu_cpu_engine.REPO_ROOT


def _row_slots(indices, row: int):
    """The recurrent state slots of one row of a prompt batch's load or store indices ([groups, rows] or
    [groups, rows, width]; [rows] without groups), as a list; None when the batch has none."""
    if indices is None:
        return None
    t = indices.detach().cpu()
    if t.dim() == 1:
        return [int(t[row])]
    return t.select(1, row).reshape(-1).tolist()


def _record_prompt_forwards(runner, clock: dict) -> list[dict]:
    """Every prompt batch ``_form_prefill_batch`` builds: the step, its requests, query and context lengths, and
    each row's recurrent state slots (load and store, per cache group)."""
    records: list[dict] = []
    real = runner._form_prefill_batch

    def recording(contents):
        rows = len(contents.req_ids)
        record = None
        if rows > 0:
            record = {
                "step": clock["step"],
                "req_ids": list(contents.req_ids),
                "query_lens": [len(t) for t in contents.token_ids],
                "context_lens": [int(c) for c in contents.context_lens],
                "logits": [len(p) for p in contents.logits_positions],
            }
        result = real(contents)
        if record is not None:
            md = result.attn_metadata[0] if result.attn_metadata else None
            load = getattr(md, "load_indices_tensor", None)
            store = getattr(md, "store_indices_tensor", None)
            record["load_slots"] = [_row_slots(load, i) for i in range(rows)]
            record["store_slots"] = [_row_slots(store, i) for i in range(rows)]
            records.append(record)
        return result

    runner._form_prefill_batch = recording
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.unit_tests.models.flash_next.prefill_batch_engine")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-prefix-caching", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    hpu_cpu_engine.install()
    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.ckpt,
                  tokenizer=args.ckpt,
                  skip_tokenizer_init=True,
                  dtype="bfloat16",
                  max_model_len=int(plan.get("max_model_len", 4096)),
                  max_num_seqs=int(plan.get("max_num_seqs", 8)),
                  language_model_only=True,
                  trust_remote_code=True,
                  enable_prefix_caching=not args.no_prefix_caching)
    if plan.get("max_num_batched_tokens"):
        kwargs["max_num_batched_tokens"] = int(plan["max_num_batched_tokens"])
    if plan.get("long_prefill_token_threshold"):
        kwargs["long_prefill_token_threshold"] = int(plan["long_prefill_token_threshold"])
    k = int(plan.get("k", 0))
    if k:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": k}
    llm = LLM(**kwargs)
    runner = hpu_cpu_engine.find_runner(llm)
    clock = {"step": 0}
    forwards = _record_prompt_forwards(runner, clock)
    engine = llm.llm_engine

    requests = {r["id"]: dict(r) for r in plan["requests"]}
    order = [r["id"] for r in plan["requests"]]
    pending = list(order)
    finished: dict[str, dict] = {}
    finished_at: dict[str, int] = {}
    logprobs = plan.get("logprobs")
    guard = hpu_cpu_engine.make_index_guard()

    def ready(rid: str, step: int) -> bool:
        r = requests[rid]
        source = r.get("turn_of") or r.get("after")
        if source is not None:
            return source in finished_at and finished_at[source] < step
        return int(r.get("arrive_step", 0)) <= step

    def prompt_of(rid: str) -> list[int]:
        r = requests[rid]
        if "turn_of" in r:
            source = finished[r["turn_of"]]
            return list(source["prompt_ids"]) + list(source["generated_ids"]) + list(r.get("extra_ids", []))
        return list(r["prompt_ids"])

    with guard:
        step = 0
        limit = int(plan.get("max_steps", 2000))
        while (pending or engine.has_unfinished_requests()) and step < limit:
            for rid in list(pending):
                if ready(rid, step):
                    prompt = prompt_of(rid)
                    requests[rid]["prompt_ids_used"] = prompt
                    params = SamplingParams(temperature=0.0, max_tokens=int(requests[rid].get("max_tokens", 8)),
                                            detokenize=False, logprobs=logprobs)
                    engine.add_request(rid, {"prompt_token_ids": prompt}, params)
                    requests[rid]["added_at"] = step
                    pending.remove(rid)
            clock["step"] = step
            if engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        result = {
                            "id": out.request_id,
                            "prompt_ids": requests[out.request_id]["prompt_ids_used"],
                            "generated_ids": list(out.outputs[0].token_ids),
                            "num_cached_tokens": getattr(out, "num_cached_tokens", None),
                            "added_at": requests[out.request_id]["added_at"],
                            "finished_at": step,
                        }
                        if logprobs:
                            result["top_logprobs"] = [{str(t): float(lp.logprob) for t, lp in pos.items()}
                                                      for pos in (out.outputs[0].logprobs or [])]
                        finished[out.request_id] = result
                        finished_at[out.request_id] = step
            step += 1
    record = {
        "plan": plan,
        "streams": [finished[rid] for rid in order if rid in finished],
        "unfinished": [rid for rid in order if rid not in finished],
        "prompt_forwards": forwards,
        "steps": step,
        "index_events": guard.events(),
        "runner": hpu_cpu_engine.runner_facts(runner),
        "prefill_batch_rows": int(getattr(runner, "prefill_batch_rows", 0)),
    }
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")
    return 0


def run(ckpt: Path, plan: dict, out: Path, *options: str, env: dict | None = None, timeout_s: float = 1800.0) -> dict:
    """Run one plan in a process of its own (``hpu_cpu_engine.ENGINE_ENV`` + ``env``) and return its record."""
    out = Path(out)
    plan_path = Path(str(out) + ".plan.json")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    run_env = dict(os.environ)
    run_env.update(hpu_cpu_engine.ENGINE_ENV)
    run_env.update(env or {})
    inherited = [entry for entry in run_env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    run_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + inherited)
    command = [
        sys.executable, "-m", "tests.unit_tests.models.flash_next.prefill_batch_engine", "--ckpt",
        str(ckpt), "--plan",
        str(plan_path), "--out",
        str(out), *options
    ]
    log = Path(str(out) + ".log")
    with open(log, "wb") as stream:
        done = subprocess.run(command, cwd=REPO_ROOT, env=run_env, stdout=stream, stderr=subprocess.STDOUT,
                              timeout=timeout_s, check=False)
    if done.returncode != 0 or not out.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"the engine rehearsal failed (exit {done.returncode}, log {log}):\n{tail}")
    return json.loads(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
