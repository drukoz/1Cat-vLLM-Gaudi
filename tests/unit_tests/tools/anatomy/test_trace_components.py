# SPDX-License-Identifier: Apache-2.0
"""trace_components.py on a synthetic HPU trace in the layout of job 00113's profile_api_light captures."""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools" / "flash_next" / "anatomy"))

import trace_components as tc  # noqa: E402


def _dev(tid, name, ts, dur, event_name):
    return {"ph": "X", "pid": 0, "tid": tid, "name": name, "ts": ts, "dur": dur, "cat": "kernel",
            "args": {"EventName": event_name}}


def _trace(steps=2):
    """Per step: a GDN node (10 us on TPC), 3 us gap, an all-reduce (HCL 4 us) inside a 6 us gap, a fused node
    without a path (2 us), 2 us gap, a vendor MoE GEMM chunk (8 us on MME) beside a MoE prep node (8 us on TPC, the
    same time), 1,000 us to the next step; one graph launch of 200 us per step on the host."""
    ev = [{"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "/device:HPU:0"}},
          {"ph": "M", "name": "process_name", "pid": 7, "args": {"name": "VLLM::Worker_TP0"}},
          {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "[D0] TPC 0"}},
          {"ph": "M", "name": "thread_name", "pid": 0, "tid": 2, "args": {"name": "[D0] MME"}},
          {"ph": "M", "name": "thread_name", "pid": 0, "tid": 3, "args": {"name": "[D3] Internal DMA 1 (HCL dedicated)"}}]
    t = 1000.0
    for _ in range(steps):
        ev.append({"ph": "X", "pid": 7, "tid": 7, "name": "step_timing:step", "ts": t - 100.0, "dur": 1040.0})
        ev.append({"ph": "X", "pid": 7, "tid": 7, "name": "ExecuteCachedGraph", "ts": t - 90.0, "dur": 200.0})
        ev.append(_dev(1, "gdn_fused_core_decode", t, 10.0,
                       "id_1_model/0/linear_attn/flashinfer_gaudi__gdn_core_decode Unique ID:5/x"))
        ev.append(_dev(3, "sibo stream_index 2 chunk_id 0", t + 11.0, 4.0, ""))
        ev.append(_dev(1, "fused_kernel_0x1_f32", t + 16.0, 2.0, "fusedTPCNode_0_1"))
        ev.append(_dev(2, "GEMM", t + 20.0, 8.0, "MOE/layer_0/chunk_1/id_9_model/0/mlp/experts/moe_op/x"))
        ev.append(_dev(1, "abs_fwd_bf16", t + 20.0, 8.0,
                       "id_3_model/0/mlp/experts/moe_op/aten__abs Unique ID:6/abs"))
        t += 1040.0
    return {"traceEvents": ev}


def test_components_gaps_and_launches(tmp_path):
    res = tc.analyze(_trace()["traceEvents"])
    assert res["steps"] == 2
    ex = res["exclusive_ms"]
    assert abs(ex["gdn"] - (10.0 + 2.0) / 1e3) < 1e-9  # the fused node without a path follows the GDN node
    assert abs(ex["collective"] - 4.0 / 1e3) < 1e-9
    assert abs(ex["moe_experts"] - 4.0 / 1e3) < 1e-9 and abs(ex["moe_prep"] - 4.0 / 1e3) < 1e-9  # 8 us shared
    assert abs(res["busy_ms"] - 24.0 / 1e3) < 1e-9  # 10 + 4 (HCL) + 2 + 8
    assert abs(res["collective_gap_ms"]["gdn"] - 6.0 / 1e3) < 1e-9  # 10..16 us after the GDN node, HCL inside
    assert abs(res["small_gap_ms"]["gdn"] - 2.0 / 1e3) < 1e-9  # 18..20 us after the fused node (given to gdn)
    assert res["launches"] == 1.0 and abs(res["launch_ms"] - 0.2) < 1e-9
    assert res["nodes"]["gdn"] == 2.0 and res["nodes"]["moe_experts"] == 1.0 and res["nodes"]["moe_prep"] == 1.0
    no = tc.analyze(_trace()["traceEvents"], assign=False)
    assert abs(no["exclusive_ms"]["unattributed"] - 2.0 / 1e3) < 1e-9


def test_component_paths():
    assert tc.component("model/3/self_attn/sparse_attn/aten__matmul") == "qsa_attention"
    assert tc.component("model/3/self_attn/index_qk_proj/aten__linear") == "qsa_proj"
    assert tc.component("model/3/self_attn/hpu__topk") == "qsa_indexer"
    assert tc.component("model/3/mlp/experts/_shared_experts/shared_expert/down_proj/aten__linear") == "moe_shared"
    assert tc.component("model/3/mlp/experts/moe_op/hpu__mixture_of_experts") == "moe_experts"
    assert tc.component("model/3/mlp/experts/gate/aten__linear") == "moe_router"
    assert tc.component("model/3/flashinfer_gaudi__hc_gate_mix") == "hc"
    assert tc.component("model/3/input_mix_weight_up/aten__linear") == "hc"
    assert tc.component("logits_processor/aten__matmul") == "logits"
    assert tc.component("") is None


def test_main_reads_gzip(tmp_path, capsys):
    p = tmp_path / "t.pt.trace.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(_trace(), fh)
    out = tmp_path / "r.json"
    assert tc.main([str(p), "--json", str(out)]) == 0
    assert "| gdn |" in capsys.readouterr().out
    assert json.loads(out.read_text())[0]["steps"] == 2
