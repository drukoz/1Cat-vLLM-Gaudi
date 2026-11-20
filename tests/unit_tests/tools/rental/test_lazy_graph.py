# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/flash_next/rental/lazy_graph.py: reading the lazy bridge's "Failing Graph" dumps.

The dumps below are cut from the card logs of 2026-11-20 (Gaudi2, bridge 1.24.1, PT_HPU_LAZY_ACC_PAR_MODE=1):
the hyper-connection card test, the GDN replay tool and a served tiny model. Lines that do not matter to
the failing tensor were removed and no line was rewritten; the one dump that joins lines of two logs says so.
"""

from __future__ import annotations

from tools.flash_next.rental import lazy_graph

# tests/unit_tests/hc_fused/test_hpu_op.py::test_hpu_op_matches_reference[norm_tiny_r1], as pytest prints it.
HC_NORM = """\
>   got = {name: tensor.cpu() for name, tensor in _run(case, hpu).items()}
E   RuntimeError: Empty tensor optional Uniqueid: 3 Strided Params Has Value: 0 Failing Graph: IR {
E      %id_24_prim__constant Unique ID:0 = n6_prim::constant{} , value=4
E      %id_22_prim__constant Unique ID:0 = n5_prim::constant{} , value=9.9999999999999995e-07
E      %id_12_hpu__input Unique ID:2 = n3_hpu::input{}
E      %id_20_hpu__input Unique ID:3 = n4_hpu::input{}
E      %id_17_flashinfer_gaudi__hc_norm Unique ID:4 = n7_flashinfer_gaudi::hc_norm{id_20_hpu__input Uniqueid: 3 id_12_hpu__input Uniqueid: 2 id_22_prim__constant id_24_prim__constant } , ROOT=0
E      %id_27_prim__constant Unique ID:0 = n8_prim::constant{} , value=0
E      %id_5_hpu__input Unique ID:1 = n1_hpu::input{}
E      %id_20_aten__clone Unique ID:3 = n9_aten::clone{id_5_hpu__input Uniqueid: 1 id_27_prim__constant } , ROOT=1
E   }

tests/unit_tests/hc_fused/test_hpu_op.py:47: RuntimeError
"""

# tools/flash_next/gdn_core_decode_replay.py, op `recurrent`, the other wording of the same failure.
GDN_REPLAY = """\
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_LAZY Error, ValidateSyncInputTensors tensor_data is empty. Tensorid:6018 QueueStatus: ThreadPool m_tasks size: 0 irValue:id_79120_hpu__input Uniqueid: 6018 Strided Params Has Value: 0 Failing Graph: IR {
   %id_79122_prim__constant Unique ID:0 = n27026_prim::constant{} , value=0.088388347648318447
   %id_117_hpu__input Unique ID:17 = n33_hpu::input{}
   %id_79120_hpu__input Unique ID:6018 = n27025_hpu::input{}
   %id_79117_flashinfer_gaudi__gdn_recurrent_decode Unique ID:6019 = n27027_flashinfer_gaudi::gdn_recurrent_decode{id_79120_hpu__input Uniqueid: 6018 id_117_hpu__input Uniqueid: 17 id_79122_prim__constant }
   %id_12_hpu__input Unique ID:2 = n3_hpu::input{}
   %id_79120_aten__clone Unique ID:6018 = n27028_aten::clone{id_12_hpu__input Uniqueid: 2 } @1=None
}
"""

# A served tiny model with the fused hyper-connection kernels (04:35 UTC): the dump as a worker logs it. The
# first input of hc_combine is linked to its clone, the second and third are graph inputs although their clones
# are in the graph. The last three lines are from another dump of the same day (the GDN server of 06:19 UTC): an
# in-place op on a real graph input, which writes the same Unique ID twice and is no failure.
_P = "(Worker_TP4 pid=93150) ERROR 11-20 04:35:28 [multiproc_executor.py:1055] "
HC_SERVER = "".join(_P + line + "\n" for line in """\
RuntimeError: Empty tensor optional Uniqueid: 3275 Strided Params Has Value: 0 Failing Graph: IR {
   %id_21342_model/1/prim__constant Unique ID:0 = n3046_prim::constant{} , scope=model/1, value=0
   %id_21307_model/0/mlp/experts/hpu__input Unique ID:3272 = n3041_hpu::input{} , scope=model/0/mlp/experts
   %id_21315_model/1/aten__view Unique ID:3273 = n3044_aten::view{id_21307_model/0/mlp/experts/hpu__input Uniqueid: 3272 } , scope=model/1, View Size = [128, 128]
   %id_21341_model/1/aten__clone Unique ID:3275 = n3047_aten::clone{id_21315_model/1/aten__view Uniqueid: 3273 id_21342_model/1/prim__constant } , scope=model/1, ROOT=0
   %id_21347_model/1/prim__constant Unique ID:0 = n3049_prim::constant{} , scope=model/1, value=4
   %id_21345_model/1/hpu__input Unique ID:3276 = n3048_hpu::input{} , scope=model/1
   %id_21341_model/1/hpu__input Unique ID:3275 = n3045_hpu::input{} , scope=model/1
   %id_21334_model/1/prim__constant Unique ID:0 = n3042_prim::constant{} , scope=model/1, value=0
   %id_21084_model/0/mlp/experts/hpu__input Unique ID:3239 = n2932_hpu::input{} , scope=model/0/mlp/experts
   %id_21325_model/1/aten__clone Unique ID:3274 = n3043_aten::clone{id_21084_model/0/mlp/experts/hpu__input Uniqueid: 3239 id_21334_model/1/prim__constant } , scope=model/1
   %id_21337_model/1/flashinfer_gaudi__hc_combine Unique ID:3277 = n3050_flashinfer_gaudi::hc_combine{id_21325_model/1/aten__clone id_21341_model/1/hpu__input Uniqueid: 3275 id_21345_model/1/hpu__input Uniqueid: 3276 id_21347_model/1/prim__constant } , scope=model/1, ROOT=1
   %id_21351_model/1/prim__constant Unique ID:0 = n3051_prim::constant{} , scope=model/1, value=0
   %id_21086_model/0/mlp/experts/hpu__input Unique ID:3244 = n2933_hpu::input{} , scope=model/0/mlp/experts
   %id_21345_model/1/aten__clone Unique ID:3276 = n3052_aten::clone{id_21086_model/0/mlp/experts/hpu__input Uniqueid: 3244 id_21351_model/1/prim__constant } , scope=model/1, ROOT=2
   %id_45198_model/1/linear_attn/hpu__input Unique ID:6298 = n9648_hpu::input{} , scope=model/1/linear_attn
   %id_45198_model/1/linear_attn/hpu__control_edge_ Unique ID:6298 = n9649_hpu::control_edge_{id_45198_model/1/linear_attn/hpu__input Uniqueid: 6298 } , scope=model/1/linear_attn
   %id_45198_model/1/linear_attn/aten__zero_ Unique ID:6298 = n9650_aten::zero_{id_45198_model/1/linear_attn/hpu__control_edge_ Uniqueid: 6298 } , scope=model/1/linear_attn
}
""".splitlines())


def test_the_card_test_dump_is_a_queued_producer():
    (graph, ) = lazy_graph.failing_graphs(HC_NORM)

    assert graph["error"] == "Empty tensor optional"
    assert graph["uid"] == 3
    assert graph["placeholder"] == "n4"  # the op's input, wired as a graph input
    assert graph["consumers"] == ["flashinfer_gaudi::hc_norm (n7)"]
    assert graph["producer"] == "aten::clone (n9)"  # the op that makes the same tensor, in the same graph
    assert graph["queued_producer"] is True


def test_the_other_wording_of_the_error_is_read_too():
    (graph, ) = lazy_graph.failing_graphs(GDN_REPLAY)

    assert graph["error"] == "ValidateSyncInputTensors tensor_data is empty"
    assert graph["uid"] == 6018
    assert graph["consumers"] == ["flashinfer_gaudi::gdn_recurrent_decode (n27027)"]
    assert graph["producer"] == "aten::clone (n27028)"
    assert graph["queued_producer"] is True


def test_a_server_log_with_prefixed_lines_and_a_scope_in_the_names():
    (graph, ) = lazy_graph.failing_graphs(HC_SERVER)

    assert graph["uid"] == 3275
    assert graph["placeholder"] == "n3045"
    assert graph["consumers"] == ["flashinfer_gaudi::hc_combine (n3050)"]
    # the producer was built between the placeholder and the op: two threads building one graph
    assert graph["producer"] == "aten::clone (n3047)"
    assert graph["queued_producer"] is True
    # every input of a custom op that is a graph input with a producer in the graph, not only the reported one
    assert graph["custom_op_inputs"] == [
        {"uid": 3275, "op": "flashinfer_gaudi::hc_combine (n3050)", "producer": "aten::clone (n3047)"},
        {"uid": 3276, "op": "flashinfer_gaudi::hc_combine (n3050)", "producer": "aten::clone (n3052)"},
    ]


def test_an_in_place_op_on_a_real_graph_input_is_not_a_queued_producer():
    """aten::zero_ writes Unique ID 6298 and reads it: the input is real, nothing is queued behind it."""
    text = HC_SERVER.replace("Uniqueid: 3275 Strided", "Uniqueid: 6298 Strided")

    (graph, ) = lazy_graph.failing_graphs(text)

    assert graph["uid"] == 6298 and graph["placeholder"] == "n9648"
    assert graph["producer"] is None
    assert graph["queued_producer"] is False


def test_an_input_nobody_produces_is_reported_as_such():
    text = "\n".join(line for line in HC_NORM.splitlines() if "aten::clone" not in line)

    (graph, ) = lazy_graph.failing_graphs(text)

    assert graph["placeholder"] == "n4" and graph["producer"] is None
    assert graph["queued_producer"] is False


def test_a_dump_repeated_by_the_worker_the_engine_and_the_api_server_counts_once():
    assert len(lazy_graph.failing_graphs(HC_SERVER + "some line\n" + HC_SERVER.replace("Worker_TP4", "EngineCore"))) == 1
    assert len(lazy_graph.failing_graphs(HC_NORM + GDN_REPLAY + HC_SERVER)) == 3


def test_a_log_without_a_dump():
    assert lazy_graph.failing_graphs("90 passed, 14 warnings in 122.06s (0:02:02)\n") == []
    assert lazy_graph.summary("90 passed") == {"dumps": 0, "queued_producer": 0, "ops": []}


def test_the_summary_names_the_ops():
    assert lazy_graph.summary(HC_NORM + GDN_REPLAY + HC_SERVER) == {
        "dumps": 3,
        "queued_producer": 3,
        "ops": ["flashinfer_gaudi::gdn_recurrent_decode", "flashinfer_gaudi::hc_combine", "flashinfer_gaudi::hc_norm"],
    }


def test_the_lines_say_what_happened():
    lines = lazy_graph.describe(HC_NORM)

    assert lines == [
        "tensor 3 (Empty tensor optional): flashinfer_gaudi::hc_norm (n7) took it as a graph input (n4) while "
        "aten::clone (n9) of the same graph produces it: the op was built before its input's producer "
        "(lazy accumulation thread, PT_HPU_LAZY_ACC_PAR_MODE=1)"
    ]
    other = lazy_graph.describe("\n".join(line for line in HC_NORM.splitlines() if "aten::clone" not in line))
    assert other == ["tensor 3 (Empty tensor optional): a graph input (n4) of flashinfer_gaudi::hc_norm (n7) that "
                     "nothing in the graph produces: not the accumulation race"]


def test_the_command_line(tmp_path, capsys):
    log = tmp_path / "hc_hpu_op.log"
    log.write_text(HC_NORM + GDN_REPLAY, encoding="utf-8")
    clean = tmp_path / "clean.log"
    clean.write_text("90 passed\n", encoding="utf-8")

    assert lazy_graph.main([str(log), str(clean)]) == 0

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "%s: 2 failing graphs, 2 with a queued producer" % log
    assert out[1].startswith("  tensor 3 (Empty tensor optional): flashinfer_gaudi::hc_norm (n7)")
    assert out[3] == "%s: no failing graph" % clean
