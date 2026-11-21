# SPDX-License-Identifier: Apache-2.0
"""graph_input_log (VLLM_HPU_LOG_UNMATCHED_GRAPH_INPUTS): the HPU graph wrapper's copy_to, wrapped and logged."""
import os
import sys
import types

os.environ.setdefault("VLLM_PLUGINS", "")

import torch  # noqa: E402

from vllm_gaudi.v1.worker import graph_input_log  # noqa: E402


def test_the_copied_inputs_are_logged_with_their_paths(monkeypatch):
    graphs = types.ModuleType("habana_frameworks.torch.hpu.graphs")

    def copy_to(dst, src):
        if isinstance(src, dict):
            for key in src:
                graphs.copy_to(dst[key], src[key])  # recursion through the module global, as the wrapper does
            return
        dst.copy_(src)

    graphs.copy_to = copy_to
    for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.hpu"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.hpu.graphs", graphs)
    monkeypatch.setattr(graph_input_log, "_installed", False)
    lines = []
    assert graph_input_log.install(lambda fmt, *args: lines.append(fmt % args))
    src = {"positions": torch.arange(8)[2:6], "ids": torch.ones(4)}
    dst = {"positions": torch.zeros(4, dtype=torch.long), "ids": torch.zeros(4)}
    for _ in range(100):
        graphs.copy_to(dst, src)
    assert dst["positions"].tolist() == [2, 3, 4, 5]
    assert len(lines) == 2  # the 1st and the 100th time, one line per outer call (not per recursion)
    assert "input.positions [4] int64 offset 2" in lines[0] and "input.ids [4] float32 offset 0" in lines[0]
    assert "(x100)" in lines[1]
