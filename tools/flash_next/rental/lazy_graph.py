# SPDX-License-Identifier: Apache-2.0
"""Read the "Failing Graph: IR {...}" dumps the Habana lazy bridge prints with an empty input tensor.

The bridge (Gaudi 1.24.1, PT_HPU_LAZY_MODE=1) fails a graph with

    RuntimeError: Empty tensor optional Uniqueid: N Strided Params Has Value: 0 Failing Graph: IR {...}
    ... ValidateSyncInputTensors tensor_data is empty. Tensorid:N ... Failing Graph: IR {...}

when a graph input has no data. On 2026-11-20 every such failure of our custom ops had one shape: the op took
tensor N as a graph input (`hpu::input`) while the op that produces N (a clone, a linear, a cast, the vendor
MoE op) sat in the same graph with its own node. The op's node had been built before its input's producer:
with PT_HPU_LAZY_ACC_PAR_MODE=1 the bridge builds queued ops on a second thread, and a custom op
(`UserCustomOpDescriptor::execute`) is built at once on the calling thread. With the flag at 0 the same card
tests pass (`docs/design/flash-next/native-extensions.md`, "Custom ops and the lazy accumulation thread").

`failing_graphs(text)` gives one record per dump, `summary(text)` the counts the rental records, `describe(text)`
one sentence per dump. As a command it reads logs:

    python -m tools.flash_next.rental.lazy_graph <log>...

Python 3.10 standard library only.
"""

from __future__ import annotations

import re
import sys

_HEAD = re.compile(r"(Empty tensor optional|ValidateSyncInputTensors tensor_data is empty)[^\n]*?"
                   r"Uniqueid: (\d+)[^\n]*Failing Graph: IR \{")
_NODE = re.compile(r"%(?P<outs>\S.*?) = n(?P<n>\d+)_(?P<kind>\w+::\w+)\{(?P<args>[^}]*)\}")
_OUTPUT = re.compile(r"(\S+) Unique ID:(\d+)")
_BUILT_IN = ("aten", "hpu", "prim")  # every other namespace is a custom op


def _nodes(lines) -> list[dict]:
    nodes = []
    for line in lines:
        match = _NODE.search(line)
        if match:
            outputs = [(name.lstrip("%"), int(uid)) for name, uid in _OUTPUT.findall(match.group("outs"))]
            nodes.append({"n": int(match.group("n")), "kind": match.group("kind"), "outputs": outputs,
                          "args": " %s " % match.group("args")})
    return nodes


def _label(node: dict) -> str:
    return "%s (n%d)" % (node["kind"], node["n"])


def _producer(nodes: list[dict], uid: int) -> dict | None:
    """The node that makes tensor `uid` from other tensors, or None.

    An in-place op on a real graph input (`aten::zero_`, `hpu::control_edge_`) writes the Unique ID it reads;
    that is the same tensor updated, not a producer the input was waiting for.
    """
    for node in nodes:
        if node["kind"] == "hpu::input" or not any(out_uid == uid for _, out_uid in node["outputs"]):
            continue
        if not re.search(r"Uniqueid: %d\b" % uid, node["args"]):
            return node
    return None


def _placeholder(nodes: list[dict], uid: int) -> dict | None:
    for node in nodes:
        if node["kind"] == "hpu::input" and any(out_uid == uid for _, out_uid in node["outputs"]):
            return node
    return None


def _consumers(nodes: list[dict], placeholder: dict) -> list[dict]:
    name = placeholder["outputs"][0][0]
    return [node for node in nodes if node is not placeholder and " %s " % name in node["args"]]


def failing_graphs(text: str) -> list[dict]:
    """One record per distinct failing-graph dump in `text`, in order.

    {"error", "uid": the empty tensor, "placeholder": its `hpu::input` node or None, "consumers": the nodes
    that take it, "producer": the node of the same graph that produces it or None, "queued_producer": True
    when all three are there, "custom_op_inputs": every input of a custom op in the dump that is a graph input
    with a producer in the graph ({"uid", "op", "producer"})}. A dump a server repeats for its worker, its
    engine and its API server counts once.
    """
    graphs, seen = [], set()
    for head in _HEAD.finditer(text):
        lines = []
        for line in text[head.end():].split("\n")[1:]:
            if "%id_" not in line:
                break
            lines.append(line)
        nodes = _nodes(lines)
        uid = int(head.group(2))
        key = (uid, tuple(node["n"] for node in nodes))
        if key in seen:
            continue
        seen.add(key)
        placeholder = _placeholder(nodes, uid)
        producer = _producer(nodes, uid) if placeholder else None
        consumers = _consumers(nodes, placeholder) if placeholder else []
        custom_inputs = []
        for node in nodes:
            if node["kind"].split("::")[0] in _BUILT_IN:
                continue
            for other in nodes:
                if other["kind"] != "hpu::input" or " %s " % other["outputs"][0][0] not in node["args"]:
                    continue
                made_by = _producer(nodes, other["outputs"][0][1])
                if made_by is not None:
                    custom_inputs.append({"uid": other["outputs"][0][1], "op": _label(node),
                                          "producer": _label(made_by)})
        graphs.append({
            "error": head.group(1),
            "uid": uid,
            "placeholder": "n%d" % placeholder["n"] if placeholder else None,
            "consumers": [_label(node) for node in consumers],
            "producer": _label(producer) if producer else None,
            "queued_producer": bool(placeholder and consumers and producer),
            "custom_op_inputs": sorted(custom_inputs, key=lambda entry: entry["uid"]),
        })
    return graphs


def summary(text: str) -> dict:
    """{"dumps", "queued_producer", "ops"}: how many dumps `text` has, how many of them have the op built before
    its input's producer, and the ops that took such an input (sorted, without node numbers)."""
    graphs = failing_graphs(text)
    queued = [graph for graph in graphs if graph["queued_producer"]]
    ops = sorted({consumer.split(" (")[0] for graph in queued for consumer in graph["consumers"]})
    return {"dumps": len(graphs), "queued_producer": len(queued), "ops": ops}


def describe(text: str) -> list[str]:
    """One sentence per dump of `text`."""
    lines = []
    for graph in failing_graphs(text):
        start = "tensor %d (%s): " % (graph["uid"], graph["error"])
        consumers = ", ".join(graph["consumers"]) or "no node of the dump"
        if graph["queued_producer"]:
            lines.append(start + "%s took it as a graph input (%s) while %s of the same graph produces it: the op "
                         "was built before its input's producer (lazy accumulation thread, "
                         "PT_HPU_LAZY_ACC_PAR_MODE=1)" % (consumers, graph["placeholder"], graph["producer"]))
        elif graph["placeholder"]:
            lines.append(start + "a graph input (%s) of %s that nothing in the graph produces: not the accumulation "
                         "race" % (graph["placeholder"], consumers))
        else:
            lines.append(start + "not a graph input of the dump: not the accumulation race")
    return lines


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python -m tools.flash_next.rental.lazy_graph <log>...", file=sys.stderr)
        return 2
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        counts = summary(text)
        if not counts["dumps"]:
            print("%s: no failing graph" % path)
            continue
        print("%s: %d failing graphs, %d with a queued producer" % (path, counts["dumps"], counts["queued_producer"]))
        for line in describe(text):
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
