# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_WORKER_CPU_PIN: the bookkeeping of the workers' CPU shares (no card, no affinity call of the OS)."""

import os

import pytest

from vllm_gaudi.v1.worker import hpu_worker_cpus as cpus

# the card farm's containers: module m has CPUs 19m .. 19m+17 (the top CPU of each range is kept for image builds)
FARM_ALL = [c for m in range(8) for c in range(19 * m, 19 * m + 18)]
FARM_UPPER = [c for m in range(4, 8) for c in range(19 * m, 19 * m + 18)]


class Log:

    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(("info", msg % args))

    def warning(self, msg, *args):
        self.lines.append(("warning", msg % args))


def test_shares_are_equal_disjoint_and_in_order():
    allowed = set(range(152))
    shares = [cpus.worker_cpu_share(allowed, index, 8) for index in range(8)]
    assert shares[0] == list(range(0, 19)) and shares[7] == list(range(133, 152))
    assert all(len(share) == 19 for share in shares)
    assert sorted(cpu for share in shares for cpu in share) == list(range(152))


def test_shares_come_from_the_cpus_the_process_may_use():
    shares = [cpus.worker_cpu_share(FARM_UPPER, index, 4) for index in range(4)]
    assert shares[0] == list(range(76, 94)) and shares[3] == list(range(133, 151))
    assert [cpus.worker_cpu_share(FARM_ALL, m, 8) for m in range(8)] == [list(range(19 * m, 19 * m + 18))
                                                                         for m in range(8)]
    # CPUs that do not divide evenly: the rest stays with nobody
    assert cpus.worker_cpu_share(range(10), 2, 3) == [6, 7, 8]


@pytest.mark.parametrize("allowed, index, count",
                         [(range(4), 0, 8), (range(16), 8, 8), (range(16), -1, 8), (range(16), 0, 0)])
def test_no_share_when_it_cannot_be_given(allowed, index, count):
    assert cpus.worker_cpu_share(allowed, index, count) == []


@pytest.mark.parametrize("rank, world, module, visible, expected", [
    (0, 4, "4", "4,5,6,7", (0, 4)),  # the farm's TP4 server on modules 4-7 (server logs: rank r has module 4+r)
    (3, 4, "7", "4,5,6,7", (3, 4)),
    (2, 4, "6", "6,7,4,5", (2, 4)),  # listed in another order: the position among the sorted modules
    (5, 8, "5", "0,1,2,3,4,5,6,7", (5, 8)),
    (1, 2, "3", "2,3", (1, 2)),
    (1, 4, None, "4,5,6,7", (1, 4)),  # no module id: the rank
    (1, 4, "5", None, (1, 4)),  # no visible list: the rank
    (1, 4, "9", "4,5,6,7", (1, 4)),  # a module that is not visible: the rank
    (1, 4, "x", "4,5", (1, 4)),  # unreadable: the rank
])
def test_share_position_follows_the_module(rank, world, module, visible, expected):
    assert cpus.share_position(rank, world, module, visible) == expected


def test_format_cpus_writes_ranges():
    assert cpus.format_cpus([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert cpus.format_cpus([3]) == "3" and cpus.format_cpus([]) == ""


def test_pin_worker_moves_every_thread_to_the_share_and_logs_it():
    calls, log = [], Log()
    share = cpus.pin_worker(1, 4, cpus.PIN_SHARE, log, getaffinity=lambda pid: set(range(8)),
                            setaffinity=lambda tid, mask: calls.append((tid, list(mask))),
                            thread_ids=lambda: [100, 101, 102])
    assert share == [2, 3]
    assert calls == [(100, [2, 3]), (101, [2, 3]), (102, [2, 3])]
    assert log.lines == [("info", "VLLM_HPU_WORKER_CPU_PIN=1: share 1 of 4, CPUs 2-3 (2 of the 8 this server "
                          "may use; 3 threads moved)")]


def test_pin_worker_off_does_nothing():
    calls = []
    assert cpus.pin_worker(0, 8, cpus.PIN_OFF, Log(), getaffinity=lambda pid: set(range(152)),
                           setaffinity=lambda tid, mask: calls.append(tid), thread_ids=lambda: [1]) == []
    assert calls == []


def test_pin_worker_without_a_share_warns_and_changes_nothing():
    calls, log = [], Log()
    assert cpus.pin_worker(0, 8, cpus.PIN_SHARE, log, getaffinity=lambda pid: {0, 1},
                           setaffinity=lambda tid, mask: calls.append(tid), thread_ids=lambda: [1]) == []
    assert calls == [] and log.lines[0][0] == "warning" and "nothing is pinned" in log.lines[0][1]


def test_pin_worker_without_the_os_functions_warns(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    log = Log()
    assert cpus.pin_worker(0, 4, cpus.PIN_SHARE, log) == []
    assert log.lines[0][0] == "warning" and "no sched_setaffinity" in log.lines[0][1]


def test_pin_worker_skips_a_thread_that_is_gone():

    def setaffinity(tid, mask):
        if tid == 2:
            raise ProcessLookupError

    log = Log()
    assert cpus.pin_worker(0, 2, cpus.PIN_SHARE, log, getaffinity=lambda pid: set(range(4)), setaffinity=setaffinity,
                           thread_ids=lambda: [1, 2, 3]) == [0, 1]
    assert "2 threads moved" in log.lines[0][1]


def test_pin_from_env_gives_each_worker_the_cpus_of_its_module():
    calls, log = {}, Log()
    for rank in range(4):
        env = {"HLS_MODULE_ID": str(4 + rank), "HABANA_VISIBLE_MODULES": "4,5,6,7"}
        share = cpus.pin_from_env(rank, 4, cpus.PIN_SHARE, log, environ=env, getaffinity=lambda pid: set(FARM_UPPER),
                                  setaffinity=lambda tid, mask: calls.setdefault(rank, list(mask)),
                                  thread_ids=lambda: [10])
        assert share == list(range(19 * (4 + rank), 19 * (4 + rank) + 18))
        assert calls[rank] == share
    assert log.lines[3] == ("info", "VLLM_HPU_WORKER_CPU_PIN=1: rank 3 module 7: share 3 of 4, CPUs 133-150 (18 of "
                            "the 72 this server may use; 1 threads moved)")


def test_pin_from_env_off_reads_nothing():
    assert cpus.pin_from_env(0, 8, cpus.PIN_OFF, environ={}, getaffinity=lambda pid: 1 / 0,
                             setaffinity=lambda tid, mask: 1 / 0, thread_ids=lambda: 1 / 0) == []


def test_the_calling_thread_gets_the_first_cpu_and_the_others_the_rest():
    calls, log = [], Log()
    own = cpus.give_calling_thread_its_own_cpu([19, 20, 21, 22], cpus.PIN_SHARE_AND_OWN_CPU, log,
                                               setaffinity=lambda tid, mask: calls.append((tid, list(mask))),
                                               thread_ids=lambda: [7, 8, 9], calling_thread=7)
    assert own == 19
    assert calls == [(8, [20, 21, 22]), (9, [20, 21, 22]), (0, [19])]
    assert "CPU 19 for itself" in log.lines[0][1] and "20-22" in log.lines[0][1]


def test_only_mode_two_gives_the_calling_thread_its_own_cpu():
    calls = []
    for mode in (cpus.PIN_OFF, cpus.PIN_SHARE):
        assert cpus.give_calling_thread_its_own_cpu([1, 2, 3], mode, setaffinity=lambda *a: calls.append(a),
                                                    thread_ids=lambda: [1], calling_thread=1) is None
    assert cpus.give_calling_thread_its_own_cpu([5], cpus.PIN_SHARE_AND_OWN_CPU, setaffinity=lambda *a: calls.append(a),
                                                thread_ids=lambda: [1], calling_thread=1) is None
    assert calls == []


def test_threads_that_inherited_the_calling_threads_cpu_are_moved_off_it():
    masks = {7: {19}, 8: {20, 21}, 9: {19}, 10: {19}}
    calls = []

    def getaffinity(tid):
        if tid == 10:
            raise ProcessLookupError  # ended meanwhile
        return masks[tid]

    moved = cpus.rehome_new_threads(19, [20, 21], 7, getaffinity=getaffinity,
                                    setaffinity=lambda tid, mask: calls.append((tid, list(mask))),
                                    thread_ids=lambda: [7, 8, 9, 10])
    assert moved == 1 and calls == [(9, [20, 21])]  # not the calling thread, not one that is elsewhere already


def _method_source(name):
    """The source text of one method of HPUWorker (read from the file: importing the worker needs the card's stack)."""
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "v1" / "worker" / "hpu_worker.py"
    text = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError("HPUWorker.%s not found" % name)


def test_init_device_pins_after_set_device_and_before_the_communicators():
    source = _method_source("init_device")
    set_device = source.index("torch.hpu.set_device(device_index)")
    pin = source.index("pin_from_env(self.rank, self.parallel_config.world_size, gaudi_envs.VLLM_HPU_WORKER_CPU_PIN")
    distributed = source.index("init_worker_distributed_environment(")
    runner = source.index("self._create_model_runner()")
    assert set_device < pin < distributed < runner


def test_the_calling_thread_gets_its_cpu_after_warmup():
    source = _method_source("compile_or_warm_up_model")
    warmup = source.index("self.model_runner.warmup_model()")
    own = source.index('give_calling_thread_its_own_cpu(getattr(self, "_cpu_share", [])')
    assert warmup < own < source.index("return CompilationTimes(")


def test_flag_defaults_to_zero(monkeypatch):
    from vllm_gaudi import envs as gaudi_envs

    monkeypatch.delenv("VLLM_HPU_WORKER_CPU_PIN", raising=False)
    assert gaudi_envs.VLLM_HPU_WORKER_CPU_PIN == 0
    monkeypatch.setenv("VLLM_HPU_WORKER_CPU_PIN", "2")
    assert gaudi_envs.VLLM_HPU_WORKER_CPU_PIN == 2
