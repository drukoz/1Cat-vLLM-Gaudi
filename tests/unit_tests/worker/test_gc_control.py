# SPDX-License-Identifier: Apache-2.0
"""A worker's collector mode (``gc_control.GcControl``, VLLM_HPU_WORKER_GC_FREEZE) and its pause counter."""
import gc
import os

os.environ.setdefault("VLLM_PLUGINS", "")

from vllm_gaudi.v1.worker.gc_control import GEN2_NEVER, GcControl, GcPauses  # noqa: E402


def test_on_freezes_and_stops_generation_2_off_restores():
    saved = gc.get_threshold()
    control = GcControl(idle_s=60.0)
    try:
        control.apply(1)
        assert control.mode == 1 and gc.get_freeze_count() > 0 and control.frozen == gc.get_freeze_count()
        assert gc.get_threshold() == (saved[0], saved[1], GEN2_NEVER)
        control.apply("1")  # no change
        assert gc.get_threshold()[2] == GEN2_NEVER
    finally:
        control.apply(0)
    assert control.mode == 0 and gc.get_freeze_count() == 0 and gc.get_threshold() == saved
    control.apply("not a number")
    assert control.mode == 0


def test_the_idle_collection_runs_once_per_idle_period_and_never_in_a_step():
    now = [100.0]
    control = GcControl(clock=lambda: now[0], idle_s=0.5)
    control.mode = 1  # the mode without the thread and the freeze
    assert not control.idle_collect()  # no step yet since the start
    control.step_begin()
    now[0] += 5.0
    assert not control.idle_collect()  # a long step is not idle
    control.step_end()
    now[0] += 0.2
    assert not control.idle_collect()
    now[0] += 0.4
    assert control.idle_collect() and control.idle_collections == 1
    now[0] += 10.0
    assert not control.idle_collect()  # once per idle period
    control.step_begin()
    control.step_end()
    now[0] += 1.0
    assert control.idle_collect() and control.idle_collections == 2


def test_pauses_count_the_collections_and_generation_2():
    pauses = GcPauses()
    pauses.install()
    pauses.install()  # once
    try:
        assert gc.callbacks.count(pauses._callback) == 1
        pauses.reset()
        gc.collect(2)
        gc.collect(0)
        assert pauses.gen2 == 1 and pauses.collections == 2 and pauses.ms >= 0.0
        pauses.reset()
        assert pauses.gen2 == 0 and pauses.collections == 0 and pauses.ms == 0.0
    finally:
        gc.callbacks.remove(pauses._callback)
