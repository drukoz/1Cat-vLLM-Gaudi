# Spike S3: eight-rank CPU tensor-parallel harness (gloo)

Runs a small sharded model on 1, 2, 4 and 8 CPU processes (`gloo` backend)
through vLLM's own tensor-parallel layers
(`VocabParallelEmbedding` / `ColumnParallelLinear` / `RowParallelLinear`),
proves the sharded output equals the unsharded reference, and proves every
rank issued the same collectives in the same order. This is the CPU gate
later TP8 tasks (MoE, FP8, GDN, attention) must pass before renting Gaudi
hardware time.

Uses the spike S0 environment (`RUK-2717`): vLLM at pin
`8f816a3f665489d7f0d222115d4f72ebab01076b`, torch 2.13.0, python 3.12. Does
**not** import `vllm_gaudi` (the harness sets `VLLM_PLUGINS=""` so vLLM does
not auto-load the hpu platform plugin in the spawned ranks).

## How to run

```bash
PY=/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/python
cd spikes/cpu-tp
$PY -m pytest -q test_cpu_tp.py          # macOS has no timeout(1); 10 tests, ~5.5 min
# e.g. one case:
$PY -m pytest -q test_cpu_tp.py::test_sharded_matches_unsharded
```

The spec's `timeout 600` wrapper is not available on stock macOS; the harness
enforces its own per-rank timeouts (a hanging rank raises `TimeoutError`).

## Measured times (this Mac, 2026-11-17, `-p no:cacheprovider --durations=0`)

`10 passed in 323.92s (0:05:23)` — slightly over the 5-minute target, so
per-test times are recorded (all under the 600 s Verify cap):

| test | call time |
|---|---|
| `test_allreduce_smoke` (world 8) | 250.9 s (see "tensor over the queue" below) |
| `test_run_ranks_reports_rank_failure` (world 2) | 6.4 s |
| `test_sharded_matches_unsharded[1]` | 3.4 s |
| `test_sharded_matches_unsharded[2]` | 5.2 s |
| `test_sharded_matches_unsharded[4]` | 5.2 s |
| `test_sharded_matches_unsharded[8]` | 6.0 s |
| `test_collective_order_identical[2]` | 4.7 s |
| `test_collective_order_identical[4]` | 5.0 s |
| `test_collective_order_identical[8]` | 7.1 s |
| `test_detects_mismatch` (world 2) | 30.0 s (deliberate: rank 0 hangs a gloo collective; the harness's `timeout_s=30` fires and terminates all ranks) |

Per world size, a spawn of N gloo ranks that import vLLM, init
`init_distributed_environment` + `initialize_model_parallel`, build the model
and forward costs ~3–6 s wall for N = 1…8 (spawns run in parallel).

## What failed and how it was worked around

1. **`get_current_vllm_config()` "Current vLLM config is not set"** in
   `initialize_model_parallel` and the layers. Fix: the harness runs
   `init_distributed_environment` → `initialize_model_parallel` → `fn` →
   destroys inside `with set_current_vllm_config(VllmConfig())`. A bare
   `VllmConfig()` (no model) builds fine at this pin; no fallback config was
   needed.
2. **macOS `sockaddr_un.sun_path` 104-byte limit.** Under this sandbox the
   default `TMPDIR` is long, so three unix-socket users broke: the torch
   `FileStore` for the `file://` rendezvous, vLLM's ZMQ message-queue
   broadcaster (`ipc://`), and torch's shm manager. Fix: `run_ranks` puts the
   rendezvous dir under `/tmp` and pins `TMPDIR` /
   `VLLM_RPC_BASE_PATH` to `/tmp` for the ranks.
3. **`AttributeError: 'ColumnParallelLinear' object has no attribute
   'cpu_linear'`** on the CPU platform. The unquantized CPU GEMM packs the
   weight in `process_weights_after_loading()` (vLLM's model loader normally
   calls it; we bypass the loader). Fix: the test calls
   `layer.quant_method.process_weights_after_loading(layer)` after
   `weight_loader`.
4. **Returning a `torch.Tensor` over a spawn queue costs ~30 s per rank**
   (serialized): CPU tensors are shared via torch's `torch_shm_manager`,
   which is very slow to start on macOS (250 s for the 8-rank smoke test).
   Fix: rank functions in the tests return `tensor.numpy()` (pickled by
   value; world-8 model test drops from 250 s to 6 s). The all-reduce smoke
   test deliberately keeps a raw tensor return, which is the case that
   proves tensors do pass the queue.
5. **Collective recording on CPU:** the TP group's device communicator on
   the CPU platform is `CpuCommunicator` (a `DeviceCommunicatorBase`
   subclass) and it **overrides** `all_reduce` / `all_gather` / `gather`, so
   patching `DeviceCommunicatorBase` would record nothing.
   `record_collectives()` therefore patches the four methods
   (`all_reduce`, `all_gather`, `reduce_scatter`, `gather`) on the live
   communicator's class and restores them on exit.
6. Misc: spawn children cannot re-import a stdin/`-c` main module (tests run
   from real files); the hpu plugin import in every child is skipped via
   `VLLM_PLUGINS=""` (also required by the spec's "do not import
   vllm_gaudi").

## How to use this in a later task

`run_ranks(world_size, fn, *args, timeout_s=120)` spawns `world_size` gloo
ranks with vLLM's distributed + model-parallel state already initialized
(inside a `set_current_vllm_config(VllmConfig())` context); `fn(rank, *args)`
builds layers, loads **full** checkpoint tensors through each parameter's
`weight_loader`, forwards, and returns plain picklable data (`ndarray`,
preferably — see workaround 4). A rank that raises comes back as a
`RuntimeError` with its traceback; a rank that hangs (e.g. mismatched
collectives) raises `TimeoutError`. `record_collectives()` (used inside `fn`)
records `[(method, input_shape, dtype), ...]` for TP collectives, per rank.

Example — checking a new sharded layer at TP8 against a plain reference:

```python
# spikes/cpu-tp/test_my_layer.py
import numpy as np
import torch
from harness import run_ranks

def _inputs():
    # Same full weight and input on every rank and in the test process.
    torch.manual_seed(0)
    return torch.randn(512, 256), torch.randn(2, 16, 256)

def _rank_forward(rank):
    # runs on the rank, TP8 already initialized, current VllmConfig set
    from vllm.model_executor.layers.your_layer import YourParallelLayer
    full_w, x = _inputs()
    layer = YourParallelLayer(256, 512, params_dtype=torch.float32, prefix="l0")
    layer.weight.weight_loader(layer.weight, full_w)
    layer.quant_method.process_weights_after_loading(layer)  # CPU weight packing
    return layer(x)[0].detach().numpy()                       # ndarray, not tensor

def test_your_layer_matches_unsharded_at_tp8():
    full_w, x = _inputs()
    ref = (x @ full_w.T).numpy()  # your math, plain PyTorch, full weights
    for out in run_ranks(8, _rank_forward, timeout_s=120):
        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)
```

Collective-order checks: wrap the forward in
`with record_collectives() as rec:` inside the rank function and return
`rec`; compare the per-rank lists (see
`test_collective_order_identical` / `test_detects_mismatch`).
