# Mac CPU development environment for vllm-gaudi

Spike S0: a CPU-only dev environment on macOS arm64 (no Gaudi hardware) for
working on this plugin, plus a per-file CPU test gate over `tests/unit_tests/`.

Two scripts:

- `setup.sh <target-dir> <vllm-commit> [python]` — builds
  `<target-dir>/venv` with CPU PyTorch, vLLM at the pin (installed from
  source, `VLLM_TARGET_DEVICE=empty`, no compiled kernels) and this plugin as
  an editable install; writes `<target-dir>/PINS.txt`. Idempotent: a second
  run with the same commit finishes in ~3 s. Re-running with a **new** commit
  in the same target dir reuses the venv and the vLLM clone, fetches and
  checks out the commit, and reinstalls vLLM (the script requires the
  installed vLLM version metadata to carry the pin's commit hash, so a re-pin
  always forces the reinstall).
- `run_cpu_tests.sh <target-dir> [pytest args]` — runs every
  `tests/unit_tests/**/test_*.py` file one file per pytest process
  (`python -m pytest -q -p no:cacheprovider <file>`, 600 s cap per file) and
  writes `<target-dir>/cpu_tests.tsv`
  (`file<TAB>outcome<TAB>passed<TAB>skipped<TAB>first error line`).
  Resumable; `FRESH=1` starts over.

The venv and vLLM sources live outside the worktree.

## Requirements

- macOS arm64, Homebrew Python 3.12 (used here) or 3.11 — `setup.sh` picks
  `/opt/homebrew/bin/python3.12` first. Nothing is installed system-wide;
  everything goes into the venv.
- Network access (git clone of vLLM, PyPI).
- `timeout`/`gtimeout` (coreutils) for the per-file 600 s cap; when absent
  (stock macOS has no `timeout`), `run_cpu_tests.sh` falls back to a built-in
  Python timer with the same semantics (exit 124 on timeout).

## Pin (one pin)

vLLM commit `8f816a3f665489d7f0d222115d4f72ebab01076b` (2026-09-04) — the
commit this fork is run against on the Gaudi box, and the commit that already
contains Qwen3.8-Flash-Next (`vllm/models/qwen4_exp/`, merged 2026-08-31).
It sits between two vLLM API changes the fork depends on:
`VllmConfig._get_v1_model_runner_unsupported_features` (added 2026-08-27) and
the removal of `_Fp8OnlineLinearBase` (2026-09-04, vLLM PR #51392), which is
why older pins such as `17da4859` break `import vllm_gaudi` against this fork.

## Commands

```bash
# build (or re-pin) the environment; ~2 min on a re-pin into an existing dir
timeout 1500 bash spikes/mac-cpu-env/setup.sh \
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev \
  8f816a3f665489d7f0d222115d4f72ebab01076b

source /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/activate
python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
python -c "import vllm_gaudi"
# Flash-Next package check (import the submodule, NOT vllm.models.qwen4_exp
# itself, whose __getattr__ loads the NVIDIA implementation and needs Triton):
python -c "import vllm.models.qwen4_exp.common.hyperconnection as m; print(m.__file__)"
deactivate

# CPU test gate (~15 min for 175 files; resumable, FRESH=1 to restart)
timeout 3000 bash spikes/mac-cpu-env/run_cpu_tests.sh \
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev
```

## Pin and results (measured on this Mac, 2026-11-17)

| | value |
|---|---|
| vLLM commit | `8f816a3f665489d7f0d222115d4f72ebab01076b` |
| vLLM version (as installed here) | `0.1.dev1+g8f816a3f6` (shallow clone has no tags, so setuptools-scm falls back to this; a full clone reports `0.28.1rc1.dev392+g8f816a3f6`) |
| python | 3.12.12 |
| torch | 2.13.0 |
| plugin commit (at setup) | `72b36d3e8dff9c7943c2b2c4b7820f5ca2debb27` |
| `import vllm` | OK |
| `import vllm_gaudi` | **OK** |
| setup duration (re-pin `17da4859`→`8f816a3f`, reused clone+venv) | 1 m 46 s |
| setup re-run (idempotent) | 3.6 s |
| test files | 175 |
| outcome split | 66 pass / 32 fail / 39 skip / 38 error |
| tests | 1550 passed, 256 skipped |

- `import vllm_gaudi` works on CPU at this pin: `vllm_gaudi/__init__.py`
  guards the habana imports; the platform plugin cleanly declines
  registration and vLLM falls back to the CPU platform.
- Flash-Next package check at the pin:

  ```text
  $ python -c "import vllm.models.qwen4_exp.common.hyperconnection as m; print(m.__file__)"
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/vllm-src/vllm/models/qwen4_exp/common/hyperconnection.py
  ```

  (plugin log lines from the vLLM plugin loader precede the output; the path
  ends in `vllm/models/qwen4_exp/common/hyperconnection.py` as required).
- The per-file matrix is in [`cpu_tests.md`](cpu_tests.md) (generated from
  `/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/cpu_tests.tsv`).
- The sweep ran 14 m 26 s wall clock; per-file cost is dominated by
  interpreter + vLLM import, not by test time.

## torch pin

The pin declares `torch==2.13.0+cpu; platform_machine == "x86_64" or ... "aarch64"`
(linux-only local wheel + marker). `setup.sh` strips the `+cpu` local version
and the marker and installs plain `torch==2.13.0` from PyPI, which exists for
macOS arm64. If a pin ever lacks a macOS wheel, the script installs the newest
stable version pip offers for the platform and logs a warning.

## Known failures (70 of 175 files)

Failing tests are not fixed here (out of scope). Categories, with the first
error line seen:

| Files | Cause | First error line |
|---:|---|---|
| 46 | ops modules import the HPU runtime | `ModuleNotFoundError: No module named 'habana_frameworks'` |
| 7 | tests poke the HPU device directly | `AttributeError: module 'torch' has no attribute 'hpu'` / `ModuleNotFoundError: No module named 'torch.hpu'` |
| 3 | omni stack (optional `omni` extra) not installed | `ModuleNotFoundError: No module named 'vllm_omni'` |
| 1 | triton is not packaged for macOS | `ModuleNotFoundError: No module named 'triton'` |
| 2 | plugin config path rejected by vLLM at the pin | `pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig` |
| 11 | genuine CPU-side failures (not hardware-related) | see below |

The 11 genuine CPU-side failures:

- `ops/test_deepseek_v41_draft_packed_mla.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'draft_dense_fp8'`
- `ops/test_deepseek_v41_prefill_projections.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'draft_query_fp8'`
- `ops/test_deepseek_v41_woa_fp8.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'kv_norm_rope_decode'`
- `ops/test_deepseek_v41_prefill_index_exchange.py` — `AttributeError: 'NoneType' object has no attribute 'numel'`
- `ops/test_deepseek_v41_prefill_index_sram.py` — `assert True is False`
- `ops/test_deepseek_v41_runtime_indexer.py` — `AttributeError: 'PagedCSA2Attention' object has no attribute 'runtime_index_mme'`
- `ops/test_deepseek_v41_router_head.py` — `AttributeError: '_OpNamespace' 'custom_op' object has no attribute 'custom_deepseek_v41_bf16_identity_gaudi2'`
- `ops/test_gdn_transition_metadata.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute '_hpu_defer_ssm_writeback'`
- `ops/test_hpu_activation.py` — `AttributeError: '_OpNamespace' '_C' object has no attribute 'silu_and_mul'`
- `omni/test_compat.py` — `ModuleNotFoundError: No module named 'vllm.entrypoints.openai.engine.protocol'` (module moved in vLLM at this pin; plugin-side drift)
- `ops/test_deepseek_v41_n256_shards.py` — `FileNotFoundError` for a `config.json` under the pytest tmpdir (test-environment artifact)

## Notes and caveats

- Dependency-resolution warning at the end of setup, expected and harmless
  for CPU work: vLLM at this pin requires `numba==0.65.0` while the plugin
  requires `numba>=0.67.0`; pip keeps the plugin's newer numba (0.68.1).
- vLLM build requirements from `requirements/build/cuda.txt` are installed
  line-by-line (torch lines removed); lines pip cannot satisfy on macOS are
  skipped with a log line. At this pin every line installed cleanly.
- The 39 `skip` rows are hardware-guarded tests
  (`torch.hpu.is_available()` / `FLASHINFER_GAUDI_RUN_HARDWARE_TESTS`),
  which is the intended CPU behaviour.
- `pytest_compat.py` (registered via the plugin's `pytest11` entry point)
  keeps collection working on non-HPU machines; no pytest invocation in this
  gate imports `habana_frameworks.torch` at collection time.
