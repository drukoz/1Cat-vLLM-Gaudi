# Mac CPU development environment for vllm-gaudi

Spike S0: a CPU-only dev environment on macOS arm64 (no Gaudi hardware) for
working on this plugin, plus a per-file CPU test gate over `tests/unit_tests/`.

Two scripts:

- `setup.sh <target-dir> <vllm-commit> [python]` — builds
  `<target-dir>/venv` with CPU PyTorch, vLLM at the pin (installed from
  source, `VLLM_TARGET_DEVICE=empty`, no compiled kernels) and this plugin as
  an editable install; writes `<target-dir>/PINS.txt`. Idempotent: a second
  run with the same commit finishes in ~3 s.
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

## Commands

```bash
# primary pin
timeout 1500 bash spikes/mac-cpu-env/setup.sh \
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev \
  17da48596c98946d3e3e6896e2ebd341e809f3bd
source /Volumes/Storage/paperclip/tmp/vllm-cpu-dev/venv/bin/activate
python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
python -c "import vllm_gaudi"
deactivate

# CPU test gate (~20 min for 175 files; resumable)
timeout 3000 bash spikes/mac-cpu-env/run_cpu_tests.sh \
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev

# secondary pin
timeout 1500 bash spikes/mac-cpu-env/setup.sh \
  /Volumes/Storage/paperclip/tmp/vllm-cpu-dev-flashnext \
  e126687a9a828d513c01a07cd69f025f27d63280
```

## Pins and results (measured on this Mac, 2026-11-17)

| | primary | secondary |
|---|---|---|
| vLLM commit | `17da48596c98946d3e3e6896e2ebd341e809f3bd` | `e126687a9a828d513c01a07cd69f025f27d63280` |
| vLLM version | `0.1.dev1+g17da48596` | `0.1.dev1+ge126687a9` |
| python | 3.12.12 | 3.12.12 |
| torch | 2.13.0 | 2.13.0 |
| plugin commit | `c4d1e9a65cb7c58f52e550d0bdf52716eda608fa` | same |
| `import vllm` | OK | OK |
| `import vllm_gaudi` | **OK** | **OK** |
| setup duration (fresh) | 3 m 24 s | 2 m 59 s |
| setup re-run (idempotent) | 3 s | 2 s |
| test files | 175 | 175 |
| outcome split | 59 pass / 32 fail / 46 skip / 38 error | 66 pass / 32 fail / 39 skip / 38 error |
| tests | 1550 passed, 256 skipped | 1550 passed, 256 skipped |

- `import vllm_gaudi` works on CPU: `vllm_gaudi/__init__.py` guards the
  habana imports; the platform plugin cleanly declines registration and vLLM
  falls back to the CPU platform.
- The per-file matrix for the primary pin is in
  [`cpu_tests.md`](cpu_tests.md) (generated from
  `/Volumes/Storage/paperclip/tmp/vllm-cpu-dev/cpu_tests.tsv`; the secondary
  TSV is at `/Volumes/Storage/paperclip/tmp/vllm-cpu-dev-flashnext/cpu_tests.tsv`,
  not committed).
- Secondary pin differs from primary in 7 files that skip on the primary pin
  and pass on the secondary: `test_deepseek_v41_dense_fp8`,
  `test_deepseek_v41_expert_n256`, `test_deepseek_v41_mhc_post_collapse`,
  `test_dflash2_score_edges_graph`, `test_flashinfer_gdn_normalization`,
  `test_flashinfer_silu_quant`, `test_qwen38_native_qk`. All other rows are
  identical across the two pins.
- The sweep ran ~20 min (primary) / ~14 min (secondary) wall clock; per-file
  cost is dominated by interpreter + vLLM import, not by test time.

## torch pin

Both pins declare `torch==2.13.0+cpu; platform_machine == "x86_64" or ... "aarch64"`
(linux-only local wheel + marker). `setup.sh` strips the `+cpu` local version
and the marker and installs plain `torch==2.13.0` from PyPI, which exists for
macOS arm64. If a pin ever lacks a macOS wheel, the script installs the newest
stable version pip offers for the platform and logs a warning.

## Known failures (identical for both pins; 70 of 175 files)

Failing tests are not fixed here (out of scope). Categories, with the first
error line seen:

| Files | Cause | First error line |
|---:|---|---|
| 48 | ops modules import the HPU runtime | `ModuleNotFoundError: No module named 'habana_frameworks'` |
| 7 | tests poke the HPU device directly | `AttributeError: module 'torch' has no attribute 'hpu'` / `ModuleNotFoundError: No module named 'torch.hpu'` |
| 4 | omni stack (optional `omni` extra) not installed | `ModuleNotFoundError: No module named 'vllm_omni'` |
| 1 | triton is not packaged for macOS | `ModuleNotFoundError: No module named 'triton'` |
| 2 | plugin config path rejected by pinned vLLM | `pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig` |
| 8 | genuine CPU-side failures (not hardware-related) | see below |

The 8 genuine CPU-side failures (primary pin):

- `ops/test_deepseek_v41_draft_packed_mla.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'draft_dense_fp8'`
- `ops/test_deepseek_v41_prefill_projections.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'draft_query_fp8'`
- `ops/test_deepseek_v41_woa_fp8.py` — `AttributeError: 'types.SimpleNamespace' object has no attribute 'kv_norm_rope_decode'`
- `ops/test_deepseek_v41_prefill_index_exchange.py` — `AttributeError: 'NoneType' object has no attribute 'numel'`
- `ops/test_deepseek_v41_prefill_index_sram.py` — `assert True is False`
- `ops/test_deepseek_v41_runtime_indexer.py` — `AttributeError: 'PagedCSA2Attention' object has no attribute 'runtime_index_mme'`
- `ops/test_deepseek_v41_router_head.py` — `AttributeError: '_OpNamespace' 'custom_op' object has no attribute 'custom_deepseek_v41_bf16_identity_gaudi2'`
- `ops/test_hpu_activation.py` — `AttributeError: '_OpNamespace' '_C' object has no attribute 'silu_and_mul'`

(`ops/test_deepseek_v41_n256_shards.py` fails with a `FileNotFoundError` for a
`config.json` under the pytest tmpdir — a test-environment artifact; the
triton case is counted in the table above.)

## Notes and caveats

- Dependency-resolution warning at the end of setup, expected and harmless
  for CPU work: `vllm requires numba==0.65.0` vs the plugin's
  `numba>=0.67.0`; pip keeps the plugin's newer numba (0.68.1).
- vLLM build requirements from `requirements/build/cuda.txt` are installed
  line-by-line (torch lines removed); lines pip cannot satisfy on macOS are
  skipped with a log line. At both pins every line installed cleanly.
- The 46 `skip` rows (primary) are hardware-guarded tests
  (`torch.hpu.is_available()` / `FLASHINFER_GAUDI_RUN_HARDWARE_TESTS`),
  which is the intended CPU behaviour.
- `pytest_compat.py` (registered via the plugin's `pytest11` entry point)
  keeps collection working on non-HPU machines; no pytest invocation in this
  gate imports `habana_frameworks.torch` at collection time.
