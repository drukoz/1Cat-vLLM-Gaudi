# Native extensions and the bridge mode (RUK-3103 of RUK-2709)

Which Habana library each native extension links, which extensions a lazy-mode server has, and the
checks for the next rental. Source: the 2026-11-18 rental (RUK-3100), image
`pytorch-installer-2.11.0` 1.24.1, where `htorch.is_torch_fork` is `True`.

## The two bridge modes load different plugin libraries

| `PT_HPU_LAZY_MODE` | Libraries the interpreter maps (fork PyTorch) |
|---|---|
| `0` (eager / torch.compile) | `libhabana_pytorch2_plugin.so` + `libhabana_pytorch_backend.so` |
| `1` (lazy; every rental server, `tools/flash_next/rental/serve.py:91`) | `libhabana_pytorch_plugin.so` + `libhabana_pytorch_backend.so` |

An upstream PyTorch build uses the same names with `.upstream` before `.so`. A library that is not
mapped in the mode at hand is also not on the loader path, so an extension whose `NEEDED` list names
the other mode's plugin fails to load: `libhabana_pytorch2_plugin.so: cannot open shared object file`.

## Link rule: the backend library only

The custom-op extensions call three Habana symbols:

| Symbol | Exported by |
|---|---|
| `habana::custom_op::registerUserCustomOp` | `libhabana_pytorch_backend.so` |
| `habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor` | `libhabana_pytorch_backend.so` |
| `habana::custom_op::UserCustomOpDescriptor::execute(std::vector<c10::IValue> const&)` | both plugin libraries |

So `gdn_fused_ops`, `flashinfer_gaudi_ops` and `flashqla_pair_transform_pt2` link
`habana_pytorch_backend` (`habana_pytorch_backend.upstream` when `htorch.is_torch_fork` is false)
and no plugin library. `execute` then binds to whichever plugin the interpreter has already mapped,
so one build serves both modes. Consequence: the extension must be loaded after
`import habana_frameworks.torch` (the plugin loads there). Loaded before it, `dlopen` fails with
`undefined symbol: _ZN6habana9custom_op22UserCustomOpDescriptor7execute...`.

Measured on the card with `patchelf` on the built `gdn_fused_ops` (source unchanged):

| `NEEDED` entry | eager | lazy |
|---|---|---|
| `libhabana_pytorch2_plugin.so` (PR #6) | loads; `test_hpu_op.py` 10 passed | does not load |
| `libhabana_pytorch_plugin.so` | not tried | loads; `test_hpu_op.py` 10 passed; replay `core lazy+hpu_graph: exact` (1,000 steps, tp8_b16) |
| `libhabana_pytorch_backend.so` only (this rule) | loads; `test_hpu_op.py` 10 passed | loads; `test_hpu_op.py` 10 passed |
| no Habana entry | loads | loads (load only checked) |

Not yet checked on a card: `flashinfer_gaudi_ops` in lazy mode, `flashqla_pair_transform_pt2` in
either mode, and a lazy server start with the backend-only build. The run sheet lines below cover
the first and the last.

## What a lazy-mode server has

| Native piece | Loaded by | Eager | Lazy |
|---|---|---|---|
| `gdn_fused_ops` + kernel database `libgdn_fused_gaudi2.so` (GDN core decode and verify) | `vllm_gaudi.register_ops` → `load_gdn_fused_extension()`, when `VLLM_HPU_GDN_CORE_DECODE` or `VLLM_HPU_GDN_CORE_VERIFY` is nonzero | yes | yes (backend-only link) |
| `flashinfer_gaudi_ops` + `libflashinfer_gaudi_kernels.so` (`silu_and_mul`, `add_rmsnorm_quant`) | `vllm_gaudi.register_ops` → `load_native_extensions()` | yes | yes (backend-only link) |
| `flashqla_pair_transform_pt2` (Qwen3.8 native Q/K preparation) | `vllm_gaudi/ops/qwen38_native_qk.py`, opt-in `VLLM_GDN_QWEN38_NATIVE_QK_PREP`; not built by the rental | yes | yes (backend-only link) |
| Native Bridge adapter (`flashinfer_gaudi_bridge_ops` with its private Bridge libraries and `bridge_artifact_v1.json`: `silu_mul_quant`, `block_fp8_dequant`, `block_fp8_linear`, `gemm_silu`, the Bridge packed GDN op) | `flashinfer_gaudi._bridge.load_bridge_adapter()`, explicit opt-in only | yes | **no, by design**: `_bridge.py` raises `Native Bridge adapter currently requires PT_HPU_LAZY_MODE=0`; those ops report unavailable and callers keep the PyTorch path |
| DeepSeek V4 / V4.1 extensions (`csrc/deepseek_v4/*`, `csrc/deepseek_v41_unique`, `csrc/flashqla_hpu/native_backend`, `tools/communication/setup_dsv41_micro_compute.py`) | the DeepSeek entry points | as before | not part of Flash-Next |

The Bridge adapter and the DeepSeek extensions compile against the private headers of one matching
upstream-PyTorch Bridge build (`GAUDI_PYTORCH_BRIDGE_ROOT`) and call Bridge internals beyond the
custom-op API, so they keep their `habana_pytorch2_plugin.upstream` + `habana_pytorch_backend.upstream`
link and stay eager-only. RUK-3103 does not change them.

## When an extension does not load

`gdn_fused_extension_errors()` and `native_diagnostics()["load_errors"]` carry the `dlopen` error:
`torch.ops.load_library` raises `OSError: Could not load this library: <path>` and keeps the real
error in `__cause__`, so each entry reads `<path>: OSError: Could not load this library: <path>;
caused by OSError: <dlopen message>`. The plugin's startup warning (`has no HPU kernel: ...`)
prints the same text.

## Run sheet lines for the next rental

In the container, from the plugin checkout (`cd /bundle/plugin`), after the install (gate G1).

1. Neither extension names a plugin library:

   ```bash
   LIB=$(python -c 'import flashinfer_gaudi, os; print(os.path.join(os.path.dirname(flashinfer_gaudi.__file__), "lib"))')
   for f in "$LIB"/gdn_fused_ops*.so "$LIB"/flashinfer_gaudi_ops*.so; do echo "$f"; readelf -d "$f" | grep NEEDED; done
   ```

   Expect `libhabana_pytorch_backend.so` in both lists and no line containing `plugin`.
   Stop rule: a `plugin` entry means the bundle was built from a commit before RUK-3103.

2. The op tests pass in both modes, one process per mode:

   ```bash
   PT_HPU_LAZY_MODE=1 python -m pytest -q -p no:cacheprovider tests/unit_tests/gdn_core_decode/test_hpu_op.py
   PT_HPU_LAZY_MODE=0 python -m pytest -q -p no:cacheprovider tests/unit_tests/gdn_core_decode/test_hpu_op.py
   ```

   Expect `10 passed` from each (2026-11-18 rental, `patchelf` backend-only build).

3. `flashinfer_gaudi_ops` loads in lazy mode:

   ```bash
   PT_HPU_LAZY_MODE=1 python -c 'import habana_frameworks.torch, flashinfer_gaudi._native as n; d = n.native_diagnostics(); print(d["loaded_libraries"], d["load_errors"], d["silu_and_mul"], d["fused_add_rmsnorm_quant"])'
   ```

   Expect the `flashinfer_gaudi_ops` path in the first tuple, an empty second tuple and `True True`.

4. A lazy server runs the fused kernel: in the first rental server's log (all run lazy),

   ```bash
   grep -E "has no HPU kernel|GDN core (decode|verify) .* is off for" <server log>
   ```

   prints nothing (the pattern is `GDN_FALLBACK_RE` in `tools/flash_next/rental/gates.py`).
