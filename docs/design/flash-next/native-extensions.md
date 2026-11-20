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

So `gdn_fused_ops`, `hc_fused_ops`, `flashinfer_gaudi_ops` and `flashqla_pair_transform_pt2` link
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
either mode, `hc_fused_ops` in either mode (it has never been built against the real bridge), and a
lazy server start with the backend-only build. The run sheet lines below cover
the first and the last.

## What a lazy-mode server has

| Native piece | Loaded by | Eager | Lazy |
|---|---|---|---|
| `gdn_fused_ops` + kernel database `libgdn_fused_gaudi2.so` (GDN core decode and verify) | `vllm_gaudi.register_ops` → `load_gdn_fused_extension()`, when `VLLM_HPU_GDN_CORE_DECODE` or `VLLM_HPU_GDN_CORE_VERIFY` is nonzero | yes | yes (backend-only link) |
| `hc_fused_ops` + kernel database `libhc_fused_gaudi2.so` (the six fused hyper-connection ops; [hc-fused.md](hc-fused.md)) | `vllm_gaudi.register_ops` → `load_hc_fused_extension()`, only when `VLLM_HPU_HC_FUSED=1` (default 0: not loaded, its schemas not defined) | not run on a card | not run on a card (backend-only link, as `gdn_fused_ops`) |
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

The fused hyper-connection kernels follow the same pattern with their own names:
`hc_fused_extension_errors()`, the startup warning `HC fused (VLLM_HPU_HC_FUSED=1) has no HPU kernel:
...`, and one line per layer, `HC fused (VLLM_HPU_HC_FUSED=1) is off for <layer>: <reason>; using the
torch path`. Every layer then runs the torch hyper-connection; nothing stops.

## The fused hyper-connection kernels (`VLLM_HPU_HC_FUSED`)

Off by default. With the flag at 0, `register_ops` does not call the loader, nothing imports
`flashinfer_gaudi.hc_fused` (so the six `flashinfer_gaudi::hc_*` schemas are not defined), and
`libhc_fused_gaudi2.so` is not added to `GC_KERNEL_PATH` by the loader. The one thing that happens with
the flag at 0 is inside the combined kernel database: when `libhc_fused_gaudi2.so` is installed next to
it, the adapter opens it as a member and lists its six GUIDs, which no graph uses.

| Piece | Where |
|---|---|
| Flag | `VLLM_HPU_HC_FUSED` (0 default, 1 fused), read once per layer at construction |
| Loader | `flashinfer_gaudi._native.load_hc_fused_extension()`, `hc_fused_extension_errors()`, `hc_fused_extension_loaded()`; status key `hc_fused_loaded` |
| Path overrides | `HC_FUSED_GAUDI_KERNEL_LIBRARY` (the kernel database), `HC_FUSED_GAUDI_NATIVE_LIBRARY` (the op extension); default: the package's `lib` folder |
| `GC_KERNEL_PATH` | the rules of `libgdn_fused_gaudi2.so`: the combined database when it is active and serves the library (its bytes name `libhc_fused_gaudi2.so`), otherwise first in the colon list, never next to `PT_HPU_RECIPE_CACHE_CONFIG` |
| Build | `VLLM_GAUDI_BUILD_HC_FUSED=1` at install time, or `tools/build_flashinfer_gaudi.py --hc-fused-only`; a failed build removes its own two files only |
| Card test | `tests/unit_tests/hc_fused/test_hpu_op.py`, run inside rental gate G1b (`gates.hc_fused`) |


## The MoE glue kernels (`VLLM_HPU_MOE_GLUE`)

Four more kernels of the hyper-connection library ([hc-fused.md](hc-fused.md), "MoE glue kernels"): no file of
their own. With the flag at 0 (the default) `vllm_gaudi.register_ops` does not call their loader, the four
`flashinfer_gaudi::moe_*` schemas are not defined (unless `VLLM_HPU_HC_FUSED=1` loads the shared extension, which
implements them), no MoE block gets a state object or a merged gate weight, and the status line has no key for them.

| | |
|---|---|
| Flag | `VLLM_HPU_MOE_GLUE` (0 default, 1 fused), read once per MoE block at construction |
| Files | `libhc_fused_gaudi2.so` and `hc_fused_ops*.so`: the hyper-connection files, their path overrides and their `GC_KERNEL_PATH` rules |
| Loader | `flashinfer_gaudi._native.load_moe_glue_extension()` = `load_hc_fused_extension()`, then `kernel_library_guids()` asks the library which of the four MoE GUIDs it serves; `moe_glue_extension_errors()`, `moe_glue_extension_loaded()`, `moe_glue_missing_kernels()`; status key `moe_glue_loaded`, present only when the loader was called (`hc_fused_loaded` is then set too) |
| Build | the hyper-connection step. Each MoE kernel is optional: `csrc/hc_fused/CMakeLists.txt` leaves out one that does not compile, and the step prints `MoE glue kernels in libhc_fused_gaudi2.so: ...` and, when some are missing, `MoE glue kernels missing from ...` |
| Messages | once at startup `MoE glue (VLLM_HPU_MOE_GLUE=1) has no HPU kernel: ...`; one line per block `MoE glue (VLLM_HPU_MOE_GLUE=1) is off for <layer>: <reason>; using the torch path` |
| Card test | `tests/unit_tests/hc_fused/test_moe_hpu_op.py`, run inside rental gate G1b after the hyper-connection one (`gates.moe_glue`) |
| Combined kernel database | nothing to do: the library is already a member, and the adapter forwards whatever GUIDs it lists |

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

## The recipe cache and GC_KERNEL_PATH (RUK-3241)

**Rule.** With a recipe cache configured (`PT_HPU_RECIPE_CACHE_CONFIG` set to a non-empty value) the Habana
PyTorch bridge (Gaudi 1.24.1) validates `GC_KERNEL_PATH` as **one regular file** and aborts on a colon-separated
list. Without the recipe cache the bridge accepts the colon list (Intel's TPC guide, "Multiple Kernel Libraries").

**Evidence.** On the Gaudi node (2026-11-19, 17:15Z) every vLLM worker aborted in `init_device` with
`RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_SYNHELPER Flag GC_KERNEL_PATH: Path is not a regular file:
<...>/flashinfer_gaudi/lib/libflashinfer_gaudi_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so`; the bridge
printout of that run shows `PT_HPU_RECIPE_CACHE_CONFIG = /results/recipe-cache,false,8192`. The same
colon-separated `GC_KERNEL_PATH` was accepted earlier the same day without the recipe cache (the servers started
and compiled graphs).

**What the loader does now.** When `PT_HPU_RECIPE_CACHE_CONFIG` is set to a non-empty value, `flashinfer_gaudi._native`
prepends no kernel database and leaves `GC_KERNEL_PATH` untouched. The op extensions that depend on a database
(`flashinfer_gaudi_ops*.so`, `gdn_fused_ops*.so`, `hc_fused_ops*.so`) are not loaded either, because their ops name TPC
GUIDs the graph compiler could not find and the first graph compile would fail instead of start-up. The reason is
recorded once in `_LOAD_ERRORS` (and, for the fused paths, in `gdn_fused_extension_errors()` and
`hc_fused_extension_errors()`). With the variable unset or empty,
registration proceeds exactly as before.

**What the operator sees.** `register_ops()` logs one warning naming `PT_HPU_RECIPE_CACHE_CONFIG`,
`GC_KERNEL_PATH` and `libflashinfer_gaudi_kernels.so`; and when a fused GDN mode is on, the existing startup line
prints `GDN core decode ... has no HPU kernel: ... PT_HPU_RECIPE_CACHE_CONFIG is set ...`. The callers fall back:
`silu_and_mul_op()` and friends return `None` and the Python paths run; the GDN layers run the torch decode path.

**Remedies.** (1) Unset `PT_HPU_RECIPE_CACHE_CONFIG` to use the native TPC kernels, or (2) install the combined
kernel database. The combined kernel database (`libflashinfer_gaudi_combined_gaudi2.so`, RUK-3243, below) bundles the
FlashInfer, stock and fused-GDN kernels into the single regular file the bridge accepts, which lifts the
restriction so a recipe-cache server can keep the native kernels.

**With the combined database.** When `flashinfer_gaudi/lib/libflashinfer_gaudi_combined_gaudi2.so` exists and
`FLASHINFER_GAUDI_COMBINED_KERNEL_DB` is not `0`, `flashinfer_gaudi._native` sets `GC_KERNEL_PATH` to **exactly that
one file** (it obeys the bridge's single-regular-file rule above) and treats the FlashInfer database as registered,
so both loaders load their op extensions and the recipe-cache block of this section does not apply. It replaces the
variable only when that is safe: when it is unset, empty, already the combined file, or the stock
`/usr/lib/habanalabs/libtpc_kernels.so` default. A foreign `GC_KERNEL_PATH` is never overwritten; the loader records
one error naming the existing value and the remedy (`unset it or set FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0`) and falls
back to the per-library behaviour, the recipe-cache block included. `FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0` is the
kill switch: with it the loader behaves exactly as if the combined file were absent, wherever the file is (this is
what the rental sets when gate **G1b** has not passed). Gate G1b of the rental runs a lazy server through the single
combined file before any real server is started on it.

**Where the combined file is looked for** (`_combined_kernel_db_path()`): first in the imported package's `lib`
folder. When it is not there, next to the library named by `GDN_FUSED_GAUDI_KERNEL_LIBRARY`. Gate G1b runs its
checks with `PYTHONPATH` on the bundle's source tree, whose `lib` is empty, and passes the installed libraries by
that variable; without the second location the gate never loaded the combined file it qualifies. The combined file
serves only the libraries next to itself, so the second location counts only when it serves what the process would
otherwise register one by one: the variable names an existing file called `libgdn_fused_gaudi2.so`, and the
package's own `lib` holds no `libflashinfer_gaudi_kernels.so` (the other folder's combined file would leave it out).

**What the loader decided** is reported by `flashinfer_gaudi._native.kernel_database_status()` as plain,
JSON-serializable data; it loads nothing and changes nothing:

| Key | Value |
|---|---|
| `state` | the `KernelDbState` name of the FlashInfer database: `MISSING`, `REGISTERED`, `BLOCKED` or `COMBINED` |
| `gc_kernel_path` | `GC_KERNEL_PATH` as it is now (the value SynapseAI reads), `""` when unset |
| `combined_active` | `True` only when this loader activated the combined file and `GC_KERNEL_PATH` is still exactly that file |
| `combined_kernel_db` | the combined file this loader activated, or `None` |
| `gdn_fused_loaded` | the result of `load_gdn_fused_extension()`, `None` when it was never called |
| `hc_fused_loaded` | the result of `load_hc_fused_extension()`, `None` when it was never called (every process without `VLLM_HPU_HC_FUSED=1` and without `VLLM_HPU_MOE_GLUE=1`) |
| `moe_glue_loaded` | the result of `load_moe_glue_extension()`; the key is absent when it was never called (every process without `VLLM_HPU_MOE_GLUE=1`) |
| `load_errors` | every recorded loader error, in order |

It cannot see inside the combined file: the members are opened by that file inside SynapseAI (next section), and a
member it cannot use is reported on stderr there.

## The combined kernel database (RUK-3243)

With `PT_HPU_RECIPE_CACHE_CONFIG` set, the Habana PyTorch bridge (1.24.1) accepts only **one regular
file** in `GC_KERNEL_PATH`, while SynapseAI itself accepts a colon list. On the Gaudi node every vLLM
server aborted at `init_device` with `PT_SYNHELPER Flag GC_KERNEL_PATH: Path is not a regular file`.
`csrc/kernel_db_adapter/adapter.cpp` is a TPC kernel library (`libflashinfer_gaudi_combined_gaudi2.so`)
that forwards every entry point to an ordered list of member libraries, so `GC_KERNEL_PATH` can name
that one file. It generalizes the DeepSeek V4 glue (`csrc/deepseek_v4/memory_ready/glue.cpp`) from one
base library to a list of members. The host test `csrc/kernel_db_adapter/test/` checks it against fake
members.

**Members and order (= GUID precedence, the same order the Python loader prepends today; first member
wins on a duplicate name):**

1. `libgdn_fused_gaudi2.so` (optional: when the file is not next to the adapter it is not a member)
2. `libhc_fused_gaudi2.so` (optional, same rule; the fused hyper-connection kernels, [hc-fused.md](hc-fused.md))
3. `libflashinfer_gaudi_kernels.so` (optional, same rule)
4. the stock library (required): `FLASHINFER_GAUDI_STOCK_KERNEL_DB` if set, else
   `/usr/lib/habanalabs/libtpc_kernels.so`

An optional sibling that is not there is not a failure. One that is there and cannot be used is, as for
every member (next paragraph). `libhc_fused_gaudi2.so` is a member whenever the file is installed,
whatever `VLLM_HPU_HC_FUSED` says, so `tools/build_flashinfer_gaudi.py` checks the built library the way
the adapter uses it (it loads, exports the three required entry points, lists its Gaudi2 kernels) and
removes it when the check fails, before the combined database is built.

Siblings are found next to the adapter with `dladdr`. Members are opened once (`std::call_once`,
`dlopen(RTLD_NOW | RTLD_LOCAL)`). The stock member is known by its role in this list, never by its
position among the members that happened to open.

**A member that cannot be used stops the database.** A member fails when `dlopen` fails, when it lacks
one of the three entry points the graph compiler requires of every kernel library (`GetKernelGuids`,
`InstantiateTpcKernel`, `GetShapeInference`), or when its own `GetKernelGuids` fails. Each failure is one
stderr line with the member's path and the `dlerror` text, for example:

```text
flashinfer_gaudi combined kernel database: cannot load the stock kernel library /usr/lib/habanalabs/libtpc_kernels.so: /usr/lib/habanalabs/libtpc_kernels.so: cannot open shared object file: No such file or directory
flashinfer_gaudi combined kernel database: <this file> is unusable: 1 member failure(s) above, so GetKernelGuids returns GLUE_FAILED and no kernel is served from this file. Fix the member, or set FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 to run without the combined kernel database.
```

After a failure `GetKernelGuids` returns `GLUE_FAILED` with a count of 0 for every device, every other
entry point returns `GLUE_FAILED`, and `GetLibVersion` is the stock member's value or 0 when the stock
member did not load (never another member's). This holds for the stock member (without it every stock
kernel would be gone and unknown GUIDs would go to the wrong library) and for a custom member that is
present but unusable (the Python loader would still report its GUIDs as registered). In the public
graph-compiler sources (release 1.16/1.17, `KernelDB::loadKernelsFromPath`) a failing `GetKernelGuids`
is logged as `Failed loading kernels from <file>` and the library is unloaded; the process is not
stopped there, so the stderr lines above are the readable message and the first graph compile is where
it stops.

**The six exported entry points** (the graph compiler calls these; see `tpc_kernel_lib_interface.h`):

| Entry point | Routing rule |
|---|---|
| `GetKernelGuids` | merged list of all members for `DEVICE_ID_GAUDI2` (first wins on a duplicate); a non-Gaudi2 device forwards to the stock member only; the count protocol is below |
| `InstantiateTpcKernel` | by `p->guid.name` to the owning member; a name no member lists goes to the stock member (as the DeepSeek glue does) |
| `GetShapeInference` | by `(p->pGuid ? p->pGuid : &p->guid)->name`; every member has it (required) |
| `GetSuggestedManipulation` | by `p->guid.name`; a member without the symbol is `GLUE_FAILED` |
| `GetSupportedDataLayouts` | by `p->guid.name`; a member without the symbol gets one all-`x` (don't-care) layout and `GLUE_SUCCESS` |
| `GetLibVersion` | the stock member's value |

**The `GetKernelGuids` count protocol.** The header documents `kernelCount` as in/out, the size of
`guids`, and "if the pointed value is zero, the callee is expected to fill requested size". The graph
compiler (`KernelDB::loadKernelGuids`) calls `(&0, nullptr)`, then `(&count, buffer)`. The adapter
always writes the required count first, as the DeepSeek glue does:

| Call | Return | `*kernelCount` after | `guids` |
|---|---|---|---|
| count pointer null | `GLUE_FAILED` | — | untouched |
| `guids` null, or `*kernelCount == 0` | `GLUE_SUCCESS` | the required count | untouched |
| `0 < *kernelCount <` required | `GLUE_FAILED` | the required count | untouched |
| `*kernelCount >=` required | `GLUE_SUCCESS` | the required count | the first *required* entries filled |

**The optional entry points, and why the adapter answers as it does.** The two custom members export
no `GetSupportedDataLayouts` (and `libflashinfer_gaudi_kernels.so` no `GetSuggestedManipulation`). The
adapter must export both for the stock member, so the graph compiler never sees a missing symbol for our
GUIDs and the adapter has to give the answer a missing symbol stands for. From the public sources:

| Entry point | Symbol absent from the library | Symbol present, non-success return | The adapter, for a member without it |
|---|---|---|---|
| `GetSupportedDataLayouts` | every layout is don't-care, the pass goes on (`node_io_manager.cpp`: `NotImplementedException` is caught) | the supported-layouts pass fails (`Glue Code GetSupportedLayouts failed`), also for 0 layouts | `*layoutCount = 1`, every layout byte `'x'` for inputs, outputs and shape tensors when a buffer is given, `GLUE_SUCCESS` (what `csrc/deepseek_v4/host/entry_points.cpp` does for its own GUIDs) |
| `GetSuggestedManipulation` | the pointer is stored null and called without a check (`kernel_db.cpp`) | no manipulation is applied, a debug line, the pass goes on (`optimize_tpc_kernels.cpp`) | `GLUE_FAILED` (what the DeepSeek glue does for its own GUIDs) |
| `GetShapeInference` | the whole library is refused (`No Entry point GetShapeInference`) | shape inference of that node fails; only called for GUIDs flagged `supportsDynamicShapes` | not synthesized: a member without it is refused at load, as the graph compiler would refuse it |

**Environment variables.** Each one leaves a stderr line when it is set, so neither can change the
member list of a production process unnoticed:

- `FLASHINFER_GAUDI_STOCK_KERNEL_DB` — replaces `/usr/lib/habanalabs/libtpc_kernels.so` as the stock
  member.
- `FLASHINFER_GAUDI_COMBINED_MEMBERS=<colon-separated paths>` — for the host test; replaces the custom
  members (the libraries next to the adapter). The stock member is not part of this list: it is always
  loaded as above, so this variable cannot remove the stock kernels.

**Build and test.** The library is host C++ only (no `tpc-clang`, no `objcopy`); it links
`${CMAKE_DL_LIBS}` and needs the header folder (in tpcsim that is `/usr/lib/habanatools/include`, the
default for `TPC_INCLUDE_DIR`/`HABANA_INCLUDE_DIR`):

```bash
tools/flash_next/tpcsim.sh 'cmake -S csrc/kernel_db_adapter -B /tmp/kda -DKERNEL_DB_ADAPTER_TESTS=ON \
    && cmake --build /tmp/kda --parallel && ctest --test-dir /tmp/kda --output-on-failure'   # 100% tests passed
tools/flash_next/tpcsim.sh 'nm -D /tmp/kda/libflashinfer_gaudi_combined_gaudi2.so | grep " T "'  # the six entry points
```

The host test (`test/test_adapter.cpp`) uses five fake members built from `test/fake_member.cpp`: three
with all six entry points (one of them stands for `libhc_fused_gaudi2.so`), one with only the three
required ones (as the GDN and FlashInfer members), and one without `GetShapeInference`. It checks the
merging and routing, the count protocol, the all-`x` layout for the three-entry-point member, the
installed layout (a copy of the adapter with `libgdn_fused_gaudi2.so` and
`libflashinfer_gaudi_kernels.so` next to it and no members override; the absent `libhc_fused_gaudi2.so`
is not reported and fails nothing), the installed layout with `libhc_fused_gaudi2.so` (five merged
GUIDs in member order, its GUID routed to it), the same layout with a `libhc_fused_gaudi2.so` that is
not a library (the database serves nothing and names the file), and that a missing stock member, a
missing custom member and a member without a required entry point each make `GetKernelGuids` fail
with the path on stderr. Each scenario loads a fresh image of the adapter;
the test fails if `dlclose` does not unload it (glibc never unloads a library that defines a
`STB_GNU_UNIQUE` symbol, which is why the adapter formats numbers with `snprintf`, not `std::to_string`).

The same executable checks a real build: `test_adapter --installed <folder> [<stock library>]` loads
`<folder>/libflashinfer_gaudi_combined_gaudi2.so` with the member libraries next to it and requires that
every GUID that is not the stock library's own is in the merged list and gets an accepted
supported-layouts answer. Without the second argument the stock library is Intel's
`/usr/lib/habanalabs/libtpc_kernels.so`, so on the node it is a check of the merged list against the
real stock library before any server starts; in tpcsim, which has no stock library, pass a fake:

```bash
tools/flash_next/tpcsim.sh 'cmake -S csrc/kernel_db_adapter -B /tmp/kda >/dev/null && cmake --build /tmp/kda --parallel >/dev/null \
    && python3 tools/build_flashinfer_gaudi.py --kernel-only --output-dir /tmp/out --build-root /tmp/b1 >/dev/null \
    && cmake -S csrc/gdn_fused -B /tmp/b2 >/dev/null && cmake --build /tmp/b2 --target gdn_fused_gaudi2 --parallel >/dev/null \
    && cp /tmp/b2/libgdn_fused_gaudi2.so /tmp/out/ \
    && /tmp/kda/test_adapter --installed /tmp/out /tmp/kda/libfake_member_b.so'   # 14 GUIDs = 2 stock + 12 from the members
```

`tools/build_flashinfer_gaudi.py` builds it after the other libraries (as
`flashinfer_gaudi/lib/libflashinfer_gaudi_combined_gaudi2.so`), only when at least one member database
was built; a failure removes the file and prints `Continuing without the combined kernel database:
GC_KERNEL_PATH stays a colon list.` so the install never fails. `--combined-only` forces the build and
exits 3 on failure.

**Unverified until a card runs it.** The host test uses a fake stock library, so on-card proof is still
needed that the graph compiler drives the adapter for the **stock** kernels under real graphs, not just
for the fused GDN / FlashInfer GUIDs: that the merged GUID list is accepted, that a stock kernel the
adapter forwards to `libtpc_kernels.so` instantiates and runs, and that `GetLibVersion` matches the
installed SynapseAI. Gate **G1b** of the rental is exactly that check (a lazy server starts and generates
with `GC_KERNEL_PATH` set to the single combined file). The table of graph-compiler behaviour above is
read from the public sources of release 1.16/1.17; the node runs 1.24.1, whose graph compiler is not
public. Two things rest on that gap: that 1.24.1 still reads an all-`x` layout as don't-care (its
`tpc_kernel_lib_interface.h` still documents `'xxxxx'` as layout agnostic, and the DeepSeek database
answers this way on cards), and what 1.24.1 does after `GetKernelGuids` fails (in 1.16/1.17 it logs and
goes on without this file's kernels).
