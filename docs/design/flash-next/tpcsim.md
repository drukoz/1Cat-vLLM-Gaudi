# TPC compiler and simulator on this Mac (tpcsim)

`tools/flash_next/tpcsim.sh <command...>` runs a command inside an Ubuntu 22.04
amd64 container that has the Habana TPC toolchain (`tpc-clang` 2.7.33) and the
Gaudi2 instruction simulator, with the current directory mounted at `/work`.
The kernel compiles natively for x86-64 under Rosetta and the simulator
executes TPC instructions; 1CatAI's simulator test
(`tests/standalone/deepseek_v41/test_simulation_descriptor.py`) passes in it.

The VM, the image and all configuration live under
`/Volumes/Storage/paperclip/linux-vm/` (never `$HOME`, which is temporary
during an agent run).

## One-time setup (done; recorded for rebuilds)

```bash
brew install colima docker          # colima 0.10.3, docker CLI 29.10.0
export XDG_CONFIG_HOME=/Volumes/Storage/paperclip/linux-vm
colima start --profile tpcsim --vm-type vz --vz-rosetta --arch aarch64 \
    --cpu 6 --memory 12 --disk 80 --mount /Volumes/Storage:w
docker build --platform linux/amd64 -t tpcsim:1.24.1 tools/flash_next/tpcsim
```

`tpcsim.sh` does all of this automatically: it starts the VM when
`colima status --profile tpcsim` says it is not running and builds the image
when `docker image inspect tpcsim:1.24.1` fails. The VM stays up afterwards.
`tools/flash_next/tpcsim.sh --stop` stops it.

## Running things

```bash
tools/flash_next/tpcsim.sh 'tpc-clang --version | head -2'
tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest \
    tests/standalone/deepseek_v41/test_simulation_descriptor.py'    # 2 passed
tools/flash_next/tpcsim.sh 'tpc-clang -Wall -Werror -O2 -march=gaudi2 \
    -I/usr/lib/habanatools/include csrc/gdn_fused/kernels/gdn_core_decode_gaudi2.c \
    -c -o /tmp/k.o'
tools/flash_next/tpcsim.sh 'cmake -S csrc/gdn_fused -B /tmp/b && cmake --build /tmp/b --target gdn_fused_gaudi2'
```

Pass one quoted string, or a command plus its arguments. The wrapper sets
`COLIMA_HOME`, `DOCKER_CONFIG`, `XDG_CONFIG_HOME` and `DOCKER_HOST`
(`unix:///Volumes/Storage/paperclip/linux-vm/colima/tpcsim/docker.sock`)
itself and exports `PYTHONPATH=/work`, so run it from a worktree root.

## Measured times (Apple Silicon, Rosetta, 6 CPU / 12 GB VM)

| Step | Time |
|------|------|
| `colima start` (first boot, image download included) | 26 s |
| `docker build` (full: habanalabs-tools + CPU torch download) | 151 s |
| `docker build` (incremental, cached layers) | 52 s |
| simulator test `test_simulation_descriptor.py` (2 passed) | 3.2 s |
| docker container start overhead | < 1 s |

## Versions in the image

- Ubuntu 22.04, Python 3.10.12
- `habanalabs-tools` 1.24.1-482 (pinned): `tpc-clang` 2.7.33,
  `tpc-llvm-objdump`, `/usr/lib/habanatools/` (`libtpcsim_shared.so`,
  `libtpc_tests_core_ext.so`, headers incl. `tpc_test_core_api.h`,
  `gc_interface.h`, `tpc_kernel_lib_interface.h`)
- torch 2.14.1+cpu (2.13.0 is not on the CPU wheel index for Python 3.10;
  newest available recorded here), pytest 9.2.0, numpy 2.2.6
- Docker server 29.5.2 inside the VM (CLI 29.10.0 on the Mac)

## Known limits

- **`half` is a reserved token of tpc-clang.** `int half = 0;` fails with
  "cannot combine with previous 'int' declaration specifier" even in an empty
  file. Never name a variable `half` in TPC code. The loop variable in
  `gdn_core_decode_recurrence.h` was renamed to `vh` (commit `07290d4`); since
  then both kernels compile with `-Wall -Werror` and the CMake target
  `gdn_fused_gaudi2` links in this container.
- **tpc-clang prints a performance warning** for both kernels: "number of
  requested special function IDs exceeds LUT cache capacity for the Gen2+
  architecture". It is not an error; it matters for kernel performance work.
- **colima ≥ 0.10 uses `$XDG_CONFIG_HOME/colima` and ignores `COLIMA_HOME`.**
  The wrapper exports `XDG_CONFIG_HOME=/Volumes/Storage/paperclip/linux-vm`.
  A pre-existing `~/.colima` on the machine takes priority over that, and
  colima prints `found ~/.colima, ignoring $XDG_CONFIG_HOME`.
- **`libtpc_tests_core_ext.so` carries no RPATH/RUNPATH**, so its transitive
  deps (`libTpcElfReader.so` …) exit 127 at load time unless
  `/usr/lib/habanatools` is in the loader cache. The image registers it via
  `/etc/ld.so.conf.d/habanatools.conf` + `ldconfig`; no test changes needed.
- **Rosetta works.** No SIGILL (exit 132) anywhere: `tpc-clang`, `tpc-llvm-objdump`
  and the simulator all run under `--vm-type vz --vz-rosetta --arch aarch64`.
  The QEMU fallback (`--vm-type qemu --arch x86_64`) was not needed.
- **No Gaudi driver headers.** `/usr/include/habanalabs` does not exist (the
  driver package is out of scope); `gc_interface.h` and
  `tpc_kernel_lib_interface.h` come from `/usr/lib/habanatools/include`, so
  point `HABANA_INCLUDE_DIR` at that folder if a build needs them.
- **VM lifetime across agent runs is verified (2026-11-17).** The VM is
  started by `tpcsim.sh` and stays up when a shell exits: a later run found
  `XDG_CONFIG_HOME=/Volumes/Storage/paperclip/linux-vm colima status
  --profile tpcsim` reporting the VM running, with the socket under
  `/Volumes/Storage/paperclip/linux-vm/colima/tpcsim/docker.sock`. It
  survived the end of the previous agent run.
- **pytest 9.2.0 has no `--keep-dirs`.** For simulator-test debugging,
  replicate the test's compile/run steps in `/tmp` inside the container.
- Do not run pytest without `--noconftest` in the container:
  `tests/unit_tests/conftest.py` imports vLLM, which is not installed.

## Verify kernel: default flags, local memory and the bitcode diagnostic

`gdn_core_verify_gaudi2.c` is compiled with the flags of the other two GDN
kernels (`-Wall -Werror -O2 -march=gaudi2`) and no local-memory override. It
was once built with `-vlm 32`, passed this simulator and segfaulted on a card
(RUK-3105): a kernel that calls lookup-table functions has 16 KiB of vector
local memory, tpc-clang checks that only without `-vlm`, and **the simulator
does not check it at all**. A green simulator run therefore says nothing
about local memory. What does:

- `tpc-clang ... -reg-mem-count` prints `Total VLM used: N bytes` (statics
  plus spills; 12544 for the verify kernel, which holds 6 nodes), and
  `test_vector_local_memory` asserts that each kernel leaves 3 KiB of the
  16384 free;
- `tests/standalone/gdn_core_decode/test_kernel_build.py` builds
  `csrc/gdn_fused` with CMake and fails when `-vlm`, `-slm` or `-spill-vlm`
  reaches any kernel.

```bash
tools/flash_next/tpcsim.sh 'tpc-clang -Wall -Werror -O2 -march=gaudi2 -reg-mem-count \
    -I/usr/lib/habanatools/include csrc/gdn_fused/kernels/gdn_core_verify_gaudi2.c \
    -c -o /tmp/k.o 2>&1 | grep "VLM used"'                           # Total VLM used: 12544 bytes
tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest \
    tests/standalone/gdn_core_decode/test_simulation.py \
    tests/standalone/gdn_core_decode/test_kernel_build.py'           # 35 passed
```

`cmake -DGDN_FUSED_VERIFY_STRIP_BITCODE=ON` (default OFF) adds an
`objcopy --remove-section=.llvmbc` step after the compile of
`gdn_core_verify_gaudi2.o` in `csrc/gdn_fused/CMakeLists.txt`, so the verify
kernel's ELF carries no embedded LLVM bitcode. It is a card diagnostic kept
from RUK-3105, never the shipped build: one hypothesis there was that the
graph compiler recompiles that bitcode in-process, without the `-vlm 32` the
kernel then needed. The kernel needs no such flag any more. The verify
simulator fixture mirrors the option with `GDN_VERIFY_SIM_STRIP_BITCODE=1`;
both runs must pass (19 passed; with the strip 18 passed, 1 skipped, the
test that the ELF keeps its bitcode):

```bash
tools/flash_next/tpcsim.sh 'python3 -m pytest -q -p no:cacheprovider --noconftest \
    tests/standalone/gdn_core_decode/test_simulation.py -k verify'
tools/flash_next/tpcsim.sh 'GDN_VERIFY_SIM_STRIP_BITCODE=1 python3 -m pytest -q -p no:cacheprovider \
    --noconftest tests/standalone/gdn_core_decode/test_simulation.py -k verify'
```

The simulator's "instructions executed" counts every instruction word,
including the all-`nop` words tpc-clang inserts to wait out the latency of a
dependent chain (it schedules one basic block at a time). The count follows
the longest chain of a loop body, not the number of operations in it; an
operation that is off that chain is free.

## Known compiler hazard: in-place tensors

tpc-clang treats `v_bf16_ld_tnsr_b` and `v_bf16_st_tnsr` on the same tensor as
independent and freely reorders them, even though they touch the same
addresses. A kernel that reads and writes a tensor in place (e.g.
`conv_silu_block` in `gdn_core_decode_gaudi2.c` when
`load_slot == store_slot`) must make every store data-depend on every load of
the same addresses, otherwise the compiler hoists the stores above the loads
and the loads read the new data. The guard pattern is to predicate each stored
value on a comparison with the load that is never true (`h0 > +inf` is false
for every value, NaN included), which makes the store depend on the load
without changing the bits:

```c
const bool128 never = v_bf16_cmp_grt_b(h0, (bf16)(1.0f / 0.0f));
const bfloat128 s1 = v_bf16_mov_vb(h0, 0, h1, never, 0);
```

`aso(1, 1)` does not fix the reorder and tpc-clang has no aliasing option. The
aliased rows of the GDN simulator test (`tp8_mixed`, `tp4_silu` in
`tests/standalone/gdn_core_decode/test_simulation.py`, the rows with
`load_indices[r] == store_indices[r]`) catch a broken guard.
