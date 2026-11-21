// SPDX-License-Identifier: Apache-2.0
// VLLM_HPU_GDN_VERIFY_VARIANT=6 (and 7, with the preferred split of the key heads): the verification kernel with the
// recurrence of gdn_core_verify_recurrence_fold.h (four value rows per iteration whose row sums are folded together
// by v_f32_reduce_add's own additions; node t's output overlapped with node t + 1's update, and a row group's last
// output with the next group's first update) and the gates of all nodes computed at once
// (GDN_VERIFY_BATCHED_GATES). The same operations on every value in the same order, so the output and both states
// are bit-identical to the interleaved kernel's (simulator, every verify case); 37 to 39 % fewer instructions
// (docs/design/flash-next/hc-kernels.md, "GDN verify").
#define GDN_VERIFY_BATCHED_GATES 1
#define GDN_CORE_VERIFY_RECURRENCE_HEADER "gdn_core_verify_recurrence_fold.h"
#include "gdn_core_verify_gaudi2.c"
