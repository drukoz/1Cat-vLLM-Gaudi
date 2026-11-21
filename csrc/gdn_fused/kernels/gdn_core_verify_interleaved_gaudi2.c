// SPDX-License-Identifier: Apache-2.0
// The fused GDN verification kernel with the recurrence of gdn_core_verify_recurrence_interleaved.h: two
// independent value rows per iteration instead of one (row r of each half of the head). The same operations in
// the same order for every row, so the output and both states are bit-identical to gdn_core_verify_gaudi2.c
// (idea I27; docs/design/flash-next/gdn-core-decode.md, "Rows per iteration"). The host glue hands this ELF to
// the graph compiler for the core-verify GUID when the process has VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 and the
// library was built with it.
#define GDN_CORE_VERIFY_RECURRENCE_HEADER "gdn_core_verify_recurrence_interleaved.h"
#include "gdn_core_verify_gaudi2.c"
