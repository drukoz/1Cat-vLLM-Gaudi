// SPDX-License-Identifier: Apache-2.0
// The fused GDN core-decode kernel with the recurrence of gdn_core_decode_recurrence_interleaved.h: four
// independent value rows per iteration instead of one. The same operations in the same order for every row,
// so the output and both states are bit-identical to gdn_core_decode_gaudi2.c; the four dependent chains share
// one basic block and fill each other's latency (idea I27; docs/design/flash-next/gdn-core-decode.md, "Rows per
// iteration"). The host glue hands this ELF to the graph compiler for the core-decode GUID when the process
// has VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 and the library was built with it.
#define GDN_CORE_DECODE_RECURRENCE_HEADER "gdn_core_decode_recurrence_interleaved.h"
#include "gdn_core_decode_gaudi2.c"
