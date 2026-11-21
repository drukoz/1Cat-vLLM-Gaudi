// SPDX-License-Identifier: Apache-2.0
// VLLM_HPU_GDN_VERIFY_VARIANT=2: the interleaved verification kernel with the row's v element taken by one broadcast
// and the output sum broadcast by one shuffle (GDN_VERIFY_LEAN_BROADCAST in gdn_core_verify_recurrence_interleaved.h):
// the same values in every lane, so the output and both states are bit-identical; 11 % fewer instructions in the
// simulator at TP8. Today's index space (docs/design/flash-next/hc-kernels.md, "GDN verify").
#define GDN_VERIFY_LEAN_BROADCAST 1
#include "gdn_core_verify_interleaved_gaudi2.c"
