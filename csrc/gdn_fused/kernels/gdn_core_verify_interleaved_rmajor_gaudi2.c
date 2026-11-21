// SPDX-License-Identifier: Apache-2.0
// VLLM_HPU_GDN_VERIFY_VARIANT=1: the interleaved verification kernel (gdn_core_verify_interleaved_gaudi2.c) with a
// request-major index space, {requests, key heads} instead of {key heads, requests}. Every point does the work of the
// interleaved kernel's point, the same operations in the same order, so the output and both states are
// bit-identical; only the order of the index-space dimensions changes. Today's layout lets the graph compiler cut
// only the request dimension: at TP8 (2 key heads) with 32 requests it ran on 16 of the 24 TPC engines, 4 points
// each (job 00205's trace). Served when the process has VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 and the library was built
// with it (docs/design/flash-next/hc-kernels.md, "GDN verify").
#define GDN_VERIFY_REQUEST_MAJOR 1
#include "gdn_core_verify_interleaved_gaudi2.c"
