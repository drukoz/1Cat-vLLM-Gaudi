// SPDX-License-Identifier: Apache-2.0
// VLLM_HPU_GDN_VERIFY_VARIANT=3: both changes, the request-major index space (gdn_core_verify_interleaved_rmajor_
// gaudi2.c) and the lean broadcasts (gdn_core_verify_interleaved_lean_gaudi2.c); bit-identical outputs and states
// (docs/design/flash-next/hc-kernels.md, "GDN verify").
#define GDN_VERIFY_REQUEST_MAJOR 1
#define GDN_VERIFY_LEAN_BROADCAST 1
#include "gdn_core_verify_interleaved_gaudi2.c"
