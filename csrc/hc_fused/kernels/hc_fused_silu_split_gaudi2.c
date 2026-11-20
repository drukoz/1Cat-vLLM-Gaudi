// SPDX-License-Identifier: Apache-2.0
// hc_fused_silu_split: silu of the lora columns of the merged down+inject GEMM output, plus the
// injection logits as a second output. The body is hc_fused_silu.h.
#define HC_FUSED_SILU_SPLIT
#include "hc_fused_silu.h"
