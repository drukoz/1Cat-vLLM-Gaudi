// SPDX-License-Identifier: Apache-2.0
#pragma once
// Which ELF the glue hands the graph compiler for the core-decode and core-verify GUIDs.
//
// VLLM_HPU_GDN_KERNEL_INTERLEAVE=1 in the process (vllm_gaudi/envs.py) asks for the interleaved builds of the two
// kernels (kernels/gdn_core_*_interleaved_gaudi2.c): several independent value rows per iteration of the
// recurrence, bit-identical outputs and states, fewer TPC instructions. The library carries them only when they
// compiled (CMakeLists.txt, GDN_FUSED_INTERLEAVE); their ELF symbols are weak, so a library or a test host without
// them links and serves today's kernels. Any other value of the variable, or none, is today's kernel.
#include <cstdlib>
#include <cstring>

inline bool GdnKernelInterleaveRequested()
{
    const char* value = std::getenv("VLLM_HPU_GDN_KERNEL_INTERLEAVE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct GdnKernelElf {
    const unsigned char* start;
    unsigned size;
};

// VLLM_HPU_GDN_VERIFY_VARIANT (with the interleave variable at "1") picks a build of the interleaved verify kernel,
// each with bit-identical outputs and states (docs/design/flash-next/hc-kernels.md, "GDN verify"):
//   1 the request-major index space {requests, key heads} (kernels/gdn_core_verify_interleaved_rmajor_gaudi2.c)
//   2 the lean broadcasts of the recurrence (kernels/gdn_core_verify_interleaved_lean_gaudi2.c)
//   3 both (kernels/gdn_core_verify_interleaved_rmajor_lean_gaudi2.c)
//   4 today's index space and ELF, with the graph compiler asked to split the key-head dimension into partitions
//     of one (preferredSplitDim) and to go on with the request dimension, so that every (key head, request) point
//     can land on its own TPC engine
//   5 as 4, with the lean ELF
//   6 the fold recurrence (kernels/gdn_core_verify_interleaved_fold_gaudi2.c): four rows' sums folded together by
//     the additions of v_f32_reduce_add, node t's output overlapped with node t + 1's update; today's index space
//   7 as 6, with the preferred split of 4
// Any other value, or none, is 0: the interleaved kernel as it is.
inline int GdnVerifyVariantRequested()
{
    const char* value = std::getenv("VLLM_HPU_GDN_VERIFY_VARIANT");
    if (value == nullptr) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "2") == 0) return 2;
    if (std::strcmp(value, "3") == 0) return 3;
    if (std::strcmp(value, "4") == 0) return 4;
    if (std::strcmp(value, "5") == 0) return 5;
    if (std::strcmp(value, "6") == 0) return 6;
    if (std::strcmp(value, "7") == 0) return 7;
    return 0;
}

// True when the (weak) ELF symbols of an optional kernel build are linked in.
inline bool GdnElfLinked(const unsigned char* start, const unsigned char* end)
{
    return start != nullptr && end != nullptr && end > start;
}

// The interleaved ELF when it was asked for and is linked in, otherwise today's.
inline GdnKernelElf GdnSelectKernelElf(const unsigned char* baseStart, const unsigned char* baseEnd,
                                       const unsigned char* interleavedStart, const unsigned char* interleavedEnd)
{
    if (GdnKernelInterleaveRequested() && interleavedStart != nullptr && interleavedEnd != nullptr &&
        interleavedEnd > interleavedStart) {
        return {interleavedStart, static_cast<unsigned>(interleavedEnd - interleavedStart)};
    }
    return {baseStart, static_cast<unsigned>(baseEnd - baseStart)};
}
