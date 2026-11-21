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
