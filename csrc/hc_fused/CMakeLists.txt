# SPDX-License-Identifier: Apache-2.0
#
# Standalone Gaudi2 kernel database for the fused hyper-connection kernels. It is built as its own
# library so a failure here cannot touch the fused GDN kernels (csrc/gdn_fused) or the flashinfer_gaudi
# kernel database; see docs/design/flash-next/hc-fused.md.
cmake_minimum_required(VERSION 3.18)
project(hc_fused_gaudi2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

# One .c per GUID. Every kernel is compiled with the flags of the proven GDN decode kernel and no
# -vlm: these kernels call lookup-table special functions, which leaves 16 KiB of vector local memory,
# and the graph compiler may recompile the embedded bitcode without any flag given here.
set(HC_FUSED_KERNELS
    hc_fused_norm_gaudi2
    hc_fused_silu_gaudi2
    hc_fused_silu_split_gaudi2
    hc_fused_gate_mix_gaudi2
    hc_fused_combine_gaudi2
    hc_fused_combine_norm_gaudi2)
set(HC_FUSED_KERNEL_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hc_fused_common.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hc_fused_silu.h")

set(HC_FUSED_EMBEDDED_OBJECTS "")
foreach(kernel IN LISTS HC_FUSED_KERNELS)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 "-I${TPC_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c"
                -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c" ${HC_FUSED_KERNEL_HEADERS}
        VERBATIM)
    # objcopy names the symbols after the input file name as given, so it runs in the build directory
    # on the bare name: _binary_<kernel>_o_start / _end, which host/hc_fused_gaudi2.cpp declares.
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                ${kernel}.o ${kernel}_embedded.o
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND HC_FUSED_EMBEDDED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o")
endforeach()

add_library(
    hc_fused_gaudi2
    SHARED
    host/hc_fused_gaudi2.cpp
    host/entry_points.cpp
    ${HC_FUSED_EMBEDDED_OBJECTS}
)
target_include_directories(
    hc_fused_gaudi2
    PRIVATE
    host
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(hc_fused_gaudi2 PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(hc_fused_gaudi2 PRIVATE -Wl,-z,noexecstack)
