# SPDX-License-Identifier: Apache-2.0
#
# Standalone Gaudi2 kernel database for the fused QSA decode kernels (VLLM_HPU_QSA_FUSED=2). Its own library,
# so a failure here cannot touch the fused GDN or hyper-connection kernels; see docs/design/flash-next/qsa-fused.md.
cmake_minimum_required(VERSION 3.18)
project(qsa_fused_gaudi2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

# One .c per GUID, compiled with the flags of the proven GDN decode kernel and no -vlm (the prepare kernel
# calls a lookup-table special function, which leaves 16 KiB of vector local memory; it keeps none).
set(QSA_FUSED_KERNELS
    qsa_fused_prepare_gaudi2
    qsa_fused_emit_gaudi2
    qsa_fused_threshold_gaudi2
    qsa_fused_select_gaudi2)
set(QSA_FUSED_KERNEL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qsa_fused_common.h")

set(QSA_FUSED_EMBEDDED_OBJECTS "")
foreach(kernel IN LISTS QSA_FUSED_KERNELS)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 "-I${TPC_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c"
                -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c" ${QSA_FUSED_KERNEL_HEADERS}
        VERBATIM)
    # objcopy names the symbols after the bare input file name: _binary_<kernel>_o_start / _end.
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                ${kernel}.o ${kernel}_embedded.o
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND QSA_FUSED_EMBEDDED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o")
endforeach()

add_library(
    qsa_fused_gaudi2
    SHARED
    host/qsa_fused_gaudi2.cpp
    host/entry_points.cpp
    ${QSA_FUSED_EMBEDDED_OBJECTS}
)
target_include_directories(
    qsa_fused_gaudi2
    PRIVATE
    host
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(qsa_fused_gaudi2 PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(qsa_fused_gaudi2 PRIVATE -Wl,-z,noexecstack)
