# SPDX-License-Identifier: Apache-2.0
#
# Standalone Gaudi2 kernel database for the fused GDN core-decode op. It is
# built as its own library so a failure here cannot touch the existing
# flashinfer_gaudi kernel database; see docs/design/flash-next/gdn-core-decode.md.
cmake_minimum_required(VERSION 3.18)
project(gdn_fused_gaudi2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2.o"
    COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 -I"${TPC_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_gaudi2.c"
            -c -o "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2.o"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_recurrence.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_norm.h"
    VERBATIM)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2_embedded.o"
    COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            gdn_core_decode_gaudi2.o gdn_core_decode_gaudi2_embedded.o
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2.o"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2_embedded.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2.o"
    COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 -I"${TPC_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_recurrent_decode_gaudi2.c"
            -c -o "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2.o"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_recurrent_decode_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_recurrence.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_norm.h"
    VERBATIM)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2_embedded.o"
    COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
            gdn_recurrent_decode_gaudi2.o gdn_recurrent_decode_gaudi2_embedded.o
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2.o"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2_embedded.o"
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

add_library(
    gdn_fused_gaudi2
    SHARED
    host/gdn_core_decode_gaudi2.cpp
    host/gdn_recurrent_decode_gaudi2.cpp
    host/entry_points.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2_embedded.o"
    "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2_embedded.o"
)
target_include_directories(
    gdn_fused_gaudi2
    PRIVATE
    host
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(gdn_fused_gaudi2 PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(gdn_fused_gaudi2 PRIVATE -Wl,-z,noexecstack)
