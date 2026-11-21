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

# One flag list for all three kernels. Every kernel here calls lookup-table functions, and such a
# kernel has 16 KiB of vector local memory for its __local__ vectors plus the compiler's vector
# spills. tpc-clang checks that total, so a kernel that needs more fails to compile. No
# local-memory override (-vlm, -slm, -spill-vlm) may appear here: -vlm raises the compiler's limit
# only, and the verify kernel built with -vlm 32 (28928 bytes) passed the simulator and segfaulted
# on a card (RUK-3105).
set(GDN_TPC_FLAGS -Wall -Werror -O2 -march=gaudi2)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2.o"
    COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} -I"${TPC_INCLUDE_DIR}"
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
    COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} -I"${TPC_INCLUDE_DIR}"
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

# Build isolation (RUK-3105): the verify kernel must not be able to take the decode and recurrent
# kernels with it. Its compile is tried here, at configure time, with the build's own command. When
# that fails (or with -DGDN_FUSED_VERIFY=OFF) the library is built without the verify kernel:
# GetKernelGuids then reports the decode and recurrent GUIDs only, and InstantiateTpcKernel answers
# GLUE_NODE_NOT_FOUND for the verify GUID (GDN_FUSED_NO_VERIFY in host/entry_points.cpp). The op
# extension (pytorch/) still registers gdn_core_verify, so keep VLLM_HPU_GDN_CORE_VERIFY=0 with such
# a library. GDN_FUSED_VERIFY_BUILT in CMakeCache.txt records the outcome.
option(GDN_FUSED_VERIFY "Build the verify kernel (GUID gdn_fused_core_verify_f32state_gaudi2) into the library" ON)
set(GDN_VERIFY_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_gaudi2.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_recurrence.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_norm.h")
set(GDN_VERIFY_BUILT OFF)
if(GDN_FUSED_VERIFY)
    # Configure again, and so try the compile again, whenever a verify kernel source changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GDN_VERIFY_KERNEL_SOURCES})
    execute_process(
        COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} "-I${TPC_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_gaudi2.c"
                -c -o "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.probe.o"
        RESULT_VARIABLE GDN_VERIFY_PROBE_RESULT
        OUTPUT_VARIABLE GDN_VERIFY_PROBE_LOG
        ERROR_VARIABLE GDN_VERIFY_PROBE_LOG)
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.probe.o")
    if(GDN_VERIFY_PROBE_RESULT EQUAL 0)
        set(GDN_VERIFY_BUILT ON)
    else()
        string(REGEX MATCH "[^\n]*error[^\n]*" GDN_VERIFY_PROBE_ERROR "${GDN_VERIFY_PROBE_LOG}")
        message(WARNING
            "gdn_core_verify_gaudi2.c does not compile with the default TPC flags "
            "(tpc-clang exit ${GDN_VERIFY_PROBE_RESULT}: ${GDN_VERIFY_PROBE_ERROR}). "
            "Building libgdn_fused_gaudi2.so with the decode and recurrent kernels only: "
            "the verify GUID is absent, keep VLLM_HPU_GDN_CORE_VERIFY=0.")
    endif()
endif()
set(GDN_FUSED_VERIFY_BUILT "${GDN_VERIFY_BUILT}" CACHE INTERNAL "ON when the verify kernel is in the library" FORCE)
message(STATUS "gdn_fused: verify kernel in the library: ${GDN_VERIFY_BUILT}")

# Card diagnostic only, never for a shipped library (default OFF): drop the embedded LLVM bitcode
# from the verify kernel ELF, so that nothing on the card can recompile the kernel from it. The
# kernel no longer needs a flag the bitcode does not carry (it used to need -vlm 32, RUK-3105), so
# a recompile now uses the same flags as this build; tools/flash_next/gdn_verify_crash_probe.sh
# still uses the option to tell a recompile problem from a kernel problem.
option(GDN_FUSED_VERIFY_STRIP_BITCODE "Diagnostic: remove .llvmbc from gdn_core_verify_gaudi2.o" OFF)
set(GDN_VERIFY_STRIP_COMMAND "")
if(GDN_FUSED_VERIFY_STRIP_BITCODE)
    set(GDN_VERIFY_STRIP_COMMAND
        COMMAND "${ELF_OBJCOPY}" --remove-section=.llvmbc "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.o")
endif()

set(GDN_VERIFY_LIBRARY_SOURCES "")
if(GDN_VERIFY_BUILT)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.o"
        COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} -I"${TPC_INCLUDE_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_gaudi2.c"
                -c -o "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.o"
        ${GDN_VERIFY_STRIP_COMMAND}
        DEPENDS ${GDN_VERIFY_KERNEL_SOURCES}
        VERBATIM)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2_embedded.o"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                gdn_core_verify_gaudi2.o gdn_core_verify_gaudi2_embedded.o
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2.o"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2_embedded.o"
        PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    set(GDN_VERIFY_LIBRARY_SOURCES
        host/gdn_core_verify_gaudi2.cpp
        "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_verify_gaudi2_embedded.o")
endif()

# The interleaved builds of the decode and verify kernels (VLLM_HPU_GDN_KERNEL_INTERLEAVE=1; idea I27): the same
# kernels with several value rows per iteration of the recurrence, bit-identical outputs. Optional like the verify
# kernel: each compile is tried here first, and one that fails (or -DGDN_FUSED_INTERLEAVE=OFF) is left out. The
# glue's references to them are weak, so the library then serves today's kernels whatever the variable says.
# GDN_FUSED_INTERLEAVE_BUILT in CMakeCache.txt records which ones are in.
option(GDN_FUSED_INTERLEAVE "Build the interleaved decode and verify kernels into the library" ON)
set(GDN_INTERLEAVE_LIBRARY_SOURCES "")
set(GDN_INTERLEAVE_BUILT "")
set(GDN_INTERLEAVE_CANDIDATES gdn_core_decode_interleaved_gaudi2)
if(GDN_VERIFY_BUILT)
    list(APPEND GDN_INTERLEAVE_CANDIDATES gdn_core_verify_interleaved_gaudi2)
    # The variant builds of the interleaved verify kernel (VLLM_HPU_GDN_VERIFY_VARIANT 1, 2, 3,
    # docs/design/flash-next/hc-kernels.md, "GDN verify"): optional like the others.
    list(APPEND GDN_INTERLEAVE_CANDIDATES gdn_core_verify_interleaved_rmajor_gaudi2
         gdn_core_verify_interleaved_lean_gaudi2 gdn_core_verify_interleaved_rmajor_lean_gaudi2)
    # VLLM_HPU_GDN_VERIFY_VARIANT 6 and 7: the fold recurrence (kernels/gdn_core_verify_recurrence_fold.h).
    list(APPEND GDN_INTERLEAVE_CANDIDATES gdn_core_verify_interleaved_fold_gaudi2)
endif()
if(GDN_FUSED_INTERLEAVE)
    foreach(kernel IN LISTS GDN_INTERLEAVE_CANDIDATES)
        set(kernel_sources
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_interleaved_gaudi2.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_recurrence_interleaved.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_recurrence_interleaved.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_verify_recurrence_fold.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_core_decode_norm.h")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${kernel_sources})
        execute_process(
            COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} "-I${TPC_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c"
                    -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.probe.o"
            RESULT_VARIABLE probe_result
            OUTPUT_VARIABLE probe_log
            ERROR_VARIABLE probe_log)
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.probe.o")
        if(NOT probe_result EQUAL 0)
            string(REGEX MATCH "[^\n]*error[^\n]*" probe_error "${probe_log}")
            message(WARNING "${kernel}.c does not compile with the default TPC flags (tpc-clang exit "
                            "${probe_result}: ${probe_error}); the library serves today's kernel for it.")
            continue()
        endif()
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            COMMAND "${TPC_COMPILER}" ${GDN_TPC_FLAGS} -I"${TPC_INCLUDE_DIR}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c"
                    -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            DEPENDS ${kernel_sources}
            VERBATIM)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
            COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64 ${kernel}.o ${kernel}_embedded.o
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            VERBATIM)
        set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND GDN_INTERLEAVE_LIBRARY_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o")
        list(APPEND GDN_INTERLEAVE_BUILT ${kernel})
    endforeach()
endif()
set(GDN_FUSED_INTERLEAVE_BUILT "${GDN_INTERLEAVE_BUILT}" CACHE INTERNAL "the interleaved kernels in the library" FORCE)
message(STATUS "gdn_fused: interleaved kernels in the library: ${GDN_INTERLEAVE_BUILT}")

add_library(
    gdn_fused_gaudi2
    SHARED
    host/gdn_core_decode_gaudi2.cpp
    host/gdn_recurrent_decode_gaudi2.cpp
    host/entry_points.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/gdn_core_decode_gaudi2_embedded.o"
    "${CMAKE_CURRENT_BINARY_DIR}/gdn_recurrent_decode_gaudi2_embedded.o"
    ${GDN_VERIFY_LIBRARY_SOURCES}
    ${GDN_INTERLEAVE_LIBRARY_SOURCES}
)
if(NOT GDN_VERIFY_BUILT)
    target_compile_definitions(gdn_fused_gaudi2 PRIVATE GDN_FUSED_NO_VERIFY=1)
endif()
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
