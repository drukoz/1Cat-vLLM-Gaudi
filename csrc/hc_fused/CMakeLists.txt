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

# The MoE glue kernels (docs/design/flash-next/hc-fused.md, "MoE glue kernels") live in this library
# so that nothing new has to be loaded, registered in GC_KERNEL_PATH or added to the combined kernel
# database. Build isolation, as for the GDN verify kernel (csrc/gdn_fused/CMakeLists.txt): the
# compile of each one is tried here, at configure time, with the build's own command. A kernel that
# does not compile (or every one of them with -DHC_FUSED_MOE_GLUE=OFF) is left out: GetKernelGuids
# then does not list its GUID and InstantiateTpcKernel answers GLUE_NODE_NOT_FOUND for it, and the
# six hyper-connection kernels are served as usual. HC_FUSED_MOE_GLUE_BUILT in CMakeCache.txt
# records which were built. The same flags and the same rule as above: no local-memory override.
option(HC_FUSED_MOE_GLUE "Build the MoE glue kernels (GUIDs moe_glue_*_bf16_gaudi2) into the library" ON)
set(MOE_GLUE_KERNELS
    moe_glue_route_gaudi2
    moe_glue_silu_mul_gaudi2
    moe_glue_combine_gaudi2
    moe_glue_row_quant_gaudi2
    moe_glue_act_quant_gaudi2)
set(MOE_GLUE_KERNEL_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hc_fused_common.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/moe_glue_common.h")
set(MOE_GLUE_BUILT "")
set(MOE_GLUE_DEFINITIONS "")
set(MOE_GLUE_EMBEDDED_OBJECTS "")
if(HC_FUSED_MOE_GLUE)
    foreach(kernel IN LISTS MOE_GLUE_KERNELS)
        set(kernel_source "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}.c")
        # Configure again, and so try the compile again, whenever a source of the kernel changes.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${kernel_source}" ${MOE_GLUE_KERNEL_HEADERS})
        execute_process(
            COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 "-I${TPC_INCLUDE_DIR}"
                    "${kernel_source}" -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.probe.o"
            RESULT_VARIABLE MOE_GLUE_PROBE_RESULT
            OUTPUT_VARIABLE MOE_GLUE_PROBE_LOG
            ERROR_VARIABLE MOE_GLUE_PROBE_LOG)
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.probe.o")
        if(NOT MOE_GLUE_PROBE_RESULT EQUAL 0)
            string(REGEX MATCH "[^\n]*error[^\n]*" MOE_GLUE_PROBE_ERROR "${MOE_GLUE_PROBE_LOG}")
            message(WARNING
                "${kernel}.c does not compile with the default TPC flags "
                "(tpc-clang exit ${MOE_GLUE_PROBE_RESULT}: ${MOE_GLUE_PROBE_ERROR}). "
                "Building libhc_fused_gaudi2.so without it: its GUID is absent and the MoE glue "
                "(VLLM_HPU_MOE_GLUE=1) stays on the torch path.")
            continue()
        endif()
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 "-I${TPC_INCLUDE_DIR}"
                    "${kernel_source}" -c -o "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            DEPENDS "${kernel_source}" ${MOE_GLUE_KERNEL_HEADERS}
            VERBATIM)
        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
            COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                    ${kernel}.o ${kernel}_embedded.o
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.o"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            VERBATIM)
        set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
        list(APPEND MOE_GLUE_EMBEDDED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${kernel}_embedded.o")
        list(APPEND MOE_GLUE_BUILT "${kernel}")
        # moe_glue_route_gaudi2 -> HC_FUSED_HAS_MOE_GLUE_ROUTE (host/moe_glue_gaudi2.cpp)
        string(REGEX REPLACE "_gaudi2$" "" kernel_stem "${kernel}")
        string(TOUPPER "${kernel_stem}" kernel_define)
        list(APPEND MOE_GLUE_DEFINITIONS "HC_FUSED_HAS_${kernel_define}=1")
    endforeach()
endif()
set(HC_FUSED_MOE_GLUE_BUILT "${MOE_GLUE_BUILT}" CACHE INTERNAL "The MoE glue kernels in the library" FORCE)
message(STATUS "hc_fused: MoE glue kernels in the library: [${MOE_GLUE_BUILT}]")

add_library(
    hc_fused_gaudi2
    SHARED
    host/hc_fused_gaudi2.cpp
    host/moe_glue_gaudi2.cpp
    host/entry_points.cpp
    ${HC_FUSED_EMBEDDED_OBJECTS}
    ${MOE_GLUE_EMBEDDED_OBJECTS}
)
target_compile_definitions(hc_fused_gaudi2 PRIVATE ${MOE_GLUE_DEFINITIONS})
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
