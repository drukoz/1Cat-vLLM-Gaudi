# SPDX-License-Identifier: Apache-2.0
#
# Combined Gaudi2 kernel database (RUK-3243): a host C++ adapter that forwards
# every TPC kernel-library entry point to its member libraries, so GC_KERNEL_PATH
# can name one regular file. This is host C++ only: no tpc-clang step, no objcopy.
# See docs/design/flash-next/native-extensions.md.
cmake_minimum_required(VERSION 3.18)
project(flashinfer_gaudi_combined LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "Gaudi host API include directory")

add_library(flashinfer_gaudi_combined_gaudi2 SHARED adapter.cpp)
set_target_properties(flashinfer_gaudi_combined_gaudi2 PROPERTIES OUTPUT_NAME "flashinfer_gaudi_combined_gaudi2")
target_include_directories(
    flashinfer_gaudi_combined_gaudi2
    PRIVATE
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(flashinfer_gaudi_combined_gaudi2 PRIVATE -Wall -Wextra -Werror)
target_link_options(flashinfer_gaudi_combined_gaudi2 PRIVATE -Wl,-z,noexecstack)
target_link_libraries(flashinfer_gaudi_combined_gaudi2 PRIVATE ${CMAKE_DL_LIBS})

option(KERNEL_DB_ADAPTER_TESTS "Build the combined kernel database host tests" ON)
if(KERNEL_DB_ADAPTER_TESTS)
    enable_testing()

    # A fake member library: fake_member(<name> <tag> [extra compile definitions...]).
    function(fake_member name tag)
        add_library(${name} SHARED test/fake_member.cpp)
        target_compile_definitions(${name} PRIVATE FAKE_TAG=${tag} ${ARGN})
        target_include_directories(${name} PRIVATE "${TPC_INCLUDE_DIR}")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Werror)
        target_link_options(${name} PRIVATE -Wl,-z,noexecstack)
    endfunction()

    fake_member(fake_member_a a)  # all six entry points
    fake_member(fake_member_b b)  # all six entry points
    fake_member(fake_member_h h)  # all six entry points; stands for libhc_fused_gaudi2.so
    fake_member(fake_member_min m FAKE_MINIMAL)  # only the three required entry points
    fake_member(fake_member_nosif n FAKE_MINIMAL FAKE_NO_SHAPE_INFERENCE)  # lacks a required entry point

    add_executable(test_adapter test/test_adapter.cpp)
    target_include_directories(test_adapter PRIVATE "${TPC_INCLUDE_DIR}")
    target_compile_definitions(
        test_adapter
        PRIVATE
        ADAPTER_PATH="$<TARGET_FILE:flashinfer_gaudi_combined_gaudi2>"
        FAKE_A_PATH="$<TARGET_FILE:fake_member_a>"
        FAKE_B_PATH="$<TARGET_FILE:fake_member_b>"
        FAKE_H_PATH="$<TARGET_FILE:fake_member_h>"
        FAKE_MIN_PATH="$<TARGET_FILE:fake_member_min>"
        FAKE_NOSIF_PATH="$<TARGET_FILE:fake_member_nosif>"
    )
    target_compile_options(test_adapter PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(test_adapter PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(
        test_adapter
        flashinfer_gaudi_combined_gaudi2
        fake_member_a
        fake_member_b
        fake_member_h
        fake_member_min
        fake_member_nosif
    )

    add_test(NAME kernel_db_adapter COMMAND test_adapter)
endif()
