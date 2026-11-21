// SPDX-License-Identifier: Apache-2.0
// Lists what a built libgdn_fused_gaudi2.so serves, for test_kernel_build.py.
//
// argv[1] is the library; every further argument is a GUID name. The program loads the library,
// prints one "guid <name>" line per GUID that GetKernelGuids reports for Gaudi2, then one
// "instantiate <name> <GlueCodeReturn>" line per GUID argument: InstantiateTpcKernel called with
// that name and no tensors, so a served GUID answers with its glue's own input-count code and an
// unknown one with GLUE_NODE_NOT_FOUND (1).
#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    using namespace tpc_lib_api;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <libgdn_fused_gaudi2.so> [guid ...]\n", argv[0]);
        return 2;
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "guid_list_host: cannot load %s: %s\n", argv[1], dlerror());
        return 2;
    }
    using GetKernelGuidsFn = GlueCodeReturn (*)(DeviceId, uint32_t*, GuidInfo*);
    using InstantiateFn = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);
    const auto getKernelGuids = reinterpret_cast<GetKernelGuidsFn>(dlsym(library, "GetKernelGuids"));
    const auto instantiate = reinterpret_cast<InstantiateFn>(dlsym(library, "InstantiateTpcKernel"));
    if (!getKernelGuids || !instantiate) {
        std::fprintf(stderr, "guid_list_host: %s lacks GetKernelGuids or InstantiateTpcKernel\n", argv[1]);
        return 2;
    }

    uint32_t count = 0;
    if (getKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) != GLUE_SUCCESS) {
        std::fprintf(stderr, "guid_list_host: the GetKernelGuids size query failed\n");
        return 3;
    }
    std::vector<GuidInfo> guids(count);
    uint32_t filled = count;
    if (count && getKernelGuids(DEVICE_ID_GAUDI2, &filled, guids.data()) != GLUE_SUCCESS) {
        std::fprintf(stderr, "guid_list_host: GetKernelGuids failed\n");
        return 3;
    }
    for (uint32_t i = 0; i < count; ++i) {
        std::printf("guid %s\n", guids[i].name);
    }

    for (int i = 2; i < argc; ++i) {
        HabanaKernelParams params{};
        HabanaKernelInstantiation instance{};
        params.deviceId = DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, argv[i], sizeof(params.guid.name) - 1);
        std::printf("instantiate %s %d\n", argv[i], static_cast<int>(instantiate(&params, &instance)));
    }
    return 0;
}
