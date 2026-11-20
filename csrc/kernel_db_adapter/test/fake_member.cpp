// SPDX-License-Identifier: Apache-2.0
//
// Fake kernel-library member for the combined kernel database host test
// (RUK-3243). -DFAKE_TAG=<t> exports the GUIDs fake_<t>_only + fake_shared.
// Each entry point records its name in the exported fake_last_call() string and
// returns values that let the test tell the members apart.
//
// Variants:
//   FAKE_TAG=a, FAKE_TAG=b   all six entry points (like Intel's libtpc_kernels.so)
//   -DFAKE_MINIMAL           only the three entry points the graph compiler requires
//                            (GetKernelGuids, InstantiateTpcKernel, GetShapeInference),
//                            like libgdn_fused_gaudi2.so / libflashinfer_gaudi_kernels.so
//   -DFAKE_NO_SHAPE_INFERENCE  additionally drops GetShapeInference: not a loadable
//                            kernel library (the graph compiler refuses it)
#include <tpc_kernel_lib_interface.h>

#include <cstdio>
#include <cstring>

using namespace tpc_lib_api;

#define KDA_STR(x) #x
#define KDA_XSTR(x) KDA_STR(x)

namespace {

const char* const kTag = KDA_XSTR(FAKE_TAG);  // "a", "b", "m" (minimal) or "n" (no shape inference)
const char* g_last = "none";

// What InstantiateTpcKernel writes to kernel.paramsNr, so the test can tell which member answered.
unsigned member_number() {
    switch (kTag[0]) {
        case 'a':
            return 1U;
        case 'b':
            return 2U;
        case 'm':
            return 3U;
        default:
            return 4U;
    }
}

const char* guid_name(char* buffer, size_t size) {
    std::snprintf(buffer, size, "fake_%s_only", kTag);
    return buffer;
}

void set_guid(GuidInfo* info, const char* name) {
    std::memset(info, 0, sizeof(*info));
    std::snprintf(info->name, MAX_NODE_NAME, "%s", name);
}

}  // namespace

extern "C" const char* fake_last_call() { return g_last; }

extern "C" GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t* kernelCount, GuidInfo* guids) {
    g_last = "GetKernelGuids";
    if (deviceId != DEVICE_ID_GAUDI2) {
        if (kernelCount) {
            *kernelCount = 0;
        }
        return GLUE_SUCCESS;
    }
    char unique[MAX_NODE_NAME];
    guid_name(unique, sizeof(unique));
    const char* names[2] = {unique, "fake_shared"};
    const uint32_t total = 2;
    const uint32_t capacity = kernelCount ? *kernelCount : 0;
    if (!guids || capacity == 0) {
        if (kernelCount) {
            *kernelCount = total;
        }
        return GLUE_SUCCESS;
    }
    if (capacity < total) {
        return GLUE_FAILED;
    }
    for (uint32_t i = 0; i < total; ++i) {
        set_guid(&guids[i], names[i]);
    }
    if (kernelCount) {
        *kernelCount = total;
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    g_last = "InstantiateTpcKernel";
    if (!params || !instance) {
        return GLUE_FAILED;
    }
    instance->kernel.paramsNr = member_number();
    return GLUE_SUCCESS;
}

#ifndef FAKE_NO_SHAPE_INFERENCE
extern "C" GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* inputParams,
                                            ShapeInferenceOutput* outputData) {
    g_last = "GetShapeInference";
    (void)deviceId;
    (void)inputParams;
    (void)outputData;
    return GLUE_SUCCESS;
}
#endif  // FAKE_NO_SHAPE_INFERENCE

#ifndef FAKE_MINIMAL
extern "C" GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, void* manipulation) {
    g_last = "GetSuggestedManipulation";
    (void)params;
    (void)manipulation;
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params, NodeDataLayouts* supportedLayouts,
                                                  uint32_t* layoutCount) {
    g_last = "GetSupportedDataLayouts";
    (void)params;
    (void)supportedLayouts;
    if (layoutCount) {
        *layoutCount = 1;
    }
    return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() {
    g_last = "GetLibVersion";
    return 1000ULL + member_number();
}
#endif  // FAKE_MINIMAL
