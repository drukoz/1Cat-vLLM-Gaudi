// SPDX-License-Identifier: Apache-2.0
//
// Fake kernel-library member for the combined kernel database host test
// (RUK-3243). Compiled twice: -DFAKE_TAG=a exports GUIDs fake_a_only +
// fake_shared, -DFAKE_TAG=b exports fake_b_only + fake_shared. Each entry point
// records its name in the exported fake_last_call() string and returns values
// that let the test tell the members apart.
#include <tpc_kernel_lib_interface.h>

#include <cstdio>
#include <cstring>

using namespace tpc_lib_api;

#define KDA_STR(x) #x
#define KDA_XSTR(x) KDA_STR(x)

namespace {

const char* const kTag = KDA_XSTR(FAKE_TAG);  // "a" or "b"
const char* g_last = "none";

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
    instance->kernel.paramsNr = (kTag[0] == 'a') ? 1U : 2U;
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* inputParams,
                                            ShapeInferenceOutput* outputData) {
    g_last = "GetShapeInference";
    (void)deviceId;
    (void)inputParams;
    (void)outputData;
    return GLUE_SUCCESS;
}

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
    return (kTag[0] == 'a') ? 1001ULL : 1002ULL;
}
