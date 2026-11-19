// SPDX-License-Identifier: Apache-2.0
//
// Combined Gaudi2 kernel database (RUK-3243).
//
// One regular .so that forwards every TPC kernel-library entry point to its
// member libraries, so GC_KERNEL_PATH can name a single file. The Habana PyTorch
// bridge (1.24.1) accepts only one regular file in GC_KERNEL_PATH when
// PT_HPU_RECIPE_CACHE_CONFIG is set, while SynapseAI itself accepts a colon list;
// this generalizes csrc/deepseek_v4/memory_ready/glue.cpp from one base library
// to an ordered list of members. See docs/design/flash-next/native-extensions.md.
#include <tpc_kernel_lib_interface.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tpc_lib_api;

namespace {

// GetSuggestedManipulation has no typedef in tpc_kernel_lib_interface.h; the
// DeepSeek glue casts the symbol to this raw signature.
using GetSuggestedManipulationFn = GlueCodeReturn (*)(const HabanaKernelParams*, void*);

struct Member {
    void* handle = nullptr;
    std::string path;
    pfnGetKernelGuids getKernelGuids = nullptr;
    pfnInstantiateTpcKernel instantiate = nullptr;
    pfnGetShapeInference shapeInference = nullptr;
    GetSuggestedManipulationFn suggestedManipulation = nullptr;
    pfnGetSupportedDataLayout supportedDataLayouts = nullptr;
    pfnGetLibVersion getLibVersion = nullptr;
};

std::once_flag g_once;
std::vector<Member> g_members;                    // in precedence order (first wins)
std::unordered_map<std::string, size_t> g_owner;  // guid name -> owning member index
std::vector<GuidInfo> g_mergedGuids;              // merged GUID list, member order
std::string g_lastError;                          // last member path that failed to open

std::string own_directory() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&own_directory), &info) || !info.dli_fname) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
}

std::vector<std::string> default_member_paths() {
    std::vector<std::string> paths;
    const std::string dir = own_directory();
    // Optional siblings, in the same precedence order the Python loader prepends today.
    for (const char* sibling : {"libgdn_fused_gaudi2.so", "libflashinfer_gaudi_kernels.so"}) {
        if (dir.empty()) {
            continue;
        }
        std::filesystem::path candidate = std::filesystem::path(dir) / sibling;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            paths.push_back(candidate.string());
        }
    }
    // Required stock library (the "stock" position is always the last member).
    const char* stock = std::getenv("FLASHINFER_GAUDI_STOCK_KERNEL_DB");
    paths.push_back((stock && *stock) ? stock : "/usr/lib/habanalabs/libtpc_kernels.so");
    return paths;
}

std::vector<std::string> member_paths() {
    const char* list = std::getenv("FLASHINFER_GAUDI_COMBINED_MEMBERS");  // test-only override
    if (!list || !*list) {
        return default_member_paths();
    }
    std::vector<std::string> paths;
    std::string s(list);
    size_t start = 0;
    while (start <= s.size()) {
        size_t colon = s.find(':', start);
        std::string token = (colon == std::string::npos) ? s.substr(start) : s.substr(start, colon - start);
        if (!token.empty()) {
            paths.push_back(token);
        }
        if (colon == std::string::npos) {
            break;
        }
        start = colon + 1;
    }
    return paths;
}

template <class F>
F sym(void* handle, const char* name) {
    return reinterpret_cast<F>(dlsym(handle, name));
}

// Query one member's Gaudi2 GUIDs with the two-call protocol, first-wins into the map.
void collect_guids(size_t index) {
    Member& m = g_members[index];
    if (!m.getKernelGuids) {
        return;
    }
    uint32_t count = 0;
    if (m.getKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) != GLUE_SUCCESS || count == 0) {
        return;
    }
    std::vector<GuidInfo> guids(count);
    uint32_t capacity = count;
    if (m.getKernelGuids(DEVICE_ID_GAUDI2, &capacity, guids.data()) != GLUE_SUCCESS) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        std::string name(guids[i].name);
        if (g_owner.find(name) == g_owner.end()) {
            g_owner.emplace(name, index);  // first member wins on a duplicate name
            g_mergedGuids.push_back(guids[i]);
        }
    }
}

void load_members() {
    for (const std::string& path : member_paths()) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            const char* err = dlerror();
            g_lastError = path + ": " + (err ? err : "dlopen failed");
            continue;  // a missing sibling is skipped silently
        }
        Member m;
        m.handle = handle;
        m.path = path;
        m.getKernelGuids = sym<pfnGetKernelGuids>(handle, "GetKernelGuids");
        m.instantiate = sym<pfnInstantiateTpcKernel>(handle, "InstantiateTpcKernel");
        m.shapeInference = sym<pfnGetShapeInference>(handle, "GetShapeInference");
        m.suggestedManipulation = sym<GetSuggestedManipulationFn>(handle, "GetSuggestedManipulation");
        m.supportedDataLayouts = sym<pfnGetSupportedDataLayout>(handle, "GetSupportedDataLayouts");
        m.getLibVersion = sym<pfnGetLibVersion>(handle, "GetLibVersion");
        g_members.push_back(m);
    }
    for (size_t index = 0; index < g_members.size(); ++index) {
        collect_guids(index);
    }
}

void ensure_loaded() { std::call_once(g_once, load_members); }

// The "stock" member is the last opened member (the required stock library position).
Member* stock_member() { return g_members.empty() ? nullptr : &g_members.back(); }

// A name no member listed goes to the stock member, as the DeepSeek glue does: the stock library
// may serve names it does not enumerate, and non-Gaudi2 GUIDs are not in the map at all.
Member& route(const char* name) {
    auto it = g_owner.find(std::string(name ? name : ""));
    return it == g_owner.end() ? g_members.back() : g_members[it->second];
}

}  // namespace

extern "C" GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t* kernelCount, GuidInfo* guids) {
    ensure_loaded();
    if (deviceId != DEVICE_ID_GAUDI2) {
        Member* stock = stock_member();
        if (!stock || !stock->getKernelGuids) {
            return GLUE_FAILED;
        }
        return stock->getKernelGuids(deviceId, kernelCount, guids);
    }
    if (g_members.empty()) {
        if (kernelCount) {
            *kernelCount = 0;
        }
        return GLUE_FAILED;
    }
    const uint32_t total = static_cast<uint32_t>(g_mergedGuids.size());
    const uint32_t capacity = kernelCount ? *kernelCount : 0;
    if (!guids || capacity == 0) {  // two-call protocol: report the required size
        if (kernelCount) {
            *kernelCount = total;
        }
        return GLUE_SUCCESS;
    }
    if (capacity < total) {
        return GLUE_FAILED;
    }
    for (uint32_t i = 0; i < total; ++i) {
        guids[i] = g_mergedGuids[i];
    }
    if (kernelCount) {
        *kernelCount = total;
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    ensure_loaded();
    if (g_members.empty() || !params) {
        return GLUE_FAILED;
    }
    Member& m = route(params->guid.name);
    return m.instantiate ? m.instantiate(params, instance) : GLUE_FAILED;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* inputParams,
                                            ShapeInferenceOutput* outputData) {
    ensure_loaded();
    if (g_members.empty() || !inputParams) {
        return GLUE_FAILED;
    }
    const GuidInfo* guid = inputParams->pGuid ? inputParams->pGuid : &inputParams->guid;
    Member& m = route(guid->name);
    return m.shapeInference ? m.shapeInference(deviceId, inputParams, outputData) : GLUE_FAILED;
}

extern "C" GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, void* manipulation) {
    ensure_loaded();
    if (g_members.empty() || !params) {
        return GLUE_FAILED;
    }
    Member& m = route(params->guid.name);
    return m.suggestedManipulation ? m.suggestedManipulation(params, manipulation) : GLUE_FAILED;
}

extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params, NodeDataLayouts* supportedLayouts,
                                                  uint32_t* layoutCount) {
    ensure_loaded();
    if (g_members.empty() || !params) {
        return GLUE_FAILED;
    }
    Member& m = route(params->guid.name);
    return m.supportedDataLayouts ? m.supportedDataLayouts(params, supportedLayouts, layoutCount) : GLUE_FAILED;
}

extern "C" uint64_t GetLibVersion() {
    ensure_loaded();
    Member* stock = stock_member();
    return (stock && stock->getLibVersion) ? stock->getLibVersion() : 0;
}
