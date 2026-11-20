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
//
// Members: the custom libraries next to this file (libgdn_fused_gaudi2.so,
// libflashinfer_gaudi_kernels.so; each optional), then Intel's stock library
// (required). A member that is present but cannot be used (dlopen fails, a
// required entry point is missing, its GetKernelGuids fails) is reported on
// stderr with its path, and the database then serves nothing: GetKernelGuids
// returns GLUE_FAILED rather than a list without that member's kernels.
#include <tpc_kernel_lib_interface.h>

#include <cstdio>
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

const char* const kLogPrefix = "flashinfer_gaudi combined kernel database: ";
const char* const kDefaultStockPath = "/usr/lib/habanalabs/libtpc_kernels.so";
const char* const kMembersEnv = "FLASHINFER_GAUDI_COMBINED_MEMBERS";
const char* const kStockEnv = "FLASHINFER_GAUDI_STOCK_KERNEL_DB";
const size_t kNoMember = static_cast<size_t>(-1);

// GetSuggestedManipulation has no typedef in tpc_kernel_lib_interface.h; the
// DeepSeek glue casts the symbol to this raw signature.
using GetSuggestedManipulationFn = GlueCodeReturn (*)(const HabanaKernelParams*, void*);

// A library the adapter was told to load. `stock` marks Intel's library: it is known by this flag,
// never by its position in a list.
struct MemberSpec {
    std::string path;
    bool stock = false;
};

struct Member {
    void* handle = nullptr;
    std::string path;
    bool stock = false;
    // Required: the graph compiler refuses a kernel library without these three.
    pfnGetKernelGuids getKernelGuids = nullptr;
    pfnInstantiateTpcKernel instantiate = nullptr;
    pfnGetShapeInference shapeInference = nullptr;
    // Optional: null when the member does not export them.
    GetSuggestedManipulationFn suggestedManipulation = nullptr;
    pfnGetSupportedDataLayout supportedDataLayouts = nullptr;
    pfnGetLibVersion getLibVersion = nullptr;
};

std::once_flag g_once;
std::vector<Member> g_members;                    // loaded members in precedence order (first wins)
size_t g_stock = kNoMember;                       // index of the stock member in g_members
std::unordered_map<std::string, size_t> g_owner;  // guid name -> owning member index
std::vector<GuidInfo> g_mergedGuids;              // merged GUID list, member order
unsigned g_failures = 0;                          // members that were listed and could not be used

// Every member failure goes to stderr with the member's path: SynapseAI logs only "Failed loading
// kernels" for this file, and the Python loader cannot see what happens in here.
void report_failure(const std::string& message) {
    ++g_failures;
    std::fprintf(stderr, "%s%s\n", kLogPrefix, message.c_str());
}

std::string own_path() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&own_path), &info) || !info.dli_fname) {
        return {};
    }
    return info.dli_fname;
}

std::vector<std::string> split_paths(const std::string& list) {
    std::vector<std::string> paths;
    size_t start = 0;
    while (start <= list.size()) {
        size_t colon = list.find(':', start);
        std::string token = (colon == std::string::npos) ? list.substr(start) : list.substr(start, colon - start);
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

// The custom members first (precedence order), then the stock member. An environment override
// never changes the list silently: each one leaves a line on stderr.
std::vector<MemberSpec> member_specs() {
    std::vector<MemberSpec> specs;
    const std::string self = own_path();
    const char* list = std::getenv(kMembersEnv);  // for the host test
    if (list && *list) {
        std::fprintf(stderr, "%s%s is set: the custom members are '%s', not the libraries next to %s\n", kLogPrefix,
                     kMembersEnv, list, self.empty() ? "this file" : self.c_str());
        for (const std::string& path : split_paths(list)) {
            specs.push_back({path, false});
        }
    } else if (!self.empty()) {
        // Optional siblings, in the same precedence order the Python loader prepends today.
        const std::filesystem::path dir = std::filesystem::path(self).parent_path();
        for (const char* sibling : {"libgdn_fused_gaudi2.so", "libflashinfer_gaudi_kernels.so"}) {
            std::filesystem::path candidate = dir / sibling;
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) {
                specs.push_back({candidate.string(), false});
            }
        }
    }
    const char* stock = std::getenv(kStockEnv);
    if (stock && *stock) {
        std::fprintf(stderr, "%s%s is set: the stock member is %s, not %s\n", kLogPrefix, kStockEnv, stock,
                     kDefaultStockPath);
        specs.push_back({stock, true});
    } else {
        specs.push_back({kDefaultStockPath, true});
    }
    return specs;
}

const char* kind(bool stock) { return stock ? "the stock kernel library" : "the member"; }

template <class F>
F optional_symbol(const Member& m, const char* name) {
    F fn = reinterpret_cast<F>(dlsym(m.handle, name));
    if (!fn && m.stock) {
        // Not fatal, but Intel's library is expected to export all six entry points.
        std::fprintf(stderr, "%swarning: the stock kernel library %s does not export %s\n", kLogPrefix, m.path.c_str(),
                     name);
    }
    return fn;
}

template <class F>
F required_symbol(const Member& m, const char* name, bool* ok) {
    dlerror();  // clear a stale message
    F fn = reinterpret_cast<F>(dlsym(m.handle, name));
    if (!fn) {
        const char* err = dlerror();
        report_failure(std::string("cannot use ") + kind(m.stock) + " " + m.path + ": required entry point " + name +
                       " is missing: " + (err ? err : "dlsym returned null"));
        *ok = false;
    }
    return fn;
}

// Opens one member; false (after report_failure) when it cannot be used.
bool open_member(const MemberSpec& spec, Member* out) {
    Member m;
    m.path = spec.path;
    m.stock = spec.stock;
    // RTLD_LAZY, as the graph compiler itself opens a kernel library: a member that loads from a
    // colon-separated GC_KERNEL_PATH must load here too (RTLD_NOW would refuse a library whose
    // lazily bound symbols only resolve inside the SynapseAI process).
    m.handle = dlopen(spec.path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!m.handle) {
        const char* err = dlerror();
        report_failure(std::string("cannot load ") + kind(m.stock) + " " + m.path + ": " +
                       (err ? err : "dlopen failed"));
        return false;
    }
    // The graph compiler (KernelDB::loadFunctions) refuses a library without one of these three,
    // so a member without one is not a kernel library.
    bool ok = true;
    m.getKernelGuids = required_symbol<pfnGetKernelGuids>(m, "GetKernelGuids", &ok);
    m.instantiate = required_symbol<pfnInstantiateTpcKernel>(m, "InstantiateTpcKernel", &ok);
    m.shapeInference = required_symbol<pfnGetShapeInference>(m, "GetShapeInference", &ok);
    if (!ok) {
        dlclose(m.handle);
        return false;
    }
    m.suggestedManipulation = optional_symbol<GetSuggestedManipulationFn>(m, "GetSuggestedManipulation");
    m.supportedDataLayouts = optional_symbol<pfnGetSupportedDataLayout>(m, "GetSupportedDataLayouts");
    m.getLibVersion = optional_symbol<pfnGetLibVersion>(m, "GetLibVersion");
    *out = m;
    return true;
}

// Query one member's Gaudi2 GUIDs with the two-call protocol, first-wins into the map.
void collect_guids(size_t index) {
    const Member& m = g_members[index];
    uint32_t count = 0;
    GlueCodeReturn rc = m.getKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    std::vector<GuidInfo> guids;
    uint32_t filled = 0;
    if (rc == GLUE_SUCCESS && count != 0) {
        guids.resize(count);
        filled = count;
        rc = m.getKernelGuids(DEVICE_ID_GAUDI2, &filled, guids.data());
    }
    if (rc != GLUE_SUCCESS) {
        // snprintf, not std::to_string: libstdc++'s to_string emits a STB_GNU_UNIQUE symbol into this
        // library, and glibc never unloads a library that defines one.
        char code[16];
        std::snprintf(code, sizeof(code), "%d", static_cast<int>(rc));
        report_failure(std::string("cannot use ") + kind(m.stock) + " " + m.path +
                       ": its GetKernelGuids failed for Gaudi2 (GlueCodeReturn " + code + ")");
        return;
    }
    if (filled > count) {
        filled = count;
    }
    for (uint32_t i = 0; i < filled; ++i) {
        guids[i].name[MAX_NODE_NAME - 1] = '\0';
        std::string name(guids[i].name);
        if (g_owner.find(name) == g_owner.end()) {
            g_owner.emplace(name, index);  // first member wins on a duplicate name
            g_mergedGuids.push_back(guids[i]);
        }
    }
}

void load_members() {
    for (const MemberSpec& spec : member_specs()) {
        Member m;
        if (!open_member(spec, &m)) {
            continue;
        }
        if (m.stock) {
            g_stock = g_members.size();
        }
        g_members.push_back(m);
    }
    for (size_t index = 0; index < g_members.size(); ++index) {
        collect_guids(index);
    }
    if (g_failures != 0) {
        const std::string self = own_path();
        std::fprintf(stderr,
                     "%s%s is unusable: %u member failure(s) above, so GetKernelGuids returns GLUE_FAILED and no "
                     "kernel is served from this file. Fix the member, or set FLASHINFER_GAUDI_COMBINED_KERNEL_DB=0 "
                     "to run without the combined kernel database.\n",
                     kLogPrefix, self.empty() ? "this kernel database" : self.c_str(), g_failures);
    }
}

void ensure_loaded() { std::call_once(g_once, load_members); }

// A database with a member that failed serves nothing: without the stock member every stock kernel
// is gone, and without a custom member the Python loader would still report its GUIDs as registered.
bool usable() { return g_failures == 0 && g_stock != kNoMember; }

// The member that serves `name`, or null when the database is unusable. A name no member listed goes
// to the stock member, as the DeepSeek glue does: the stock library may serve names it does not
// enumerate, and non-Gaudi2 GUIDs are not in the map at all.
const Member* route(const char* name) {
    if (!usable()) {
        return nullptr;
    }
    auto it = g_owner.find(std::string(name ? name : ""));
    return &g_members[it == g_owner.end() ? g_stock : it->second];
}

}  // namespace

// tpc_kernel_lib_interface.h: kernelCount is [in/out], the size of `guids`; "if the pointed value is
// zero, the callee is expected to fill requested size". The graph compiler calls (&0, nullptr) and then
// (&count, buffer). The required size is always written before the capacity check (as the DeepSeek glue
// does), so a caller whose buffer is too small gets GLUE_FAILED together with the size it needs.
extern "C" GlueCodeReturn GetKernelGuids(DeviceId deviceId, uint32_t* kernelCount, GuidInfo* guids) {
    ensure_loaded();
    if (!kernelCount) {
        return GLUE_FAILED;
    }
    if (!usable()) {
        *kernelCount = 0;
        return GLUE_FAILED;
    }
    if (deviceId != DEVICE_ID_GAUDI2) {
        return g_members[g_stock].getKernelGuids(deviceId, kernelCount, guids);
    }
    const uint32_t total = static_cast<uint32_t>(g_mergedGuids.size());
    const uint32_t capacity = *kernelCount;
    *kernelCount = total;
    if (!guids || capacity == 0) {  // the size query
        return GLUE_SUCCESS;
    }
    if (capacity < total) {
        return GLUE_FAILED;
    }
    for (uint32_t i = 0; i < total; ++i) {
        guids[i] = g_mergedGuids[i];
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    ensure_loaded();
    const Member* m = params ? route(params->guid.name) : nullptr;
    return m ? m->instantiate(params, instance) : GLUE_FAILED;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId deviceId, const ShapeInferenceParams* inputParams,
                                            ShapeInferenceOutput* outputData) {
    ensure_loaded();
    if (!inputParams) {
        return GLUE_FAILED;
    }
    const GuidInfo* guid = inputParams->pGuid ? inputParams->pGuid : &inputParams->guid;
    const Member* m = route(guid->name);
    return m ? m->shapeInference(deviceId, inputParams, outputData) : GLUE_FAILED;
}

// A member without this entry point gets GLUE_FAILED, as the DeepSeek glue answers its own GUIDs: the
// graph compiler then applies no manipulation to the node (optimize_tpc_kernels.cpp logs "may not be
// supported" at debug level and goes on). The adapter cannot leave the symbol out instead, because the
// stock member needs it.
extern "C" GlueCodeReturn GetSuggestedManipulation(const HabanaKernelParams* params, void* manipulation) {
    ensure_loaded();
    const Member* m = params ? route(params->guid.name) : nullptr;
    return (m && m->suggestedManipulation) ? m->suggestedManipulation(params, manipulation) : GLUE_FAILED;
}

// A member without this entry point (libgdn_fused_gaudi2.so, libflashinfer_gaudi_kernels.so) gets one
// explicit don't-care layout. Loaded directly, the graph compiler treats the missing symbol as "every
// layout is don't-care"; but this file always exports the symbol, and a non-success return or zero
// layouts fails the supported-layouts pass (node_io_manager.cpp, "Glue Code GetSupportedLayouts
// failed"). Same answer as csrc/deepseek_v4/host/entry_points.cpp gives for its own GUIDs.
extern "C" GlueCodeReturn GetSupportedDataLayouts(const HabanaKernelParams* params, NodeDataLayouts* supportedLayouts,
                                                  uint32_t* layoutCount) {
    ensure_loaded();
    const Member* m = params ? route(params->guid.name) : nullptr;
    if (!m) {
        return GLUE_FAILED;
    }
    if (m->supportedDataLayouts) {
        return m->supportedDataLayouts(params, supportedLayouts, layoutCount);
    }
    if (!layoutCount) {
        return GLUE_FAILED;
    }
    *layoutCount = 1;
    if (supportedLayouts) {
        NodeDataLayouts* l = supportedLayouts;
        for (uint32_t i = 0; l->inputs && i < l->inputTensorNr; ++i) {
            std::memset(l->inputs[i].layout, 'x', sizeof(l->inputs[i].layout));
        }
        for (uint32_t i = 0; l->outputs && i < l->outputTensorNr; ++i) {
            std::memset(l->outputs[i].layout, 'x', sizeof(l->outputs[i].layout));
        }
        for (uint32_t i = 0; l->shapeTensors && i < l->shapeTensorNr; ++i) {
            std::memset(l->shapeTensors[i].layout, 'x', sizeof(l->shapeTensors[i].layout));
        }
    }
    return GLUE_SUCCESS;
}

// The stock member's version; 0 when the stock member did not load (never another member's value).
extern "C" uint64_t GetLibVersion() {
    ensure_loaded();
    if (g_stock == kNoMember || !g_members[g_stock].getLibVersion) {
        return 0;
    }
    return g_members[g_stock].getLibVersion();
}
