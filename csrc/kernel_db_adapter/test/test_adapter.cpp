// SPDX-License-Identifier: Apache-2.0
//
// Host test for the combined kernel database adapter (RUK-3243). It dlopens the
// built adapter with FLASHINFER_GAUDI_COMBINED_MEMBERS pointed at two fake
// members and checks GUID merging, the two-call size protocol, routing of all
// six entry points, and graceful behaviour when no member can be opened.
#include <tpc_kernel_lib_interface.h>

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>

using namespace tpc_lib_api;

using GetSuggestedManipulationFn = GlueCodeReturn (*)(const HabanaKernelParams*, void*);

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

template <class F>
F sym(void* handle, const char* name) {
    void* f = dlsym(handle, name);
    if (!f) {
        std::printf("FAIL: dlsym(%s) returned null: %s\n", name, dlerror());
        ++g_failures;
    }
    return reinterpret_cast<F>(f);
}

void set_name(GuidInfo* info, const char* name) {
    std::memset(info, 0, sizeof(*info));
    std::strncpy(info->name, name, MAX_NODE_NAME - 1);
}

// Opens a member's image so the test can read its exported fake_last_call().
const char* member_last_call(void* handle) {
    auto fn = sym<const char* (*)()>(handle, "fake_last_call");
    return fn ? fn() : "<null>";
}

void scenario_two_members() {
    std::string members = std::string(FAKE_A_PATH) + ":" + FAKE_B_PATH;
    setenv("FLASHINFER_GAUDI_COMBINED_MEMBERS", members.c_str(), 1);

    void* adapter = dlopen(ADAPTER_PATH, RTLD_NOW | RTLD_LOCAL);
    check(adapter != nullptr, std::string("dlopen adapter: ") + (adapter ? "" : dlerror()));
    if (!adapter) {
        return;
    }
    void* fake_a = dlopen(FAKE_A_PATH, RTLD_NOW | RTLD_LOCAL);
    void* fake_b = dlopen(FAKE_B_PATH, RTLD_NOW | RTLD_LOCAL);
    check(fake_a && fake_b, "dlopen both fake members");
    if (!fake_a || !fake_b) {
        dlclose(adapter);
        return;
    }

    auto getGuids = sym<pfnGetKernelGuids>(adapter, "GetKernelGuids");
    auto instantiate = sym<pfnInstantiateTpcKernel>(adapter, "InstantiateTpcKernel");
    auto shapeInference = sym<pfnGetShapeInference>(adapter, "GetShapeInference");
    auto suggested = sym<GetSuggestedManipulationFn>(adapter, "GetSuggestedManipulation");
    auto dataLayouts = sym<pfnGetSupportedDataLayout>(adapter, "GetSupportedDataLayouts");
    auto libVersion = sym<pfnGetLibVersion>(adapter, "GetLibVersion");

    // Two-call protocol: the size call reports the merged total (first wins on the shared GUID).
    uint32_t count = 0;
    GlueCodeReturn rc = getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    check(rc == GLUE_SUCCESS, "GetKernelGuids size call returns GLUE_SUCCESS");
    check(count == 3, "merged GUID count is 3 (fake_shared de-duplicated)");

    GuidInfo guids[3];
    std::memset(guids, 0, sizeof(guids));
    count = 3;
    rc = getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_SUCCESS, "GetKernelGuids fill returns GLUE_SUCCESS");
    check(std::string(guids[0].name) == "fake_a_only", "guid[0] = fake_a_only");
    check(std::string(guids[1].name) == "fake_shared", "guid[1] = fake_shared");
    check(std::string(guids[2].name) == "fake_b_only", "guid[2] = fake_b_only (member order)");

    GuidInfo small[2];
    count = 2;
    rc = getGuids(DEVICE_ID_GAUDI2, &count, small);
    check(rc == GLUE_FAILED, "GetKernelGuids with capacity 2 returns GLUE_FAILED");

    // InstantiateTpcKernel routes by guid.name; first member wins on fake_shared.
    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    set_name(&params.guid, "fake_b_only");
    rc = instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 2, "Instantiate fake_b_only -> member b (paramsNr 2)");
    check(std::strcmp(member_last_call(fake_b), "InstantiateTpcKernel") == 0, "member b recorded InstantiateTpcKernel");

    set_name(&params.guid, "fake_shared");
    inst.kernel.paramsNr = 0;
    rc = instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 1, "Instantiate fake_shared -> member a (first wins)");

    set_name(&params.guid, "totally_unknown_guid");
    rc = instantiate(&params, &inst);
    check(rc == GLUE_NODE_NOT_FOUND, "Instantiate unknown GUID -> GLUE_NODE_NOT_FOUND");

    // GetLibVersion comes from the stock (last) member: member b.
    check(libVersion() == 1002ULL, "GetLibVersion is the last member's (stock) value 1002");

    // GetShapeInference routes through pGuid when set, then through guid otherwise.
    ShapeInferenceParams sip{};
    ShapeInferenceOutput out{};
    GuidInfo viaGuid{};
    set_name(&viaGuid, "fake_b_only");
    sip.pGuid = &viaGuid;
    set_name(&sip.guid, "fake_a_only");  // pGuid must win over guid
    const char* a_before = member_last_call(fake_a);
    rc = shapeInference(DEVICE_ID_GAUDI2, &sip, &out);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_b), "GetShapeInference") == 0,
          "GetShapeInference routes through pGuid (-> member b)");
    check(std::strcmp(member_last_call(fake_a), a_before) == 0, "member a untouched by the pGuid routing");

    ShapeInferenceParams sip2{};
    sip2.pGuid = nullptr;
    set_name(&sip2.guid, "fake_a_only");
    rc = shapeInference(DEVICE_ID_GAUDI2, &sip2, &out);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_a), "GetShapeInference") == 0,
          "GetShapeInference routes through guid when pGuid is null (-> member a)");

    // GetSuggestedManipulation and GetSupportedDataLayouts route by guid.name.
    set_name(&params.guid, "fake_b_only");
    rc = suggested(&params, nullptr);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_b), "GetSuggestedManipulation") == 0,
          "GetSuggestedManipulation routes -> member b");

    set_name(&params.guid, "fake_a_only");
    NodeDataLayouts layouts{};
    uint32_t layoutCount = 0;
    rc = dataLayouts(&params, &layouts, &layoutCount);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_a), "GetSupportedDataLayouts") == 0,
          "GetSupportedDataLayouts routes -> member a");

    dlclose(fake_a);
    dlclose(fake_b);
    dlclose(adapter);  // unloads the adapter so its statics reset for the next scenario
}

void scenario_no_member() {
    setenv("FLASHINFER_GAUDI_COMBINED_MEMBERS", "/nonexistent.so", 1);
    void* adapter = dlopen(ADAPTER_PATH, RTLD_NOW | RTLD_LOCAL);
    check(adapter != nullptr, std::string("dlopen adapter (no member): ") + (adapter ? "" : dlerror()));
    if (!adapter) {
        return;
    }
    auto getGuids = sym<pfnGetKernelGuids>(adapter, "GetKernelGuids");
    auto instantiate = sym<pfnInstantiateTpcKernel>(adapter, "InstantiateTpcKernel");

    uint32_t count = 0;
    getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    check(count == 0, "no openable member: GetKernelGuids size call reports 0");

    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    set_name(&params.guid, "fake_a_only");
    check(instantiate(&params, &inst) == GLUE_FAILED, "no openable member: InstantiateTpcKernel -> GLUE_FAILED");

    dlclose(adapter);
}

}  // namespace

int main() {
    scenario_two_members();
    scenario_no_member();
    if (g_failures == 0) {
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
