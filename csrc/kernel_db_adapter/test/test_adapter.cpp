// SPDX-License-Identifier: Apache-2.0
//
// Host test for the combined kernel database adapter (RUK-3243). It dlopens the
// built adapter against fake member libraries and checks GUID merging, the
// GetKernelGuids count protocol, routing of all six entry points, the answers
// the adapter gives for a member that exports only the three required entry
// points, and that a member that cannot be loaded is reported on stderr and
// makes GetKernelGuids fail instead of producing a database without it.
//
// The custom members come from FLASHINFER_GAUDI_COMBINED_MEMBERS (or, in
// scenario_installed_layout and scenario_installed_layout_with_hc, from the
// files next to a copy of the adapter, as in production); the stock member is
// always FLASHINFER_GAUDI_STOCK_KERNEL_DB.
#include <tpc_kernel_lib_interface.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace tpc_lib_api;

using GetSuggestedManipulationFn = GlueCodeReturn (*)(const HabanaKernelParams*, void*);

namespace {

const char* const kMembersEnv = "FLASHINFER_GAUDI_COMBINED_MEMBERS";
const char* const kStockEnv = "FLASHINFER_GAUDI_STOCK_KERNEL_DB";

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

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

// Sends file descriptor 2 to a file until finish(), which returns what was written there
// (and repeats it on stdout so a failing run shows it).
class StderrCapture {
public:
    StderrCapture() {
        std::fflush(stderr);
        char path[] = "/tmp/kda_stderr_XXXXXX";
        int fd = mkstemp(path);
        path_ = path;
        saved_ = dup(STDERR_FILENO);
        dup2(fd, STDERR_FILENO);
        close(fd);
    }
    std::string finish() {
        std::fflush(stderr);
        dup2(saved_, STDERR_FILENO);
        close(saved_);
        std::ifstream in(path_);
        std::stringstream text;
        text << in.rdbuf();
        std::remove(path_.c_str());
        if (!text.str().empty()) {
            std::printf("      stderr: %s", text.str().c_str());
        }
        return text.str();
    }

private:
    std::string path_;
    int saved_ = -1;
};

struct Adapter {
    void* handle = nullptr;
    pfnGetKernelGuids getGuids = nullptr;
    pfnInstantiateTpcKernel instantiate = nullptr;
    pfnGetShapeInference shapeInference = nullptr;
    GetSuggestedManipulationFn suggested = nullptr;
    pfnGetSupportedDataLayout dataLayouts = nullptr;
    pfnGetLibVersion libVersion = nullptr;
};

void set_env(const char* name, const std::string& value) {
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
}

// Loads a fresh image of the adapter at `path`. `members` is the value of the members override (empty:
// unset, the adapter uses the libraries next to itself) and `stock` the stock member's path.
Adapter open_adapter(const std::string& path, const std::string& members, const std::string& stock,
                     const std::string& label) {
    set_env(kMembersEnv, members);
    set_env(kStockEnv, stock);
    Adapter a;
    a.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(a.handle != nullptr, "dlopen adapter (" + label + "): " + (a.handle ? "" : dlerror()));
    if (!a.handle) {
        return a;
    }
    a.getGuids = sym<pfnGetKernelGuids>(a.handle, "GetKernelGuids");
    a.instantiate = sym<pfnInstantiateTpcKernel>(a.handle, "InstantiateTpcKernel");
    a.shapeInference = sym<pfnGetShapeInference>(a.handle, "GetShapeInference");
    a.suggested = sym<GetSuggestedManipulationFn>(a.handle, "GetSuggestedManipulation");
    a.dataLayouts = sym<pfnGetSupportedDataLayout>(a.handle, "GetSupportedDataLayouts");
    a.libVersion = sym<pfnGetLibVersion>(a.handle, "GetLibVersion");
    return a;
}

// Unloads the adapter so its statics reset for the next scenario, and checks that it really left the
// process (a library that defines a STB_GNU_UNIQUE symbol, for example, is never unloaded by glibc, and
// the next scenario would then silently reuse this one's members).
void close_adapter(Adapter& a) {
    if (!a.handle) {
        return;
    }
    Dl_info info{};
    const bool located = dladdr(reinterpret_cast<void*>(a.getGuids), &info) && info.dli_fname;
    const std::string path = located ? info.dli_fname : "";
    dlclose(a.handle);
    a.handle = nullptr;
    void* still = path.empty() ? nullptr : dlopen(path.c_str(), RTLD_NOLOAD | RTLD_NOW);
    if (still) {
        dlclose(still);
    }
    if (still || path.empty()) {
        check(false, "the adapter unloads on dlclose (" + path + ")");
    }
}

// The production layout in a temporary folder: a copy of the adapter with libgdn_fused_gaudi2.so (the
// full fake member a) and libflashinfer_gaudi_kernels.so (the minimal fake member) next to it. With the
// members override unset the adapter finds these siblings itself, as it does when installed.
// `hcMember` adds the optional third sibling, libhc_fused_gaudi2.so, as a copy of that file (empty: the
// layout of an install whose hyper-connection build was off or failed, which has no such file).
struct InstalledLayout {
    std::filesystem::path dir;
    std::string adapter;
    bool ok = false;
};

InstalledLayout make_installed_layout(const std::string& hcMember = "") {
    namespace fs = std::filesystem;
    InstalledLayout layout;
    char tmpl[] = "/tmp/kda_layout_XXXXXX";
    const char* made = mkdtemp(tmpl);
    if (made) {
        layout.dir = made;
        layout.adapter = (layout.dir / "libflashinfer_gaudi_combined_gaudi2.so").string();
        std::error_code ec;
        layout.ok = fs::copy_file(ADAPTER_PATH, layout.adapter, ec) &&
                    fs::copy_file(FAKE_A_PATH, layout.dir / "libgdn_fused_gaudi2.so", ec) &&
                    fs::copy_file(FAKE_MIN_PATH, layout.dir / "libflashinfer_gaudi_kernels.so", ec) &&
                    (hcMember.empty() || fs::copy_file(hcMember, layout.dir / "libhc_fused_gaudi2.so", ec));
    }
    check(layout.ok, std::string("installed layout: the adapter and ") + (hcMember.empty() ? "two" : "three") +
                         " members copied into one folder");
    return layout;
}

void remove_layout(const InstalledLayout& layout) {
    if (!layout.dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(layout.dir, ec);
    }
}

void scenario_two_members() {
    std::printf("-- two members: custom member a, stock member b\n");
    Adapter adapter = open_adapter(ADAPTER_PATH, FAKE_A_PATH, FAKE_B_PATH, "two members");
    if (!adapter.handle) {
        return;
    }
    void* fake_a = dlopen(FAKE_A_PATH, RTLD_NOW | RTLD_LOCAL);
    void* fake_b = dlopen(FAKE_B_PATH, RTLD_NOW | RTLD_LOCAL);
    check(fake_a && fake_b, "dlopen both fake members");
    if (!fake_a || !fake_b) {
        close_adapter(adapter);
        return;
    }

    // Two-call protocol: the size call reports the merged total (first wins on the shared GUID).
    StderrCapture capture;
    uint32_t count = 0;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    const std::string err = capture.finish();
    check(rc == GLUE_SUCCESS, "GetKernelGuids size call returns GLUE_SUCCESS");
    check(count == 3, "merged GUID count is 3 (fake_shared de-duplicated)");
    // The members override is for tests: using it must leave a line on stderr.
    check(contains(err, kMembersEnv) && contains(err, FAKE_A_PATH),
          "stderr says FLASHINFER_GAUDI_COMBINED_MEMBERS replaced the custom members, with the path");
    check(contains(err, kStockEnv) && contains(err, FAKE_B_PATH),
          "stderr says FLASHINFER_GAUDI_STOCK_KERNEL_DB replaced the stock member, with the path");

    GuidInfo guids[3];
    std::memset(guids, 0, sizeof(guids));
    count = 3;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_SUCCESS, "GetKernelGuids fill returns GLUE_SUCCESS");
    check(std::string(guids[0].name) == "fake_a_only", "guid[0] = fake_a_only");
    check(std::string(guids[1].name) == "fake_shared", "guid[1] = fake_shared");
    check(std::string(guids[2].name) == "fake_b_only", "guid[2] = fake_b_only (member order)");

    // InstantiateTpcKernel routes by guid.name; first member wins on fake_shared.
    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    set_name(&params.guid, "fake_b_only");
    rc = adapter.instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 2, "Instantiate fake_b_only -> member b (paramsNr 2)");
    check(std::strcmp(member_last_call(fake_b), "InstantiateTpcKernel") == 0, "member b recorded InstantiateTpcKernel");

    set_name(&params.guid, "fake_shared");
    inst.kernel.paramsNr = 0;
    rc = adapter.instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 1, "Instantiate fake_shared -> member a (first wins)");

    set_name(&params.guid, "totally_unknown_guid");
    inst.kernel.paramsNr = 0;
    rc = adapter.instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 2, "Instantiate unknown GUID -> stock member b");

    // GetLibVersion comes from the stock member: member b.
    check(adapter.libVersion() == 1002ULL, "GetLibVersion is the stock member's value 1002");

    // GetShapeInference routes through pGuid when set, then through guid otherwise.
    ShapeInferenceParams sip{};
    ShapeInferenceOutput out{};
    GuidInfo viaGuid{};
    set_name(&viaGuid, "fake_b_only");
    sip.pGuid = &viaGuid;
    set_name(&sip.guid, "fake_a_only");  // pGuid must win over guid
    const char* a_before = member_last_call(fake_a);
    rc = adapter.shapeInference(DEVICE_ID_GAUDI2, &sip, &out);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_b), "GetShapeInference") == 0,
          "GetShapeInference routes through pGuid (-> member b)");
    check(std::strcmp(member_last_call(fake_a), a_before) == 0, "member a untouched by the pGuid routing");

    ShapeInferenceParams sip2{};
    sip2.pGuid = nullptr;
    set_name(&sip2.guid, "fake_a_only");
    rc = adapter.shapeInference(DEVICE_ID_GAUDI2, &sip2, &out);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_a), "GetShapeInference") == 0,
          "GetShapeInference routes through guid when pGuid is null (-> member a)");

    // GetSuggestedManipulation and GetSupportedDataLayouts route by guid.name.
    set_name(&params.guid, "fake_b_only");
    rc = adapter.suggested(&params, nullptr);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_b), "GetSuggestedManipulation") == 0,
          "GetSuggestedManipulation routes -> member b");

    set_name(&params.guid, "fake_a_only");
    NodeDataLayouts layouts{};
    uint32_t layoutCount = 0;
    rc = adapter.dataLayouts(&params, &layouts, &layoutCount);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_a), "GetSupportedDataLayouts") == 0,
          "GetSupportedDataLayouts routes -> member a (a member that exports it answers itself)");

    dlclose(fake_a);
    dlclose(fake_b);
    close_adapter(adapter);
}

// tpc_kernel_lib_interface.h: kernelCount is [in/out], the size of the guids array, and "if the pointed
// value is zero, the callee is expected to fill requested size". The graph compiler (KernelDB::loadKernelGuids)
// calls with (&0, nullptr), then with (&count, buffer). The required size is written before the capacity
// check, as the DeepSeek glue does, so a caller with a small buffer learns the size it needs.
void scenario_count_protocol(const InstalledLayout& layout) {
    std::printf("-- GetKernelGuids count protocol (installed layout, 4 GUIDs)\n");
    Adapter adapter = open_adapter(layout.adapter, "", FAKE_B_PATH, "count protocol");
    if (!adapter.handle) {
        return;
    }
    StderrCapture capture;
    uint32_t total = 0;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &total, nullptr);
    capture.finish();
    check(rc == GLUE_SUCCESS && total == 4, "(&0, nullptr) -> GLUE_SUCCESS and the required count 4");

    GuidInfo untouched[5];
    std::memset(untouched, 0x5a, sizeof(untouched));
    GuidInfo guids[5];

    // A buffer with a zero count is the size query too: the count is filled, the buffer is not.
    std::memset(guids, 0x5a, sizeof(guids));
    uint32_t count = 0;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_SUCCESS && count == 4, "(&0, buffer) -> GLUE_SUCCESS and the required count 4");
    check(std::memcmp(guids, untouched, sizeof(untouched)) == 0, "(&0, buffer) leaves the buffer untouched");

    // No buffer, a non-zero count: still the size query.
    count = 99;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    check(rc == GLUE_SUCCESS && count == 4, "(&99, nullptr) -> GLUE_SUCCESS and the required count 4");

    // A buffer that is too small: GLUE_FAILED, the required count is written, nothing is filled.
    std::memset(guids, 0x5a, sizeof(guids));
    count = 3;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_FAILED, "capacity 3 of 4 -> GLUE_FAILED");
    check(count == 4, "capacity 3 of 4 writes the required count 4");
    check(std::memcmp(guids, untouched, sizeof(untouched)) == 0, "capacity 3 of 4 leaves the buffer untouched");

    // The exact size, the graph compiler's second call.
    std::memset(guids, 0x5a, sizeof(guids));
    count = 4;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_SUCCESS && count == 4, "capacity 4 of 4 -> GLUE_SUCCESS and the count 4");
    check(std::string(guids[3].name) == "fake_b_only" && std::memcmp(&guids[4], untouched, sizeof(GuidInfo)) == 0,
          "capacity 4 of 4 fills exactly 4 entries");

    // A larger buffer: filled with the total, the count comes back as the total.
    std::memset(guids, 0x5a, sizeof(guids));
    count = 5;
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
    check(rc == GLUE_SUCCESS && count == 4, "capacity 5 of 4 -> GLUE_SUCCESS and the count 4");
    check(std::string(guids[3].name) == "fake_b_only" && std::memcmp(&guids[4], untouched, sizeof(GuidInfo)) == 0,
          "capacity 5 of 4 fills exactly 4 entries");

    // No count pointer: there is no size to honour, the call fails and fills nothing.
    std::memset(guids, 0x5a, sizeof(guids));
    rc = adapter.getGuids(DEVICE_ID_GAUDI2, nullptr, guids);
    check(rc == GLUE_FAILED, "a null count pointer -> GLUE_FAILED");
    check(std::memcmp(guids, untouched, sizeof(untouched)) == 0, "a null count pointer leaves the buffer untouched");

    close_adapter(adapter);
}

// libgdn_fused_gaudi2.so and libflashinfer_gaudi_kernels.so export no GetSupportedDataLayouts. Loaded
// directly, the graph compiler treats the missing symbol as "every layout is don't-care"; an exported
// function that returns non-success (or zero layouts) fails its supported-layouts pass. The adapter always
// exports the symbol, so for such a member it must answer one all-'x' layout itself.
void scenario_minimal_member() {
    std::printf("-- a member with only the three required entry points\n");
    Adapter adapter = open_adapter(ADAPTER_PATH, FAKE_MIN_PATH, FAKE_B_PATH, "minimal member");
    if (!adapter.handle) {
        return;
    }
    void* fake_min = dlopen(FAKE_MIN_PATH, RTLD_NOW | RTLD_LOCAL);
    check(fake_min != nullptr, "dlopen the minimal fake member");
    if (!fake_min) {
        close_adapter(adapter);
        return;
    }
    check(dlsym(fake_min, "GetKernelGuids") && dlsym(fake_min, "InstantiateTpcKernel") &&
              dlsym(fake_min, "GetShapeInference"),
          "the minimal member exports the three required entry points");
    check(!dlsym(fake_min, "GetSupportedDataLayouts") && !dlsym(fake_min, "GetSuggestedManipulation") &&
              !dlsym(fake_min, "GetLibVersion"),
          "the minimal member exports none of the three optional entry points");

    StderrCapture capture;
    uint32_t count = 0;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    capture.finish();
    check(rc == GLUE_SUCCESS && count == 3, "minimal member + stock: 3 merged GUIDs");

    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    set_name(&params.guid, "fake_m_only");
    rc = adapter.instantiate(&params, &inst);
    check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 3, "Instantiate fake_m_only -> the minimal member");

    // The graph compiler's first call: no buffer, it asks for the number of layouts.
    uint32_t layoutCount = 0;
    rc = adapter.dataLayouts(&params, nullptr, &layoutCount);
    check(rc == GLUE_SUCCESS, "GetSupportedDataLayouts (count query) for the minimal member's GUID -> GLUE_SUCCESS");
    check(layoutCount == 1, "GetSupportedDataLayouts (count query) reports 1 layout");

    // Its second call: a buffer for the inputs, the outputs and the shape tensors.
    TensorDataLayout inputs[3];
    TensorDataLayout outputs[2];
    TensorDataLayout shapes[2];
    std::memset(inputs, 0, sizeof(inputs));
    std::memset(outputs, 0, sizeof(outputs));
    std::memset(shapes, 0, sizeof(shapes));
    NodeDataLayouts layouts{};
    layouts.inputs = inputs;
    layouts.inputTensorNr = 2;  // the third input entry is outside the node and must stay untouched
    layouts.outputs = outputs;
    layouts.outputTensorNr = 1;
    layouts.shapeTensors = shapes;
    layouts.shapeTensorNr = 1;
    layoutCount = 1;
    rc = adapter.dataLayouts(&params, &layouts, &layoutCount);
    check(rc == GLUE_SUCCESS && layoutCount == 1, "GetSupportedDataLayouts (fill) -> GLUE_SUCCESS with 1 layout");
    char all_x[sizeof(inputs[0].layout)];
    char all_zero[sizeof(inputs[0].layout)];
    std::memset(all_x, 'x', sizeof(all_x));
    std::memset(all_zero, 0, sizeof(all_zero));
    check(std::memcmp(inputs[0].layout, all_x, sizeof(all_x)) == 0 &&
              std::memcmp(inputs[1].layout, all_x, sizeof(all_x)) == 0,
          "every input layout byte is 'x' (don't-care)");
    check(std::memcmp(outputs[0].layout, all_x, sizeof(all_x)) == 0, "every output layout byte is 'x'");
    check(std::memcmp(shapes[0].layout, all_x, sizeof(all_x)) == 0, "every shape-tensor layout byte is 'x'");
    check(std::memcmp(inputs[2].layout, all_zero, sizeof(all_zero)) == 0 &&
              std::memcmp(outputs[1].layout, all_zero, sizeof(all_zero)) == 0 &&
              std::memcmp(shapes[1].layout, all_zero, sizeof(all_zero)) == 0,
          "layouts beyond the node's tensor counts are untouched");
    check(std::strcmp(member_last_call(fake_min), "InstantiateTpcKernel") == 0,
          "the minimal member was not called for the layouts");

    // No count pointer: nothing can be reported.
    rc = adapter.dataLayouts(&params, &layouts, nullptr);
    check(rc == GLUE_FAILED, "GetSupportedDataLayouts with a null count pointer -> GLUE_FAILED");

    // GetSuggestedManipulation stays a non-success for a member without it: the graph compiler then
    // applies no manipulation (the DeepSeek glue answers its own GUIDs the same way).
    rc = adapter.suggested(&params, nullptr);
    check(rc == GLUE_FAILED, "GetSuggestedManipulation for the minimal member's GUID -> GLUE_FAILED");

    // GetShapeInference is one of the three required entry points: the member answers it.
    ShapeInferenceParams sip{};
    ShapeInferenceOutput out{};
    set_name(&sip.guid, "fake_m_only");
    rc = adapter.shapeInference(DEVICE_ID_GAUDI2, &sip, &out);
    check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_min), "GetShapeInference") == 0,
          "GetShapeInference routes -> the minimal member");

    // The stock member still answers its own GUIDs and the library version.
    set_name(&params.guid, "fake_b_only");
    layoutCount = 0;
    rc = adapter.dataLayouts(&params, nullptr, &layoutCount);
    check(rc == GLUE_SUCCESS && layoutCount == 1, "GetSupportedDataLayouts for a stock GUID -> the stock member");
    check(adapter.libVersion() == 1002ULL, "GetLibVersion is the stock member's value 1002");

    dlclose(fake_min);
    close_adapter(adapter);
}

// The production layout: no members override, the adapter finds its siblings next to itself.
// The stock member exists in the first run and does not in the second.
void scenario_installed_layout(const InstalledLayout& layout) {
    std::printf("-- installed layout: siblings next to the adapter\n");
    const std::string& adapterPath = layout.adapter;

    {
        Adapter adapter = open_adapter(adapterPath, "", FAKE_B_PATH, "installed layout");
        if (adapter.handle) {
            StderrCapture capture;
            uint32_t count = 0;
            GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
            const std::string err = capture.finish();
            check(rc == GLUE_SUCCESS && count == 4,
                  "siblings + stock: 4 merged GUIDs (a_only, shared, m_only, b_only)");
            check(!contains(err, kMembersEnv), "no members-override line when the override is unset");
            // libhc_fused_gaudi2.so is an optional sibling and this folder has none: not a failure, no line.
            check(!contains(err, "libhc_fused_gaudi2.so"),
                  "an absent optional sibling (libhc_fused_gaudi2.so) is not reported and fails nothing");
            GuidInfo guids[4];
            std::memset(guids, 0, sizeof(guids));
            count = 4;
            rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
            check(rc == GLUE_SUCCESS && std::string(guids[0].name) == "fake_a_only" &&
                      std::string(guids[1].name) == "fake_shared" && std::string(guids[2].name) == "fake_m_only" &&
                      std::string(guids[3].name) == "fake_b_only",
                  "order: libgdn_fused_gaudi2.so, libflashinfer_gaudi_kernels.so, then the stock member");
            HabanaKernelParams params{};
            HabanaKernelInstantiation inst{};
            set_name(&params.guid, "totally_unknown_guid");
            rc = adapter.instantiate(&params, &inst);
            check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 2, "unknown GUID -> the stock member");
            check(adapter.libVersion() == 1002ULL, "GetLibVersion is the stock member's value 1002");
            // libflashinfer_gaudi_kernels.so here is the minimal member: no GetSupportedDataLayouts.
            set_name(&params.guid, "fake_m_only");
            uint32_t layoutCount = 0;
            rc = adapter.dataLayouts(&params, nullptr, &layoutCount);
            check(rc == GLUE_SUCCESS && layoutCount == 1,
                  "GetSupportedDataLayouts for a GUID of libflashinfer_gaudi_kernels.so -> GLUE_SUCCESS, 1 layout");
            close_adapter(adapter);
        }
    }

    // Intel's library cannot be opened. The last member that did open (libflashinfer_gaudi_kernels.so)
    // must not be taken for the stock member: the database has no stock kernels, so it reports none.
    {
        const std::string missingStock = (layout.dir / "no_such_libtpc_kernels.so").string();
        Adapter adapter = open_adapter(adapterPath, "", missingStock, "installed layout, stock missing");
        if (adapter.handle) {
            StderrCapture capture;
            uint32_t count = 77;
            GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
            GuidInfo guids[4];
            uint32_t capacity = 4;
            GlueCodeReturn rcFill = adapter.getGuids(DEVICE_ID_GAUDI2, &capacity, guids);
            HabanaKernelParams params{};
            HabanaKernelInstantiation inst{};
            set_name(&params.guid, "totally_unknown_guid");
            GlueCodeReturn rcUnknown = adapter.instantiate(&params, &inst);
            uint32_t otherCount = 77;
            GlueCodeReturn rcOther = adapter.getGuids(DEVICE_ID_GAUDI, &otherCount, nullptr);
            const uint64_t version = adapter.libVersion();
            const std::string err = capture.finish();
            check(rc == GLUE_FAILED, "stock member missing: GetKernelGuids (size call) -> GLUE_FAILED");
            check(count == 0, "stock member missing: the count is 0");
            check(rcFill == GLUE_FAILED && capacity == 0, "stock member missing: GetKernelGuids (fill) -> GLUE_FAILED");
            check(rcOther == GLUE_FAILED, "stock member missing: GetKernelGuids for another device -> GLUE_FAILED");
            check(contains(err, missingStock), "stock member missing: stderr names the stock library's path");
            check(contains(err, "cannot open shared object file"), "stock member missing: stderr has the dlerror text");
            check(rcUnknown == GLUE_FAILED, "stock member missing: an unknown GUID is not routed to another member");
            check(version == 0, "stock member missing: GetLibVersion is 0, not another member's version");
            close_adapter(adapter);
        }
    }
}

// The production layout with the optional hyper-connection library: libhc_fused_gaudi2.so (the full fake
// member h) sits between libgdn_fused_gaudi2.so and libflashinfer_gaudi_kernels.so in the merged list, and
// its GUID is routed to it. Like the real library, member h exports GetSupportedDataLayouts itself.
void scenario_installed_layout_with_hc() {
    std::printf("-- installed layout with libhc_fused_gaudi2.so next to the adapter\n");
    const InstalledLayout layout = make_installed_layout(FAKE_H_PATH);
    if (layout.ok) {
        Adapter adapter = open_adapter(layout.adapter, "", FAKE_B_PATH, "installed layout with hc");
        const std::string hcPath = (layout.dir / "libhc_fused_gaudi2.so").string();
        void* fake_h = dlopen(hcPath.c_str(), RTLD_NOW | RTLD_LOCAL);
        check(fake_h != nullptr, "dlopen the hyper-connection member of the layout");
        if (adapter.handle && fake_h) {
            StderrCapture capture;
            uint32_t count = 0;
            GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
            capture.finish();
            check(rc == GLUE_SUCCESS && count == 5,
                  "three siblings + stock: 5 merged GUIDs (a_only, shared, h_only, m_only, b_only)");
            GuidInfo guids[5];
            std::memset(guids, 0, sizeof(guids));
            count = 5;
            rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, guids);
            check(rc == GLUE_SUCCESS && std::string(guids[0].name) == "fake_a_only" &&
                      std::string(guids[1].name) == "fake_shared" && std::string(guids[2].name) == "fake_h_only" &&
                      std::string(guids[3].name) == "fake_m_only" && std::string(guids[4].name) == "fake_b_only",
                  "order: libgdn_fused_gaudi2.so, libhc_fused_gaudi2.so, libflashinfer_gaudi_kernels.so, then "
                  "the stock member");
            HabanaKernelParams params{};
            HabanaKernelInstantiation inst{};
            set_name(&params.guid, "fake_h_only");
            rc = adapter.instantiate(&params, &inst);
            check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 5,
                  "Instantiate fake_h_only -> libhc_fused_gaudi2.so (paramsNr 5)");
            uint32_t layoutCount = 0;
            rc = adapter.dataLayouts(&params, nullptr, &layoutCount);
            check(rc == GLUE_SUCCESS && layoutCount == 1 &&
                      std::strcmp(member_last_call(fake_h), "GetSupportedDataLayouts") == 0,
                  "GetSupportedDataLayouts for a GUID of libhc_fused_gaudi2.so -> the member answers itself");
            ShapeInferenceParams sip{};
            ShapeInferenceOutput out{};
            set_name(&sip.guid, "fake_h_only");
            rc = adapter.shapeInference(DEVICE_ID_GAUDI2, &sip, &out);
            check(rc == GLUE_SUCCESS && std::strcmp(member_last_call(fake_h), "GetShapeInference") == 0,
                  "GetShapeInference for a GUID of libhc_fused_gaudi2.so -> the member");
            // The other members still answer their own GUIDs.
            set_name(&params.guid, "fake_shared");
            inst.kernel.paramsNr = 0;
            rc = adapter.instantiate(&params, &inst);
            check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 1,
                  "fake_shared still goes to libgdn_fused_gaudi2.so (the first member wins)");
            set_name(&params.guid, "fake_m_only");
            inst.kernel.paramsNr = 0;
            rc = adapter.instantiate(&params, &inst);
            check(rc == GLUE_SUCCESS && inst.kernel.paramsNr == 3, "fake_m_only -> libflashinfer_gaudi_kernels.so");
        }
        if (fake_h) {
            dlclose(fake_h);
        }
        close_adapter(adapter);
    }
    remove_layout(layout);
}

// libhc_fused_gaudi2.so is present next to the adapter but is not a loadable kernel library (here: a text
// file). The existing rule applies to it as to every member that is there: the database serves nothing
// and says why, with the file's path. Only an absent optional sibling is not a failure.
void scenario_installed_layout_with_an_unusable_hc() {
    std::printf("-- installed layout with an unusable libhc_fused_gaudi2.so\n");
    const InstalledLayout layout = make_installed_layout();
    if (layout.ok) {
        const std::string hcPath = (layout.dir / "libhc_fused_gaudi2.so").string();
        {
            std::ofstream broken(hcPath);
            broken << "not a shared object\n";
        }
        Adapter adapter = open_adapter(layout.adapter, "", FAKE_B_PATH, "installed layout, unusable hc");
        if (adapter.handle) {
            StderrCapture capture;
            uint32_t count = 77;
            GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
            HabanaKernelParams params{};
            HabanaKernelInstantiation inst{};
            set_name(&params.guid, "fake_a_only");
            GlueCodeReturn rcOther = adapter.instantiate(&params, &inst);
            const std::string err = capture.finish();
            check(rc == GLUE_FAILED && count == 0,
                  "unusable libhc_fused_gaudi2.so: GetKernelGuids -> GLUE_FAILED, count 0");
            check(contains(err, hcPath), "unusable libhc_fused_gaudi2.so: stderr names its path");
            check(rcOther == GLUE_FAILED,
                  "unusable libhc_fused_gaudi2.so: no kernel is served, not even another member's");
            close_adapter(adapter);
        }
    }
    remove_layout(layout);
}

// A custom member that is listed but cannot be loaded: the database would silently lack its GUIDs
// while the Python loader reports them as registered, so it reports none and says why.
void scenario_broken_custom_member() {
    std::printf("-- a listed custom member cannot be loaded\n");
    const std::string missing = "/nonexistent/libgdn_fused_gaudi2.so";
    Adapter adapter = open_adapter(ADAPTER_PATH, missing + ":" + FAKE_A_PATH, FAKE_B_PATH, "broken custom member");
    if (!adapter.handle) {
        return;
    }
    StderrCapture capture;
    uint32_t count = 77;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    const std::string err = capture.finish();
    check(rc == GLUE_FAILED && count == 0, "custom member missing: GetKernelGuids -> GLUE_FAILED, count 0");
    check(contains(err, missing) && contains(err, "cannot open shared object file"),
          "custom member missing: stderr names its path and the dlerror text");
    close_adapter(adapter);
}

// A member without GetShapeInference is not a kernel library: the graph compiler itself refuses a
// library without that entry point (KernelDB::loadFunctions), so the adapter refuses the member.
void scenario_member_without_a_required_entry_point() {
    std::printf("-- a member lacks a required entry point\n");
    Adapter adapter = open_adapter(ADAPTER_PATH, FAKE_NOSIF_PATH, FAKE_B_PATH, "member without GetShapeInference");
    if (!adapter.handle) {
        return;
    }
    StderrCapture capture;
    uint32_t count = 77;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    const std::string err = capture.finish();
    check(rc == GLUE_FAILED && count == 0, "member without GetShapeInference: GetKernelGuids -> GLUE_FAILED, count 0");
    check(contains(err, FAKE_NOSIF_PATH) && contains(err, "GetShapeInference"),
          "member without GetShapeInference: stderr names its path and the missing entry point");
    close_adapter(adapter);
}

void scenario_no_member() {
    std::printf("-- no member can be loaded\n");
    Adapter adapter = open_adapter(ADAPTER_PATH, "/nonexistent.so", "/nonexistent_stock.so", "no member");
    if (!adapter.handle) {
        return;
    }
    StderrCapture capture;
    uint32_t count = 0;
    adapter.getGuids(DEVICE_ID_GAUDI2, &count, nullptr);
    HabanaKernelParams params{};
    HabanaKernelInstantiation inst{};
    set_name(&params.guid, "fake_a_only");
    GlueCodeReturn rc = adapter.instantiate(&params, &inst);
    const std::string err = capture.finish();
    check(count == 0, "no openable member: GetKernelGuids size call reports 0");
    check(rc == GLUE_FAILED, "no openable member: InstantiateTpcKernel -> GLUE_FAILED");
    check(contains(err, "/nonexistent.so") && contains(err, "/nonexistent_stock.so"),
          "no openable member: stderr names both paths");
    close_adapter(adapter);
}

// test_adapter --installed <folder> [<stock library>]
//
// Checks a real build instead of the fakes: <folder>/libflashinfer_gaudi_combined_gaudi2.so with the member
// libraries next to it (libgdn_fused_gaudi2.so, libhc_fused_gaudi2.so, libflashinfer_gaudi_kernels.so; each
// optional) and the given stock library
// (default: Intel's /usr/lib/habanalabs/libtpc_kernels.so). Every GUID that is not the stock library's own
// must be in the merged list and must get a supported-layouts answer the graph compiler accepts.
int check_installed(const std::string& folder, const std::string& stockArg) {
    const std::string stock = stockArg.empty() ? "/usr/lib/habanalabs/libtpc_kernels.so" : stockArg;
    std::set<std::string> stockNames;
    void* stockHandle = dlopen(stock.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(stockHandle != nullptr,
          "dlopen the stock library " + stock + (stockHandle ? "" : std::string(": ") + dlerror()));
    if (!stockHandle) {
        return 1;
    }
    auto stockGuids = sym<pfnGetKernelGuids>(stockHandle, "GetKernelGuids");
    uint32_t stockCount = 0;
    if (stockGuids && stockGuids(DEVICE_ID_GAUDI2, &stockCount, nullptr) == GLUE_SUCCESS && stockCount != 0) {
        std::vector<GuidInfo> guids(stockCount);
        if (stockGuids(DEVICE_ID_GAUDI2, &stockCount, guids.data()) == GLUE_SUCCESS) {
            for (uint32_t i = 0; i < stockCount && i < guids.size(); ++i) {
                stockNames.insert(guids[i].name);
            }
        }
    }
    std::printf("stock library: %zu Gaudi2 GUIDs\n", stockNames.size());

    const std::string adapterPath = folder + "/libflashinfer_gaudi_combined_gaudi2.so";
    Adapter adapter = open_adapter(adapterPath, "", stockArg, "installed folder");
    if (!adapter.handle) {
        return 1;
    }
    uint32_t total = 0;
    GlueCodeReturn rc = adapter.getGuids(DEVICE_ID_GAUDI2, &total, nullptr);
    check(rc == GLUE_SUCCESS && total != 0,
          "GetKernelGuids size call -> GLUE_SUCCESS, " + std::to_string(total) + " GUIDs");
    std::vector<GuidInfo> merged(total);
    uint32_t capacity = total;
    rc = total ? adapter.getGuids(DEVICE_ID_GAUDI2, &capacity, merged.data()) : GLUE_FAILED;
    check(rc == GLUE_SUCCESS && capacity == total, "GetKernelGuids fill -> GLUE_SUCCESS");
    if (rc != GLUE_SUCCESS) {
        close_adapter(adapter);
        return 1;
    }
    std::set<std::string> mergedNames;
    unsigned custom = 0;
    for (const GuidInfo& guid : merged) {
        mergedNames.insert(guid.name);
        if (stockNames.count(guid.name)) {
            continue;
        }
        ++custom;
        HabanaKernelParams params{};
        set_name(&params.guid, guid.name);
        params.deviceId = DEVICE_ID_GAUDI2;
        uint32_t layoutCount = 0;
        rc = adapter.dataLayouts(&params, nullptr, &layoutCount);  // the graph compiler's count query
        TensorDataLayout inputs[1];
        TensorDataLayout outputs[1];
        std::memset(inputs, 0, sizeof(inputs));
        std::memset(outputs, 0, sizeof(outputs));
        NodeDataLayouts layouts{};
        layouts.inputs = inputs;
        layouts.inputTensorNr = 1;
        layouts.outputs = outputs;
        layouts.outputTensorNr = 1;
        uint32_t filled = layoutCount;
        GlueCodeReturn rcFill = adapter.dataLayouts(&params, &layouts, &filled);
        check(rc == GLUE_SUCCESS && layoutCount >= 1 && rcFill == GLUE_SUCCESS,
              std::string("GetSupportedDataLayouts accepted for ") + guid.name + " (layout '" +
                  std::string(inputs[0].layout, sizeof(inputs[0].layout)) + "')");
    }
    check(custom != 0, std::to_string(custom) + " GUIDs come from the member libraries next to the adapter");
    bool allStock = true;
    for (const std::string& name : stockNames) {
        allStock = allStock && mergedNames.count(name) != 0;
    }
    check(allStock, "every stock GUID is in the merged list");
    check(mergedNames.size() == total, "the merged list has no duplicate name");
    std::printf("combined: %u GUIDs = %zu stock + %u from the members\n", total, stockNames.size(), custom);
    close_adapter(adapter);
    return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc >= 3 && std::string(argv[1]) == "--installed") {
        const int result = check_installed(argv[2], argc >= 4 ? argv[3] : "");
        std::printf(result == 0 ? "ALL CHECKS PASSED\n" : "%d CHECK(S) FAILED\n", g_failures);
        return result;
    }
    scenario_two_members();
    scenario_minimal_member();
    const InstalledLayout layout = make_installed_layout();
    if (layout.ok) {
        scenario_count_protocol(layout);
        scenario_installed_layout(layout);
    }
    remove_layout(layout);
    scenario_installed_layout_with_hc();
    scenario_installed_layout_with_an_unusable_hc();
    scenario_broken_custom_member();
    scenario_member_without_a_required_entry_point();
    scenario_no_member();
    if (g_failures == 0) {
        std::printf("ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
