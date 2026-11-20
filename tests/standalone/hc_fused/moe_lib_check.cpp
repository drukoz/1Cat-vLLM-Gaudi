// SPDX-License-Identifier: Apache-2.0
// Entry-point checks for the MoE glue kernels of libhc_fused_gaudi2.so (test_moe_library.py builds
// and runs this program; lib_check.cpp does the same for the hyper-connection kernels).
//
// It opens the library given as argv[1] with dlopen, the way the graph compiler and the combined
// kernel database (csrc/kernel_db_adapter) do, resolves the entry points by name and calls them
// with good and bad nodes. One line per check: "PASS <name>" or "FAIL <name>: <detail>". Exit code
// 0 only when every check passed. With "--combined" as argv[2], argv[1] is the combined kernel
// database whose member list holds libhc_fused_gaudi2.so.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hc_fused_params.h"

using namespace tpc_lib_api;

namespace {

using InstantiateFn = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);

pfnGetKernelGuids g_guids = nullptr;
InstantiateFn g_instantiate = nullptr;
pfnGetShapeInference g_sif = nullptr;
pfnGetSupportedDataLayout g_layouts = nullptr;
int g_failures = 0;

void Report(bool ok, const std::string& name, const std::string& detail = "")
{
    if (ok) {
        std::printf("PASS %s\n", name.c_str());
    } else {
        ++g_failures;
        std::printf("FAIL %s: %s\n", name.c_str(), detail.c_str());
    }
}

void Expect(GlueCodeReturn got, GlueCodeReturn want, const std::string& name)
{
    Report(got == want, name, "returned " + std::to_string(static_cast<int>(got)) + ", expected " +
                                  std::to_string(static_cast<int>(want)));
}

Tensor Matrix(std::initializer_list<uint64_t> dims, TensorDataType type = DATA_BF16)
{
    Tensor tensor{};
    tensor.geometry.dataType = type;
    tensor.geometry.dims = static_cast<uint32_t>(dims.size());
    unsigned index = 0;
    for (uint64_t dim : dims) {
        tensor.geometry.maxSizes[index] = dim;
        tensor.geometry.minSizes[index] = dim;
        ++index;
    }
    return tensor;
}

// One node: tensors, node parameters and the buffers InstantiateTpcKernel writes.
struct Node {
    std::vector<Tensor> in;
    std::vector<Tensor> out;
    MoeGlueParams scalar{512, 10, 512, kMoeGlueFp8Max};
    uint32_t scalarSize = sizeof(MoeGlueParams);
    HabanaKernelParams params{};
    HabanaKernelInstantiation instance{};
    std::vector<TensorAccessPattern> inPatterns;
    std::vector<TensorAccessPattern> outPatterns;
    std::vector<char> elf;

    Node(const char* guid, std::vector<Tensor> inputs, std::vector<Tensor> outputs)
        : in(std::move(inputs)), out(std::move(outputs))
    {
        params.deviceId = DEVICE_ID_GAUDI2;
        params.maxAvailableTpc = 1;
        std::strncpy(params.guid.name, guid, MAX_NODE_NAME - 1);
        inPatterns.resize(8);
        outPatterns.resize(8);
    }

    GlueCodeReturn Instantiate()
    {
        params.inputTensors = in.data();
        params.inputTensorNr = static_cast<uint32_t>(in.size());
        params.outputTensors = out.data();
        params.outputTensorNr = static_cast<uint32_t>(out.size());
        params.nodeParams.nodeParams = &scalar;
        params.nodeParams.nodeParamsSize = scalarSize;
        instance = HabanaKernelInstantiation{};
        instance.inputTensorAccessPattern = inPatterns.data();
        instance.outputTensorAccessPattern = outPatterns.data();
        GlueCodeReturn result = g_instantiate(&params, &instance);
        if (result == GLUE_INSUFFICIENT_ELF_BUFFER) {
            elf.assign(instance.kernel.elfSize, 0);
            instance.kernel.kernelElf = elf.data();
            result = g_instantiate(&params, &instance);
        }
        return result;
    }

    MoeGlueKernelArgs Args() const
    {
        MoeGlueKernelArgs args{};
        std::memcpy(&args, instance.kernel.scalarParams, sizeof(args));
        return args;
    }

    bool HasElf() const { return elf.size() > 4 && !std::memcmp(elf.data(), "\x7f" "ELF", 4); }
};

bool MappingIs(const TensorAccessPattern& pattern, unsigned dim, float a, float startB, float endB)
{
    const DimIndexSpaceMapping& m = pattern.mapping[dim];
    return m.indexSpaceDim == 0 && m.a == a && m.start_b == startB && m.end_b == endB;
}

// A whole-row pattern of a {columns, tokens} tensor: columns [first, last] of the point's token.
bool RowIs(const TensorAccessPattern& pattern, float first, float last)
{
    return MappingIs(pattern, 0, 0, first, last) && MappingIs(pattern, 1, 1, 0, 0);
}

Node RouteNode() { return Node(kMoeGlueRouteGuid, {Matrix({513, 3})}, {Matrix({10, 3}, DATA_I32), Matrix({10, 3})}); }
Node SiluMulNode() { return Node(kMoeGlueSiluMulGuid, {Matrix({160, 3})}, {Matrix({80, 3})}); }
Node CombineNode()
{
    return Node(kMoeGlueCombineGuid, {Matrix({2560, 3}), Matrix({513, 3}), Matrix({2560, 3})}, {Matrix({2560, 3})});
}
Node RowQuantNode()
{
    return Node(kMoeGlueRowQuantGuid, {Matrix({2560, 3})}, {Matrix({2560, 3}, DATA_F8_143), Matrix({1, 3}, DATA_F32)});
}

void CheckGuids()
{
    uint32_t count = 0;
    Expect(g_guids(DEVICE_ID_GAUDI2, &count, nullptr), GLUE_SUCCESS, "guids/size_query");
    std::vector<GuidInfo> guids(count);
    uint32_t filled = count;
    Expect(g_guids(DEVICE_ID_GAUDI2, &filled, guids.data()), GLUE_SUCCESS, "guids/fill");
    const char* const ours[] = {kMoeGlueRouteGuid, kMoeGlueSiluMulGuid, kMoeGlueCombineGuid, kMoeGlueRowQuantGuid};
    Report(count == kHcFusedGuidCount + kMoeGlueGuidCount, "guids/count", "count " + std::to_string(count));
    for (unsigned index = 0; index < kMoeGlueGuidCount && kHcFusedGuidCount + index < guids.size(); ++index) {
        Report(!std::strcmp(guids[kHcFusedGuidCount + index].name, ours[index]),
               std::string("guids/after_the_hyper_connection_six/") + ours[index]);
    }
}

void CheckRoute()
{
    {
        Node node = RouteNode();
        Expect(node.Instantiate(), GLUE_SUCCESS, "route/instantiate");
        Report(node.instance.indexSpaceRank == 1 && node.instance.indexSpaceGeometry[0] == 3, "route/index_space");
        const MoeGlueKernelArgs args = node.Args();
        Report(node.instance.kernel.paramsNr == 4 && args.experts == 512 && args.top_k == 10 &&
                   args.inv_fp8_max == 0.0f,
               "route/kernel_args");
        Report(RowIs(node.inPatterns[0], 0, 512) && RowIs(node.outPatterns[0], 0, 9) && RowIs(node.outPatterns[1], 0, 9),
               "route/access_patterns");
        Report(node.HasElf(), "route/elf_magic");
    }
    {
        Node node(kMoeGlueRouteGuid, {Matrix({16, 5})}, {Matrix({2, 5}, DATA_I32), Matrix({2, 5})});
        node.scalar = MoeGlueParams{16, 2, 0, 0.0f};
        Expect(node.Instantiate(), GLUE_SUCCESS, "route/router_only_gates");
    }
    {
        Node node = RouteNode();
        node.in.push_back(Matrix({513, 3}));
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_COUNT, "route/input_count");
        Report(node.params.inputTensorNr == 1, "route/input_count_reported");
    }
    {
        Node node = RouteNode();
        node.out.pop_back();
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_OUTPUT_COUNT, "route/output_count");
        Report(node.params.outputTensorNr == 2, "route/output_count_reported");
    }
    {
        Node node(kMoeGlueRouteGuid, {Matrix({513, 3}, DATA_F32)}, {Matrix({10, 3}, DATA_I32), Matrix({10, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_DATA_TYPE, "route/f32_gates");
    }
    {
        Node node(kMoeGlueRouteGuid, {Matrix({513, 3, 2})}, {Matrix({10, 3}, DATA_I32), Matrix({10, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "route/rank3_gates");
    }
    {
        Node node = RouteNode();
        node.scalar.top_k = 0;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "route/top_k_zero");
        node.scalar.top_k = kMoeGlueMaxTopK + 1;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "route/top_k_too_large");
        node.scalar.top_k = 10;
        node.scalar.experts = 9;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "route/fewer_experts_than_top_k");
        node.scalar.experts = 514;
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "route/more_experts_than_columns");
        node.scalar.experts = kMoeGlueMaxExperts + 1;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "route/too_many_experts");
    }
    {
        Node node = RouteNode();
        node.scalarSize = 12;  // the hyper-connection kernels' parameter size
        Expect(node.Instantiate(), GLUE_FAILED, "route/node_params_size");
    }
    {
        // int64 ids are not what the kernel writes: the glue answers with the int32 geometry it needs.
        Node node(kMoeGlueRouteGuid, {Matrix({513, 3})}, {Matrix({10, 3}, DATA_I64), Matrix({10, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_OUTPUT_SIZE, "route/int64_ids");
        Report(node.out[0].geometry.dataType == DATA_I32 && node.out[0].geometry.maxSizes[0] == 10 &&
                   node.out[1].geometry.dataType == DATA_BF16,
               "route/output_fix_reported");
        Expect(node.Instantiate(), GLUE_SUCCESS, "route/instantiate_after_output_fix");
    }
    {
        Node node = RouteNode();
        node.params.deviceId = DEVICE_ID_GAUDI3;
        Expect(node.Instantiate(), GLUE_NODE_NOT_FOUND, "route/other_device");
    }
}

void CheckSiluMul()
{
    {
        Node node = SiluMulNode();
        Expect(node.Instantiate(), GLUE_SUCCESS, "silu_mul/instantiate");
        Report(node.instance.indexSpaceRank == 1 && node.instance.indexSpaceGeometry[0] == 3 &&
                   RowIs(node.inPatterns[0], 0, 159) && RowIs(node.outPatterns[0], 0, 79) && node.HasElf(),
               "silu_mul/index_space_and_patterns");
    }
    {
        Node node(kMoeGlueSiluMulGuid, {Matrix({161, 3})}, {Matrix({80, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "silu_mul/odd_columns");
    }
    {
        Node node(kMoeGlueSiluMulGuid, {Matrix({160, 3})}, {Matrix({160, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_OUTPUT_SIZE, "silu_mul/output_size");
        Report(node.out[0].geometry.maxSizes[0] == 80 && node.out[0].geometry.maxSizes[1] == 3,
               "silu_mul/output_size_reported");
    }
}

void CheckCombine()
{
    {
        Node node = CombineNode();
        Expect(node.Instantiate(), GLUE_SUCCESS, "combine/instantiate");
        Report(node.instance.indexSpaceRank == 1 && node.instance.indexSpaceGeometry[0] == 3 &&
                   node.Args().gate_column == 512,
               "combine/index_space_and_args");
        Report(RowIs(node.inPatterns[0], 0, 2559) && RowIs(node.inPatterns[1], 512, 512) &&
                   RowIs(node.inPatterns[2], 0, 2559) && RowIs(node.outPatterns[0], 0, 2559) && node.HasElf(),
               "combine/access_patterns");
    }
    {
        Node node(kMoeGlueCombineGuid, {Matrix({400, 3}), Matrix({513, 3}), Matrix({400, 3})}, {Matrix({400, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "combine/hidden_not_multiple_of_128");
    }
    {
        Node node(kMoeGlueCombineGuid, {Matrix({2560, 3}), Matrix({513, 3}), Matrix({1280, 3})}, {Matrix({2560, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "combine/routed_width");
    }
    {
        Node node(kMoeGlueCombineGuid, {Matrix({2560, 3}), Matrix({513, 4}), Matrix({2560, 3})}, {Matrix({2560, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "combine/gates_tokens");
    }
    {
        Node node = CombineNode();
        node.scalar.gate_column = 513;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "combine/gate_column_outside");
        node.scalar.gate_column = -1;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "combine/gate_column_negative");
    }
}

void CheckRowQuant()
{
    {
        Node node = RowQuantNode();
        Expect(node.Instantiate(), GLUE_SUCCESS, "row_quant/instantiate");
        // The reciprocal is rounded to bf16, 0x1.12p-8: a card divides a bf16 tensor by 240 as a product
        // with this constant (first card run, 2026-11-20), and the kernel must give the card's scale.
        Report(node.instance.indexSpaceRank == 1 && node.instance.indexSpaceGeometry[0] == 3 &&
                   node.Args().inv_fp8_max == 0x1.12p-8f && node.Args().inv_fp8_max != 1.0f / 240.0f,
               "row_quant/index_space_and_args");
        Report(RowIs(node.inPatterns[0], 0, 2559) && RowIs(node.outPatterns[0], 0, 2559) &&
                   RowIs(node.outPatterns[1], 0, 0) && node.HasElf(),
               "row_quant/access_patterns");
    }
    {
        Node node = RowQuantNode();
        node.scalar.fp8_max = 448.0f;  // the Gaudi3 range
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "row_quant/fp8_max_448");
        node.scalar.fp8_max = 0.0f;
        Expect(node.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "row_quant/fp8_max_zero");
    }
    {
        Node node(kMoeGlueRowQuantGuid, {Matrix({200, 3})}, {Matrix({200, 3}, DATA_F8_143), Matrix({1, 3}, DATA_F32)});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_INPUT_SIZE, "row_quant/hidden_not_multiple_of_128");
    }
    {
        Node node(kMoeGlueRowQuantGuid, {Matrix({2560, 3})}, {Matrix({2560, 3}), Matrix({1, 3})});
        Expect(node.Instantiate(), GLUE_INCOMPATIBLE_OUTPUT_SIZE, "row_quant/bf16_outputs");
        Report(node.out[0].geometry.dataType == DATA_F8_143 && node.out[1].geometry.dataType == DATA_F32 &&
                   node.out[1].geometry.maxSizes[0] == 1,
               "row_quant/output_types_reported");
        Expect(node.Instantiate(), GLUE_SUCCESS, "row_quant/instantiate_after_output_fix");
    }
}

void CheckLayouts()
{
    const char* const ours[] = {kMoeGlueRouteGuid, kMoeGlueSiluMulGuid, kMoeGlueCombineGuid, kMoeGlueRowQuantGuid};
    for (const char* guid : ours) {
        HabanaKernelParams params{};
        params.deviceId = DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, guid, MAX_NODE_NAME - 1);
        TensorDataLayout inputs[3];
        TensorDataLayout outputs[2];
        std::memset(inputs, 0, sizeof(inputs));
        std::memset(outputs, 0, sizeof(outputs));
        NodeDataLayouts layouts{};
        layouts.inputs = inputs;
        layouts.inputTensorNr = 3;
        layouts.outputs = outputs;
        layouts.outputTensorNr = 2;
        uint32_t count = 0;
        const GlueCodeReturn result = g_layouts(&params, &layouts, &count);
        Report(result == GLUE_SUCCESS && count == 1 && inputs[0].layout[0] == 'x' && outputs[1].layout[0] == 'x',
               std::string("layouts/") + guid);
    }
}

// Shape inference of one node. `how`: 0 names the GUID through pGuid, 1 through guid, 2 not at all.
GlueCodeReturn Infer(const char* guid, int how, std::vector<Tensor> in, unsigned outputs, MoeGlueParams scalar,
                     std::vector<TensorShapeInfo>& out)
{
    std::vector<TensorShapeInfo> inputShapes(in.size());
    std::vector<TensorShapeInfo*> inputPointers(in.size());
    for (size_t index = 0; index < in.size(); ++index) {
        inputShapes[index].geometry = in[index].geometry;
        inputPointers[index] = &inputShapes[index];
    }
    out.assign(outputs, TensorShapeInfo{});
    std::vector<TensorShapeInfo*> outputPointers(outputs);
    for (unsigned index = 0; index < outputs; ++index) outputPointers[index] = &out[index];
    GuidInfo info{};
    std::strncpy(info.name, guid, MAX_NODE_NAME - 1);
    ShapeInferenceParams params{};
    if (how == 0) params.pGuid = &info;
    if (how == 1) params.guid = info;
    params.inputTensors = inputPointers.data();
    params.inputTensorsNr = static_cast<uint32_t>(in.size());
    params.outputTensorsNr = outputs;
    params.nodeParams.nodeParams = &scalar;
    params.nodeParams.nodeParamsSize = sizeof(scalar);
    uint32_t invalid = 0;
    ShapeInferenceOutput output{};
    output.outputTensors = outputPointers.data();
    output.invalidMask = &invalid;
    return g_sif(DEVICE_ID_GAUDI2, &params, &output);
}

bool ShapeIs(const TensorShapeInfo& shape, uint64_t columns, uint64_t tokens)
{
    return shape.geometry.dims == 2 && shape.geometry.maxSizes[0] == columns && shape.geometry.maxSizes[1] == tokens;
}

// `modes`: 3 checks the call without a GUID name too (the adapter needs a name to find the member).
void CheckShapeInference(const std::string& prefix, int modes)
{
    const char* const how[] = {"pguid", "guid", "no_guid"};
    for (int mode = 0; mode < modes; ++mode) {
        const std::string tag = prefix + how[mode] + "/";
        std::vector<TensorShapeInfo> out;
        GlueCodeReturn result = Infer(kMoeGlueRouteGuid, mode, {Matrix({513, 7})}, 2, MoeGlueParams{512, 10, 0, 0.0f}, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 10, 7) && ShapeIs(out[1], 10, 7), tag + "route");
        result = Infer(kMoeGlueSiluMulGuid, mode, {Matrix({160, 7})}, 1, MoeGlueParams{0, 0, 0, 0.0f}, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 80, 7), tag + "silu_mul");
        result = Infer(kMoeGlueCombineGuid, mode, {Matrix({2560, 7}), Matrix({513, 7}), Matrix({2560, 7})}, 1,
                       MoeGlueParams{0, 0, 512, 0.0f}, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 2560, 7), tag + "combine");
        // one input and two outputs, like route: without a GUID name the missing top_k tells them apart
        result = Infer(kMoeGlueRowQuantGuid, mode, {Matrix({2560, 7})}, 2, MoeGlueParams{0, 0, 0, kMoeGlueFp8Max}, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 2560, 7) && ShapeIs(out[1], 1, 7), tag + "row_quant");
    }
    std::vector<TensorShapeInfo> out;
    Expect(Infer(kMoeGlueRouteGuid, 0, {Matrix({513, 7})}, 2, MoeGlueParams{514, 10, 0, 0.0f}, out),
           GLUE_INCOMPATIBLE_INPUT_SIZE, prefix + "more_experts_than_columns");
    Expect(Infer(kMoeGlueRouteGuid, 0, {Matrix({513, 7})}, 1, MoeGlueParams{512, 10, 0, 0.0f}, out),
           GLUE_INCOMPATIBLE_OUTPUT_COUNT, prefix + "output_count");
}

}  // namespace

int main(int argc, char** argv)
{
    const bool combined = argc == 3 && !std::strcmp(argv[2], "--combined");
    if (argc != 2 && !combined) {
        std::fprintf(stderr, "usage: %s <libhc_fused_gaudi2.so> | <combined kernel database> --combined\n", argv[0]);
        return 2;
    }
    void* handle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror());
        return 2;
    }
    g_guids = reinterpret_cast<pfnGetKernelGuids>(dlsym(handle, KERNEL_GUIDS_ENTRY_POINT_NAME));
    g_instantiate = reinterpret_cast<InstantiateFn>(dlsym(handle, KERNEL_INSTANTIATION_ENTRY_POINT_NAME));
    g_sif = reinterpret_cast<pfnGetShapeInference>(dlsym(handle, GET_SHAPE_INFERENCE_ENTRY_POINT_NAME));
    g_layouts = reinterpret_cast<pfnGetSupportedDataLayout>(dlsym(handle, GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME));
    Report(g_guids && g_instantiate && g_sif && g_layouts, "symbols/entry_points");
    if (g_failures) return 1;
    if (combined) {
        // Through the adapter: the GUIDs are listed among the other members' and every entry point
        // reaches our library.
        uint32_t count = 0;
        Expect(g_guids(DEVICE_ID_GAUDI2, &count, nullptr), GLUE_SUCCESS, "combined/guids_size_query");
        std::vector<GuidInfo> guids(count);
        uint32_t filled = count;
        Expect(g_guids(DEVICE_ID_GAUDI2, &filled, guids.data()), GLUE_SUCCESS, "combined/guids_fill");
        const char* const ours[] = {kMoeGlueRouteGuid, kMoeGlueSiluMulGuid, kMoeGlueCombineGuid, kMoeGlueRowQuantGuid};
        for (const char* guid : ours) {
            bool listed = false;
            for (const GuidInfo& info : guids) listed = listed || !std::strcmp(info.name, guid);
            Report(listed, std::string("combined/lists/") + guid);
        }
        Node route = RouteNode();
        Expect(route.Instantiate(), GLUE_SUCCESS, "combined/instantiate_route");
        Report(route.HasElf() && route.Args().top_k == 10, "combined/instantiate_reaches_our_glue");
        Node quant = RowQuantNode();
        quant.scalar.fp8_max = 448.0f;
        Expect(quant.Instantiate(), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "combined/rejection_reaches_the_caller");
        CheckLayouts();
        CheckShapeInference("combined/sif/", 2);
        std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
        return g_failures ? 1 : 0;
    }

    CheckGuids();
    CheckRoute();
    CheckSiluMul();
    CheckCombine();
    CheckRowQuant();
    CheckLayouts();
    CheckShapeInference("sif/", 3);
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
