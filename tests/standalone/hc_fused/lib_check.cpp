// SPDX-License-Identifier: Apache-2.0
// Entry-point checks for libhc_fused_gaudi2.so (test_hc_library.py builds and runs this program).
//
// It opens the library given as argv[1] with dlopen, the way the graph compiler and the combined
// kernel database (csrc/kernel_db_adapter) do, resolves the entry points by name and calls them
// with good and bad nodes. One line per check: "PASS <name>" or "FAIL <name>: <detail>"; the GUID
// list is printed as "GUID <index> <name>". Exit code 0 only when every check passed.
//
// With "--combined" as argv[2], argv[1] is the combined kernel database
// (libflashinfer_gaudi_combined_gaudi2.so) whose member list holds libhc_fused_gaudi2.so: the same
// entry points are then called through the adapter, which must list our six GUIDs and route
// InstantiateTpcKernel, GetShapeInference and GetSupportedDataLayouts for them to our library.
#include <tpc_kernel_lib_interface.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hc_fused_params.h"

using namespace tpc_lib_api;

namespace {

using InstantiateFn = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);
using SuggestedManipulationFn = GlueCodeReturn (*)(const HabanaKernelParams*, void*);

pfnGetKernelGuids g_guids = nullptr;
InstantiateFn g_instantiate = nullptr;
pfnGetShapeInference g_sif = nullptr;
pfnGetSupportedDataLayout g_layouts = nullptr;
SuggestedManipulationFn g_manipulation = nullptr;
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

Tensor Bf16(std::initializer_list<uint64_t> dims, TensorDataType type = DATA_BF16)
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
    HcFusedParams scalar{1e-6f, 4, 0};
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

    // Binds the pointers (after the caller edited the vectors or the scalars) and instantiates.
    GlueCodeReturn Instantiate(bool withElf)
    {
        params.inputTensors = in.data();
        params.inputTensorNr = static_cast<uint32_t>(in.size());
        params.outputTensors = out.data();
        params.outputTensorNr = static_cast<uint32_t>(out.size());
        if (!params.nodeParams.nodeParams) {
            params.nodeParams.nodeParams = &scalar;
            params.nodeParams.nodeParamsSize = sizeof(scalar);
        }
        instance = HabanaKernelInstantiation{};
        instance.inputTensorAccessPattern = inPatterns.data();
        instance.outputTensorAccessPattern = outPatterns.data();
        GlueCodeReturn result = g_instantiate(&params, &instance);
        if (withElf && result == GLUE_INSUFFICIENT_ELF_BUFFER) {
            elf.assign(instance.kernel.elfSize, 0);
            instance.kernel.kernelElf = elf.data();
            result = g_instantiate(&params, &instance);
        }
        return result;
    }
};

bool MappingIs(const TensorAccessPattern& pattern, unsigned dim, unsigned indexDim, float a, float startB, float endB)
{
    const DimIndexSpaceMapping& m = pattern.mapping[dim];
    return m.indexSpaceDim == indexDim && m.a == a && m.start_b == startB && m.end_b == endB;
}

void CheckGuids()
{
    uint32_t count = 0;
    Expect(g_guids(DEVICE_ID_GAUDI2, &count, nullptr), GLUE_SUCCESS, "guids/size_query");
    // The six hyper-connection GUIDs, then the MoE glue GUIDs that were built into the library (0 to 4;
    // moe_lib_check.cpp checks those).
    Report(count >= kHcFusedGuidCount && count <= kHcFusedGuidCount + kMoeGlueGuidCount, "guids/count",
           "count " + std::to_string(count));
    std::vector<GuidInfo> guids(count);
    uint32_t filled = count;
    Expect(g_guids(DEVICE_ID_GAUDI2, &filled, guids.data()), GLUE_SUCCESS, "guids/fill");
    for (uint32_t index = 0; index < filled && index < guids.size(); ++index) {
        std::printf("GUID %u %s\n", index, guids[index].name);
    }
    std::vector<GuidInfo> few(3);
    std::memset(few.data(), 0x7f, few.size() * sizeof(GuidInfo));
    uint32_t capacity = 2;
    Expect(g_guids(DEVICE_ID_GAUDI2, &capacity, few.data()), GLUE_SUCCESS, "guids/short_buffer");
    Report(capacity == count && few[2].name[0] == 0x7f && !std::strcmp(few[0].name, guids[0].name),
           "guids/short_buffer_fills_only_capacity");
    uint32_t other = 7;
    Expect(g_guids(DEVICE_ID_GAUDI, &other, guids.data()), GLUE_SUCCESS, "guids/other_device");
    Report(other == 0, "guids/other_device_count", "count " + std::to_string(other));
    Expect(g_guids(DEVICE_ID_GAUDI2, nullptr, nullptr), GLUE_FAILED, "guids/null_count");
}

void CheckInstantiate()
{
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3})});
        Expect(node.Instantiate(false), GLUE_INSUFFICIENT_ELF_BUFFER, "norm/elf_size_query");
        Report(node.instance.kernel.elfSize > 0, "norm/elf_size_reported");
        Expect(node.Instantiate(true), GLUE_SUCCESS, "norm/instantiate");
        Report(node.instance.indexSpaceRank == 2 && node.instance.indexSpaceGeometry[0] == 4 &&
                   node.instance.indexSpaceGeometry[1] == 3,
               "norm/index_space");
        HcFusedKernelArgs args{};
        std::memcpy(&args, node.instance.kernel.scalarParams, sizeof(args));
        Report(node.instance.kernel.paramsNr == 4 && args.eps == 1e-6f && args.inv_group == 1.0f / 128.0f &&
                   args.inv_hc == 0.25f && args.hc_count == 4,
               "norm/kernel_args");
        Report(MappingIs(node.inPatterns[0], 0, 0, 128, 0, 127) && MappingIs(node.inPatterns[0], 1, 1, 1, 0, 0) &&
                   MappingIs(node.inPatterns[1], 0, 0, 128, 0, 127) &&
                   MappingIs(node.outPatterns[0], 0, 0, 128, 0, 127) &&
                   MappingIs(node.outPatterns[0], 1, 1, 1, 0, 0),
               "norm/access_patterns_per_branch_weight");
        Report(node.elf.size() > 4 && !std::memcmp(node.elf.data(), "\x7f" "ELF", 4), "norm/elf_magic");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({128})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "norm/shared_weight");
        Report(MappingIs(node.inPatterns[1], 0, 0, 0, 0, 127), "norm/access_pattern_shared_weight");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_COUNT, "norm/input_count");
        Report(node.params.inputTensorNr == 2, "norm/input_count_reported");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3}), Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_OUTPUT_COUNT, "norm/output_count");
        Report(node.params.outputTensorNr == 1, "norm/output_count_reported");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}, DATA_F32), Bf16({512})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_DATA_TYPE, "norm/f32_input");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({400, 3}), Bf16({400})}, {Bf16({400, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "norm/hidden_not_multiple_of_128");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({514, 3}), Bf16({514})}, {Bf16({514, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "norm/width_not_divisible_by_hc");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({256})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "norm/weight_size");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512, 1})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "norm/weight_rank");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3, 2}), Bf16({512})}, {Bf16({512, 3, 2})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "norm/rank3_input");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3})});
        node.scalar.eps = 0.0f;
        Expect(node.Instantiate(true), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "norm/eps_zero");
        node.scalar.eps = std::nanf("");
        Expect(node.Instantiate(true), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "norm/eps_nan");
        node.scalar.eps = 1e-6f;
        node.scalar.hc_count = 0;
        Expect(node.Instantiate(true), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "norm/hc_count_zero");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3})});
        node.params.nodeParams.nodeParams = &node.scalar;
        node.params.nodeParams.nodeParamsSize = 8;
        Expect(node.Instantiate(true), GLUE_FAILED, "norm/node_params_size");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({128, 3}, DATA_F32)});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_OUTPUT_SIZE, "norm/output_size");
        const TensorGeometry& fixed = node.out[0].geometry;
        Report(fixed.dataType == DATA_BF16 && fixed.dims == 2 && fixed.maxSizes[0] == 512 && fixed.maxSizes[1] == 3,
               "norm/output_size_reported");
        Expect(node.Instantiate(true), GLUE_SUCCESS, "norm/instantiate_after_output_fix");
    }
    {
        Node node("hc_fused_no_such_kernel", {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3})});
        Expect(node.Instantiate(true), GLUE_NODE_NOT_FOUND, "unknown_guid");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({512, 3}), Bf16({512})}, {Bf16({512, 3})});
        node.params.deviceId = DEVICE_ID_GAUDI3;
        Expect(node.Instantiate(true), GLUE_NODE_NOT_FOUND, "other_device");
    }
    {
        Node node(kHcFusedSiluGuid, {Bf16({320, 5})}, {Bf16({320, 5})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "silu/instantiate");
        Report(node.instance.indexSpaceRank == 1 && node.instance.indexSpaceGeometry[0] == 5 &&
                   MappingIs(node.inPatterns[0], 0, 0, 0, 0, 319) && MappingIs(node.inPatterns[0], 1, 0, 1, 0, 0) &&
                   MappingIs(node.outPatterns[0], 0, 0, 0, 0, 319),
               "silu/index_space_and_patterns");
    }
    {
        Node node(kHcFusedSiluSplitGuid, {Bf16({336, 5})}, {Bf16({320, 5}), Bf16({4, 5})});
        node.scalar.lora_rank = 320;
        Expect(node.Instantiate(true), GLUE_SUCCESS, "silu_split/instantiate");
        Report(MappingIs(node.inPatterns[0], 0, 0, 0, 0, 335) && MappingIs(node.outPatterns[0], 0, 0, 0, 0, 319) &&
                   MappingIs(node.outPatterns[1], 0, 0, 0, 0, 3) && MappingIs(node.outPatterns[1], 1, 0, 1, 0, 0),
               "silu_split/patterns");
        node.scalar.lora_rank = 0;
        Expect(node.Instantiate(true), GLUE_KERNEL_INVALID_SCALAR_ARGUMENT, "silu_split/lora_rank_zero");
        node.scalar.lora_rank = 334;
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "silu_split/lora_rank_too_large");
    }
    {
        Node node(kHcFusedGateMixGuid, {Bf16({10240, 2}), Bf16({10240, 2})}, {Bf16({2560, 2})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "gate_mix/instantiate");
        Report(node.instance.indexSpaceRank == 2 && node.instance.indexSpaceGeometry[0] == 20 &&
                   node.instance.indexSpaceGeometry[1] == 2 && MappingIs(node.inPatterns[0], 0, 0, 0, 0, 10239) &&
                   MappingIs(node.inPatterns[1], 1, 1, 1, 0, 0) && MappingIs(node.outPatterns[0], 0, 0, 128, 0, 127) &&
                   MappingIs(node.outPatterns[0], 1, 1, 1, 0, 0),
               "gate_mix/index_space_and_patterns");
        node.in[1] = Bf16({10240, 3});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "gate_mix/gate_shape");
    }
    {
        Node node(kHcFusedCombineGuid, {Bf16({10240, 2}), Bf16({2560, 2}), Bf16({4, 2})}, {Bf16({10240, 2})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "combine/instantiate");
        Report(node.instance.indexSpaceRank == 2 && node.instance.indexSpaceGeometry[0] == 4 &&
                   MappingIs(node.inPatterns[0], 0, 0, 2560, 0, 2559) &&
                   MappingIs(node.inPatterns[1], 0, 0, 0, 0, 2559) &&
                   MappingIs(node.inPatterns[2], 0, 0, 1, 0, 0) && MappingIs(node.inPatterns[2], 1, 1, 1, 0, 0) &&
                   MappingIs(node.outPatterns[0], 0, 0, 2560, 0, 2559),
               "combine/index_space_and_patterns");
        node.in[2] = Bf16({3, 2});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "combine/injection_width");
        node.in[2] = Bf16({4, 2});
        node.in[1] = Bf16({2560, 1});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "combine/block_rows");
    }
    {
        Node node(kHcFusedCombineNormGuid, {Bf16({10240, 2}), Bf16({2560, 2}), Bf16({4, 2}), Bf16({2560})},
                  {Bf16({10240, 2}), Bf16({10240, 2})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "combine_norm/instantiate");
        Report(MappingIs(node.inPatterns[3], 0, 0, 0, 0, 2559) && MappingIs(node.outPatterns[1], 0, 0, 2560, 0, 2559) &&
                   MappingIs(node.outPatterns[1], 1, 1, 1, 0, 0),
               "combine_norm/patterns");
        HcFusedKernelArgs args{};
        std::memcpy(&args, node.instance.kernel.scalarParams, sizeof(args));
        Report(args.inv_group == 1.0f / 2560.0f && args.inv_hc == 0.25f, "combine_norm/kernel_args");
    }
    Expect(g_instantiate(nullptr, nullptr), GLUE_FAILED, "instantiate/null");
}

void CheckLayouts()
{
    const char* const guids[] = {kHcFusedNormGuid,    kHcFusedSiluGuid,    kHcFusedSiluSplitGuid,
                                 kHcFusedGateMixGuid, kHcFusedCombineGuid, kHcFusedCombineNormGuid};
    for (const char* guid : guids) {
        HabanaKernelParams params{};
        params.deviceId = DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, guid, MAX_NODE_NAME - 1);
        TensorDataLayout inputs[4];
        TensorDataLayout outputs[2];
        std::memset(inputs, 0, sizeof(inputs));
        std::memset(outputs, 0, sizeof(outputs));
        NodeDataLayouts layouts{};
        layouts.inputs = inputs;
        layouts.inputTensorNr = 4;
        layouts.outputs = outputs;
        layouts.outputTensorNr = 2;
        uint32_t count = 0;
        const GlueCodeReturn result = g_layouts(&params, &layouts, &count);
        bool allX = true;
        for (const TensorDataLayout& layout : inputs) {
            for (char c : layout.layout) allX = allX && c == 'x';
        }
        for (const TensorDataLayout& layout : outputs) {
            for (char c : layout.layout) allX = allX && c == 'x';
        }
        Report(result == GLUE_SUCCESS && count == 1 && allX, std::string("layouts/") + guid);
        uint32_t countOnly = 0;
        Report(g_layouts(&params, nullptr, &countOnly) == GLUE_SUCCESS && countOnly == 1,
               std::string("layouts/count_only/") + guid);
    }
    HabanaKernelParams unknown{};
    std::strcpy(unknown.guid.name, "relu_fwd_f32");
    uint32_t count = 5;
    Expect(g_layouts(&unknown, nullptr, &count), GLUE_NODE_NOT_FOUND, "layouts/unknown_guid");
    HabanaKernelParams known{};
    std::strcpy(known.guid.name, kHcFusedNormGuid);
    Expect(g_layouts(&known, nullptr, nullptr), GLUE_FAILED, "layouts/null_count");
}

// Shape inference of one node. `how`: 0 names the GUID through pGuid, 1 through guid, 2 not at all.
GlueCodeReturn Infer(const char* guid, int how, std::vector<Tensor> in, unsigned outputs, HcFusedParams scalar,
                     std::vector<TensorShapeInfo>& out, DeviceId device = DEVICE_ID_GAUDI2, bool withParams = true)
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
    if (withParams) {
        params.nodeParams.nodeParams = &scalar;
        params.nodeParams.nodeParamsSize = sizeof(scalar);
    }
    uint32_t invalid = 0;
    ShapeInferenceOutput output{};
    output.outputTensors = outputPointers.data();
    output.invalidMask = &invalid;
    return g_sif(device, &params, &output);
}

bool ShapeIs(const TensorShapeInfo& shape, uint64_t columns, uint64_t tokens)
{
    return shape.geometry.dims == 2 && shape.geometry.maxSizes[0] == columns && shape.geometry.maxSizes[1] == tokens;
}

void CheckShapeInference()
{
    const char* const how[] = {"pguid", "guid", "no_guid"};
    for (int mode = 0; mode < 3; ++mode) {
        const std::string tag = std::string("sif/") + how[mode] + "/";
        std::vector<TensorShapeInfo> out;
        HcFusedParams scalar{1e-6f, 4, 0};
        GlueCodeReturn result = Infer(kHcFusedNormGuid, mode, {Bf16({10240, 7}), Bf16({10240})}, 1, scalar, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 10240, 7), tag + "norm");
        result = Infer(kHcFusedGateMixGuid, mode, {Bf16({10240, 7}), Bf16({10240, 7})}, 1, scalar, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 2560, 7), tag + "gate_mix");
        result = Infer(kHcFusedSiluGuid, mode, {Bf16({320, 7})}, 1, scalar, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 320, 7), tag + "silu");
        HcFusedParams split{0.0f, 4, 320};
        result = Infer(kHcFusedSiluSplitGuid, mode, {Bf16({336, 7})}, 2, split, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 320, 7) && ShapeIs(out[1], 4, 7), tag + "silu_split");
        result = Infer(kHcFusedCombineGuid, mode, {Bf16({10240, 7}), Bf16({2560, 7}), Bf16({4, 7})}, 1, scalar, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 10240, 7), tag + "combine");
        result = Infer(kHcFusedCombineNormGuid, mode,
                       {Bf16({10240, 7}), Bf16({2560, 7}), Bf16({4, 7}), Bf16({10240})}, 2, scalar, out);
        Report(result == GLUE_SUCCESS && ShapeIs(out[0], 10240, 7) && ShapeIs(out[1], 10240, 7), tag + "combine_norm");
    }
    std::vector<TensorShapeInfo> out;
    HcFusedParams scalar{1e-6f, 4, 0};
    Expect(Infer(kHcFusedNormGuid, 0, {Bf16({10240, 7}), Bf16({10240})}, 1, scalar, out, DEVICE_ID_GAUDI3),
           GLUE_NODE_NOT_FOUND, "sif/other_device");
    Expect(Infer(kHcFusedNormGuid, 0, {Bf16({10240, 7}), Bf16({10240})}, 1, scalar, out, DEVICE_ID_GAUDI2, false),
           GLUE_MISSING_PRIVATE_STRUCTURE, "sif/no_node_params");
    Expect(Infer(kHcFusedNormGuid, 0, {Bf16({400, 7}), Bf16({400})}, 1, scalar, out), GLUE_INCOMPATIBLE_INPUT_SIZE,
           "sif/hidden_not_multiple_of_128");
    Expect(Infer(kHcFusedNormGuid, 0, {Bf16({10240, 7}), Bf16({10240})}, 2, scalar, out),
           GLUE_INCOMPATIBLE_OUTPUT_COUNT, "sif/output_count");
    Expect(Infer("", 2, {Bf16({1, 1}), Bf16({1, 1}), Bf16({1, 1}), Bf16({1, 1}), Bf16({1, 1})}, 1, scalar, out),
           GLUE_NODE_NOT_FOUND, "sif/no_such_kernel");
    Expect(g_sif(DEVICE_ID_GAUDI2, nullptr, nullptr), GLUE_SIF_NULL_PTR, "sif/null");
}

// Through the combined kernel database: our GUIDs are listed among the other members' and every
// entry point reaches our library.
void CheckThroughCombined()
{
    const char* const ours[] = {kHcFusedNormGuid,    kHcFusedSiluGuid,    kHcFusedSiluSplitGuid,
                                kHcFusedGateMixGuid, kHcFusedCombineGuid, kHcFusedCombineNormGuid};
    uint32_t count = 0;
    Expect(g_guids(DEVICE_ID_GAUDI2, &count, nullptr), GLUE_SUCCESS, "combined/guids_size_query");
    std::vector<GuidInfo> guids(count);
    uint32_t filled = count;
    Expect(g_guids(DEVICE_ID_GAUDI2, &filled, guids.data()), GLUE_SUCCESS, "combined/guids_fill");
    Report(count > kHcFusedGuidCount, "combined/guids_include_other_members", "count " + std::to_string(count));
    for (const char* guid : ours) {
        bool listed = false;
        for (const GuidInfo& info : guids) listed = listed || !std::strcmp(info.name, guid);
        Report(listed, std::string("combined/lists/") + guid);

        HabanaKernelParams params{};
        params.deviceId = DEVICE_ID_GAUDI2;
        std::strncpy(params.guid.name, guid, MAX_NODE_NAME - 1);
        TensorDataLayout inputs[4];
        TensorDataLayout outputs[2];
        std::memset(inputs, 0, sizeof(inputs));
        std::memset(outputs, 0, sizeof(outputs));
        NodeDataLayouts layouts{};
        layouts.inputs = inputs;
        layouts.inputTensorNr = 4;
        layouts.outputs = outputs;
        layouts.outputTensorNr = 2;
        uint32_t layoutCount = 0;
        const GlueCodeReturn result = g_layouts(&params, &layouts, &layoutCount);
        Report(result == GLUE_SUCCESS && layoutCount == 1 && inputs[0].layout[0] == 'x' && outputs[1].layout[0] == 'x',
               std::string("combined/layouts/") + guid);
    }
    {
        Node node(kHcFusedCombineNormGuid, {Bf16({10240, 2}), Bf16({2560, 2}), Bf16({4, 2}), Bf16({10240})},
                  {Bf16({10240, 2}), Bf16({10240, 2})});
        Expect(node.Instantiate(true), GLUE_SUCCESS, "combined/instantiate_combine_norm");
        Report(node.instance.indexSpaceRank == 2 && node.instance.indexSpaceGeometry[0] == 4 &&
                   node.instance.indexSpaceGeometry[1] == 2 && node.instance.kernel.paramsNr == 4 &&
                   node.elf.size() > 4 && !std::memcmp(node.elf.data(), "\x7f" "ELF", 4),
               "combined/instantiate_reaches_our_glue");
    }
    {
        Node node(kHcFusedNormGuid, {Bf16({400, 3}), Bf16({400})}, {Bf16({400, 3})});
        Expect(node.Instantiate(true), GLUE_INCOMPATIBLE_INPUT_SIZE, "combined/rejection_reaches_the_caller");
    }
    std::vector<TensorShapeInfo> out;
    HcFusedParams scalar{0.0f, 4, 320};
    Report(Infer(kHcFusedSiluSplitGuid, 0, {Bf16({336, 7})}, 2, scalar, out) == GLUE_SUCCESS &&
               ShapeIs(out[0], 320, 7) && ShapeIs(out[1], 4, 7),
           "combined/shape_inference_pguid");
    Report(Infer(kHcFusedGateMixGuid, 1, {Bf16({10240, 7}), Bf16({10240, 7})}, 1, scalar, out) == GLUE_SUCCESS &&
               ShapeIs(out[0], 2560, 7),
           "combined/shape_inference_guid");
}

}  // namespace

int main(int argc, char** argv)
{
    const bool combined = argc == 3 && !std::strcmp(argv[2], "--combined");
    if (argc != 2 && !combined) {
        std::fprintf(stderr, "usage: %s <libhc_fused_gaudi2.so> | <combined kernel database> --combined\n", argv[0]);
        return 2;
    }
    // RTLD_LAZY | RTLD_LOCAL: how csrc/kernel_db_adapter/adapter.cpp opens a member library.
    void* handle = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "cannot load %s: %s\n", argv[1], dlerror());
        return 2;
    }
    g_guids = reinterpret_cast<pfnGetKernelGuids>(dlsym(handle, KERNEL_GUIDS_ENTRY_POINT_NAME));
    g_instantiate = reinterpret_cast<InstantiateFn>(dlsym(handle, KERNEL_INSTANTIATION_ENTRY_POINT_NAME));
    g_sif = reinterpret_cast<pfnGetShapeInference>(dlsym(handle, GET_SHAPE_INFERENCE_ENTRY_POINT_NAME));
    g_layouts = reinterpret_cast<pfnGetSupportedDataLayout>(dlsym(handle, GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME));
    g_manipulation = reinterpret_cast<SuggestedManipulationFn>(dlsym(handle, "GetSuggestedManipulation"));
    Report(g_guids && g_instantiate && g_sif && g_layouts && g_manipulation, "symbols/five_entry_points");
    if (g_failures) return 1;
    if (combined) {
        CheckThroughCombined();
        std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
        return g_failures ? 1 : 0;
    }
    Report(dlsym(handle, GET_LIB_VERSION_ENTRY_POINT_NAME) == nullptr, "symbols/no_lib_version");

    CheckGuids();
    CheckInstantiate();
    CheckLayouts();
    CheckShapeInference();
    {
        HabanaKernelParams params{};
        std::strcpy(params.guid.name, kHcFusedNormGuid);
        int suggestion = 0;
        Expect(g_manipulation(&params, &suggestion), GLUE_UNSUPPORTED_LAYER_CONFIGURATION, "manipulation/unsupported");
        Expect(g_manipulation(nullptr, nullptr), GLUE_FAILED, "manipulation/null");
    }
    std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
