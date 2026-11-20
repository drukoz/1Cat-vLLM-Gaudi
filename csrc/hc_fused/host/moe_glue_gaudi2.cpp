// SPDX-License-Identifier: Apache-2.0
// Host glue of the fused MoE glue kernels: validation, index space, access patterns, scalar
// arguments and the embedded kernel ELF, for the MoE GUIDs of hc_fused_params.h.
//
// Every size comes from the tensor geometry and the node parameters. The gates are: rank-2
// activations {columns, tokens}; bf16 inputs; route: 1 <= top_k <= 64, top_k <= experts <= the
// columns of gates and <= 32767, outputs int32 ids and bf16 weights; silu_mul: an even number of
// input columns; combine and row_quant: a hidden size divisible by one 128-lane vector; combine:
// a gate column inside gates; row_quant: fp8_max exactly 240, outputs fp8 (1-4-3) and fp32.
// Anything else is rejected here with a GlueCodeReturn; no kernel runs on a shape it cannot handle.

#include <cstdint>
#include <cstring>

#include "moe_glue_gaudi2.hpp"

#ifdef HC_FUSED_HAS_MOE_GLUE_ROUTE
extern unsigned char _binary_moe_glue_route_gaudi2_o_start;
extern unsigned char _binary_moe_glue_route_gaudi2_o_end;
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_SILU_MUL
extern unsigned char _binary_moe_glue_silu_mul_gaudi2_o_start;
extern unsigned char _binary_moe_glue_silu_mul_gaudi2_o_end;
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_COMBINE
extern unsigned char _binary_moe_glue_combine_gaudi2_o_start;
extern unsigned char _binary_moe_glue_combine_gaudi2_o_end;
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_ROW_QUANT
extern unsigned char _binary_moe_glue_row_quant_gaudi2_o_start;
extern unsigned char _binary_moe_glue_row_quant_gaudi2_o_end;
#endif

namespace moe_glue {
namespace {

using namespace tpc_lib_api;

constexpr unsigned kMaxInputs = 3;
constexpr unsigned kMaxOutputs = 2;

// The kernels that were built, in the order of hc_fused_params.h. The array always has one
// element more than the kernels, so that it is never empty.
const KernelInfo kKernels[] = {
#ifdef HC_FUSED_HAS_MOE_GLUE_ROUTE
    {Kind::Route, kMoeGlueRouteGuid, 1, 2},
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_SILU_MUL
    {Kind::SiluMul, kMoeGlueSiluMulGuid, 1, 1},
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_COMBINE
    {Kind::Combine, kMoeGlueCombineGuid, 3, 1},
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_ROW_QUANT
    {Kind::RowQuant, kMoeGlueRowQuantGuid, 1, 2},
#endif
    {Kind::Route, nullptr, 0, 0},
};
constexpr unsigned kKernelCount = sizeof(kKernels) / sizeof(kKernels[0]) - 1;

struct Elf {
    unsigned char* start;
    unsigned char* end;
};

Elf KernelElf(Kind kind)
{
    switch (kind) {
#ifdef HC_FUSED_HAS_MOE_GLUE_ROUTE
        case Kind::Route:
            return {&_binary_moe_glue_route_gaudi2_o_start, &_binary_moe_glue_route_gaudi2_o_end};
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_SILU_MUL
        case Kind::SiluMul:
            return {&_binary_moe_glue_silu_mul_gaudi2_o_start, &_binary_moe_glue_silu_mul_gaudi2_o_end};
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_COMBINE
        case Kind::Combine:
            return {&_binary_moe_glue_combine_gaudi2_o_start, &_binary_moe_glue_combine_gaudi2_o_end};
#endif
#ifdef HC_FUSED_HAS_MOE_GLUE_ROW_QUANT
        case Kind::RowQuant:
            return {&_binary_moe_glue_row_quant_gaudi2_o_start, &_binary_moe_glue_row_quant_gaudi2_o_end};
#endif
        default:
            break;
    }
    return {nullptr, nullptr};
}

// What the shapes of one node resolve to.
struct Resolved {
    uint64_t tokens = 0;
    unsigned outputs = 0;
    uint64_t outputColumns[kMaxOutputs] = {};  // every output is rank 2 {columns, tokens}
    TensorDataType outputType[kMaxOutputs] = {DATA_BF16, DATA_BF16};
};

bool IsMatrix(const TensorGeometry& geometry)
{
    return geometry.dims == 2 && geometry.maxSizes[0] > 0 && geometry.maxSizes[1] > 0;
}

bool IsMatrixOf(const TensorGeometry& geometry, uint64_t columns, uint64_t tokens)
{
    return geometry.dims == 2 && geometry.maxSizes[0] == columns && geometry.maxSizes[1] == tokens;
}

// Shape validation shared by Instantiate and InferShapes. `in` has kernel.inputs entries.
GlueCodeReturn Resolve(const KernelInfo& kernel, const TensorGeometry* const* in, const MoeGlueParams& scalar,
                       Resolved& r)
{
    if (!IsMatrix(*in[0])) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t columns = in[0]->maxSizes[0];
    r.tokens = in[0]->maxSizes[1];
    r.outputs = kernel.outputs;

    switch (kernel.kind) {
        case Kind::Route:
            if (scalar.top_k < 1 || scalar.top_k > kMoeGlueMaxTopK || scalar.experts < scalar.top_k ||
                scalar.experts > kMoeGlueMaxExperts) {
                return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
            }
            if (columns < static_cast<uint64_t>(scalar.experts)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            r.outputColumns[0] = static_cast<uint64_t>(scalar.top_k);
            r.outputColumns[1] = static_cast<uint64_t>(scalar.top_k);
            r.outputType[0] = DATA_I32;
            return GLUE_SUCCESS;
        case Kind::SiluMul:
            if (columns % 2 != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            r.outputColumns[0] = columns / 2;
            return GLUE_SUCCESS;
        case Kind::Combine:
            if (columns % kHcFusedVectorLanes != 0 || !IsMatrix(*in[1]) || in[1]->maxSizes[1] != r.tokens ||
                !IsMatrixOf(*in[2], columns, r.tokens)) {
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            }
            if (scalar.gate_column < 0 || static_cast<uint64_t>(scalar.gate_column) >= in[1]->maxSizes[0]) {
                return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
            }
            r.outputColumns[0] = columns;
            return GLUE_SUCCESS;
        case Kind::RowQuant:
            if (columns % kHcFusedVectorLanes != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if (!(scalar.fp8_max == kMoeGlueFp8Max)) return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
            r.outputColumns[0] = columns;
            r.outputColumns[1] = 1;
            r.outputType[0] = DATA_F8_143;
            r.outputType[1] = DATA_F32;
            return GLUE_SUCCESS;
    }
    return GLUE_NODE_NOT_FOUND;
}

void MapDimension(TensorAccessPattern& pattern, unsigned tensorDim, uint64_t a, uint64_t startB, uint64_t endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = 0;
    pattern.mapping[tensorDim].a = static_cast<float>(a);
    pattern.mapping[tensorDim].start_b = static_cast<float>(startB);
    pattern.mapping[tensorDim].end_b = static_cast<float>(endB);
}

// An fp32 value rounded to the nearest bf16 (ties to even), as fp32. For finite values.
float RoundToBf16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    bits &= 0xFFFF0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// {columns, tokens} tensor of which the point `token` reads or writes columns [first, last] of its row.
void MapRow(TensorAccessPattern& pattern, uint64_t first, uint64_t last)
{
    MapDimension(pattern, 0, 0, first, last);
    MapDimension(pattern, 1, 1, 0, 0);
}

}  // namespace

const KernelInfo* Kernels() { return kKernels; }

unsigned KernelCount() { return kKernelCount; }

const KernelInfo* FindKernel(const char* name)
{
    if (!name) return nullptr;
    for (unsigned index = 0; index < kKernelCount; ++index) {
        if (!std::strncmp(name, kKernels[index].guid, MAX_NODE_NAME)) return &kKernels[index];
    }
    return nullptr;
}

const KernelInfo* KernelForShapes(const ShapeInferenceParams* params)
{
    if (!params || !params->nodeParams.nodeParams || params->nodeParams.nodeParamsSize != sizeof(MoeGlueParams)) {
        return nullptr;
    }
    MoeGlueParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    for (unsigned index = 0; index < kKernelCount; ++index) {
        const KernelInfo& kernel = kKernels[index];
        if (kernel.inputs != params->inputTensorsNr || kernel.outputs != params->outputTensorsNr) continue;
        // route and row_quant both have one input and two outputs: only route has a top_k.
        if (kernel.kind == Kind::Route && scalar.top_k < 1) continue;
        if (kernel.kind == Kind::RowQuant && scalar.top_k >= 1) continue;
        return &kernel;
    }
    return nullptr;
}

GlueCodeReturn Instantiate(const KernelInfo& kernel, HabanaKernelParams* params, HabanaKernelInstantiation* instance)
{
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != kernel.inputs) {
        params->inputTensorNr = kernel.inputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (params->outputTensorNr != kernel.outputs) {
        params->outputTensorNr = kernel.outputs;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(MoeGlueParams)) {
        return GLUE_FAILED;
    }
    MoeGlueParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));

    const TensorGeometry* in[kMaxInputs] = {};
    for (unsigned input = 0; input < kernel.inputs; ++input) {
        if (params->inputTensors[input].geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
        in[input] = &params->inputTensors[input].geometry;
    }
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;

    bool outputsMatch = true;
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        const TensorGeometry& geometry = params->outputTensors[output].geometry;
        outputsMatch = outputsMatch && geometry.dataType == r.outputType[output] &&
                       IsMatrixOf(geometry, r.outputColumns[output], r.tokens);
    }
    if (!outputsMatch) {
        for (unsigned output = 0; output < kernel.outputs; ++output) {
            TensorGeometry& geometry = params->outputTensors[output].geometry;
            geometry.dataType = r.outputType[output];
            geometry.dims = 2;
            geometry.maxSizes[0] = r.outputColumns[output];
            geometry.maxSizes[1] = r.tokens;
        }
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    for (unsigned input = 0; input < kernel.inputs; ++input) {
        std::memset(&instance->inputTensorAccessPattern[input], 0, sizeof(TensorAccessPattern));
    }
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        std::memset(&instance->outputTensorAccessPattern[output], 0, sizeof(TensorAccessPattern));
    }
    TensorAccessPattern* inputPattern = instance->inputTensorAccessPattern;
    TensorAccessPattern* outputPattern = instance->outputTensorAccessPattern;

    // Every kernel: one index-space point is one token and owns the token's row of every output.
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = r.tokens;
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        MapRow(outputPattern[output], 0, r.outputColumns[output] - 1);
    }
    // An input is declared as the token's whole row: the last 128-lane load of a row reads what
    // follows the columns the kernel needs. combine reads one element of gates.
    for (unsigned input = 0; input < kernel.inputs; ++input) {
        MapRow(inputPattern[input], 0, in[input]->maxSizes[0] - 1);
    }
    if (kernel.kind == Kind::Combine) {
        const uint64_t column = static_cast<uint64_t>(scalar.gate_column);
        MapRow(inputPattern[1], column, column);
    }

    MoeGlueKernelArgs args{};
    args.experts = scalar.experts;
    args.top_k = scalar.top_k;
    args.gate_column = scalar.gate_column;
    // row_quant: the reciprocal of fp8_max, rounded to bf16 (0x1.12p-8 for 240). The torch path this
    // kernel replaces divides a bf16 tensor by FP8_MAX, and a Gaudi2 card computes that as a product with
    // the reciprocal rounded to bf16, not as a quotient. The kernel has to return the card's scale: with
    // the unrounded reciprocal its scale was CPU torch's, one bf16 step below the card's in two rows of
    // three (first card run, 2026-11-20; tests/unit_tests/hc_fused/hpu_arithmetic.py).
    args.inv_fp8_max = kernel.kind == Kind::RowQuant ? RoundToBf16(1.0f / scalar.fp8_max) : 0.0f;
    instance->kernel.paramsNr = sizeof(MoeGlueKernelArgs) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));

    const Elf elf = KernelElf(kernel.kind);
    if (!elf.start) return GLUE_NODE_NOT_FOUND;
    const unsigned elfSize = static_cast<unsigned>(elf.end - elf.start);
    const unsigned providedSize = instance->kernel.elfSize;
    instance->kernel.elfSize = elfSize;
    if (providedSize < elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, elf.start, elfSize);
    return GLUE_SUCCESS;
}

GlueCodeReturn InferShapes(const KernelInfo& kernel, const ShapeInferenceParams* params, ShapeInferenceOutput* output)
{
    if (!params || !output || !params->inputTensors || !output->outputTensors) return GLUE_SIF_NULL_PTR;
    if (params->inputTensorsNr != kernel.inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorsNr != kernel.outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const TensorGeometry* in[kMaxInputs] = {};
    for (unsigned input = 0; input < kernel.inputs; ++input) {
        if (!params->inputTensors[input]) return GLUE_SIF_NULL_PTR;
        in[input] = &params->inputTensors[input]->geometry;
    }
    for (unsigned index = 0; index < kernel.outputs; ++index) {
        if (!output->outputTensors[index]) return GLUE_SIF_NULL_PTR;
    }
    if (!params->nodeParams.nodeParams || params->nodeParams.nodeParamsSize != sizeof(MoeGlueParams)) {
        return GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    MoeGlueParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;
    for (unsigned index = 0; index < kernel.outputs; ++index) {
        TensorGeometry& geometry = output->outputTensors[index]->geometry;
        geometry.dims = 2;
        geometry.maxSizes[0] = r.outputColumns[index];
        geometry.maxSizes[1] = r.tokens;
    }
    return GLUE_SUCCESS;
}

}  // namespace moe_glue
