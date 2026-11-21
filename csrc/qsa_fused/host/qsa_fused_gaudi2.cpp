// SPDX-License-Identifier: Apache-2.0
// Host glue of the fused QSA decode kernels: validation, index space, access patterns, scalar arguments and
// the embedded kernel ELF, for the GUIDs of qsa_fused_params.h.
//
// Every size comes from the tensor geometry and the node parameters. Anything outside the gates of
// qsa_fused_params.h is rejected here with a GlueCodeReturn: no kernel runs on a shape it cannot handle.
//
//   prepare  inputs  key_ring {D, S*R} bf16, key_cache {D, C} bf16, qk {(H+1)*D, T} bf16, rows {4, T} i32,
//                    q_norm_weight {D} bf16, k_norm_weight {D} bf16, cos_sin {2*R2, P} bf16
//            output  q_idx {H*D, T} bf16
//   emit     inputs  blocks {topk, T} i32, positions {T} i32, block_table {MB, T} i32
//            outputs slots {topk*ratio + ratio - 1, T} i32, valid {same} i32
//   threshold inputs scores {N, T} f32 (N a multiple of 128), positions {T} i32; output meta {4 + N / 16, T} i32
//   select   input   meta {4 + 2 NW, T} i32; output blocks {topk, T} i32
//   topk     input   masked {N, T} f32 (N a multiple of 512, at most 32768); outputs ids {topk, T} i32 and the
//                    kernel's workspace {4 + N / 16, T} i32

#include <cstdint>
#include <cstring>

#include "qsa_fused_gaudi2.hpp"

extern unsigned char _binary_qsa_fused_prepare_gaudi2_o_start;
extern unsigned char _binary_qsa_fused_prepare_gaudi2_o_end;
extern unsigned char _binary_qsa_fused_emit_gaudi2_o_start;
extern unsigned char _binary_qsa_fused_emit_gaudi2_o_end;
extern unsigned char _binary_qsa_fused_threshold_gaudi2_o_start;
extern unsigned char _binary_qsa_fused_threshold_gaudi2_o_end;
extern unsigned char _binary_qsa_fused_select_gaudi2_o_start;
extern unsigned char _binary_qsa_fused_select_gaudi2_o_end;
extern unsigned char _binary_qsa_fused_topk_gaudi2_o_start;
extern unsigned char _binary_qsa_fused_topk_gaudi2_o_end;

namespace qsa_fused {
namespace {

using namespace tpc_lib_api;

constexpr unsigned kMaxInputs = 7;
constexpr unsigned kMaxOutputs = 2;

const KernelInfo kKernels[] = {
    {Kind::Prepare, kQsaFusedPrepareGuid, 7, 1},
    {Kind::Emit, kQsaFusedEmitGuid, 3, 2},
    {Kind::Threshold, kQsaFusedThresholdGuid, 2, 1},
    {Kind::Select, kQsaFusedSelectGuid, 1, 1},
    {Kind::Topk, kQsaFusedTopkGuid, 1, 2},
};

struct Elf {
    unsigned char* start;
    unsigned char* end;
};

Elf KernelElf(Kind kind)
{
    switch (kind) {
        case Kind::Prepare:
            return {&_binary_qsa_fused_prepare_gaudi2_o_start, &_binary_qsa_fused_prepare_gaudi2_o_end};
        case Kind::Emit:
            return {&_binary_qsa_fused_emit_gaudi2_o_start, &_binary_qsa_fused_emit_gaudi2_o_end};
        case Kind::Threshold:
            return {&_binary_qsa_fused_threshold_gaudi2_o_start, &_binary_qsa_fused_threshold_gaudi2_o_end};
        case Kind::Select:
            return {&_binary_qsa_fused_select_gaudi2_o_start, &_binary_qsa_fused_select_gaudi2_o_end};
        case Kind::Topk:
            return {&_binary_qsa_fused_topk_gaudi2_o_start, &_binary_qsa_fused_topk_gaudi2_o_end};
    }
    return {nullptr, nullptr};
}

bool PowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

int Log2(int64_t value)
{
    int bits = 0;
    while ((int64_t{1} << bits) < value) ++bits;
    return bits;
}

bool Is(const TensorGeometry& g, TensorDataType type, unsigned dims)
{
    if (g.dataType != type || g.dims != dims) return false;
    for (unsigned d = 0; d < dims; ++d) {
        if (g.maxSizes[d] == 0) return false;
    }
    return true;
}

// What the shapes of one node resolve to.
struct Resolved {
    uint64_t rows = 0;                         // T: the index space
    uint64_t outputColumns[kMaxOutputs] = {};  // every output is {columns, T}
    TensorDataType outputType[kMaxOutputs] = {DATA_BF16, DATA_I32};
    uint64_t dim = 0, ringRows = 0, cacheRows = 0, qkColumns = 0, cosColumns = 0, positions = 0;  // prepare
    uint64_t topk = 0, tableColumns = 0;                                                         // emit
};

// Shape and dtype validation shared by Instantiate and InferShapes. `in` has kernel.inputs entries.
GlueCodeReturn Resolve(const KernelInfo& kernel, const TensorGeometry* const* in, const QsaFusedParams& s, Resolved& r)
{
    if (kernel.kind == Kind::Prepare) {
        const TensorGeometry& ring = *in[0];
        const TensorGeometry& cache = *in[1];
        const TensorGeometry& qk = *in[2];
        const TensorGeometry& rows = *in[3];
        const TensorGeometry& qw = *in[4];
        const TensorGeometry& kw = *in[5];
        const TensorGeometry& cs = *in[6];
        if (!Is(ring, DATA_BF16, 2) || !Is(cache, DATA_BF16, 2) || !Is(qk, DATA_BF16, 2) || !Is(qw, DATA_BF16, 1) ||
            !Is(kw, DATA_BF16, 1) || !Is(cs, DATA_BF16, 2)) {
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
        if (!Is(rows, DATA_I32, 2)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!(s.eps > 0.0f) || !(s.eps < 1.0f) || s.heads < 1 || s.heads > kQsaFusedMaxHeads || !PowerOfTwo(s.ratio) ||
            s.ratio < 2 || s.ratio > kQsaFusedMaxRatio || s.ring_rows < s.ratio || s.ring_rows > kQsaFusedMaxRingRows ||
            !PowerOfTwo(s.rotary_half)) {
            return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        }
        const uint64_t dim = cache.maxSizes[0];
        const uint64_t tokens = qk.maxSizes[1];
        if (dim > kQsaFusedMaxDim || ring.maxSizes[0] != dim || qw.maxSizes[0] != dim || kw.maxSizes[0] != dim) {
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        if (ring.maxSizes[1] % static_cast<uint64_t>(s.ring_rows) != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (qk.maxSizes[0] != (static_cast<uint64_t>(s.heads) + 1) * dim) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (rows.maxSizes[0] != 4 || rows.maxSizes[1] != tokens) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (cs.maxSizes[0] != 2 * static_cast<uint64_t>(s.rotary_half) || cs.maxSizes[0] > dim) {
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        // positions and slots are int32 and the kernel's float modulo needs n < 2^24
        if (cs.maxSizes[1] > (uint64_t{1} << 24) || ring.maxSizes[1] > (uint64_t{1} << 30) ||
            cache.maxSizes[1] > (uint64_t{1} << 30)) {
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        r.rows = tokens;
        r.dim = dim;
        r.ringRows = ring.maxSizes[1];
        r.cacheRows = cache.maxSizes[1];
        r.qkColumns = qk.maxSizes[0];
        r.cosColumns = cs.maxSizes[0];
        r.positions = cs.maxSizes[1];
        r.outputColumns[0] = static_cast<uint64_t>(s.heads) * dim;
        r.outputType[0] = DATA_BF16;
        return GLUE_SUCCESS;
    }
    if (kernel.kind == Kind::Threshold) {
        const TensorGeometry& scores = *in[0];
        const TensorGeometry& positions = *in[1];
        if (!Is(scores, DATA_F32, 2) || !Is(positions, DATA_I32, 1)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (!PowerOfTwo(s.ratio) || s.ratio < 2 || s.ratio > kQsaFusedMaxRatio || s.topk < 1) {
            return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        }
        const uint64_t columns = scores.maxSizes[0];
        if (columns % 128 != 0 || columns > (uint64_t{1} << 24) || positions.maxSizes[0] != scores.maxSizes[1]) {
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        r.rows = scores.maxSizes[1];
        r.topk = static_cast<uint64_t>(s.topk);
        r.tableColumns = columns;
        r.outputColumns[0] = 4 + columns / 16;  // two bitmaps of columns / 32 words
        r.outputType[0] = DATA_I32;
        return GLUE_SUCCESS;
    }
    if (kernel.kind == Kind::Topk) {
        const TensorGeometry& masked = *in[0];
        if (!Is(masked, DATA_F32, 2)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (s.topk < 1 || s.topk > (1 << 20)) return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        const uint64_t columns = masked.maxSizes[0];
        if (columns % static_cast<uint64_t>(kQsaFusedTopkColumnStep) != 0 ||
            columns > static_cast<uint64_t>(kQsaFusedTopkMaxColumns)) {
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
        r.rows = masked.maxSizes[1];
        r.topk = static_cast<uint64_t>(s.topk);
        r.tableColumns = columns;
        r.outputColumns[0] = r.topk;
        r.outputColumns[1] = 4 + columns / 16;  // the workspace: two bitmaps of columns / 32 words
        r.outputType[0] = DATA_I32;
        r.outputType[1] = DATA_I32;
        return GLUE_SUCCESS;
    }
    if (kernel.kind == Kind::Select) {
        const TensorGeometry& meta = *in[0];
        if (!Is(meta, DATA_I32, 2)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (s.topk < 1) return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
        if (meta.maxSizes[0] < 4 + 8 || (meta.maxSizes[0] - 4) % 8 != 0) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        r.rows = meta.maxSizes[1];
        r.topk = static_cast<uint64_t>(s.topk);
        r.tableColumns = meta.maxSizes[0];
        r.outputColumns[0] = r.topk;
        r.outputType[0] = DATA_I32;
        return GLUE_SUCCESS;
    }
    const TensorGeometry& blocks = *in[0];
    const TensorGeometry& positions = *in[1];
    const TensorGeometry& table = *in[2];
    if (!Is(blocks, DATA_I32, 2) || !Is(positions, DATA_I32, 1) || !Is(table, DATA_I32, 2)) {
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (!PowerOfTwo(s.ratio) || s.ratio < 2 || s.ratio > kQsaFusedMaxRatio || !PowerOfTwo(s.block_size) ||
        s.block_size < s.ratio || s.block_size > kQsaFusedMaxBlockSize) {
        return GLUE_KERNEL_INVALID_SCALAR_ARGUMENT;
    }
    const uint64_t tokens = blocks.maxSizes[1];
    if (positions.maxSizes[0] != tokens || table.maxSizes[1] != tokens) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (blocks.maxSizes[0] > (uint64_t{1} << 20)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    r.rows = tokens;
    r.topk = blocks.maxSizes[0];
    r.tableColumns = table.maxSizes[0];
    const uint64_t width = r.topk * static_cast<uint64_t>(s.ratio) + static_cast<uint64_t>(s.ratio) - 1;
    r.outputColumns[0] = width;
    r.outputColumns[1] = width;
    r.outputType[0] = DATA_I32;
    r.outputType[1] = DATA_I32;
    return GLUE_SUCCESS;
}

void MapDimension(TensorAccessPattern& pattern, unsigned tensorDim, unsigned indexDim, uint64_t a, uint64_t startB,
                  uint64_t endB)
{
    pattern.mapping[tensorDim].indexSpaceDim = indexDim;
    pattern.mapping[tensorDim].a = static_cast<float>(a);
    pattern.mapping[tensorDim].start_b = static_cast<float>(startB);
    pattern.mapping[tensorDim].end_b = static_cast<float>(endB);
}

// Every point reads the whole tensor (data-dependent rows: the ring, the cache, other rows of the step).
void MapWhole(TensorAccessPattern& pattern, const TensorGeometry& geometry)
{
    for (unsigned d = 0; d < geometry.dims; ++d) MapDimension(pattern, d, 0, 0, 0, geometry.maxSizes[d] - 1);
}

// A {columns, T} tensor of which point t reads or writes columns [0, columns) of row t.
void MapRow(TensorAccessPattern& pattern, uint64_t columns)
{
    MapDimension(pattern, 0, 0, 0, 0, columns - 1);
    MapDimension(pattern, 1, 0, 1, 0, 0);
}

}  // namespace

const KernelInfo* Kernels() { return kKernels; }

const KernelInfo* FindKernel(const char* name)
{
    if (!name) return nullptr;
    for (const KernelInfo& kernel : kKernels) {
        if (!std::strncmp(name, kernel.guid, MAX_NODE_NAME)) return &kernel;
    }
    return nullptr;
}

const KernelInfo* KernelForCounts(unsigned inputs, unsigned outputs)
{
    for (const KernelInfo& kernel : kKernels) {
        if (kernel.inputs == inputs && kernel.outputs == outputs) return &kernel;
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
        params->nodeParams.nodeParamsSize != sizeof(QsaFusedParams)) {
        return GLUE_FAILED;
    }
    QsaFusedParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const TensorGeometry* in[kMaxInputs] = {};
    for (unsigned input = 0; input < kernel.inputs; ++input) in[input] = &params->inputTensors[input].geometry;
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;

    bool outputsMatch = true;
    for (unsigned output = 0; output < kernel.outputs; ++output) {
        const TensorGeometry& g = params->outputTensors[output].geometry;
        outputsMatch = outputsMatch && g.dataType == r.outputType[output] && g.dims == 2 &&
                       g.maxSizes[0] == r.outputColumns[output] && g.maxSizes[1] == r.rows;
    }
    if (!outputsMatch) {
        for (unsigned output = 0; output < kernel.outputs; ++output) {
            TensorGeometry& g = params->outputTensors[output].geometry;
            g.dataType = r.outputType[output];
            g.dims = 2;
            g.maxSizes[0] = r.outputColumns[output];
            g.maxSizes[1] = r.rows;
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
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = r.rows;

    if (kernel.kind == Kind::Prepare) {
        // The ring and the cache are read and written at data-dependent rows, the row table and qk of every
        // row are read by the write-then-read search, cos_sin at data-dependent positions: whole tensors.
        for (unsigned input = 0; input < kernel.inputs; ++input) MapWhole(inputPattern[input], *in[input]);
        MapRow(outputPattern[0], r.outputColumns[0]);
        QsaFusedPrepareArgs args{};
        args.eps = scalar.eps;
        args.inv_dim = 1.0f / static_cast<float>(r.dim);
        args.inv_ring = 1.0f / static_cast<float>(scalar.ring_rows);
        args.inv_ratio = 1.0f / static_cast<float>(scalar.ratio);
        args.heads = scalar.heads;
        args.ring_rows = scalar.ring_rows;
        args.ratio = scalar.ratio;
        args.rotary_half = scalar.rotary_half;
        instance->kernel.paramsNr = sizeof(args) / sizeof(uint32_t);
        std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));
    } else if (kernel.kind == Kind::Topk) {
        MapRow(inputPattern[0], r.tableColumns);  // the row's masked scores
        MapRow(outputPattern[0], r.outputColumns[0]);
        MapRow(outputPattern[1], r.outputColumns[1]);
        QsaFusedTopkArgs args{};
        args.topk = scalar.topk;
        instance->kernel.paramsNr = sizeof(args) / sizeof(uint32_t);
        std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));
    } else if (kernel.kind == Kind::Threshold || kernel.kind == Kind::Select) {
        if (kernel.kind == Kind::Threshold) {
            MapRow(inputPattern[0], r.tableColumns);  // the row's scores
            MapDimension(inputPattern[1], 0, 0, 1, 0, 0);
        } else {
            MapRow(inputPattern[0], r.tableColumns);  // the row's meta
        }
        MapRow(outputPattern[0], r.outputColumns[0]);
        QsaFusedSelectArgs args{};
        args.ratio_log2 = Log2(scalar.ratio);
        args.topk = scalar.topk;
        instance->kernel.paramsNr = sizeof(args) / sizeof(uint32_t);
        std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));
    } else {
        MapRow(inputPattern[0], r.topk);
        MapDimension(inputPattern[1], 0, 0, 1, 0, 0);
        MapRow(inputPattern[2], r.tableColumns);
        MapRow(outputPattern[0], r.outputColumns[0]);
        MapRow(outputPattern[1], r.outputColumns[1]);
        QsaFusedEmitArgs args{};
        args.ratio_log2 = Log2(scalar.ratio);
        args.block_log2 = Log2(scalar.block_size);
        args.topk = static_cast<int32_t>(r.topk);
        instance->kernel.paramsNr = sizeof(args) / sizeof(uint32_t);
        std::memcpy(instance->kernel.scalarParams, &args, sizeof(args));
    }

    const Elf elf = KernelElf(kernel.kind);
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
    if (!params->nodeParams.nodeParams || params->nodeParams.nodeParamsSize != sizeof(QsaFusedParams)) {
        return GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    QsaFusedParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    Resolved r;
    const GlueCodeReturn resolved = Resolve(kernel, in, scalar, r);
    if (resolved != GLUE_SUCCESS) return resolved;
    for (unsigned index = 0; index < kernel.outputs; ++index) {
        TensorGeometry& g = output->outputTensors[index]->geometry;
        g.dataType = r.outputType[index];
        g.dims = 2;
        g.maxSizes[0] = r.outputColumns[index];
        g.maxSizes[1] = r.rows;
    }
    return GLUE_SUCCESS;
}

}  // namespace qsa_fused
