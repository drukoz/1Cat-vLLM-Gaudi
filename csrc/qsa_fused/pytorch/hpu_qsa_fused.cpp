// SPDX-License-Identifier: Apache-2.0
// HPU kernels of the fused QSA decode ops (torch.ops.flashinfer_gaudi.qsa_index_prepare / qsa_index_emit).
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>
#include "hpu_custom_op_pt2.h"
#include "../qsa_fused_params.h"

namespace {
// The schemas and the fake kernels are defined in Python (flashinfer_gaudi/qsa_fused.py, imported by
// flashinfer_gaudi._native.load_qsa_fused_extension before this library is loaded); this file registers only the
// HPU kernels. There is no CPU kernel outside the tests (tests/unit_tests/qsa_fused/qsa_ops.py adds the references).
//   qsa_index_prepare(Tensor key_ring, Tensor qk, Tensor rows, Tensor q_norm_weight, Tensor k_norm_weight,
//                     Tensor cos_sin, float eps, int heads, int ring_rows, int ratio) -> (Tensor, Tensor)
//                     (q_idx and pooled; no input is written: card rule 10, the bridge's custom-op API declares no
//                     in-place output and a kernel's stores into an input are lost on the card)
//   qsa_index_emit(Tensor blocks, Tensor positions, Tensor block_table, int ratio, int block_size)
//                  -> (Tensor, Tensor)
constexpr auto kPrepareSchema = "flashinfer_gaudi::qsa_index_prepare";
constexpr auto kEmitSchema = "flashinfer_gaudi::qsa_index_emit";
constexpr auto kThresholdSchema = "flashinfer_gaudi::qsa_index_threshold";
constexpr auto kSelectSchema = "flashinfer_gaudi::qsa_index_select";

bool power_of_two(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

void check_tensor(const at::Tensor& t, at::ScalarType type, int64_t dims, const char* name) {
    TORCH_CHECK(t.scalar_type() == type, "QSA fused ", name, " has the wrong dtype");
    TORCH_CHECK(t.dim() == dims, "QSA fused ", name, " must be ", dims, "-D");
    TORCH_CHECK(t.numel() > 0, "QSA fused ", name, " must not be empty");
    TORCH_CHECK(t.is_contiguous(), "QSA fused ", name, " must be contiguous");
}

void check_same_device(const at::Tensor& first, const at::Tensor& other, const char* name) {
    TORCH_CHECK(other.device() == first.device(), "QSA fused ", name, " must be on the device of the first input");
}

void validate_prepare(const at::Tensor& key_ring, const at::Tensor& qk, const at::Tensor& rows,
                      const at::Tensor& q_norm_weight, const at::Tensor& k_norm_weight, const at::Tensor& cos_sin,
                      double eps, int64_t heads, int64_t ring_rows, int64_t ratio) {
    check_tensor(key_ring, at::kBFloat16, 2, "key_ring");
    check_tensor(qk, at::kBFloat16, 2, "qk");
    check_tensor(rows, at::kInt, 2, "rows");
    check_tensor(q_norm_weight, at::kBFloat16, 1, "q_norm_weight");
    check_tensor(k_norm_weight, at::kBFloat16, 1, "k_norm_weight");
    check_tensor(cos_sin, at::kBFloat16, 2, "cos_sin");
    for (const auto* t : {&qk, &rows, &q_norm_weight, &k_norm_weight, &cos_sin}) {
        check_same_device(key_ring, *t, "every tensor");
    }
    const int64_t dim = key_ring.size(1);
    TORCH_CHECK(dim >= 1 && dim <= kQsaFusedMaxDim, "QSA fused head dim must be in [1, ", kQsaFusedMaxDim, "]");
    TORCH_CHECK(q_norm_weight.size(0) == dim && k_norm_weight.size(0) == dim,
                "QSA fused key_ring and the weights must share the head dim");
    TORCH_CHECK(heads >= 1 && heads <= kQsaFusedMaxHeads, "QSA fused heads must be in [1, ", kQsaFusedMaxHeads, "]");
    TORCH_CHECK(qk.size(1) == (heads + 1) * dim, "QSA fused qk must be [T, (heads + 1) * D]");
    TORCH_CHECK(rows.size(0) == qk.size(0) && rows.size(1) == 4, "QSA fused rows must be [T, 4]");
    TORCH_CHECK(power_of_two(ratio) && ratio >= 2 && ratio <= kQsaFusedMaxRatio, "QSA fused ratio must be 2, 4 or 8");
    TORCH_CHECK(ring_rows >= ratio && ring_rows <= kQsaFusedMaxRingRows && key_ring.size(0) % ring_rows == 0,
                "QSA fused key_ring must hold whole slots of ring_rows >= ratio rows");
    TORCH_CHECK(cos_sin.size(1) % 2 == 0 && power_of_two(cos_sin.size(1) / 2) && cos_sin.size(1) <= dim,
                "QSA fused cos_sin must be [P, 2 * R2] with R2 a power of two and 2 * R2 <= D");
    TORCH_CHECK(std::isnormal(static_cast<float>(eps)) && eps > 0 && eps < 1, "QSA fused eps must be in (0, 1)");
}

void validate_emit(const at::Tensor& blocks, const at::Tensor& positions, const at::Tensor& block_table,
                   int64_t ratio, int64_t block_size) {
    check_tensor(blocks, at::kInt, 2, "blocks");
    check_tensor(positions, at::kInt, 1, "positions");
    check_tensor(block_table, at::kInt, 2, "block_table");
    check_same_device(blocks, positions, "positions");
    check_same_device(blocks, block_table, "block_table");
    TORCH_CHECK(positions.size(0) == blocks.size(0) && block_table.size(0) == blocks.size(0),
                "QSA fused blocks, positions and block_table must have one row per query row");
    TORCH_CHECK(power_of_two(ratio) && ratio >= 2 && ratio <= kQsaFusedMaxRatio, "QSA fused ratio must be 2, 4 or 8");
    TORCH_CHECK(power_of_two(block_size) && block_size >= ratio && block_size <= kQsaFusedMaxBlockSize,
                "QSA fused block_size must be a power of two >= ratio");
}

void validate_threshold(const at::Tensor& scores, const at::Tensor& positions, int64_t ratio, int64_t topk) {
    check_tensor(scores, at::kFloat, 2, "scores");
    check_tensor(positions, at::kInt, 1, "positions");
    check_same_device(scores, positions, "positions");
    TORCH_CHECK(positions.size(0) == scores.size(0), "QSA fused positions must have one entry per score row");
    TORCH_CHECK(scores.size(1) % 128 == 0, "QSA fused scores must have a multiple of 128 columns");
    TORCH_CHECK(power_of_two(ratio) && ratio >= 2 && ratio <= kQsaFusedMaxRatio, "QSA fused ratio must be 2, 4 or 8");
    TORCH_CHECK(topk >= 1, "QSA fused topk must be >= 1");
}

void validate_select(const at::Tensor& meta, int64_t topk) {
    check_tensor(meta, at::kInt, 2, "meta");
    TORCH_CHECK(meta.size(1) >= 12 && (meta.size(1) - 4) % 8 == 0, "QSA fused meta must be [T, 4 + N / 16]");
    TORCH_CHECK(topk >= 1, "QSA fused topk must be >= 1");
}

std::shared_ptr<void> node_params(size_t& size, double eps, int64_t heads, int64_t ring_rows, int64_t ratio,
                                  int64_t block_size, int64_t rotary_half, int64_t topk = 0) {
    size = sizeof(QsaFusedParams);
    return std::make_shared<QsaFusedParams>(QsaFusedParams{
        static_cast<float>(eps), static_cast<int32_t>(heads), static_cast<int32_t>(ring_rows),
        static_cast<int32_t>(ratio), static_cast<int32_t>(block_size), static_cast<int32_t>(rotary_half),
        static_cast<int32_t>(topk)});
}

const bool prepare_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kPrepareSchema, kQsaFusedPrepareGuid,
        [](const at::Stack& stack) {
            const auto& key_ring = stack.at(0).toTensor();
            const auto& qk = stack.at(1).toTensor();
            const int64_t heads = stack.at(7).toInt();
            validate_prepare(key_ring, qk, stack.at(2).toTensor(), stack.at(3).toTensor(), stack.at(4).toTensor(),
                             stack.at(5).toTensor(), stack.at(6).toDouble(), heads, stack.at(8).toInt(),
                             stack.at(9).toInt());
            const int64_t dim = key_ring.size(1);
            return habana::PartialOutputMetaDataVector{
                habana::PartialOutputMetaData{at::kBFloat16, {qk.size(0), heads * dim}},
                habana::PartialOutputMetaData{at::kBFloat16, {qk.size(0), dim}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, stack.at(6).toDouble(), stack.at(7).toInt(), stack.at(8).toInt(),
                               stack.at(9).toInt(), 0, stack.at(5).toTensor().size(1) / 2);
        });
    return true;
}();

const bool emit_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kEmitSchema, kQsaFusedEmitGuid,
        [](const at::Stack& stack) {
            const auto& blocks = stack.at(0).toTensor();
            const int64_t ratio = stack.at(3).toInt();
            validate_emit(blocks, stack.at(1).toTensor(), stack.at(2).toTensor(), ratio, stack.at(4).toInt());
            const int64_t width = blocks.size(1) * ratio + ratio - 1;
            return habana::PartialOutputMetaDataVector{habana::PartialOutputMetaData{at::kInt, {blocks.size(0), width}},
                                                       habana::PartialOutputMetaData{at::kInt, {blocks.size(0), width}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, 0, 0, stack.at(3).toInt(), stack.at(4).toInt(), 0);
        });
    return true;
}();

const bool threshold_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kThresholdSchema, kQsaFusedThresholdGuid,
        [](const at::Stack& stack) {
            const auto& scores = stack.at(0).toTensor();
            validate_threshold(scores, stack.at(1).toTensor(), stack.at(2).toInt(), stack.at(3).toInt());
            return habana::PartialOutputMetaDataVector{
                habana::PartialOutputMetaData{at::kInt, {scores.size(0), 4 + scores.size(1) / 16}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, 0, 0, stack.at(2).toInt(), 0, 0, stack.at(3).toInt());
        });
    return true;
}();

const bool select_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSelectSchema, kQsaFusedSelectGuid,
        [](const at::Stack& stack) {
            const auto& meta = stack.at(0).toTensor();
            const int64_t topk = stack.at(1).toInt();
            validate_select(meta, topk);
            return habana::PartialOutputMetaDataVector{habana::PartialOutputMetaData{at::kInt, {meta.size(0), topk}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, 0, 0, 0, 0, 0, stack.at(1).toInt());
        });
    return true;
}();

std::vector<at::Tensor> execute(const char* schema, const at::Tensor& first, std::vector<c10::IValue> inputs,
                                size_t outputs) {
    TORCH_CHECK(first.device().type() == c10::DeviceType::HPU, "QSA fused ops require HPU inputs");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<at::Tensor> result = descriptor.execute(inputs);
    TORCH_CHECK(result.size() == outputs, "QSA fused op ", schema, " returned ", result.size(),
                " outputs, expected ", outputs);
    return result;
}

std::tuple<at::Tensor, at::Tensor> run_prepare(const at::Tensor& key_ring, const at::Tensor& qk, const at::Tensor& rows,
                                               const at::Tensor& q_norm_weight, const at::Tensor& k_norm_weight,
                                               const at::Tensor& cos_sin, double eps, int64_t heads, int64_t ring_rows,
                                               int64_t ratio) {
    validate_prepare(key_ring, qk, rows, q_norm_weight, k_norm_weight, cos_sin, eps, heads, ring_rows, ratio);
    TORCH_CHECK(prepare_registered);
    std::vector<c10::IValue> inputs{key_ring, qk, rows, q_norm_weight, k_norm_weight, cos_sin, eps, heads, ring_rows,
                                    ratio};
    const auto outputs = execute(kPrepareSchema, key_ring, std::move(inputs), 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> run_emit(const at::Tensor& blocks, const at::Tensor& positions,
                                            const at::Tensor& block_table, int64_t ratio, int64_t block_size) {
    validate_emit(blocks, positions, block_table, ratio, block_size);
    TORCH_CHECK(emit_registered);
    const auto outputs = execute(kEmitSchema, blocks, {blocks, positions, block_table, ratio, block_size}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}
at::Tensor run_threshold(const at::Tensor& scores, const at::Tensor& positions, int64_t ratio, int64_t topk) {
    validate_threshold(scores, positions, ratio, topk);
    TORCH_CHECK(threshold_registered);
    return execute(kThresholdSchema, scores, {scores, positions, ratio, topk}, 1).at(0);
}

at::Tensor run_select(const at::Tensor& meta, int64_t topk) {
    validate_select(meta, topk);
    TORCH_CHECK(select_registered);
    return execute(kSelectSchema, meta, {meta, topk}, 1).at(0);
}
}  // namespace

TORCH_LIBRARY_IMPL(flashinfer_gaudi, HPU, m) {
    m.impl("qsa_index_prepare", run_prepare);
    m.impl("qsa_index_emit", run_emit);
    m.impl("qsa_index_threshold", run_threshold);
    m.impl("qsa_index_select", run_select);
}
