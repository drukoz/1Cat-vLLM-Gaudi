// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include <tuple>
#include "hpu_custom_op_pt2.h"
#include "../hc_fused_params.h"

namespace {
// The schemas and the fake kernels are defined in Python (flashinfer_gaudi/hc_fused.py, imported by
// flashinfer_gaudi._native.load_hc_fused_extension before this library is loaded); this file
// registers only the HPU kernels. There is no CPU kernel outside the tests
// (tests/unit_tests/hc_fused/hc_ops.py adds the torch oracle). Schemas, in argument order:
//   hc_norm(Tensor x, Tensor norm_weight, float eps, int hc_count) -> Tensor
//   hc_silu(Tensor down, int hc_count) -> Tensor
//   hc_silu_split(Tensor down, int hc_count, int lora_rank) -> (Tensor, Tensor)
//   hc_gate_mix(Tensor xn, Tensor gate, int hc_count) -> Tensor
//   hc_combine(Tensor residual, Tensor block_output, Tensor injection, int hc_count) -> Tensor
//   hc_combine_norm(Tensor residual, Tensor block_output, Tensor injection, Tensor norm_weight,
//                   float eps, int hc_count) -> (Tensor, Tensor)
constexpr auto kNormSchema = "flashinfer_gaudi::hc_norm";
constexpr auto kSiluSchema = "flashinfer_gaudi::hc_silu";
constexpr auto kSiluSplitSchema = "flashinfer_gaudi::hc_silu_split";
constexpr auto kGateMixSchema = "flashinfer_gaudi::hc_gate_mix";
constexpr auto kCombineSchema = "flashinfer_gaudi::hc_combine";
constexpr auto kCombineNormSchema = "flashinfer_gaudi::hc_combine_norm";
// The interleaved kernels (docs/design/flash-next/hc-kernels.md): the schemas of hc_gate_mix and
// hc_combine_norm under their own names, served by the GUIDs kHcFusedGateMixIlGuid and
// kHcFusedCombineNormIlGuid (each optional in the kernel library: flashinfer_gaudi._native says whether it is
// served, and the layer asks before it calls one):
//   hc_gate_mix_il(Tensor xn, Tensor gate, int hc_count) -> Tensor
//   hc_combine_norm_il(Tensor residual, Tensor block_output, Tensor injection, Tensor norm_weight,
//                      float eps, int hc_count) -> (Tensor, Tensor)
constexpr auto kGateMixIlSchema = "flashinfer_gaudi::hc_gate_mix_il";
constexpr auto kCombineNormIlSchema = "flashinfer_gaudi::hc_combine_norm_il";
constexpr int64_t kVectorLanes = kHcFusedVectorLanes;
constexpr int64_t kMaxStreams = 64;

// The MoE glue ops of the same extension (schemas and fake kernels: flashinfer_gaudi/moe_glue.py,
// imported by the same loader before this library is loaded). Schemas, in argument order:
//   moe_route(Tensor gates, int experts, int top_k) -> (Tensor, Tensor)       int32 ids, bf16 weights
//   moe_silu_mul(Tensor gate_up, int first_column=0) -> Tensor
//   moe_combine(Tensor shared, Tensor gates, Tensor routed, int gate_column) -> Tensor
//   moe_row_quant(Tensor x, float fp8_max) -> (Tensor, Tensor)                fp8 values, fp32 scale
constexpr auto kMoeRouteSchema = "flashinfer_gaudi::moe_route";
constexpr auto kMoeSiluMulSchema = "flashinfer_gaudi::moe_silu_mul";
constexpr auto kMoeCombineSchema = "flashinfer_gaudi::moe_combine";
constexpr auto kMoeRowQuantSchema = "flashinfer_gaudi::moe_row_quant";

// A [tokens, columns] activation: 2-D, bf16, contiguous, not empty.
void check_matrix(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "HC fused ", name, " must be bf16");
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0, "HC fused ", name,
                " must be 2-D [tokens, columns] with tokens > 0 and columns > 0");
    TORCH_CHECK(tensor.is_contiguous(), "HC fused ", name, " must be contiguous");
}

void check_same_device(const at::Tensor& first, const at::Tensor& other, const char* name) {
    TORCH_CHECK(other.device() == first.device(), "HC fused ", name, " must be on the device of the first input");
}

void check_hc_count(int64_t hc_count) {
    TORCH_CHECK(hc_count >= 1 && hc_count <= kMaxStreams, "HC fused hc_count must be in [1, ", kMaxStreams, "]");
}

void check_eps(double eps) {
    TORCH_CHECK(std::isnormal(static_cast<float>(eps)) && eps > 0, "HC fused eps must be a positive normal FP32 value");
}

// The hidden size of a multi-stream row of `width` columns: hc_count streams of whole 128-lane vectors.
int64_t hidden_size(int64_t width, int64_t hc_count, const char* name) {
    check_hc_count(hc_count);
    TORCH_CHECK(width % hc_count == 0, "HC fused ", name, " must have hc_count * hidden columns");
    const int64_t hidden = width / hc_count;
    TORCH_CHECK(hidden % kVectorLanes == 0, "HC fused hidden size must be a multiple of ", kVectorLanes,
                " (one TPC vector), got ", hidden);
    return hidden;
}

void check_norm_weight(const at::Tensor& norm_weight, const at::Tensor& first, int64_t hidden, int64_t width) {
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "HC fused norm_weight must be bf16");
    TORCH_CHECK(norm_weight.dim() == 1 && (norm_weight.size(0) == hidden || norm_weight.size(0) == width),
                "HC fused norm_weight must be 1-D with hidden or hc_count * hidden elements");
    TORCH_CHECK(norm_weight.is_contiguous(), "HC fused norm_weight must be contiguous");
    check_same_device(first, norm_weight, "norm_weight");
}

void validate_norm(const at::Tensor& x, const at::Tensor& norm_weight, double eps, int64_t hc_count) {
    check_matrix(x, "x");
    const int64_t hidden = hidden_size(x.size(1), hc_count, "x");
    check_norm_weight(norm_weight, x, hidden, x.size(1));
    check_eps(eps);
}

void validate_silu(const at::Tensor& down, int64_t hc_count) {
    check_matrix(down, "down");
    check_hc_count(hc_count);
}

void validate_silu_split(const at::Tensor& down, int64_t hc_count, int64_t lora_rank) {
    check_matrix(down, "down");
    check_hc_count(hc_count);
    TORCH_CHECK(lora_rank >= 1 && lora_rank + hc_count <= down.size(1),
                "HC fused lora_rank must be >= 1 and lora_rank + hc_count must fit the columns of down");
}

void validate_gate_mix(const at::Tensor& xn, const at::Tensor& gate, int64_t hc_count) {
    check_matrix(xn, "xn");
    hidden_size(xn.size(1), hc_count, "xn");
    check_matrix(gate, "gate");
    TORCH_CHECK(gate.sizes() == xn.sizes(), "HC fused gate must have the shape of xn");
    check_same_device(xn, gate, "gate");
}

int64_t validate_combine(const at::Tensor& residual, const at::Tensor& block_output, const at::Tensor& injection,
                         int64_t hc_count) {
    check_matrix(residual, "residual");
    const int64_t hidden = hidden_size(residual.size(1), hc_count, "residual");
    check_matrix(block_output, "block_output");
    TORCH_CHECK(block_output.size(0) == residual.size(0) && block_output.size(1) == hidden,
                "HC fused block_output must be [tokens, hidden]");
    check_matrix(injection, "injection");
    TORCH_CHECK(injection.size(0) == residual.size(0) && injection.size(1) == hc_count,
                "HC fused injection must be [tokens, hc_count]");
    check_same_device(residual, block_output, "block_output");
    check_same_device(residual, injection, "injection");
    return hidden;
}

void validate_combine_norm(const at::Tensor& residual, const at::Tensor& block_output, const at::Tensor& injection,
                           const at::Tensor& norm_weight, double eps, int64_t hc_count) {
    const int64_t hidden = validate_combine(residual, block_output, injection, hc_count);
    check_norm_weight(norm_weight, residual, hidden, residual.size(1));
    check_eps(eps);
}

habana::PartialOutputMetaData bf16_output(std::vector<int64_t> shape) {
    return habana::PartialOutputMetaData{at::kBFloat16, std::move(shape)};
}

std::shared_ptr<void> node_params(size_t& size, double eps, int64_t hc_count, int64_t lora_rank) {
    size = sizeof(HcFusedParams);
    return std::make_shared<HcFusedParams>(
        HcFusedParams{static_cast<float>(eps), static_cast<int32_t>(hc_count), static_cast<int32_t>(lora_rank)});
}

const bool norm_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kNormSchema, kHcFusedNormGuid,
        [](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            validate_norm(x, stack.at(1).toTensor(), stack.at(2).toDouble(), stack.at(3).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(x.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, stack.at(2).toDouble(), stack.at(3).toInt(), 0);
        });
    return true;
}();

const bool silu_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSiluSchema, kHcFusedSiluGuid,
        [](const at::Stack& stack) {
            const auto& down = stack.at(0).toTensor();
            validate_silu(down, stack.at(1).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(down.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, stack.at(1).toInt(), 0);
        });
    return true;
}();

const bool silu_split_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSiluSplitSchema, kHcFusedSiluSplitGuid,
        [](const at::Stack& stack) {
            const auto& down = stack.at(0).toTensor();
            const int64_t hc_count = stack.at(1).toInt();
            const int64_t lora_rank = stack.at(2).toInt();
            validate_silu_split(down, hc_count, lora_rank);
            return habana::PartialOutputMetaDataVector{bf16_output({down.size(0), lora_rank}),
                                                       bf16_output({down.size(0), hc_count})};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, stack.at(1).toInt(), stack.at(2).toInt());
        });
    return true;
}();

const bool gate_mix_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kGateMixSchema, kHcFusedGateMixGuid,
        [](const at::Stack& stack) {
            const auto& xn = stack.at(0).toTensor();
            const int64_t hc_count = stack.at(2).toInt();
            validate_gate_mix(xn, stack.at(1).toTensor(), hc_count);
            return habana::PartialOutputMetaDataVector{bf16_output({xn.size(0), xn.size(1) / hc_count})};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, stack.at(2).toInt(), 0);
        });
    return true;
}();

const bool combine_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kCombineSchema, kHcFusedCombineGuid,
        [](const at::Stack& stack) {
            const auto& residual = stack.at(0).toTensor();
            validate_combine(residual, stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(residual.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, stack.at(3).toInt(), 0);
        });
    return true;
}();

const bool combine_norm_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kCombineNormSchema, kHcFusedCombineNormGuid,
        [](const at::Stack& stack) {
            const auto& residual = stack.at(0).toTensor();
            validate_combine_norm(residual, stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toTensor(),
                                  stack.at(4).toDouble(), stack.at(5).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(residual.sizes().vec()),
                                                       bf16_output(residual.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, stack.at(4).toDouble(), stack.at(5).toInt(), 0);
        });
    return true;
}();

const bool gate_mix_il_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kGateMixIlSchema, kHcFusedGateMixIlGuid,
        [](const at::Stack& stack) {
            const auto& xn = stack.at(0).toTensor();
            const int64_t hc_count = stack.at(2).toInt();
            validate_gate_mix(xn, stack.at(1).toTensor(), hc_count);
            return habana::PartialOutputMetaDataVector{bf16_output({xn.size(0), xn.size(1) / hc_count})};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, 0.0, stack.at(2).toInt(), 0);
        });
    return true;
}();

const bool combine_norm_il_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kCombineNormIlSchema, kHcFusedCombineNormIlGuid,
        [](const at::Stack& stack) {
            const auto& residual = stack.at(0).toTensor();
            validate_combine_norm(residual, stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toTensor(),
                                  stack.at(4).toDouble(), stack.at(5).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(residual.sizes().vec()),
                                                       bf16_output(residual.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return node_params(size, stack.at(4).toDouble(), stack.at(5).toInt(), 0);
        });
    return true;
}();

// ---- MoE glue ops: validation, output metadata and node parameters ---------------------------
void validate_moe_route(const at::Tensor& gates, int64_t experts, int64_t top_k) {
    check_matrix(gates, "gates");
    TORCH_CHECK(top_k >= 1 && top_k <= kMoeGlueMaxTopK, "MoE glue top_k must be in [1, ", kMoeGlueMaxTopK, "]");
    TORCH_CHECK(experts >= top_k && experts <= kMoeGlueMaxExperts && experts <= gates.size(1),
                "MoE glue experts must be at least top_k, at most ", kMoeGlueMaxExperts,
                " and at most the columns of gates");
}

// first_column: where the gate_up block starts in the input (0: the input is the gate_up GEMM output;
// experts + 1: the gate_up GEMM is merged into the gate GEMM). The block is the rest of the row.
void validate_moe_silu_mul(const at::Tensor& gate_up, int64_t first_column) {
    check_matrix(gate_up, "gate_up");
    TORCH_CHECK(first_column >= 0 && first_column < gate_up.size(1),
                "MoE glue first_column must be a column of gate_up");
    TORCH_CHECK((gate_up.size(1) - first_column) % 2 == 0,
                "MoE glue gate_up must have an even number of columns from first_column on");
}

void validate_moe_combine(const at::Tensor& shared, const at::Tensor& gates, const at::Tensor& routed,
                          int64_t gate_column) {
    check_matrix(shared, "shared");
    TORCH_CHECK(shared.size(1) % kVectorLanes == 0, "MoE glue hidden size must be a multiple of ", kVectorLanes,
                " (one TPC vector), got ", shared.size(1));
    check_matrix(gates, "gates");
    TORCH_CHECK(gates.size(0) == shared.size(0), "MoE glue gates must have the rows of shared");
    TORCH_CHECK(gate_column >= 0 && gate_column < gates.size(1), "MoE glue gate_column must be a column of gates");
    check_matrix(routed, "routed");
    TORCH_CHECK(routed.sizes() == shared.sizes(), "MoE glue routed must have the shape of shared");
    check_same_device(shared, gates, "gates");
    check_same_device(shared, routed, "routed");
}

void validate_moe_row_quant(const at::Tensor& x, double fp8_max) {
    check_matrix(x, "x");
    TORCH_CHECK(x.size(1) % kVectorLanes == 0, "MoE glue hidden size must be a multiple of ", kVectorLanes,
                " (one TPC vector), got ", x.size(1));
    TORCH_CHECK(static_cast<float>(fp8_max) == kMoeGlueFp8Max, "MoE glue fp8_max must be ", kMoeGlueFp8Max,
                " (the Gaudi2 fp8 range), got ", fp8_max);
}

std::shared_ptr<void> moe_node_params(size_t& size, int64_t experts, int64_t top_k, int64_t gate_column,
                                      double fp8_max) {
    size = sizeof(MoeGlueParams);
    return std::make_shared<MoeGlueParams>(MoeGlueParams{static_cast<int32_t>(experts), static_cast<int32_t>(top_k),
                                                         static_cast<int32_t>(gate_column),
                                                         static_cast<float>(fp8_max)});
}

const bool moe_route_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kMoeRouteSchema, kMoeGlueRouteGuid,
        [](const at::Stack& stack) {
            const auto& gates = stack.at(0).toTensor();
            const int64_t top_k = stack.at(2).toInt();
            validate_moe_route(gates, stack.at(1).toInt(), top_k);
            return habana::PartialOutputMetaDataVector{
                habana::PartialOutputMetaData{at::kInt, {gates.size(0), top_k}}, bf16_output({gates.size(0), top_k})};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return moe_node_params(size, stack.at(1).toInt(), stack.at(2).toInt(), 0, 0.0);
        });
    return true;
}();

const bool moe_silu_mul_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kMoeSiluMulSchema, kMoeGlueSiluMulGuid,
        [](const at::Stack& stack) {
            const auto& gate_up = stack.at(0).toTensor();
            const int64_t first_column = stack.at(1).toInt();
            validate_moe_silu_mul(gate_up, first_column);
            return habana::PartialOutputMetaDataVector{
                bf16_output({gate_up.size(0), (gate_up.size(1) - first_column) / 2})};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return moe_node_params(size, 0, 0, stack.at(1).toInt(), 0.0);
        });
    return true;
}();

const bool moe_combine_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kMoeCombineSchema, kMoeGlueCombineGuid,
        [](const at::Stack& stack) {
            const auto& shared = stack.at(0).toTensor();
            validate_moe_combine(shared, stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toInt());
            return habana::PartialOutputMetaDataVector{bf16_output(shared.sizes().vec())};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return moe_node_params(size, 0, 0, stack.at(3).toInt(), 0.0);
        });
    return true;
}();

const bool moe_row_quant_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kMoeRowQuantSchema, kMoeGlueRowQuantGuid,
        [](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            validate_moe_row_quant(x, stack.at(1).toDouble());
            return habana::PartialOutputMetaDataVector{
                habana::PartialOutputMetaData{at::ScalarType::Float8_e4m3fn, x.sizes().vec()},
                habana::PartialOutputMetaData{at::kFloat, {x.size(0), 1}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            return moe_node_params(size, 0, 0, 0, stack.at(1).toDouble());
        });
    return true;
}();

std::vector<at::Tensor> execute(const char* schema, const at::Tensor& first, std::vector<c10::IValue> inputs,
                                size_t outputs) {
    TORCH_CHECK(first.device().type() == c10::DeviceType::HPU, "HC fused ops require HPU inputs");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<at::Tensor> result = descriptor.execute(inputs);
    TORCH_CHECK(result.size() == outputs, "HC fused op ", schema, " returned ", result.size(), " outputs, expected ",
                outputs);
    return result;
}

at::Tensor run_norm(const at::Tensor& x, const at::Tensor& norm_weight, double eps, int64_t hc_count) {
    validate_norm(x, norm_weight, eps, hc_count);
    TORCH_CHECK(norm_registered);
    return execute(kNormSchema, x, {x, norm_weight, eps, hc_count}, 1).at(0);
}

at::Tensor run_silu(const at::Tensor& down, int64_t hc_count) {
    validate_silu(down, hc_count);
    TORCH_CHECK(silu_registered);
    return execute(kSiluSchema, down, {down, hc_count}, 1).at(0);
}

std::tuple<at::Tensor, at::Tensor> run_silu_split(const at::Tensor& down, int64_t hc_count, int64_t lora_rank) {
    validate_silu_split(down, hc_count, lora_rank);
    TORCH_CHECK(silu_split_registered);
    const auto outputs = execute(kSiluSplitSchema, down, {down, hc_count, lora_rank}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}

at::Tensor run_gate_mix(const at::Tensor& xn, const at::Tensor& gate, int64_t hc_count) {
    validate_gate_mix(xn, gate, hc_count);
    TORCH_CHECK(gate_mix_registered);
    return execute(kGateMixSchema, xn, {xn, gate, hc_count}, 1).at(0);
}

at::Tensor run_combine(const at::Tensor& residual, const at::Tensor& block_output, const at::Tensor& injection,
                       int64_t hc_count) {
    validate_combine(residual, block_output, injection, hc_count);
    TORCH_CHECK(combine_registered);
    return execute(kCombineSchema, residual, {residual, block_output, injection, hc_count}, 1).at(0);
}

std::tuple<at::Tensor, at::Tensor> run_combine_norm(const at::Tensor& residual, const at::Tensor& block_output,
                                                    const at::Tensor& injection, const at::Tensor& norm_weight,
                                                    double eps, int64_t hc_count) {
    validate_combine_norm(residual, block_output, injection, norm_weight, eps, hc_count);
    TORCH_CHECK(combine_norm_registered);
    const auto outputs = execute(kCombineNormSchema, residual,
                                 {residual, block_output, injection, norm_weight, eps, hc_count}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}

at::Tensor run_gate_mix_il(const at::Tensor& xn, const at::Tensor& gate, int64_t hc_count) {
    validate_gate_mix(xn, gate, hc_count);
    TORCH_CHECK(gate_mix_il_registered);
    return execute(kGateMixIlSchema, xn, {xn, gate, hc_count}, 1).at(0);
}

std::tuple<at::Tensor, at::Tensor> run_combine_norm_il(const at::Tensor& residual, const at::Tensor& block_output,
                                                       const at::Tensor& injection, const at::Tensor& norm_weight,
                                                       double eps, int64_t hc_count) {
    validate_combine_norm(residual, block_output, injection, norm_weight, eps, hc_count);
    TORCH_CHECK(combine_norm_il_registered);
    const auto outputs = execute(kCombineNormIlSchema, residual,
                                 {residual, block_output, injection, norm_weight, eps, hc_count}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> run_moe_route(const at::Tensor& gates, int64_t experts, int64_t top_k) {
    validate_moe_route(gates, experts, top_k);
    TORCH_CHECK(moe_route_registered);
    const auto outputs = execute(kMoeRouteSchema, gates, {gates, experts, top_k}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}

at::Tensor run_moe_silu_mul(const at::Tensor& gate_up, int64_t first_column) {
    validate_moe_silu_mul(gate_up, first_column);
    TORCH_CHECK(moe_silu_mul_registered);
    return execute(kMoeSiluMulSchema, gate_up, {gate_up, first_column}, 1).at(0);
}

at::Tensor run_moe_combine(const at::Tensor& shared, const at::Tensor& gates, const at::Tensor& routed,
                           int64_t gate_column) {
    validate_moe_combine(shared, gates, routed, gate_column);
    TORCH_CHECK(moe_combine_registered);
    return execute(kMoeCombineSchema, shared, {shared, gates, routed, gate_column}, 1).at(0);
}

std::tuple<at::Tensor, at::Tensor> run_moe_row_quant(const at::Tensor& x, double fp8_max) {
    validate_moe_row_quant(x, fp8_max);
    TORCH_CHECK(moe_row_quant_registered);
    const auto outputs = execute(kMoeRowQuantSchema, x, {x, fp8_max}, 2);
    return std::make_tuple(outputs.at(0), outputs.at(1));
}
}  // namespace

TORCH_LIBRARY_IMPL(flashinfer_gaudi, HPU, m) {
    m.impl("moe_route", run_moe_route);
    m.impl("moe_silu_mul", run_moe_silu_mul);
    m.impl("moe_combine", run_moe_combine);
    m.impl("moe_row_quant", run_moe_row_quant);
    m.impl("hc_norm", run_norm);
    m.impl("hc_silu", run_silu);
    m.impl("hc_silu_split", run_silu_split);
    m.impl("hc_gate_mix", run_gate_mix);
    m.impl("hc_combine", run_combine);
    m.impl("hc_combine_norm", run_combine_norm);
    m.impl("hc_gate_mix_il", run_gate_mix_il);
    m.impl("hc_combine_norm_il", run_combine_norm_il);
}
