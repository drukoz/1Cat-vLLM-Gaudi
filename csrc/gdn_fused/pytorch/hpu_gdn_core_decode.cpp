// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_custom_op_pt2.h"
#include "../gdn_core_decode_params.h"

namespace {
// The schema, CPU kernel and fake kernel are defined in Python by
// flashinfer_gaudi/gdn_core_decode.py (gdn_core_decode, gdn_recurrent_decode) and
// flashinfer_gaudi/gdn_core_verify.py (gdn_core_verify); this file registers only the HPU kernels.
constexpr auto kSchema = "flashinfer_gaudi::gdn_core_decode";
constexpr auto kRecurrentSchema = "flashinfer_gaudi::gdn_recurrent_decode";
constexpr auto kVerifySchema = "flashinfer_gaudi::gdn_core_verify";
constexpr int64_t kHeadDim = 128;

void validate(const at::Tensor& conv_state, const at::Tensor& ssm_state, const at::Tensor& mixed_qkv,
              const at::Tensor& z, const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
              const at::Tensor& dt_bias, const at::Tensor& conv_weight_t, const at::Tensor& norm_weight,
              const at::Tensor& load_indices, const at::Tensor& store_indices, double eps, int64_t gate_activation) {
    TORCH_CHECK(gate_activation == 0 || gate_activation == 1,
                "GDN core decode gate_activation must be 0 (silu) or 1 (sigmoid)");
    TORCH_CHECK(std::isnormal(static_cast<float>(eps)) && eps > 0,
                "GDN core decode eps must be a positive normal FP32 value");
    TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && ssm_state.scalar_type() == at::kFloat &&
                mixed_qkv.scalar_type() == at::kBFloat16 && z.scalar_type() == at::kBFloat16 &&
                a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat &&
                conv_weight_t.scalar_type() == at::kBFloat16 && norm_weight.scalar_type() == at::kBFloat16 &&
                load_indices.scalar_type() == at::kInt && store_indices.scalar_type() == at::kInt,
                "GDN core decode requires bf16 activations, fp32 state, gates and int32 indices");
    for (const auto* tensor :
         {&conv_state, &ssm_state, &mixed_qkv, &z, &a, &b, &A_log, &dt_bias, &conv_weight_t, &norm_weight,
          &load_indices, &store_indices}) {
        TORCH_CHECK(tensor->is_contiguous(), "GDN core decode requires contiguous tensors");
        TORCH_CHECK(tensor->device() == conv_state.device(), "GDN core decode inputs must use the same device");
    }
    TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(0) > 0, "GDN core decode mixed_qkv must be [B,C] with B>0");
    TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.size(2) == kHeadDim && ssm_state.size(3) == kHeadDim,
                "GDN core decode ssm_state must be [S,HV,128,128]");
    const auto batch = mixed_qkv.size(0);
    const auto channels = mixed_qkv.size(1);
    const auto value_heads = ssm_state.size(1);
    const auto key_heads = (channels - value_heads * kHeadDim) / (2 * kHeadDim);
    TORCH_CHECK(key_heads > 0 && value_heads % key_heads == 0 &&
                    channels == 2 * key_heads * kHeadDim + value_heads * kHeadDim,
                "GDN core decode C must be 2*H*128 + HV*128 with H>0 and HV divisible by H");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == 3 && conv_state.size(2) == channels &&
                    conv_state.size(0) == ssm_state.size(0),
                "GDN core decode conv_state must be [S,3,C] sharing S with ssm_state");
    TORCH_CHECK(z.dim() == 2 && z.size(0) == batch && z.size(1) == value_heads * kHeadDim,
                "GDN core decode z must be [B,HV*128]");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({batch, value_heads}) &&
                    b.sizes() == at::IntArrayRef({batch, value_heads}),
                "GDN core decode a and b must be [B,HV]");
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == value_heads && dt_bias.dim() == 1 &&
                    dt_bias.size(0) == value_heads,
                "GDN core decode A_log and dt_bias must be [HV]");
    TORCH_CHECK(conv_weight_t.dim() == 2 && conv_weight_t.sizes() == at::IntArrayRef({4, channels}),
                "GDN core decode conv_weight_t must be [4,C]");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == kHeadDim, "GDN core decode norm_weight must be [128]");
    TORCH_CHECK(load_indices.dim() == 1 && load_indices.size(0) == batch && store_indices.dim() == 1 &&
                    store_indices.size(0) == batch,
                "GDN core decode load_indices and store_indices must be [B]");
}

void validate_recurrent(const at::Tensor& ssm_state, const at::Tensor& qkv, const at::Tensor& a, const at::Tensor& b,
                        const at::Tensor& A_log, const at::Tensor& dt_bias, const at::Tensor& load_indices,
                        const at::Tensor& store_indices, double scale) {
    TORCH_CHECK(std::isnormal(static_cast<float>(scale)) && scale > 0,
                "GDN recurrent decode scale must be a positive normal FP32 value");
    TORCH_CHECK(ssm_state.scalar_type() == at::kFloat && qkv.scalar_type() == at::kBFloat16 &&
                a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat &&
                load_indices.scalar_type() == at::kInt && store_indices.scalar_type() == at::kInt,
                "GDN recurrent decode requires bf16 activations, fp32 state, gates and int32 indices");
    for (const auto* tensor : {&ssm_state, &qkv, &a, &b, &A_log, &dt_bias, &load_indices, &store_indices}) {
        TORCH_CHECK(tensor->is_contiguous(), "GDN recurrent decode requires contiguous tensors");
        TORCH_CHECK(tensor->device() == ssm_state.device(), "GDN recurrent decode inputs must use the same device");
    }
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(0) > 0, "GDN recurrent decode qkv must be [B,C] with B>0");
    TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.size(2) == kHeadDim && ssm_state.size(3) == kHeadDim,
                "GDN recurrent decode ssm_state must be [S,HV,128,128]");
    const auto batch = qkv.size(0);
    const auto channels = qkv.size(1);
    const auto value_heads = ssm_state.size(1);
    const auto key_heads = (channels - value_heads * kHeadDim) / (2 * kHeadDim);
    TORCH_CHECK(key_heads > 0 && value_heads % key_heads == 0 &&
                    channels == 2 * key_heads * kHeadDim + value_heads * kHeadDim,
                "GDN recurrent decode C must be 2*H*128 + HV*128 with H>0 and HV divisible by H");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({batch, value_heads}) &&
                    b.sizes() == at::IntArrayRef({batch, value_heads}),
                "GDN recurrent decode a and b must be [B,HV]");
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == value_heads && dt_bias.dim() == 1 &&
                    dt_bias.size(0) == value_heads,
                "GDN recurrent decode A_log and dt_bias must be [HV]");
    TORCH_CHECK(load_indices.dim() == 1 && load_indices.size(0) == batch && store_indices.dim() == 1 &&
                    store_indices.size(0) == batch,
                "GDN recurrent decode load_indices and store_indices must be [B]");
}

void validate_verify(const at::Tensor& conv_state, const at::Tensor& ssm_state, const at::Tensor& mixed_qkv,
                     const at::Tensor& z, const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                     const at::Tensor& dt_bias, const at::Tensor& conv_weight_t, const at::Tensor& norm_weight,
                     const at::Tensor& row_indices, const at::Tensor& ssm_store_indices, double scale, double eps,
                     int64_t gate_activation) {
    TORCH_CHECK(gate_activation == 0 || gate_activation == 1,
                "GDN core verify gate_activation must be 0 (silu) or 1 (sigmoid)");
    TORCH_CHECK(std::isnormal(static_cast<float>(eps)) && eps > 0,
                "GDN core verify eps must be a positive normal FP32 value");
    TORCH_CHECK(std::isnormal(static_cast<float>(scale)) && scale > 0,
                "GDN core verify scale must be a positive normal FP32 value");
    TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && ssm_state.scalar_type() == at::kFloat &&
                mixed_qkv.scalar_type() == at::kBFloat16 && z.scalar_type() == at::kBFloat16 &&
                a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat &&
                conv_weight_t.scalar_type() == at::kBFloat16 && norm_weight.scalar_type() == at::kBFloat16 &&
                row_indices.scalar_type() == at::kInt && ssm_store_indices.scalar_type() == at::kInt,
                "GDN core verify requires bf16 activations, fp32 state, gates and int32 indices");
    for (const auto* tensor :
         {&conv_state, &ssm_state, &mixed_qkv, &z, &a, &b, &A_log, &dt_bias, &conv_weight_t, &norm_weight,
          &row_indices, &ssm_store_indices}) {
        TORCH_CHECK(tensor->is_contiguous(), "GDN core verify requires contiguous tensors");
        TORCH_CHECK(tensor->device() == conv_state.device(), "GDN core verify inputs must use the same device");
    }
    TORCH_CHECK(ssm_state.dim() == 4 && ssm_state.size(2) == kHeadDim && ssm_state.size(3) == kHeadDim,
                "GDN core verify ssm_state must be [S,HV,128,128]");
    TORCH_CHECK(row_indices.dim() == 2 && row_indices.size(0) > 0 && row_indices.size(1) == 5,
                "GDN core verify row_indices must be [B,5] with B>0");
    const auto batch = row_indices.size(0);
    TORCH_CHECK(ssm_store_indices.dim() == 2 && ssm_store_indices.size(0) == batch &&
                    ssm_store_indices.size(1) >= 1 && ssm_store_indices.size(1) <= kGdnCoreVerifyMaxNodes,
                "GDN core verify ssm_store_indices must be [B,T] with 1<=T<=8");
    const auto nodes = ssm_store_indices.size(1);
    TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.size(0) == batch * nodes,
                "GDN core verify mixed_qkv must be [B*T,C] with B*T = row_indices rows x ssm_store_indices columns");
    const auto tokens = mixed_qkv.size(0);
    const auto channels = mixed_qkv.size(1);
    const auto value_heads = ssm_state.size(1);
    const auto key_heads = (channels - value_heads * kHeadDim) / (2 * kHeadDim);
    TORCH_CHECK(key_heads > 0 && value_heads % key_heads == 0 &&
                    channels == 2 * key_heads * kHeadDim + value_heads * kHeadDim,
                "GDN core verify C must be 2*H*128 + HV*128 with H>0 and HV divisible by H");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(2) == channels && conv_state.size(1) >= nodes + 2 &&
                    conv_state.size(0) == ssm_state.size(0),
                "GDN core verify conv_state must be [S,R,C] with R>=T+2 sharing S with ssm_state");
    TORCH_CHECK(z.dim() == 2 && z.size(0) == tokens && z.size(1) == value_heads * kHeadDim,
                "GDN core verify z must be [B*T,HV*128]");
    TORCH_CHECK(a.sizes() == at::IntArrayRef({tokens, value_heads}) &&
                    b.sizes() == at::IntArrayRef({tokens, value_heads}),
                "GDN core verify a and b must be [B*T,HV]");
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == value_heads && dt_bias.dim() == 1 &&
                    dt_bias.size(0) == value_heads,
                "GDN core verify A_log and dt_bias must be [HV]");
    TORCH_CHECK(conv_weight_t.dim() == 2 && conv_weight_t.sizes() == at::IntArrayRef({4, channels}),
                "GDN core verify conv_weight_t must be [4,C]");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == kHeadDim, "GDN core verify norm_weight must be [128]");
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSchema, kGdnCoreDecodeGuid,
        [](const at::Stack& stack) {
            validate(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toTensor(),
                     stack.at(4).toTensor(), stack.at(5).toTensor(), stack.at(6).toTensor(), stack.at(7).toTensor(),
                     stack.at(8).toTensor(), stack.at(9).toTensor(), stack.at(10).toTensor(), stack.at(11).toTensor(),
                     stack.at(13).toDouble(), stack.at(14).toInt());
            const auto batch = stack.at(2).toTensor().size(0);
            const auto value_heads = stack.at(1).toTensor().size(1);
            return habana::PartialOutputMetaDataVector{{at::kFloat, {batch, value_heads, kHeadDim}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(GdnCoreDecodeParams);
            return std::make_shared<GdnCoreDecodeParams>(GdnCoreDecodeParams{
                static_cast<float>(stack.at(12).toDouble()), static_cast<float>(stack.at(13).toDouble()),
                static_cast<int32_t>(stack.at(14).toInt())});
        });
    return true;
}();

const bool recurrent_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kRecurrentSchema, kGdnRecurrentDecodeGuid,
        [](const at::Stack& stack) {
            validate_recurrent(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toTensor(),
                               stack.at(3).toTensor(), stack.at(4).toTensor(), stack.at(5).toTensor(),
                               stack.at(6).toTensor(), stack.at(7).toTensor(), stack.at(8).toDouble());
            const auto batch = stack.at(1).toTensor().size(0);
            const auto value_heads = stack.at(0).toTensor().size(1);
            return habana::PartialOutputMetaDataVector{{at::kFloat, {batch, value_heads, kHeadDim}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(GdnRecurrentDecodeParams);
            return std::make_shared<GdnRecurrentDecodeParams>(
                GdnRecurrentDecodeParams{static_cast<float>(stack.at(8).toDouble())});
        });
    return true;
}();

const bool verify_registered = [] {
    habana::custom_op::registerUserCustomOp(
        kVerifySchema, kGdnCoreVerifyGuid,
        [](const at::Stack& stack) {
            validate_verify(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toTensor(),
                            stack.at(3).toTensor(), stack.at(4).toTensor(), stack.at(5).toTensor(),
                            stack.at(6).toTensor(), stack.at(7).toTensor(), stack.at(8).toTensor(),
                            stack.at(9).toTensor(), stack.at(10).toTensor(), stack.at(11).toTensor(),
                            stack.at(12).toDouble(), stack.at(13).toDouble(), stack.at(14).toInt());
            const auto tokens = stack.at(2).toTensor().size(0);
            const auto value_heads = stack.at(1).toTensor().size(1);
            return habana::PartialOutputMetaDataVector{{at::kFloat, {tokens, value_heads, kHeadDim}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(GdnCoreVerifyParams);
            return std::make_shared<GdnCoreVerifyParams>(GdnCoreVerifyParams{
                static_cast<float>(stack.at(12).toDouble()), static_cast<float>(stack.at(13).toDouble()),
                static_cast<int32_t>(stack.at(14).toInt())});
        });
    return true;
}();

at::Tensor run(const at::Tensor& conv_state, const at::Tensor& ssm_state, const at::Tensor& mixed_qkv,
               const at::Tensor& z, const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
               const at::Tensor& dt_bias, const at::Tensor& conv_weight_t, const at::Tensor& norm_weight,
               const at::Tensor& load_indices, const at::Tensor& store_indices, double scale, double eps,
               int64_t gate_activation) {
    validate(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load_indices,
             store_indices, eps, gate_activation);
    TORCH_CHECK(conv_state.device().type() == c10::DeviceType::HPU, "GDN core decode requires HPU inputs");
    TORCH_CHECK(registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    const auto outputs = descriptor.execute(
        {conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, load_indices,
         store_indices, scale, eps, gate_activation});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}

at::Tensor run_recurrent(const at::Tensor& ssm_state, const at::Tensor& qkv, const at::Tensor& a, const at::Tensor& b,
                         const at::Tensor& A_log, const at::Tensor& dt_bias, const at::Tensor& load_indices,
                         const at::Tensor& store_indices, double scale) {
    validate_recurrent(ssm_state, qkv, a, b, A_log, dt_bias, load_indices, store_indices, scale);
    TORCH_CHECK(ssm_state.device().type() == c10::DeviceType::HPU, "GDN recurrent decode requires HPU inputs");
    TORCH_CHECK(recurrent_registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kRecurrentSchema);
    const auto outputs =
        descriptor.execute({ssm_state, qkv, a, b, A_log, dt_bias, load_indices, store_indices, scale});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}

at::Tensor run_verify(const at::Tensor& conv_state, const at::Tensor& ssm_state, const at::Tensor& mixed_qkv,
                      const at::Tensor& z, const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                      const at::Tensor& dt_bias, const at::Tensor& conv_weight_t, const at::Tensor& norm_weight,
                      const at::Tensor& row_indices, const at::Tensor& ssm_store_indices, double scale, double eps,
                      int64_t gate_activation) {
    validate_verify(conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, row_indices,
                    ssm_store_indices, scale, eps, gate_activation);
    TORCH_CHECK(conv_state.device().type() == c10::DeviceType::HPU, "GDN core verify requires HPU inputs");
    TORCH_CHECK(verify_registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kVerifySchema);
    const auto outputs = descriptor.execute(
        {conv_state, ssm_state, mixed_qkv, z, a, b, A_log, dt_bias, conv_weight_t, norm_weight, row_indices,
         ssm_store_indices, scale, eps, gate_activation});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
}  // namespace

TORCH_LIBRARY_IMPL(flashinfer_gaudi, HPU, m) {
    m.impl("gdn_core_decode", run);
    m.impl("gdn_recurrent_decode", run_recurrent);
    m.impl("gdn_core_verify", run_verify);
}
