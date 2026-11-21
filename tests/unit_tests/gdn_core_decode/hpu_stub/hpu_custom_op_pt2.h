// Test-only stand-in for Habana's hpu_custom_op_pt2.h: records the callbacks that
// registerUserCustomOp receives so a CPU test can call them. Never used in a card build.
#pragma once
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace at {
using Stack = torch::jit::Stack;
}

namespace habana {
struct PartialOutputMetaData {
    c10::ScalarType dtype;
    std::vector<int64_t> shape;
};
using PartialOutputMetaDataVector = std::vector<PartialOutputMetaData>;

namespace custom_op {
using OutputMetaFn = std::function<PartialOutputMetaDataVector(const at::Stack&)>;
using FillParamsFn = std::function<std::shared_ptr<void>(const at::Stack&, size_t&)>;

struct RegisteredOp {
    std::string guid;
    OutputMetaFn output_meta;
    FillParamsFn fill_params;
};

inline std::map<std::string, RegisteredOp>& registered_ops() {
    static std::map<std::string, RegisteredOp> ops;
    return ops;
}

inline void registerUserCustomOp(const std::string& schema, const std::string& guid, OutputMetaFn output_meta,
                                 FillParamsFn fill_params) {
    registered_ops()[schema] = RegisteredOp{guid, std::move(output_meta), std::move(fill_params)};
}

struct UserCustomOpDescriptor {
    static UserCustomOpDescriptor getUserCustomOpDescriptor(const std::string&) { return {}; }
    std::vector<at::Tensor> execute(const std::vector<c10::IValue>&) {
        TORCH_CHECK(false, "the stub custom-op descriptor cannot execute");
    }
};
}  // namespace custom_op
}  // namespace habana
