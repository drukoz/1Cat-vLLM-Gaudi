// SPDX-License-Identifier: Apache-2.0
// Host-side driver for test_op_callbacks_host.py: includes the real
// csrc/hc_fused/pytorch/hpu_hc_fused.cpp against the test stub header, reads one case per line
// from the file named by argv[1] ("<schema> <case_name> <arg>...", args in schema order:
// t:<dtype>:<d0>,<d1>,... | f:<value> | i:<value>), calls the registered output_meta and
// fill_params callbacks of each case and prints one result line per case:
// "<case_name> ok <n_outputs> <dtype>:<d0>,<d1>;<dtype>:... <guid> <size> <hex>" on success, or
// "<case_name> error <first line of what()>" when a callback throws or the schema is unknown.
// Built with -fsanitize=address,undefined. Modelled on tests/unit_tests/gdn_core_decode/op_callbacks_host.cpp.
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>

#include "hpu_hc_fused.cpp"  // the real op file, found through -I csrc/hc_fused/pytorch

namespace {

at::Tensor make_tensor(const std::string& spec) {  // "<dtype>:<d0>,<d1>,..."
    const size_t sep = spec.find(':');
    if (sep == std::string::npos) throw std::invalid_argument("bad tensor arg: " + spec);
    const std::string dt = spec.substr(0, sep);
    at::ScalarType dtype;
    if (dt == "bf16") {
        dtype = at::kBFloat16;
    } else if (dt == "f32") {
        dtype = at::kFloat;
    } else if (dt == "i32") {
        dtype = at::kInt;
    } else {
        throw std::invalid_argument("unknown dtype: " + dt);
    }
    std::vector<int64_t> shape;
    std::istringstream dims(spec.substr(sep + 1));
    std::string dim;
    while (std::getline(dims, dim, ',')) shape.push_back(dim.empty() ? 0 : std::stoll(dim));
    return at::zeros(shape, at::TensorOptions().dtype(dtype));
}

at::Stack make_stack(std::istringstream& line) {
    at::Stack stack;
    std::string arg;
    while (line >> arg) {
        if (arg.size() < 2 || arg[1] != ':') throw std::invalid_argument("bad arg: " + arg);
        switch (arg[0]) {
            case 't': stack.emplace_back(make_tensor(arg.substr(2))); break;
            case 'f': stack.emplace_back(std::stod(arg.substr(2))); break;
            case 'i': stack.emplace_back(static_cast<int64_t>(std::stoll(arg.substr(2)))); break;
            default: throw std::invalid_argument("bad arg kind: " + arg);
        }
    }
    return stack;
}

const char* dtype_name(c10::ScalarType dtype) {
    switch (dtype) {
        case at::kFloat: return "f32";
        case at::kBFloat16: return "bf16";
        case at::kInt: return "i32";
        default: return "other";
    }
}

void run_case(const std::string& schema, const std::string& name, std::istringstream& line) {
    auto& ops = habana::custom_op::registered_ops();
    const auto it = ops.find(schema);
    if (it == ops.end()) {
        std::cout << name << " error unknown schema\n";
        return;
    }
    try {
        const at::Stack stack = make_stack(line);
        habana::PartialOutputMetaDataVector meta = it->second.output_meta(stack);
        if (meta.empty()) throw std::runtime_error("output_meta returned no outputs");
        size_t size = 0;
        const std::shared_ptr<void> params = it->second.fill_params(stack, size);
        std::cout << name << " ok " << meta.size() << " ";
        for (size_t output = 0; output < meta.size(); ++output) {
            std::cout << (output == 0 ? "" : ";") << dtype_name(meta[output].dtype) << ":";
            for (size_t i = 0; i < meta[output].shape.size(); ++i)
                std::cout << (i == 0 ? "" : ",") << meta[output].shape[i];
        }
        std::cout << " " << it->second.guid << " " << size << " ";
        const auto* bytes = static_cast<const unsigned char*>(params.get());
        for (size_t i = 0; i < size; ++i) std::printf("%02x", bytes[i]);
        std::cout << "\n";
    } catch (const std::exception& e) {
        const std::string what(e.what());
        std::cout << name << " error " << what.substr(0, what.find('\n')) << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <case-file>\n", argv[0]);
        return 2;
    }
    std::ifstream cases(argv[1]);
    if (!cases) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::string raw;
    while (std::getline(cases, raw)) {
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        std::istringstream line(raw);
        std::string schema, name;
        if (!(line >> schema >> name)) continue;  // blank line
        run_case(schema, name, line);
        std::cout.flush();
    }
    return 0;
}
