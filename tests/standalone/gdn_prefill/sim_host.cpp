// SPDX-License-Identifier: Apache-2.0
// Simulator host program for the GDN prefill kernels of csrc/flashqla_hpu.
//
// argv[1] is one case folder written by tests/standalone/gdn_prefill/kernel_cases.py: case.txt plus
// one raw little-endian <name>.bin per input tensor.  case.txt lines:
//   guid <kernel guid>
//   in  <name> <dtype> <dim0> <dim1> ...     TPC dims, fastest first; dtype: bf16, f32 or i32
//   out <name> <dtype> <dim0> <dim1> ...
// The program links libflashqla_hpu_perf_lib.so and goes through its exported entry point only
// (InstantiateTpcKernel with the two-call ELF handshake), exactly as the graph compiler does,
// drives the Gaudi2 instruction simulator and writes <name>.out.bin per output.  It also checks
// that no input changed and that the guard bytes around every output are intact.  The library
// has no node parameters and its GetShapeInference is empty, so neither is exercised here.
//
// With "--expect-glue <code>" the program only checks that the filling InstantiateTpcKernel call
// returns that GlueCodeReturn value (a shape the glue must refuse) and runs nothing.
// "--no-mapping-test" turns the simulator's index-space mapping test off, "--no-shuffle" keeps the
// index space in one ordered partition.
//
// The SDK headers come first so that a syntax-only check on a machine without the SDK fails only
// with "file not found" for them and checks nothing else.
#include <tpc_test_core_api.h>
#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "deepseek_v41_simulation_descriptor.hpp"

extern "C" {
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance);
}

namespace {

constexpr size_t kGuardBytes = 512;
constexpr uint8_t kGuardByte = 0x5A;
constexpr uint8_t kUnwrittenByte = 0xC3;  // bf16 0xC3C3 = -390.5, f32 0xC3C3C3C3 = -391.5: never written

struct TensorSpec {
    std::string name;
    tpc_lib_api::TensorDataType type = tpc_lib_api::DATA_BF16;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> storage;   // inputs: the data; outputs: guard + data + guard
    std::vector<uint8_t> original;  // inputs: a copy, to prove the kernel did not write them

    uint64_t elements() const
    {
        uint64_t count = 1;
        for (uint64_t dim : dims) count *= dim;
        return count;
    }
    uint64_t elementSize() const
    {
        return (type == tpc_lib_api::DATA_F32 || type == tpc_lib_api::DATA_I32) ? 4 : 2;
    }
    uint64_t bytes() const { return elements() * elementSize(); }
};

bool ParseType(const std::string& text, tpc_lib_api::TensorDataType& type)
{
    if (text == "bf16") {
        type = tpc_lib_api::DATA_BF16;
    } else if (text == "f32") {
        type = tpc_lib_api::DATA_F32;
    } else if (text == "i32") {
        type = tpc_lib_api::DATA_I32;
    } else {
        return false;
    }
    return true;
}

void FillGeometry(tpc_lib_api::TensorGeometry& geometry, const TensorSpec& spec)
{
    geometry.dataType = spec.type;
    geometry.dims = static_cast<uint32_t>(spec.dims.size());
    for (size_t dimension = 0; dimension < spec.dims.size(); ++dimension) {
        geometry.maxSizes[dimension] = spec.dims[dimension];
        geometry.minSizes[dimension] = spec.dims[dimension];
    }
}

}  // namespace

int main(int argc, char** argv)
{
    using namespace tpc_lib_api;
    using namespace tpc_tests;

    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <case-folder> [--no-mapping-test] [--no-shuffle] [--expect-glue N]\n",
                     argv[0]);
        return 1;
    }
    const std::string folder = argv[1];
    bool mappingTest = true;
    bool shuffle = true;
    bool expectGlue = false;
    int expectedGlue = 0;
    for (int arg = 2; arg < argc; ++arg) {
        if (!std::strcmp(argv[arg], "--no-mapping-test")) {
            mappingTest = false;
        } else if (!std::strcmp(argv[arg], "--no-shuffle")) {
            shuffle = false;
        } else if (!std::strcmp(argv[arg], "--expect-glue") && arg + 1 < argc) {
            expectGlue = true;
            expectedGlue = std::atoi(argv[++arg]);
        } else {
            std::fprintf(stderr, "sim_host: unknown option %s\n", argv[arg]);
            return 1;
        }
    }

    std::string guid;
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
    {
        std::ifstream manifest(folder + "/case.txt");
        if (!manifest) {
            std::fprintf(stderr, "sim_host: cannot open %s/case.txt\n", folder.c_str());
            return 1;
        }
        std::string raw;
        while (std::getline(manifest, raw)) {
            std::istringstream line(raw);
            std::string key;
            if (!(line >> key)) continue;
            if (key == "guid") {
                line >> guid;
            } else if (key == "in" || key == "out") {
                TensorSpec spec;
                std::string type;
                line >> spec.name >> type;
                uint64_t dim = 0;
                while (line >> dim) spec.dims.push_back(dim);
                if (!ParseType(type, spec.type) || spec.dims.empty() || spec.dims.size() > 5) {
                    std::fprintf(stderr, "sim_host: bad tensor line: %s\n", raw.c_str());
                    return 1;
                }
                (key == "in" ? inputs : outputs).push_back(std::move(spec));
            } else {
                std::fprintf(stderr, "sim_host: unknown manifest key %s\n", key.c_str());
                return 1;
            }
        }
    }
    if (guid.empty() || guid.size() >= MAX_NODE_NAME || inputs.empty() || outputs.empty()) {
        std::fprintf(stderr, "sim_host: case.txt needs a guid, an input and an output\n");
        return 1;
    }

    if (!expectGlue) {
        for (TensorSpec& spec : inputs) {
            const std::string path = folder + "/" + spec.name + ".bin";
            std::ifstream stream(path, std::ios::binary);
            if (!stream) {
                std::fprintf(stderr, "sim_host: cannot open %s\n", path.c_str());
                return 2;
            }
            spec.storage.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
            if (spec.storage.size() != spec.bytes()) {
                std::fprintf(stderr, "sim_host: %s holds %llu bytes, expected %llu\n", path.c_str(),
                             static_cast<unsigned long long>(spec.storage.size()),
                             static_cast<unsigned long long>(spec.bytes()));
                return 2;
            }
            spec.original = spec.storage;
        }
        for (TensorSpec& spec : outputs) {
            spec.storage.assign(spec.bytes() + 2 * kGuardBytes, kGuardByte);
            std::memset(spec.storage.data() + kGuardBytes, kUnwrittenByte, spec.bytes());
        }
    }

    std::vector<Tensor> in(inputs.size());
    std::vector<Tensor> out(outputs.size());
    for (size_t index = 0; index < inputs.size(); ++index) FillGeometry(in[index].geometry, inputs[index]);
    for (size_t index = 0; index < outputs.size(); ++index) FillGeometry(out[index].geometry, outputs[index]);

    HabanaKernelParams p{};
    p.deviceId = DEVICE_ID_GAUDI2;
    p.maxAvailableTpc = 1;
    p.inputTensorNr = static_cast<uint32_t>(in.size());
    p.outputTensorNr = static_cast<uint32_t>(out.size());
    p.inputTensors = in.data();
    p.outputTensors = out.data();
    std::strcpy(p.guid.name, guid.c_str());

    HabanaKernelInstantiation g{};
    std::vector<TensorAccessPattern> inputPatterns(in.size());
    std::vector<TensorAccessPattern> outputPatterns(out.size());
    g.inputTensorAccessPattern = inputPatterns.data();
    g.outputTensorAccessPattern = outputPatterns.data();

    // A kernel ELF of this library is far below 1 MiB; one buffer serves the sizing and the filling call.
    std::vector<char> kernelElf(1 << 20);
    g.kernel.kernelElf = nullptr;
    g.kernel.elfSize = 0;
    GlueCodeReturn result = InstantiateTpcKernel(&p, &g);
    if (expectGlue) {
        if (result == GLUE_INSUFFICIENT_ELF_BUFFER) {
            g.kernel.kernelElf = kernelElf.data();
            g.kernel.elfSize = static_cast<unsigned>(kernelElf.size());
            result = InstantiateTpcKernel(&p, &g);
        }
        std::printf("guid: %s\nglue code: %d\n", guid.c_str(), static_cast<int>(result));
        return static_cast<int>(result) == expectedGlue ? 0 : 8;
    }
    if (result != GLUE_INSUFFICIENT_ELF_BUFFER) {
        std::fprintf(stderr,
                     "sim_host: sizing InstantiateTpcKernel returned %d, expected GLUE_INSUFFICIENT_ELF_BUFFER (%d)\n",
                     static_cast<int>(result), static_cast<int>(GLUE_INSUFFICIENT_ELF_BUFFER));
        return 3;
    }
    if (g.kernel.elfSize > kernelElf.size()) {
        std::fprintf(stderr, "sim_host: the kernel ELF has %u bytes, more than the host buffer\n", g.kernel.elfSize);
        return 3;
    }
    const unsigned elfSize = g.kernel.elfSize;
    g.kernel.kernelElf = kernelElf.data();
    result = InstantiateTpcKernel(&p, &g);
    if (result != GLUE_SUCCESS) {
        std::fprintf(stderr, "sim_host: filling InstantiateTpcKernel returned %d, expected GLUE_SUCCESS (%d)\n",
                     static_cast<int>(result), static_cast<int>(GLUE_SUCCESS));
        return 3;
    }

    std::vector<TensorDesc2> descriptors;
    for (size_t index = 0; index < in.size(); ++index) {
        descriptors.push_back(gaudi_validation::descriptor(inputs[index].storage.data(), in[index]));
    }
    for (size_t index = 0; index < out.size(); ++index) {
        descriptors.push_back(
            gaudi_validation::descriptor(outputs[index].storage.data() + kGuardBytes, out[index]));
    }

    VPEStats stats;
    TestConfigurations cfg{};
    cfg.dontTestIndexSpaceMapping = !mappingTest;
    cfg.disableShuffleIndexSpacePartition = !shuffle;
    const unsigned status = RunSimulation(
        p, g, descriptors, stats, mappingTest ? e_accessPatternDefaultMode : e_accessPatternIgnoreMode, {}, {}, cfg);
    if (stats.instructionsExecuted == 0) {
        std::fprintf(stderr, "sim_host: the simulator executed 0 instructions\n");
        return 5;
    }

    for (const TensorSpec& spec : inputs) {
        if (spec.storage != spec.original) {
            std::fprintf(stderr, "sim_host: the kernel wrote input %s\n", spec.name.c_str());
            return 7;
        }
    }
    for (const TensorSpec& spec : outputs) {
        for (size_t offset = 0; offset < kGuardBytes; ++offset) {
            if (spec.storage[offset] != kGuardByte ||
                spec.storage[kGuardBytes + spec.bytes() + offset] != kGuardByte) {
                std::fprintf(stderr, "sim_host: the kernel wrote outside output %s\n", spec.name.c_str());
                return 7;
            }
        }
        const std::string path = folder + "/" + spec.name + ".out.bin";
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(spec.storage.data() + kGuardBytes),
                     static_cast<std::streamsize>(spec.bytes()));
        stream.close();
        if (!stream) {
            std::fprintf(stderr, "sim_host: cannot write %s\n", path.c_str());
            return 1;
        }
    }

    std::printf("guid: %s\n", guid.c_str());
    std::printf("kernel elf bytes: %u\n", elfSize);
    std::printf("index space: rank %u {", static_cast<unsigned>(g.indexSpaceRank));
    for (unsigned dimension = 0; dimension < g.indexSpaceRank; ++dimension) {
        std::printf("%s%llu", dimension ? ", " : "", static_cast<unsigned long long>(g.indexSpaceGeometry[dimension]));
    }
    std::printf("}\n");
    std::printf("mapping test: %s, shuffle: %s, simulator status: %u\n", mappingTest ? "on" : "off",
                shuffle ? "on" : "off", status);
    std::printf("instructions executed: %llu\n", static_cast<unsigned long long>(stats.instructionsExecuted));
    return 0;
}
