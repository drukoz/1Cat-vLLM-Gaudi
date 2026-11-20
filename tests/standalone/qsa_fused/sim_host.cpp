// SPDX-License-Identifier: Apache-2.0
// Simulator host program for the kernels of csrc/qsa_fused (the fused QSA decode kernels), derived from
// tests/standalone/hc_fused/sim_host.cpp.
//
// argv[1] is one case folder written by tests/unit_tests/qsa_fused/qsa_case.py: case.txt plus one raw
// little-endian <name>.bin per input tensor.  case.txt lines:
//   guid <kernel guid>
//   eps <float> / heads / ring_rows / ratio / block_size / rotary_half <int>   the QsaFusedParams fields
//   in    <name> <dtype> <dim0> <dim1> ...                  TPC dims, fastest first
//   inout <name> <dtype> <dim0> <dim1> ...                  an input the kernel writes in place (its
//                                                           contents after the run go to <name>.out.bin)
//   out   <name> <dtype> <dim0> <dim1> ...                  dtype: bf16, f32, i32 or f8 (1-4-3)
// The program links libqsa_fused_gaudi2.so and goes through its exported entry points only
// (InstantiateTpcKernel with the two-call ELF handshake, GetShapeInference), exactly as the graph
// compiler does, drives the Gaudi2 instruction simulator and writes <name>.out.bin per output.
// It also checks that the shape inference agrees with the manifest, that no input other than an inout
// changed and that the guard bytes around every output are intact.
//
// Optional argv[2..]: "--no-mapping-test" turns the simulator's index-space mapping test off,
// "--no-shuffle" keeps the index space in one ordered partition.
//
// The SDK headers come first so that a syntax-only check on a machine without the SDK fails only
// with "file not found" for them and checks nothing else.
#include <tpc_test_core_api.h>
#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "deepseek_v41_simulation_descriptor.hpp"
#include "qsa_fused_params.h"

extern "C" {
tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance);
tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId device, const tpc_lib_api::ShapeInferenceParams* params,
    tpc_lib_api::ShapeInferenceOutput* output);
}

namespace {

constexpr size_t kGuardBytes = 512;
constexpr uint8_t kGuardByte = 0x5A;
constexpr uint8_t kUnwrittenByte = 0xC3;  // bf16 0xC3C3 = -390.5: an output element the kernel never wrote

struct TensorSpec {
    std::string name;
    tpc_lib_api::TensorDataType type = tpc_lib_api::DATA_BF16;
    std::vector<uint64_t> dims;
    std::vector<uint8_t> storage;   // inputs: the data; outputs: guard + data + guard
    std::vector<uint8_t> original;  // inputs: a copy, to prove the kernel did not write them
    bool inout = false;             // an input the kernel may write in place

    uint64_t elements() const
    {
        uint64_t count = 1;
        for (uint64_t dim : dims) count *= dim;
        return count;
    }
    uint64_t elementSize() const
    {
        if (type == tpc_lib_api::DATA_F32 || type == tpc_lib_api::DATA_I32) return 4;
        return type == tpc_lib_api::DATA_F8_143 ? 1 : 2;
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
    } else if (text == "f8") {
        type = tpc_lib_api::DATA_F8_143;
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
        std::fprintf(stderr, "usage: %s <case-folder> [--no-mapping-test] [--no-shuffle]\n", argv[0]);
        return 1;
    }
    const std::string folder = argv[1];
    bool mappingTest = true;
    bool shuffle = true;
    for (int arg = 2; arg < argc; ++arg) {
        if (!std::strcmp(argv[arg], "--no-mapping-test")) {
            mappingTest = false;
        } else if (!std::strcmp(argv[arg], "--no-shuffle")) {
            shuffle = false;
        } else {
            std::fprintf(stderr, "sim_host: unknown option %s\n", argv[arg]);
            return 1;
        }
    }

    std::string guid;
    QsaFusedParams scalar{};
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
            } else if (key == "eps") {
                double value = 0;
                line >> value;
                scalar.eps = static_cast<float>(value);
            } else if (key == "heads") {
                line >> scalar.heads;
            } else if (key == "ring_rows") {
                line >> scalar.ring_rows;
            } else if (key == "ratio") {
                line >> scalar.ratio;
            } else if (key == "block_size") {
                line >> scalar.block_size;
            } else if (key == "rotary_half") {
                line >> scalar.rotary_half;
            } else if (key == "in" || key == "inout" || key == "out") {
                TensorSpec spec;
                std::string type;
                line >> spec.name >> type;
                uint64_t dim = 0;
                while (line >> dim) spec.dims.push_back(dim);
                if (!ParseType(type, spec.type) || spec.dims.empty() || spec.dims.size() > 5) {
                    std::fprintf(stderr, "sim_host: bad tensor line: %s\n", raw.c_str());
                    return 1;
                }
                spec.inout = key == "inout";
                (key == "out" ? outputs : inputs).push_back(std::move(spec));
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
    p.nodeParams.nodeParams = &scalar;
    p.nodeParams.nodeParamsSize = sizeof(QsaFusedParams);
    std::strcpy(p.guid.name, guid.c_str());

    // Shape inference first: it must give the manifest's output shapes from the inputs alone.
    {
        std::vector<TensorShapeInfo> inputShapes(in.size());
        std::vector<TensorShapeInfo*> inputShapePointers(in.size());
        for (size_t index = 0; index < in.size(); ++index) {
            inputShapes[index].geometry = in[index].geometry;
            inputShapePointers[index] = &inputShapes[index];
        }
        std::vector<TensorShapeInfo> outputShapes(out.size());
        std::vector<TensorShapeInfo*> outputShapePointers(out.size());
        for (size_t index = 0; index < out.size(); ++index) outputShapePointers[index] = &outputShapes[index];
        uint32_t invalidMask = 0;
        ShapeInferenceParams sifParams{};
        sifParams.guid = p.guid;
        sifParams.pGuid = &p.guid;
        sifParams.inputTensors = inputShapePointers.data();
        sifParams.inputTensorsNr = static_cast<uint32_t>(in.size());
        sifParams.nodeParams = p.nodeParams;
        sifParams.outputTensorsNr = static_cast<uint32_t>(out.size());
        ShapeInferenceOutput sifOutput{};
        sifOutput.outputTensors = outputShapePointers.data();
        sifOutput.invalidMask = &invalidMask;
        const GlueCodeReturn sif = GetShapeInference(DEVICE_ID_GAUDI2, &sifParams, &sifOutput);
        if (sif != GLUE_SUCCESS || invalidMask != 0) {
            std::fprintf(stderr, "sim_host: GetShapeInference returned %d (invalid mask %u)\n",
                         static_cast<int>(sif), invalidMask);
            return 6;
        }
        for (size_t index = 0; index < out.size(); ++index) {
            const TensorGeometry& inferred = outputShapes[index].geometry;
            bool same = inferred.dims == outputs[index].dims.size();
            for (size_t dimension = 0; same && dimension < outputs[index].dims.size(); ++dimension) {
                same = inferred.maxSizes[dimension] == outputs[index].dims[dimension];
            }
            if (!same) {
                std::fprintf(stderr, "sim_host: GetShapeInference output %zu (%s) does not match case.txt\n",
                             index, outputs[index].name.c_str());
                return 6;
            }
        }
    }

    HabanaKernelInstantiation g{};
    std::vector<TensorAccessPattern> inputPatterns(in.size());
    std::vector<TensorAccessPattern> outputPatterns(out.size());
    g.inputTensorAccessPattern = inputPatterns.data();
    g.outputTensorAccessPattern = outputPatterns.data();

    g.kernel.kernelElf = nullptr;
    g.kernel.elfSize = 0;
    GlueCodeReturn result = InstantiateTpcKernel(&p, &g);
    if (result != GLUE_INSUFFICIENT_ELF_BUFFER) {
        std::fprintf(stderr,
                     "sim_host: sizing InstantiateTpcKernel returned %d, expected GLUE_INSUFFICIENT_ELF_BUFFER (%d)\n",
                     static_cast<int>(result), static_cast<int>(GLUE_INSUFFICIENT_ELF_BUFFER));
        return 3;
    }
    std::vector<char> kernelElf(g.kernel.elfSize);
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
        if (spec.inout) {
            const std::string path = folder + "/" + spec.name + ".out.bin";
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(spec.storage.data()),
                         static_cast<std::streamsize>(spec.bytes()));
            stream.close();
            if (!stream) {
                std::fprintf(stderr, "sim_host: cannot write %s\n", path.c_str());
                return 1;
            }
            continue;
        }
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
