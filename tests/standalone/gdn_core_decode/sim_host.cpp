// SPDX-License-Identifier: Apache-2.0
// Simulator host program for the fused GDN core-decode kernel (K5b).
//
// argv[1] is one sim_case folder written by tests/unit_tests/gdn_core_decode/sim_case.py
// (12 raw little-endian <name>.bin inputs plus params.txt = "S H HV B scale eps gate").
// The program fills the tensor geometries, runs the real host glue
// (GdnCoreDecodeGaudi2::GetGcDefinitions, two-call ELF handshake), drives the Gaudi2
// instruction simulator and writes out.bin, conv_state.out.bin and ssm_state.out.bin
// back into the folder for the pytest to compare with the torch reference.
//
// The SDK headers come first so that a syntax-only check on a machine without the SDK
// (this Mac) fails only with "file not found" for them and checks nothing else.
#include <tpc_test_core_api.h>
#include <gc_interface.h>
#include <tpc_kernel_lib_interface.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "deepseek_v41_simulation_descriptor.hpp"
#include "gdn_core_decode_gaudi2.hpp"

namespace {

constexpr int kInputCount = 12;
constexpr uint64_t kHeadDim = 128;
const char* const kInputNames[kInputCount] = {"conv_state", "ssm_state",     "mixed_qkv", "z",
                                              "a",          "b",             "A_log",     "dt_bias",
                                              "conv_weight_t", "norm_weight", "load_indices", "store_indices"};

bool LoadInput(const std::string& path, std::vector<uint8_t>& buffer, uint64_t expectedBytes)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "sim_host: cannot open %s\n", path.c_str());
        return false;
    }
    buffer.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    if (buffer.size() != expectedBytes) {
        std::fprintf(
            stderr,
            "sim_host: %s holds %llu bytes, expected %llu\n",
            path.c_str(),
            static_cast<unsigned long long>(buffer.size()),
            static_cast<unsigned long long>(expectedBytes));
        return false;
    }
    return true;
}

bool SaveOutput(const std::string& path, const std::vector<uint8_t>& data)
{
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    stream.close();
    if (!stream) {
        std::fprintf(stderr, "sim_host: cannot write %s\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    using namespace tpc_lib_api;
    using namespace tpc_tests;

    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <case-folder>\n", argv[0]);
        return 1;
    }
    const std::string folder = argv[1];

    uint32_t S = 0;
    uint32_t H = 0;
    uint32_t HV = 0;
    uint32_t B = 0;
    double scale = 0.0;
    double eps = 0.0;
    int32_t gate = -1;
    {
        std::ifstream params(folder + "/params.txt");
        if (!(params >> S >> H >> HV >> B >> scale >> eps >> gate)) {
            std::fprintf(stderr, "sim_host: cannot read S H HV B scale eps gate from %s/params.txt\n", folder.c_str());
            return 1;
        }
    }
    const uint64_t channels = 2 * static_cast<uint64_t>(H) * kHeadDim + static_cast<uint64_t>(HV) * kHeadDim;

    const uint64_t inputBytes[kInputCount] = {
        channels * 3 * S * 2,            // conv_state      BF16 {C, 3, S}
        kHeadDim * kHeadDim * HV * S * 4,  // ssm_state     F32  {128, 128, HV, S}
        channels * B * 2,                // mixed_qkv       BF16 {C, B}
        static_cast<uint64_t>(HV) * kHeadDim * B * 2,  // z BF16 {HV*128, B}
        static_cast<uint64_t>(HV) * B * 2,  // a             BF16 {HV, B}
        static_cast<uint64_t>(HV) * B * 2,  // b             BF16 {HV, B}
        static_cast<uint64_t>(HV) * 4,   // A_log           F32  {HV}
        static_cast<uint64_t>(HV) * 4,   // dt_bias         F32  {HV}
        channels * 4 * 2,                // conv_weight_t   BF16 {C, 4}
        kHeadDim * 2,                    // norm_weight     BF16 {128}
        static_cast<uint64_t>(B) * 4,    // load_indices    I32  {B}
        static_cast<uint64_t>(B) * 4,    // store_indices   I32  {B}
    };
    std::vector<uint8_t> inputBuffers[kInputCount];
    for (int input = 0; input < kInputCount; ++input) {
        if (!LoadInput(folder + "/" + kInputNames[input] + ".bin", inputBuffers[input], inputBytes[input])) {
            return 2;
        }
    }
    std::vector<uint8_t> outputBuffer(kHeadDim * HV * B * 4, 0);

    const auto setGeometry = [](Tensor& tensor, TensorDataType type, unsigned dims, const uint64_t* sizes) {
        tensor.geometry.dataType = type;
        tensor.geometry.dims = dims;
        for (unsigned dimension = 0; dimension < dims; ++dimension) {
            tensor.geometry.maxSizes[dimension] = sizes[dimension];
        }
    };

    Tensor in[kInputCount]{};
    Tensor out{};
    {
        const uint64_t convState[]{channels, 3, S};
        const uint64_t ssmState[]{kHeadDim, kHeadDim, HV, S};
        const uint64_t mixedQkv[]{channels, B};
        const uint64_t z[]{static_cast<uint64_t>(HV) * kHeadDim, B};
        const uint64_t a[]{HV, B};
        const uint64_t b[]{HV, B};
        const uint64_t aLog[]{HV};
        const uint64_t dtBias[]{HV};
        const uint64_t convWeightT[]{channels, 4};
        const uint64_t normWeight[]{kHeadDim};
        const uint64_t indices[]{B};
        const uint64_t output[]{kHeadDim, HV, B};
        setGeometry(in[0], DATA_BF16, 3, convState);
        setGeometry(in[1], DATA_F32, 4, ssmState);
        setGeometry(in[2], DATA_BF16, 2, mixedQkv);
        setGeometry(in[3], DATA_BF16, 2, z);
        setGeometry(in[4], DATA_BF16, 2, a);
        setGeometry(in[5], DATA_BF16, 2, b);
        setGeometry(in[6], DATA_F32, 1, aLog);
        setGeometry(in[7], DATA_F32, 1, dtBias);
        setGeometry(in[8], DATA_BF16, 2, convWeightT);
        setGeometry(in[9], DATA_BF16, 1, normWeight);
        setGeometry(in[10], DATA_I32, 1, indices);
        setGeometry(in[11], DATA_I32, 1, indices);
        setGeometry(out, DATA_F32, 3, output);
    }

    GdnCoreDecodeParams scalar{static_cast<float>(scale), static_cast<float>(eps), gate};
    HabanaKernelParams p{};
    p.deviceId = DEVICE_ID_GAUDI2;
    p.maxAvailableTpc = 1;
    p.inputTensorNr = kInputCount;
    p.outputTensorNr = 1;
    p.inputTensors = in;
    p.outputTensors = &out;
    p.nodeParams.nodeParams = &scalar;
    p.nodeParams.nodeParamsSize = sizeof(GdnCoreDecodeParams);
    std::strcpy(p.guid.name, kGdnCoreDecodeGuid);

    HabanaKernelInstantiation g{};
    TensorAccessPattern inputPatterns[kInputCount]{};
    TensorAccessPattern outputPattern{};
    g.inputTensorAccessPattern = inputPatterns;
    g.outputTensorAccessPattern = &outputPattern;

    GdnCoreDecodeGaudi2 glue;
    g.kernel.kernelElf = nullptr;
    g.kernel.elfSize = 0;
    GlueCodeReturn result = glue.GetGcDefinitions(&p, &g);
    if (result != GLUE_INSUFFICIENT_ELF_BUFFER) {
        std::fprintf(
            stderr,
            "sim_host: sizing GetGcDefinitions returned %d, expected GLUE_INSUFFICIENT_ELF_BUFFER (%d)\n",
            static_cast<int>(result),
            static_cast<int>(GLUE_INSUFFICIENT_ELF_BUFFER));
        return 3;
    }
    std::vector<char> kernelElf(g.kernel.elfSize);
    g.kernel.kernelElf = kernelElf.data();
    result = glue.GetGcDefinitions(&p, &g);
    if (result != GLUE_SUCCESS) {
        std::fprintf(
            stderr,
            "sim_host: filling GetGcDefinitions returned %d, expected GLUE_SUCCESS (%d)\n",
            static_cast<int>(result),
            static_cast<int>(GLUE_SUCCESS));
        return 3;
    }

    if (g.indexSpaceRank != 2 || g.indexSpaceGeometry[0] != H || g.indexSpaceGeometry[1] != B) {
        std::fprintf(
            stderr,
            "sim_host: glue index space is rank %u {%llu, %llu}, expected 2 {%u, %u}\n",
            static_cast<unsigned>(g.indexSpaceRank),
            static_cast<unsigned long long>(g.indexSpaceGeometry[0]),
            static_cast<unsigned long long>(g.indexSpaceGeometry[1]),
            static_cast<unsigned>(H),
            static_cast<unsigned>(B));
        return 4;
    }

    std::vector<TensorDesc2> descriptors;
    for (int input = 0; input < kInputCount; ++input) {
        descriptors.push_back(gaudi_validation::descriptor(inputBuffers[input].data(), in[input]));
    }
    descriptors.push_back(gaudi_validation::descriptor(outputBuffer.data(), out));

    VPEStats stats;
    TestConfigurations cfg{};
    cfg.dontTestIndexSpaceMapping = true;
    cfg.disableShuffleIndexSpacePartition = true;
    RunSimulation(p, g, descriptors, stats, e_accessPatternIgnoreMode, {}, {}, cfg);
    if (stats.instructionsExecuted == 0) {
        std::fprintf(stderr, "sim_host: the simulator executed 0 instructions\n");
        return 5;
    }

    if (!SaveOutput(folder + "/out.bin", outputBuffer) ||
        !SaveOutput(folder + "/conv_state.out.bin", inputBuffers[0]) ||
        !SaveOutput(folder + "/ssm_state.out.bin", inputBuffers[1])) {
        return 1;
    }
    std::printf(
        "instructions executed: %llu\n", static_cast<unsigned long long>(stats.instructionsExecuted));
    return 0;
}
