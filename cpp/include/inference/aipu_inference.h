// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "inference/aipu_context.h"
#include "common/config.h"
#include <cstddef>
#include <cstdint>
#include <axruntime/axruntime.h>
#include <string>
#include <utility>
#include <vector>

namespace lines {
// Runtime metadata for either an input or an output, matching ZipDepth.
struct AipuTensorInfo {
    std::string name;
    std::vector<std::int64_t> shape;         // Logical, without padding.
    std::vector<std::int64_t> padded_shape;
    std::vector<std::pair<std::int64_t, std::int64_t>> padding;
    double scale = 1.0;
    std::int32_t zero_point = 0;
    std::size_t size_bytes = 0;
};

class AipuInference {
public:
    // Borrow context; connect/load one model instance on num_cores cores.
    // Disable double buffering and input/output DMA buffers, as in runtime.py.
    explicit AipuInference(AipuContext& context, const InferenceConfig& config,
                           int num_cores = 1);
    ~AipuInference();
    AipuInference(const AipuInference&) = delete;
    AipuInference& operator=(const AipuInference&) = delete;
    AipuInference(AipuInference&&) = delete;
    AipuInference& operator=(AipuInference&&) = delete;

    std::size_t input_count() const;
    std::size_t output_count() const;

    const AipuTensorInfo& input_info(std::size_t index = 0) const;
    const AipuTensorInfo& input_info(const std::string& name) const;

    // Validated ABI order: output 0 = distance_normalized, output 1 = orientation_fraction.
    const AipuTensorInfo& output_info(std::size_t index) const;
    const AipuTensorInfo& output_info(const std::string& name) const;

    // Backend-owned buffers. Input matches physical metadata; no quantization here.
    // Addresses remain valid until destruction; output contents change on Process.
    std::int8_t* input_buffer(std::size_t index = 0);
    const std::int8_t* output_buffer(std::size_t index) const;

    void Process();  // axr_run_model_instance using the owned buffers; throws on failure.

private:
    AipuContext& aipu_context_;
    axrConnection* connection_ = nullptr;
    axrModel* model_ = nullptr;
    axrModelInstance* instance_ = nullptr;

    std::vector<axrTensorInfo> raw_input_infos_;
    std::vector<axrTensorInfo> raw_output_infos_;

    std::vector<AipuTensorInfo> input_infos_;
    std::vector<AipuTensorInfo> output_infos_;

    std::vector<std::vector<std::int8_t>> input_memory_;
    std::vector<std::vector<std::int8_t>> output_memory_;

    std::vector<axrArgument> input_args_;
    std::vector<axrArgument> output_args_;

    void LoadInstance(int num_cores);
    void Release();
    // Read axr_get_model_input/output and axr_tensor_size; bind axrArgument arrays.
    void InitializeTensors();
    // Validate manifest/provenance, NHWC int8 [1,240,320,1], quantization,
    // padding, no CPU split graphs; distance ABI 0, orientation ABI 1.
    void ValidateMetadata(const InferenceConfig& config) const;

    AipuTensorInfo to_tensor_info(const axrTensorInfo& raw) const;

    std::size_t find_input_index(const std::string& name) const;
    std::size_t find_output_index(const std::string& name) const;
};
}
