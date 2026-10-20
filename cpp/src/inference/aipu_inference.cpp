// SPDX-License-Identifier: AGPL-3.0-or-later
#include "inference/aipu_inference.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace lines {
namespace {
std::runtime_error RuntimeError(AipuContext& context, const std::string& operation) {
    const char* detail = axr_last_error_string(reinterpret_cast<const axrObject*>(context.get()));
    return std::runtime_error("AipuInference: " + operation + (detail ? ": " + std::string(detail) : ""));
}

nlohmann::json ReadJson(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return nlohmann::json::parse(stream);
}

std::string Sha256(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot read provenance source: " + path.string());
    }
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("Cannot initialize SHA256");
    }
    std::array<char, 65536> buffer;
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount() > 0) {
        if (EVP_DigestUpdate(digest.get(), buffer.data(), stream.gcount()) != 1) {
            throw std::runtime_error("Cannot update SHA256");
        }
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash;
    unsigned int size = 0;
    if (!stream.eof() || EVP_DigestFinal_ex(digest.get(), hash.data(), &size) != 1) {
        throw std::runtime_error("Cannot finish SHA256 for " + path.string());
    }
    std::ostringstream result;
    for (unsigned int index = 0; index < size; ++index) {
        result << std::hex << std::setfill('0') << std::setw(2) << unsigned(hash[index]);
    }
    return result.str();
}
}

AipuInference::AipuInference(AipuContext& context, const InferenceConfig& config, int num_cores)
    : aipu_context_(context) {
    if (num_cores < 1 || num_cores > 4) {
        throw std::invalid_argument("AipuInference: num_cores must be between 1 and 4 for Metis");
    }
    if (config.model_directory.empty()) {
        throw std::invalid_argument("AipuInference: model_directory is required");
    }
    try {
        const auto model_path = config.model_directory / "model.json";
        model_ = axr_load_model(aipu_context_.get(), model_path.string().c_str());
        if (!model_) {
            throw RuntimeError(aipu_context_, "cannot load " + model_path.string());
        }
        InitializeTensors();
        ValidateMetadata(config);
        LoadInstance(num_cores);
    } catch (...) {
        Release();
        throw;
    }
}

AipuInference::~AipuInference() {
    Release();
}

void AipuInference::Release() {
    if (instance_) {
        axr_destroy(reinterpret_cast<const axrObject*>(instance_));
        instance_ = nullptr;
    }
    if (model_) {
        axr_destroy(reinterpret_cast<const axrObject*>(model_));
        model_ = nullptr;
    }
    if (connection_) {
        axr_destroy(reinterpret_cast<const axrObject*>(connection_));
        connection_ = nullptr;
    }
}

void AipuInference::LoadInstance(int num_cores) {
    connection_ = axr_device_connect(aipu_context_.get(), nullptr, 1, nullptr);
    if (!connection_) {
        throw RuntimeError(aipu_context_, "cannot connect to a device");
    }
    const std::string options = "double_buffer=0;input_dmabuf=0;output_dmabuf=0;"
                                "num_sub_devices=1;aipu_cores=" + std::to_string(num_cores);
    auto* properties = axr_create_properties(aipu_context_.get(), options.c_str());
    if (!properties) {
        throw RuntimeError(aipu_context_, "cannot create model-instance properties");
    }
    instance_ = axr_load_model_instance(connection_, model_, properties);
    if (!instance_) {
        const auto error = RuntimeError(aipu_context_, "cannot load model instance");
        axr_destroy(reinterpret_cast<const axrObject*>(properties));
        throw error;
    }
    axr_destroy(reinterpret_cast<const axrObject*>(properties));
}

AipuTensorInfo AipuInference::to_tensor_info(const axrTensorInfo& raw) const {
    if (raw.ndims != 4 || raw.bits != 8 || raw.type != AXR_SIGNED ||
        !std::isfinite(raw.scale) || raw.scale <= 0) {
        throw std::runtime_error("AipuInference: expected a rank-4 int8 tensor with a positive finite scale");
    }
    AipuTensorInfo info;
    info.name.assign(raw.name, std::find(raw.name, raw.name + sizeof(raw.name), '\0'));
    info.scale = raw.scale;
    info.zero_point = raw.zero_point;
    std::size_t elements = 1;
    for (std::size_t dimension = 0; dimension < raw.ndims; ++dimension) {
        const auto size = raw.dims[dimension];
        const auto before = raw.padding[dimension][0];
        const auto after = raw.padding[dimension][1];
        if (size == 0 || size > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
            before >= size || after >= size - before ||
            elements > std::numeric_limits<std::size_t>::max() / size) {
            throw std::runtime_error("AipuInference: invalid shape/padding for " + info.name);
        }
        elements *= size;
        info.padded_shape.push_back(static_cast<std::int64_t>(size));
        info.shape.push_back(static_cast<std::int64_t>(size - before - after));
        info.padding.emplace_back(before, after);
    }
    info.size_bytes = axr_tensor_size(&raw);
    if (info.name.empty() || info.size_bytes != elements) {
        throw std::runtime_error("AipuInference: invalid name/buffer size for " + info.name);
    }
    return info;
}

void AipuInference::InitializeTensors() {
    const auto inputs = axr_num_model_inputs(model_);
    const auto outputs = axr_num_model_outputs(model_);
    if (inputs != 1 || outputs != 2) {
        throw std::runtime_error("AipuInference: DeepLSD requires one input and two output heads");
    }
    raw_input_infos_.resize(inputs);
    input_infos_.resize(inputs);
    input_memory_.resize(inputs);
    input_args_.resize(inputs);
    for (std::size_t index = 0; index < inputs; ++index) {
        raw_input_infos_[index] = axr_get_model_input(model_, index);
        input_infos_[index] = to_tensor_info(raw_input_infos_[index]);
        input_memory_[index].resize(input_infos_[index].size_bytes);
        input_args_[index] = {input_memory_[index].data(), -1, 0, input_memory_[index].size()};
    }
    raw_output_infos_.resize(outputs);
    output_infos_.resize(outputs);
    output_memory_.resize(outputs);
    output_args_.resize(outputs);
    for (std::size_t index = 0; index < outputs; ++index) {
        raw_output_infos_[index] = axr_get_model_output(model_, index);
        output_infos_[index] = to_tensor_info(raw_output_infos_[index]);
        output_memory_[index].resize(output_infos_[index].size_bytes);
        output_args_[index] = {output_memory_[index].data(), -1, 0, output_memory_[index].size()};
    }
}

void AipuInference::ValidateMetadata(const InferenceConfig& config) const {
    const auto manifest = ReadJson(config.model_directory / "manifest.json");
    const auto description = ReadJson(config.model_directory / "model.json");
    const auto provenance = ReadJson(config.model_directory / "provenance.json");
    // The reference artifact layout keeps ONNX beside the configured AIPU directory.
    const auto onnx_path = config.model_directory.parent_path() / "onnx/deeplsd.onnx";
    if (provenance.at("onnx_sha256").get<std::string>() != Sha256(onnx_path)) {
        throw std::runtime_error("AipuInference: compiled-model provenance does not match ONNX");
    }
    if (manifest.at("input_tensor_layout") != "NHWC" ||
        !manifest.value("preprocess_graph", nlohmann::json()).empty() ||
        !manifest.value("postprocess_graph", nlohmann::json()).empty()) {
        throw std::runtime_error("AipuInference: expected NHWC without CPU split graphs");
    }
    const std::vector<std::int64_t> logical_shape{1, 240, 320, 1};
    for (bool input : {true, false}) {
        const auto& infos = input ? input_infos_ : output_infos_;
        const auto& shapes = manifest.at(input ? "input_shapes" : "output_shapes");
        const auto& paddings = manifest.at(input ? "n_padded_ch_inputs" : "n_padded_ch_outputs");
        const auto& params = manifest.at(input ? "quantize_params" : "dequantize_params");
        const auto& dtypes = manifest.at(input ? "input_dtypes" : "output_dtypes");
        const auto& tensors = description.at(input ? "inputs" : "outputs");
        if (shapes.size() != infos.size() || paddings.size() != infos.size() ||
            params.size() != infos.size() || dtypes.size() != infos.size() || tensors.size() != infos.size()) {
            throw std::runtime_error("AipuInference: tensor count differs from manifest/model.json");
        }
        for (std::size_t index = 0; index < infos.size(); ++index) {
            const auto& info = infos[index];
            std::vector<std::int64_t> padding;
            for (const auto& pair : info.padding) {
                padding.push_back(pair.first);
                padding.push_back(pair.second);
            }
            if (info.shape != logical_shape || shapes.at(index) != info.padded_shape ||
                paddings.at(index) != padding || dtypes.at(index) != "int8" ||
                params.at(index).size() != 2 || params.at(index).at(0) != info.scale ||
                params.at(index).at(1) != info.zero_point ||
                tensors.at(index).at("name") != info.name ||
                tensors.at(index).at("shape") != info.padded_shape ||
                tensors.at(index).at("dtype") != "int8" ||
                tensors.at(index).at("size") != info.size_bytes) {
                throw std::runtime_error("AipuInference: runtime/manifest/model tensor mismatch at ABI " +
                                         std::string(input ? "input " : "output ") + std::to_string(index));
            }
        }
    }
    if (output_infos_[0].name == output_infos_[1].name) {
        throw std::runtime_error("AipuInference: output head names must be distinct");
    }
}

void AipuInference::Process() {
    const auto result = axr_run_model_instance(instance_, input_args_.data(), input_args_.size(),
                                              output_args_.data(), output_args_.size());
    if (result != AXR_SUCCESS) {
        throw RuntimeError(aipu_context_, std::string("execution failed: ") + axr_error_string(result));
    }
}

std::size_t AipuInference::input_count() const { return input_infos_.size(); }
std::size_t AipuInference::output_count() const { return output_infos_.size(); }
const AipuTensorInfo& AipuInference::input_info(std::size_t index) const { return input_infos_.at(index); }
const AipuTensorInfo& AipuInference::output_info(std::size_t index) const { return output_infos_.at(index); }
const AipuTensorInfo& AipuInference::input_info(const std::string& name) const {
    return input_infos_.at(find_input_index(name));
}
const AipuTensorInfo& AipuInference::output_info(const std::string& name) const {
    return output_infos_.at(find_output_index(name));
}
std::int8_t* AipuInference::input_buffer(std::size_t index) { return input_memory_.at(index).data(); }
const std::int8_t* AipuInference::output_buffer(std::size_t index) const {
    return output_memory_.at(index).data();
}
std::size_t AipuInference::find_input_index(const std::string& name) const {
    for (std::size_t index = 0; index < input_infos_.size(); ++index) {
        if (input_infos_[index].name == name) return index;
    }
    throw std::out_of_range("AipuInference: unknown input '" + name + "'");
}
std::size_t AipuInference::find_output_index(const std::string& name) const {
    for (std::size_t index = 0; index < output_infos_.size(); ++index) {
        if (output_infos_[index].name == name) return index;
    }
    throw std::out_of_range("AipuInference: unknown output '" + name + "'");
}
}
