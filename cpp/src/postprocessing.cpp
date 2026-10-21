// SPDX-License-Identifier: AGPL-3.0-or-later
#include "postprocessing.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lines {
namespace {
void ValidateOutputInfo(const AipuTensorInfo& info) {
    if (info.shape != std::vector<std::int64_t>({1, 240, 320, 1}) ||
        info.padded_shape.size() != 4 || info.padding.size() != 4 ||
        !std::isfinite(info.scale) || !std::isfinite(float(info.scale)) || float(info.scale) <= 0) {
        throw std::invalid_argument("Postprocessing requires NHWC [1,240,320,1] metadata and a positive float32 scale");
    }
    std::size_t elements = 1;
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
        const auto size = info.padded_shape[dimension];
        const auto [before, after] = info.padding[dimension];
        if (size <= 0 || before < 0 || after < 0 || before >= size || after >= size - before ||
            size - before - after != info.shape[dimension] ||
            static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max() / elements) {
            throw std::invalid_argument("Postprocessing received invalid physical shape or padding");
        }
        elements *= static_cast<std::size_t>(size);
    }
    if (info.size_bytes != elements) {
        throw std::invalid_argument("Postprocessing buffer size does not match tensor metadata");
    }
}

void Dequantize(const std::int8_t* buffer, const AipuTensorInfo& info, std::vector<float>& field) {
    const auto height = static_cast<std::size_t>(info.padded_shape[1]);
    const auto width = static_cast<std::size_t>(info.padded_shape[2]);
    const auto channels = static_cast<std::size_t>(info.padded_shape[3]);
    const auto batch = static_cast<std::size_t>(info.padding[0].first);
    const auto top = static_cast<std::size_t>(info.padding[1].first);
    const auto left = static_cast<std::size_t>(info.padding[2].first);
    const auto channel = static_cast<std::size_t>(info.padding[3].first);
    const auto rows = static_cast<std::size_t>(info.shape[1]);
    const auto columns = static_cast<std::size_t>(info.shape[2]);
    const float scale = static_cast<float>(info.scale);
    const float zero_point = static_cast<float>(info.zero_point);

    field.resize(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto offset = ((batch * height + row + top) * width + left) * channels + channel;
        for (std::size_t column = 0; column < columns; ++column) {
            const float quantized = static_cast<float>(buffer[offset + column * channels]);
            field[row * columns + column] = (quantized - zero_point) * scale;
        }
    }
}
}

Postprocessing::Postprocessing(AipuTensorInfo distance_info, AipuTensorInfo orientation_info)
    : distance_info_(std::move(distance_info)), orientation_info_(std::move(orientation_info)) {
    ValidateOutputInfo(distance_info_);
    ValidateOutputInfo(orientation_info_);
}

void Postprocessing::Process(const std::int8_t* distance_buffer, const std::int8_t* orientation_buffer,
                             std::vector<float>& distance, std::vector<float>& orientation) const {
    if (!distance_buffer || !orientation_buffer || &distance == &orientation) {
        throw std::invalid_argument("Postprocessing requires two valid buffers and separate output vectors");
    }
    Dequantize(distance_buffer, distance_info_, distance);
    Dequantize(orientation_buffer, orientation_info_, orientation);

    constexpr float pi = 3.14159265358979323846f;
    for (std::size_t index = 0; index < distance.size(); ++index) {
        distance[index] = 5.0f * std::exp(-distance[index]);
        orientation[index] *= pi;
    }
}
}
