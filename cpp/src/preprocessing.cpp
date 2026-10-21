// SPDX-License-Identifier: AGPL-3.0-or-later
#include "preprocessing.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lines {
namespace {
void ValidateInputInfo(const AipuTensorInfo& info, std::size_t buffer_size) {
    if (info.shape != std::vector<std::int64_t>({1, 240, 320, 1}) ||
        info.padded_shape.size() != 4 || info.padding.size() != 4 ||
        !std::isfinite(info.scale) || !std::isfinite(float(info.scale)) || float(info.scale) <= 0) {
        throw std::invalid_argument("Preprocessing requires NHWC [1,240,320,1] metadata and a positive float32 scale");
    }
    std::size_t elements = 1;
    for (std::size_t dimension = 0; dimension < 4; ++dimension) {
        const auto size = info.padded_shape[dimension];
        const auto [before, after] = info.padding[dimension];
        if (size <= 0 || before < 0 || after < 0 || before >= size || after >= size - before ||
            size - before - after != info.shape[dimension] ||
            static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max() / elements) {
            throw std::invalid_argument("Preprocessing received invalid physical shape or padding");
        }
        elements *= static_cast<std::size_t>(size);
    }
    if (info.size_bytes != elements || buffer_size != info.size_bytes) {
        throw std::invalid_argument("Preprocessing input buffer size does not match tensor metadata");
    }
}

std::int8_t Quantize(float value, float scale, float zero_point) {
    const float scaled = value / scale;
    const float shifted = scaled + zero_point;
    const float bounded = std::clamp(shifted, -128.0f, 127.0f);
    const float lower = std::floor(bounded);
    const float fraction = bounded - lower;
    int rounded = static_cast<int>(lower);
    if (fraction > 0.5f || (fraction == 0.5f && rounded % 2 != 0)) {
        ++rounded;
    }
    return static_cast<std::int8_t>(rounded);
}
}
ResizeTransform Preprocessing::Process(const cv::Mat& image, cv::Mat& normalized) const {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("Preprocessing requires a nonempty BGR uint8 image");
    }

    ResizeTransform transform;
    transform.original_width = image.cols;
    transform.original_height = image.rows;
    transform.scale_x = double(image.cols) / transform.network_width;
    transform.scale_y = double(image.rows) / transform.network_height;

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(transform.network_width, transform.network_height),
               0, 0, cv::INTER_AREA);

    if (normalized.size() != gray.size() || normalized.type() != CV_32FC1 ||
        !normalized.isContinuous()) {
        normalized = cv::Mat(gray.size(), CV_32FC1);
    }
    for (int row = 0; row < gray.rows; ++row) {
        const auto* source = gray.ptr<unsigned char>(row);
        auto* destination = normalized.ptr<float>(row);
        for (int column = 0; column < gray.cols; ++column) {
            destination[column] = float(source[column]) / 255.0f;
        }
    }
    return transform;
}

ResizeTransform Preprocessing::Process(const cv::Mat& image, const AipuTensorInfo& input_info,
                                      std::int8_t* input_buffer, std::size_t buffer_size) const {
    cv::Mat normalized;
    const auto transform = Process(image, normalized);
    QuantizeAndPad(normalized, input_info, input_buffer, buffer_size);
    return transform;
}

void Preprocessing::QuantizeAndPad(const cv::Mat& normalized, const AipuTensorInfo& input_info,
                                  std::int8_t* input_buffer, std::size_t buffer_size) const {
    ValidateInputInfo(input_info, buffer_size);
    if (!input_buffer || normalized.type() != CV_32FC1 ||
        normalized.rows != 240 || normalized.cols != 320 ||
        !cv::checkRange(normalized, true, nullptr, 0.0, std::nextafter(1.0f, 2.0f))) {
        throw std::invalid_argument("QuantizeAndPad requires a valid output buffer and normalized float32 [240,320] in [0,1]");
    }

    const auto padding_value = static_cast<std::int8_t>(std::clamp(input_info.zero_point, -128, 127));
    std::fill_n(input_buffer, buffer_size, padding_value);

    const auto height = static_cast<std::size_t>(input_info.padded_shape[1]);
    const auto width = static_cast<std::size_t>(input_info.padded_shape[2]);
    const auto channels = static_cast<std::size_t>(input_info.padded_shape[3]);
    const auto batch = static_cast<std::size_t>(input_info.padding[0].first);
    const auto top = static_cast<std::size_t>(input_info.padding[1].first);
    const auto left = static_cast<std::size_t>(input_info.padding[2].first);
    const auto channel = static_cast<std::size_t>(input_info.padding[3].first);
    const float scale = static_cast<float>(input_info.scale);
    const float zero_point = static_cast<float>(input_info.zero_point);
    for (int row = 0; row < normalized.rows; ++row) {
        const auto* source = normalized.ptr<float>(row);
        const auto row_offset = ((batch * height + row + top) * width + left) * channels + channel;
        for (int column = 0; column < normalized.cols; ++column) {
            input_buffer[row_offset + column * channels] = Quantize(source[column], scale, zero_point);
        }
    }
}

}
