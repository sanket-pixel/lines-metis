// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include "inference/aipu_inference.h"

#include <opencv2/core/mat.hpp>

namespace lines {
class Preprocessing {
public:
    // BGR uint8 -> grayscale -> INTER_AREA 320x240 -> float32 /255.
    // Float reference output is contiguous CV_32FC1.
    ResizeTransform Process(const cv::Mat& image, cv::Mat& normalized) const;

    // End-to-end input preparation; also the intended boundary for a fused kernel.
    ResizeTransform Process(const cv::Mat& image, const AipuTensorInfo& input_info,
                            std::int8_t* input_buffer, std::size_t buffer_size) const;

    // Float32 affine quantization, ties-to-even, saturation, then NHWC padding.
    void QuantizeAndPad(const cv::Mat& normalized, const AipuTensorInfo& input_info,
                        std::int8_t* input_buffer, std::size_t buffer_size) const;
};
}
