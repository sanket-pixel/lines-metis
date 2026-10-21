// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include "inference/aipu_inference.h"

#include <opencv2/core/mat.hpp>

namespace lines {
class Preprocessing {
public:
    // BGR uint8 -> grayscale -> INTER_AREA 320x240 -> /255 -> quantized, padded NHWC int8.
    ResizeTransform Process(const cv::Mat& image, const AipuTensorInfo& input_info,
                            std::int8_t* input_buffer, std::size_t buffer_size) const;
};
}
