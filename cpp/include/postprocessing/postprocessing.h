// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "inference/aipu_inference.h"

namespace lines {
class Postprocessing {
public:
    Postprocessing(AipuTensorInfo distance_info, AipuTensorInfo orientation_info);
    // Crop and independently dequantize each head, then decode
    // distance = 5*exp(-normalized), orientation = pi*fraction.
    // Output vectors are resized/reused: row-major 320x240, pixels and radians.
    void Process(const std::int8_t* distance_buffer, const std::int8_t* orientation_buffer,
                 std::vector<float>& distance, std::vector<float>& orientation) const;

private:
    AipuTensorInfo distance_info_;
    AipuTensorInfo orientation_info_;
};
}
