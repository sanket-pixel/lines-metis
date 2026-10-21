// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/config.h"
#include "common/types.h"
#include "inference/aipu_inference.h"
#include "preprocessing.h"
#include "postprocessing.h"
#include "segment_extractor.h"

#include <vector>

namespace lines {
struct PipelineTimings {
    double preprocess_ms = 0;
    double inference_ms = 0;
    double postprocess_ms = 0;
    double extraction_ms = 0;
    double total_ms = 0;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    // Calls preprocessing, inference, postprocessing, extraction, then maps once.
    // Temporal consumers may use the returned result without changing these stages.
    std::vector<Segment> Process(const cv::Mat& image, PipelineTimings* timings = nullptr);  // Original-image endpoints.

private:
    static void MapToOriginal(std::vector<Segment>& segments, const ResizeTransform& transform);

    AipuContext aipu_context_;  // Outlives the attached model backend.
    AipuInference inference_;  // Supplies validated metadata to stages.
    Preprocessing preprocessing_;
    Postprocessing postprocessing_;
    SegmentExtractor segment_extractor_;
    std::vector<float> distance_;
    std::vector<float> orientation_;
};
}
