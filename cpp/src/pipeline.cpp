// SPDX-License-Identifier: AGPL-3.0-or-later
#include "pipeline.h"

#include <chrono>

namespace lines {
Pipeline::Pipeline(const PipelineConfig& config)
    : inference_(aipu_context_, config.inference),
      postprocessing_(inference_.output_info(0), inference_.output_info(1)),
      segment_extractor_(config.segment_extractor) {}

std::vector<Segment> Pipeline::Process(const cv::Mat& image, PipelineTimings* timings) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto& input_info = inference_.input_info();
    const auto transform = preprocessing_.Process(image, input_info, inference_.input_buffer(),
                                                 input_info.size_bytes);
    const auto pre_end = Clock::now();

    inference_.Process();
    const auto inf_end = Clock::now();
    postprocessing_.Process(inference_.output_buffer(0), inference_.output_buffer(1), distance_, orientation_);
    const auto post_end = Clock::now();

    auto segments = segment_extractor_.Process(distance_.data(), orientation_.data(),
                                              transform.network_width, transform.network_height);
    const auto seg_end = Clock::now();
    MapToOriginal(segments, transform);
    const auto end = Clock::now();

    if (timings) {
        using Milliseconds = std::chrono::duration<double, std::milli>;
        timings->preprocess_ms = Milliseconds(pre_end - start).count();
        timings->inference_ms = Milliseconds(inf_end - pre_end).count();
        timings->postprocess_ms = Milliseconds(post_end - inf_end).count();
        timings->extraction_ms = Milliseconds(seg_end - post_end).count();
        timings->total_ms = Milliseconds(end - start).count();
    }
    return segments;
}

void Pipeline::MapToOriginal(std::vector<Segment>& segments, const ResizeTransform& transform) {
    for (auto& segment : segments) {
        segment.x1 = float((double(segment.x1) + 0.5) * transform.scale_x - 0.5);
        segment.y1 = float((double(segment.y1) + 0.5) * transform.scale_y - 0.5);
        segment.x2 = float((double(segment.x2) + 0.5) * transform.scale_x - 0.5);
        segment.y2 = float((double(segment.y2) + 0.5) * transform.scale_y - 0.5);
    }
}
}
