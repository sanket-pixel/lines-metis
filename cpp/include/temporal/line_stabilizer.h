// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"
#include <opencv2/core.hpp>
#include <filesystem>
#include <optional>
#include <vector>

namespace lines {
struct TemporalConfig {
    int confirmation_frames = 2;
    int max_missing_frames = 2;
    float observation_weight = 0.6f;
    float max_angle_degrees = 10;
    float max_normal_distance = 4;
    double max_time_gap = 0.25;
};

TemporalConfig LoadTemporalConfig(const std::filesystem::path& path);

// Stateful consumer of original-image endpoints. One instance per stream.
class LineStabilizer {
public:
    explicit LineStabilizer(const TemporalConfig& config = {});
    std::vector<Segment> Process(const cv::Mat& image, const std::vector<Segment>& segments,
                                 std::optional<double> timestamp_seconds = std::nullopt);
    void Reset();

private:
    struct Track {
        Segment segment;
        int hits = 1;
        int missing = 0;
        bool confirmed = false;
    };
    std::vector<bool> Predict(const cv::Mat& gray);
    float MatchCost(const Segment& prediction, const Segment& observation) const;
    Segment Smooth(const Segment& prediction, Segment observation) const;

    TemporalConfig config_;
    cv::Mat previous_gray_;
    cv::Size original_size_;
    std::optional<double> previous_timestamp_;
    std::vector<Track> tracks_;
};
}
