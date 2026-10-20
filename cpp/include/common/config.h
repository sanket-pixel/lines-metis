// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <filesystem>

namespace lines {
// Filled from the selected YAML preset; missing or invalid values are errors.
struct SegmentThresholds {
    float max_distance{};
    float angle_tolerance{};
    float region_radius{};
    float min_length{};
    float min_support{};
    int min_pixels{};
    float duplicate_distance{};
    float merge_gap{};
    float merge_distance{};
    float max_mean_distance{};
};

void ValidateSegmentThresholds(const SegmentThresholds& thresholds);

// Resolved settings only; preset selection and overrides happen during loading.
struct SegmentExtractorConfig {
    SegmentThresholds thresholds;
    bool joining_enabled = true;
};

struct InferenceConfig {
    std::filesystem::path model_directory;  // Required; no implicit model path.
};

struct PipelineConfig {
    InferenceConfig inference;
    SegmentExtractorConfig segment_extractor;
};

// Resolve paths relative to their YAML file; reject incomplete presets.
// Apply min_length and max_join_gap overrides after loading the selected preset.
PipelineConfig LoadConfig(const std::filesystem::path& config_path);
}
