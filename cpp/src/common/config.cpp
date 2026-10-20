// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <initializer_list>
#include <set>
#include <stdexcept>
#include <string>

namespace lines {
namespace {
void CheckKeys(const YAML::Node& node, std::initializer_list<std::string> allowed,
               const std::string& section) {
    if (!node.IsMap()) {
        throw std::runtime_error(section + " must be a YAML mapping");
    }
    const std::set<std::string> allowed_keys(allowed);
    std::set<std::string> seen;
    for (const auto& entry : node) {
        const auto key = entry.first.as<std::string>();
        if (allowed_keys.count(key) == 0 || !seen.insert(key).second) {
            throw std::runtime_error("Unknown or duplicate key in " + section + ": " + key);
        }
    }
}

std::filesystem::path ResolvePath(const std::filesystem::path& yaml_path,
                                  const std::string& value) {
    if (value.empty()) {
        throw std::runtime_error("Empty path in " + yaml_path.string());
    }
    return (yaml_path.parent_path() / value).lexically_normal();
}

SegmentThresholds LoadThresholds(const YAML::Node& preset) {
    CheckKeys(preset, {"max_distance", "angle_tolerance", "region_radius", "min_length",
                      "min_support", "min_pixels", "duplicate_distance", "merge_gap",
                      "merge_distance", "max_mean_distance"}, "selected preset");
    SegmentThresholds thresholds;
    thresholds.max_distance = preset["max_distance"].as<float>();
    thresholds.angle_tolerance = preset["angle_tolerance"].as<float>();
    thresholds.region_radius = preset["region_radius"].as<float>();
    thresholds.min_length = preset["min_length"].as<float>();
    thresholds.min_support = preset["min_support"].as<float>();
    thresholds.min_pixels = preset["min_pixels"].as<int>();
    thresholds.duplicate_distance = preset["duplicate_distance"].as<float>();
    thresholds.merge_gap = preset["merge_gap"].as<float>();
    thresholds.merge_distance = preset["merge_distance"].as<float>();
    thresholds.max_mean_distance = preset["max_mean_distance"].as<float>();
    return thresholds;
}

}

void ValidateSegmentThresholds(const SegmentThresholds& thresholds) {
    for (float value : {thresholds.max_distance, thresholds.angle_tolerance,
                        thresholds.region_radius, thresholds.min_length,
                        thresholds.min_support, thresholds.duplicate_distance,
                        thresholds.merge_gap, thresholds.merge_distance,
                        thresholds.max_mean_distance}) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Segment thresholds must be finite");
        }
    }
    if (!(thresholds.max_distance > 0 && thresholds.angle_tolerance > 0 &&
          thresholds.angle_tolerance < std::acos(-1.0f) / 2 &&
          thresholds.region_radius > 0 && thresholds.min_length > 0 &&
          thresholds.min_support > 0 && thresholds.min_support <= 1 &&
          thresholds.min_pixels >= 2 && thresholds.duplicate_distance >= 0 &&
          thresholds.merge_gap >= 0 && thresholds.merge_distance >= 0 &&
          thresholds.max_mean_distance > 0)) {
        throw std::runtime_error("Invalid segment thresholds: distances/length/support must be "
                                 "positive, support <= 1, angle < pi/2, min_pixels >= 2; "
                                 "join gap and merge/duplicate distances may be zero");
    }
}

PipelineConfig LoadConfig(const std::filesystem::path& config_path) {
    const auto path = std::filesystem::absolute(config_path).lexically_normal();
    try {
        const auto yaml = YAML::LoadFile(path.string());
        CheckKeys(yaml, {"inference", "segment_extractor"}, "config");
        const auto inference = yaml["inference"];
        const auto extractor = yaml["segment_extractor"];
        CheckKeys(inference, {"model_directory"}, "inference");
        CheckKeys(extractor, {"presets_file", "strictness", "min_length",
                              "joining_enabled", "max_join_gap"}, "segment_extractor");

        PipelineConfig config;
        config.inference.model_directory = ResolvePath(path, inference["model_directory"].as<std::string>());
        if (!std::filesystem::is_directory(config.inference.model_directory)) {
            throw std::runtime_error("Model directory does not exist: " +
                                     config.inference.model_directory.string());
        }

        const auto presets_path = ResolvePath(path, extractor["presets_file"].as<std::string>());
        const auto presets = YAML::LoadFile(presets_path.string());
        CheckKeys(presets, {"strict", "balanced", "relaxed"}, "presets file");
        const auto strictness = extractor["strictness"].as<std::string>();
        if (strictness != "strict" && strictness != "balanced" && strictness != "relaxed") {
            throw std::runtime_error("Unknown strictness preset: " + strictness);
        }
        const auto preset = presets[strictness];
        if (!preset || preset.IsNull()) {
            throw std::runtime_error("Preset '" + strictness + "' has no configured thresholds in " +
                                     presets_path.string());
        }

        auto& settings = config.segment_extractor;
        settings.thresholds = LoadThresholds(preset);
        ValidateSegmentThresholds(settings.thresholds);
        if (extractor["min_length"]) {
            settings.thresholds.min_length = extractor["min_length"].as<float>();
        }
        if (extractor["max_join_gap"]) {
            settings.thresholds.merge_gap = extractor["max_join_gap"].as<float>();
        }
        if (extractor["joining_enabled"]) {
            settings.joining_enabled = extractor["joining_enabled"].as<bool>();
        }
        ValidateSegmentThresholds(settings.thresholds);
        return config;
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot load config " + path.string() + ": " + error.what());
    }
}
}
