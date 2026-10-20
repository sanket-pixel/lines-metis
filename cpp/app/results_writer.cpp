// SPDX-License-Identifier: AGPL-3.0-or-later
#include "results_writer.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <stdexcept>

namespace lines::app {
ResultsWriter::ResultsWriter(const Arguments& args, const PipelineConfig& config) {
    if (args.output_path.empty()) return;
    if (std::filesystem::exists(args.output_path)) {
        throw std::runtime_error("Results file already exists: " + args.output_path.string());
    }
    nlohmann::json provenance;
    std::ifstream provenance_stream(config.inference.model_directory / "provenance.json");
    if (!provenance_stream) throw std::runtime_error("Cannot read model provenance for result export");
    provenance_stream >> provenance;
    const auto preset = YAML::LoadFile(args.config_path.string())["segment_extractor"]["strictness"].as<std::string>();
    const auto& t = config.segment_extractor.thresholds;
    const char* type = args.input_type == InputType::Image ? "image" :
                       args.input_type == InputType::Sequence ? "sequence" :
                       args.input_type == InputType::Video ? "video" : "webcam";
    const char* timestamp_source = args.input_type == InputType::Webcam ? "elapsed_since_source_open_seconds" :
                                   args.input_type == InputType::Video ? "video_position_seconds" : "numeric_filename_seconds";
    nlohmann::json metadata = {
        {"type", "metadata"}, {"schema_version", 1}, {"preset", preset},
        {"coordinate_space", "original_image_pixels"}, {"segment_format", {"x1", "y1", "x2", "y2"}},
        {"input", {{"type", type}, {"path", args.input_type == InputType::Webcam ? "" : std::filesystem::absolute(args.input_path).lexically_normal().string()}, {"camera_index", args.camera_index}}},
        {"timestamp_source", timestamp_source}, {"timing_units", "milliseconds"},
        {"timing_scope", "Pipeline only, including coordinate mapping; excludes loading, capture, rendering and export"},
        {"config_file", std::filesystem::absolute(args.config_path).lexically_normal().string()},
        {"model_directory", config.inference.model_directory.string()}, {"model_provenance", provenance},
        {"extractor", {{"max_distance", t.max_distance}, {"angle_tolerance", t.angle_tolerance},
                       {"region_radius", t.region_radius}, {"min_length", t.min_length},
                       {"min_support", t.min_support}, {"min_pixels", t.min_pixels},
                       {"duplicate_distance", t.duplicate_distance}, {"merge_gap", t.merge_gap},
                       {"merge_distance", t.merge_distance}, {"max_mean_distance", t.max_mean_distance},
                       {"joining_enabled", config.segment_extractor.joining_enabled}}}
    };
    if (!args.temporal_config_path.empty()) {
        const auto temporal = YAML::LoadFile(args.temporal_config_path.string());
        metadata["schema_version"] = 2;
        metadata["temporal"] = {{"method", "motion_compensated_line_stabilizer_v1"}, {"working_max_width", 640},
            {"timing_scope", "temporal_ms includes motion estimation and association; additional to Pipeline"}};
        for (const auto& entry : temporal) {
            const auto key = entry.first.as<std::string>();
            metadata["temporal"][key] = entry.second.as<double>();
        }
    }
    stream_.open(args.output_path);
    if (!stream_) throw std::runtime_error("Cannot create results file: " + args.output_path.string());
    stream_ << metadata.dump() << '\n';
    if (!stream_) throw std::runtime_error("Cannot write results metadata");
}

void ResultsWriter::Write(const InputSource& input, const cv::Mat& frame,
                           const std::vector<Segment>& segments, const PipelineTimings& timings,
                           const std::vector<Segment>* raw_segments, double temporal_ms) {
    if (!stream_.is_open()) return;
    auto endpoints = nlohmann::json::array();
    for (const auto& segment : segments) {
        endpoints.push_back({segment.x1, segment.y1, segment.x2, segment.y2});
    }
    const auto timestamp = input.timestamp_seconds();
    nlohmann::json record = {
        {"type", "frame"}, {"index", frames_}, {"file", input.frame_path().filename().string()},
        {"timestamp_seconds", timestamp ? nlohmann::json(*timestamp) : nlohmann::json(nullptr)},
        {"width", frame.cols}, {"height", frame.rows}, {"segments", endpoints},
        {"timings_ms", {{"preprocessing", timings.preprocess_ms}, {"inference", timings.inference_ms},
                        {"postprocessing", timings.postprocess_ms}, {"extraction", timings.extraction_ms},
                        {"pipeline", timings.total_ms}}}
    };
    if (raw_segments) {
        record["temporal_ms"] = temporal_ms;
        record["raw_segments"] = nlohmann::json::array();
        for (const auto& s : *raw_segments) record["raw_segments"].push_back({s.x1,s.y1,s.x2,s.y2});
    }
    stream_ << record.dump() << '\n';
    if (!stream_) throw std::runtime_error("Cannot write frame results");
    ++frames_;
}

void ResultsWriter::Finish() {
    if (!stream_.is_open()) return;
    stream_ << nlohmann::json({{"type", "summary"}, {"frames", frames_}}).dump() << '\n';
    stream_.close();
    if (!stream_) throw std::runtime_error("Cannot finish results file");
}
}
