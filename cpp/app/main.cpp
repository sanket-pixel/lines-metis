// SPDX-License-Identifier: AGPL-3.0-or-later
#include "pipeline.h"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
constexpr const char* usage =
    "Usage: lines_metis --video <path> --config <config.yaml> [--output <path>]\n"
    "                   [--detections <path.jsonl>] [--profile]\n"
    "Processes the video to EOF. Press Ctrl+C to stop.\n";
volatile std::sig_atomic_t stop_requested = 0;

struct Arguments {
    std::filesystem::path video_path;
    std::filesystem::path config_path;
    std::filesystem::path output_path;
    std::filesystem::path detections_path;
    bool profile = false;
};

std::optional<Arguments> ParseArguments(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::cout << usage;
        return std::nullopt;
    }
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--profile") {
            if (args.profile) throw std::invalid_argument("Specify --profile only once");
            args.profile = true;
        } else if (option == "--video" || option == "--config" || option == "--output" || option == "--detections") {
            if (++index >= argc || argv[index][0] == '\0' || argv[index][0] == '-') {
                throw std::invalid_argument("Missing value for " + option);
            }
            auto& path = option == "--video" ? args.video_path
                       : option == "--config" ? args.config_path
                       : option == "--output" ? args.output_path : args.detections_path;
            if (!path.empty()) throw std::invalid_argument("Specify " + option + " only once");
            path = argv[index];
        } else {
            throw std::invalid_argument("Unexpected argument: " + option);
        }
    }
    if (args.video_path.empty() || args.config_path.empty()) throw std::invalid_argument(usage);
    return args;
}

void RequestStop(int) {
    stop_requested = 1;
}

void PrintProfile(const lines::PipelineTimings& total, std::size_t frames) {
    std::cout << std::fixed << std::setprecision(3)
              << "Profile: average ms/frame over " << frames << " frame(s), including first frame\n"
              << "  Preprocessing:  " << total.preprocess_ms / frames << " ms\n"
              << "  Inference:      " << total.inference_ms / frames << " ms\n"
              << "  Postprocessing: " << total.postprocess_ms / frames << " ms\n"
              << "  Extraction:     " << total.extraction_ms / frames << " ms\n"
              << "  Pipeline total: " << total.total_ms / frames << " ms (includes coordinate mapping)\n"
              << "  Processing FPS: " << 1000.0 * frames / total.total_ms << '\n'
              << "  Excludes model loading, video decoding, drawing, encoding and console output\n";
}

void DrawSegments(cv::Mat& frame, const std::vector<lines::Segment>& segments) {
    for (const auto& segment : segments) {
        const cv::Point start(cvRound(segment.x1 * 16), cvRound(segment.y1 * 16));
        const cv::Point end(cvRound(segment.x2 * 16), cvRound(segment.y2 * 16));
        cv::line(frame, start, end, cv::Scalar(218, 232, 99), 1, cv::LINE_AA, 4);
    }
}

void WriteMetadata(std::ostream& stream, const Arguments& args, const lines::PipelineConfig& config,
                   double fps) {
    const auto yaml = YAML::LoadFile(args.config_path.string());
    std::ifstream provenance_file(config.inference.model_directory / "provenance.json");
    const auto provenance = nlohmann::json::parse(provenance_file);
    const auto& settings = config.segment_extractor;
    const auto& t = settings.thresholds;
    const nlohmann::json metadata = {
        {"type", "metadata"}, {"schema_version", 1},
        {"coordinate_space", "original_image_pixels"}, {"segment_format", {"x1", "y1", "x2", "y2"}},
        {"preset", yaml["segment_extractor"]["strictness"].as<std::string>()},
        {"input", {{"type", "video"}, {"path", std::filesystem::canonical(args.video_path).string()}, {"fps", fps}}},
        {"config_file", std::filesystem::canonical(args.config_path).string()},
        {"model_directory", config.inference.model_directory.string()}, {"model_provenance", provenance},
        {"extractor", {{"max_distance", t.max_distance}, {"angle_tolerance", t.angle_tolerance},
            {"region_radius", t.region_radius}, {"min_length", t.min_length}, {"min_support", t.min_support},
            {"min_pixels", t.min_pixels}, {"duplicate_distance", t.duplicate_distance},
            {"merge_gap", t.merge_gap}, {"merge_distance", t.merge_distance},
            {"max_mean_distance", t.max_mean_distance}, {"joining_enabled", settings.joining_enabled}}},
        {"timing_units", "milliseconds"},
        {"timing_scope", "Pipeline including first frame and coordinate mapping; excludes model loading, video decoding, JSON export, drawing and encoding"},
        {"timestamp_source", "frame_index_divided_by_video_fps"}
    };
    stream << metadata.dump() << '\n';
}

void WriteDetections(std::ostream& stream, std::size_t index, const cv::Size& size,
                     double fps, const std::vector<lines::Segment>& segments,
                     const lines::PipelineTimings& timings) {
    auto endpoints = nlohmann::json::array();
    for (const auto& s : segments) endpoints.push_back({s.x1, s.y1, s.x2, s.y2});
    const nlohmann::json record = {
        {"type", "frame"}, {"index", index}, {"timestamp_seconds", index / fps},
        {"width", size.width}, {"height", size.height}, {"segments", endpoints},
        {"timings_ms", {{"preprocessing", timings.preprocess_ms}, {"inference", timings.inference_ms},
            {"postprocessing", timings.postprocess_ms}, {"extraction", timings.extraction_ms}, {"pipeline", timings.total_ms}}}
    };
    stream << record.dump() << '\n';
}

void RunVideo(const Arguments& args, const lines::PipelineConfig& config) {
    if (!std::filesystem::is_regular_file(args.video_path)) {
        throw std::runtime_error("Video file does not exist: " + args.video_path.string());
    }
    cv::VideoCapture capture(args.video_path.string());
    cv::Mat frame;
    if (!capture.isOpened() || !capture.read(frame) || frame.empty()) {
        throw std::runtime_error("Cannot read video: " + args.video_path.string());
    }

    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("Video has no valid frame rate");
    for (const auto& path : {args.output_path, args.detections_path}) {
        if (path.empty()) continue;
        const auto resolved = std::filesystem::weakly_canonical(path);
        if (resolved == std::filesystem::canonical(args.video_path) ||
            resolved == std::filesystem::canonical(args.config_path)) {
            throw std::invalid_argument("Output must differ from input video and config");
        }
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    }
    if (!args.detections_path.empty() && (args.detections_path.extension() != ".jsonl" ||
        (!args.output_path.empty() && std::filesystem::weakly_canonical(args.output_path) ==
                                      std::filesystem::weakly_canonical(args.detections_path)))) {
        throw std::invalid_argument("Detections must use a separate .jsonl file");
    }

    lines::Pipeline pipeline(config);
    std::ofstream detections;
    auto partial_path = args.detections_path;
    partial_path += ".partial";
    if (!args.detections_path.empty()) {
        detections.exceptions(std::ios::failbit | std::ios::badbit);
        detections.open(partial_path);
        WriteMetadata(detections, args, config, fps);
    }
    cv::VideoWriter output;
    const auto frame_size = frame.size();
    if (!args.output_path.empty()) {
        if (!output.open(args.output_path.string(), cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame_size)) {
            throw std::runtime_error("Cannot create output video: " + args.output_path.string());
        }
    }

    lines::PipelineTimings total;
    std::size_t frames = 0;
    do {
        if (stop_requested) break;
        lines::PipelineTimings timings;
        if (frame.size() != frame_size) throw std::runtime_error("Video frame size changed");
        const auto segments = pipeline.Process(frame, args.profile || detections.is_open() ? &timings : nullptr);
        if (args.profile) {
            total.preprocess_ms += timings.preprocess_ms;
            total.inference_ms += timings.inference_ms;
            total.postprocess_ms += timings.postprocess_ms;
            total.extraction_ms += timings.extraction_ms;
            total.total_ms += timings.total_ms;
        }
        if (detections.is_open()) WriteDetections(detections, frames, frame_size, fps, segments, timings);
        if (output.isOpened()) {
            DrawSegments(frame, segments);
            output.write(frame);
        }
        ++frames;
    } while (!stop_requested && capture.read(frame) && !frame.empty());

    output.release();
    if (detections.is_open()) {
        detections << nlohmann::json({{"type", "summary"}, {"frames", frames}, {"completed", !stop_requested}}).dump() << '\n';
        detections.close();
        if (!stop_requested) std::filesystem::rename(partial_path, args.detections_path);
        std::cout << "Saved detections: " << (stop_requested ? partial_path : args.detections_path).string() << '\n';
    }
    if (!args.output_path.empty()) std::cout << "Saved output: " << args.output_path.string() << '\n';
    std::cout << "Processed " << frames << " frame(s)\n";
    if (args.profile && frames > 0) PrintProfile(total, frames);
}
}

int main(int argc, char** argv) {
    try {
        const auto args = ParseArguments(argc, argv);
        if (!args) return 0;
        const auto config = lines::LoadConfig(args->config_path);
        std::signal(SIGINT, RequestStop);
        RunVideo(*args, config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
