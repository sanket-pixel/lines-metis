// SPDX-License-Identifier: AGPL-3.0-or-later
#include "pipeline.h"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
constexpr const char* usage =
    "Usage: lines_metis --video <path> --config <config.yaml> [--output <path>] [--profile]\n"
    "Processes the video to EOF. Press Ctrl+C to stop.\n";
volatile std::sig_atomic_t stop_requested = 0;

struct Arguments {
    std::filesystem::path video_path;
    std::filesystem::path config_path;
    std::filesystem::path output_path;
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
        } else if (option == "--video" || option == "--config" || option == "--output") {
            if (++index >= argc || argv[index][0] == '\0' || argv[index][0] == '-') {
                throw std::invalid_argument("Missing value for " + option);
            }
            auto& path = option == "--video" ? args.video_path
                       : option == "--config" ? args.config_path : args.output_path;
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

void RunVideo(const Arguments& args, const lines::PipelineConfig& config) {
    if (!std::filesystem::is_regular_file(args.video_path)) {
        throw std::runtime_error("Video file does not exist: " + args.video_path.string());
    }
    cv::VideoCapture capture(args.video_path.string());
    cv::Mat frame;
    if (!capture.isOpened() || !capture.read(frame) || frame.empty()) {
        throw std::runtime_error("Cannot read video: " + args.video_path.string());
    }

    cv::VideoWriter output;
    const auto frame_size = frame.size();
    if (!args.output_path.empty()) {
        if (std::filesystem::exists(args.output_path) &&
            std::filesystem::equivalent(args.video_path, args.output_path)) {
            throw std::invalid_argument("Output must be different from the input video");
        }
        if (args.output_path.has_parent_path()) std::filesystem::create_directories(args.output_path.parent_path());
        const double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("Video has no valid frame rate");
        if (!output.open(args.output_path.string(), cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame_size)) {
            throw std::runtime_error("Cannot create output video: " + args.output_path.string());
        }
    }

    lines::Pipeline pipeline(config);
    lines::PipelineTimings total;
    std::size_t frames = 0;
    do {
        if (stop_requested) break;
        lines::PipelineTimings timings;
        const auto segments = pipeline.Process(frame, args.profile ? &timings : nullptr);
        if (args.profile) {
            total.preprocess_ms += timings.preprocess_ms;
            total.inference_ms += timings.inference_ms;
            total.postprocess_ms += timings.postprocess_ms;
            total.extraction_ms += timings.extraction_ms;
            total.total_ms += timings.total_ms;
        }
        if (output.isOpened()) {
            if (frame.size() != frame_size) throw std::runtime_error("Video frame size changed");
            DrawSegments(frame, segments);
            output.write(frame);
        }
        ++frames;
    } while (!stop_requested && capture.read(frame) && !frame.empty());

    output.release();
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
