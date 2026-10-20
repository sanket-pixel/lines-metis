// SPDX-License-Identifier: AGPL-3.0-or-later
#include "pipeline/pipeline.h"

#include <opencv2/videoio.hpp>
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
    "Usage: lines_metis --video <path> --config <config.yaml> [--profile]\n"
    "Processes the video to EOF. Press Ctrl+C to stop.\n";
volatile std::sig_atomic_t stop_requested = 0;

struct Arguments {
    std::filesystem::path video_path;
    std::filesystem::path config_path;
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
        } else if (option == "--video" || option == "--config") {
            if (++index >= argc || argv[index][0] == '\0' || argv[index][0] == '-') {
                throw std::invalid_argument("Missing value for " + option);
            }
            auto& path = option == "--video" ? args.video_path : args.config_path;
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
              << "  Excludes model loading, video decoding and console output\n";
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

    lines::Pipeline pipeline(config);
    lines::PipelineTimings total;
    std::size_t frames = 0;
    do {
        if (stop_requested) break;
        lines::PipelineTimings timings;
        pipeline.Process(frame, args.profile ? &timings : nullptr);
        if (args.profile) {
            total.preprocess_ms += timings.preprocess_ms;
            total.inference_ms += timings.inference_ms;
            total.postprocess_ms += timings.postprocess_ms;
            total.extraction_ms += timings.extraction_ms;
            total.total_ms += timings.total_ms;
        }
        ++frames;
    } while (!stop_requested && capture.read(frame) && !frame.empty());

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
