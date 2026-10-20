// SPDX-License-Identifier: AGPL-3.0-or-later
#include "arguments.h"
#include "input_source.h"
#include "output_window.h"
#include "results_writer.h"
#include "pipeline/pipeline.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <csignal>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t stop_requested = 0;

void RequestStop(int) {
    stop_requested = 1;
}

void PrintConfig(const lines::PipelineConfig& config) {
    const auto& settings = config.segment_extractor;
    const auto& thresholds = settings.thresholds;
    std::cout << "Model directory: " << config.inference.model_directory.string() << '\n'
              << "Segment thresholds (network pixels; angles in radians):\n"
              << "  max_distance: " << thresholds.max_distance << '\n'
              << "  angle_tolerance: " << thresholds.angle_tolerance << '\n'
              << "  region_radius: " << thresholds.region_radius << '\n'
              << "  min_length: " << thresholds.min_length << '\n'
              << "  min_support: " << thresholds.min_support << '\n'
              << "  min_pixels: " << thresholds.min_pixels << '\n'
              << "  duplicate_distance: " << thresholds.duplicate_distance << '\n'
              << "  merge_gap: " << thresholds.merge_gap << '\n'
              << "  merge_distance: " << thresholds.merge_distance << '\n'
              << "  max_mean_distance: " << thresholds.max_mean_distance << '\n'
              << "Joining enabled: " << (settings.joining_enabled ? "true" : "false") << '\n';
}

void PrintSegments(const std::vector<lines::Segment>& segments) {
    std::cout << "Extracted segments: " << segments.size() << " (original-image coordinates)\n";
    for (std::size_t index = 0; index < std::min(segments.size(), std::size_t(5)); ++index) {
        const auto& segment = segments[index];
        std::cout << "  (" << segment.x1 << ", " << segment.y1 << ") -> ("
                  << segment.x2 << ", " << segment.y2 << ")\n";
    }
}

void PrintProfile(const lines::PipelineTimings& total, std::size_t frames) {
    std::cout << std::fixed << std::setprecision(3)
              << "Profile: average ms/frame over " << frames << " frame(s), including first frame\n"
              << "  Preprocessing:  " << total.preprocess_ms / frames << " ms\n"
              << "  Inference:      " << total.inference_ms / frames << " ms\n"
              << "  Postprocessing: " << total.postprocess_ms / frames << " ms\n"
              << "  Extraction:     " << total.extraction_ms / frames << " ms\n"
              << "  Pipeline total: " << total.total_ms / frames << " ms (includes coordinate mapping)\n"
              << "  Excludes model loading, capture, display and console output\n";
}

void RunInput(const lines::app::Arguments& args, const lines::PipelineConfig& config) {
    lines::app::InputSource input(args);
    cv::Mat frame;
    if (!input.Read(frame)) throw std::runtime_error("Input contains no readable frames");

    lines::app::ResultsWriter results(args, config);
    lines::Pipeline pipeline(config);
    lines::app::OutputWindow window(args.show);
    std::size_t frame_number = 0;
    lines::PipelineTimings total;
    do {
        if (stop_requested) break;
        lines::PipelineTimings timings;
        const auto segments = pipeline.Process(frame, (args.profile || !args.output_path.empty()) ? &timings : nullptr);
        if (args.profile) {
            total.preprocess_ms += timings.preprocess_ms;
            total.inference_ms += timings.inference_ms;
            total.postprocess_ms += timings.postprocess_ms;
            total.extraction_ms += timings.extraction_ms;
            total.total_ms += timings.total_ms;
        }
        results.Write(input, frame, segments, timings);
        ++frame_number;
        if (!args.show && args.output_path.empty()) {
            std::cout << "Frame " << frame_number << ": " << frame.cols << " x " << frame.rows << '\n';
            PrintSegments(segments);
        }
        if (!args.output_path.empty() && frame_number % 100 == 0) {
            std::cout << "Exported " << frame_number << " frames" << std::endl;
        }
        if (!window.Show(frame, segments)) break;
    } while (!stop_requested && (args.max_frames == 0 || frame_number < args.max_frames) && input.Read(frame));
    results.Finish();
    std::cout << "Processed " << frame_number << " frame(s)\n";
    if (args.profile && frame_number > 0) PrintProfile(total, frame_number);
}

}

int main(int argc, char** argv) {
    try {
        const auto args = lines::app::ParseArguments(argc, argv);
        if (!args) {
            return 0;
        }

        const lines::PipelineConfig config = lines::LoadConfig(args->config_path);
        PrintConfig(config);

        std::signal(SIGINT, RequestStop);
        RunInput(*args, config);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
