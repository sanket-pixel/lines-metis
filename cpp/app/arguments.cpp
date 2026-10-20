// SPDX-License-Identifier: AGPL-3.0-or-later
#include "arguments.h"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace lines::app {
namespace {
constexpr const char* usage =
    "Usage: lines_metis (--image <path> | --sequence <directory> | --video <path> | --webcam <index>)\n"
    "                   --config <config.yaml> [--max-frames <count>] [--show] [--profile] [--output <results.jsonl>] [--temporal-config <temporal.yaml>]\n"
    "       lines_metis <image> --config <config.yaml>\n"
    "Video ends at EOF. Use Ctrl+C, or Esc/Q in the preview window, to stop.\n";

int ParseInteger(std::string_view value, const std::string& option, int minimum) {
    int number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || number < minimum) {
        throw std::invalid_argument(option + " requires an integer >= " + std::to_string(minimum));
    }
    return number;
}
}

std::optional<Arguments> ParseArguments(int argc, char** argv) {
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::cout << usage;
        return std::nullopt;
    }

    Arguments args;
    bool source_set = false;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--profile") {
            if (args.profile) throw std::invalid_argument("Specify --profile only once");
            args.profile = true;
        } else if (option == "--show") {
            if (args.show) throw std::invalid_argument("Specify --show only once");
            args.show = true;
        } else if (option == "--image" || option == "--sequence" || option == "--video" || option == "--webcam") {
            if (source_set) throw std::invalid_argument("Choose exactly one input source");
            if (++index >= argc || std::string_view(argv[index]).empty() || argv[index][0] == '-') {
                throw std::invalid_argument("Missing value for " + option);
            }
            source_set = true;
            if (option == "--webcam") {
                args.input_type = InputType::Webcam;
                args.camera_index = ParseInteger(argv[index], option, 0);
            } else {
                args.input_type = option == "--image" ? InputType::Image :
                                  option == "--sequence" ? InputType::Sequence : InputType::Video;
                args.input_path = argv[index];
            }
        } else if (option == "--config" || option == "--output" || option == "--max-frames" || option == "--temporal-config") {
            if (++index >= argc || std::string_view(argv[index]).empty() || argv[index][0] == '-') {
                throw std::invalid_argument("Missing value for " + option);
            }
            if (option == "--config") {
                if (!args.config_path.empty()) throw std::invalid_argument("Specify --config only once");
                args.config_path = argv[index];
            } else if (option == "--temporal-config") {
                if (!args.temporal_config_path.empty()) throw std::invalid_argument("Specify --temporal-config only once");
                args.temporal_config_path = argv[index];
            } else if (option == "--output") {
                if (!args.output_path.empty()) throw std::invalid_argument("Specify --output only once");
                args.output_path = argv[index];
            } else {
                if (args.max_frames != 0) throw std::invalid_argument("Specify --max-frames only once");
                args.max_frames = static_cast<std::size_t>(ParseInteger(argv[index], option, 1));
            }
        } else if (!option.empty() && option.front() != '-' && !source_set) {
            args.input_path = option;
            source_set = true;
        } else {
            throw std::invalid_argument("Unexpected argument: " + option);
        }
    }
    if (!source_set || args.config_path.empty()) {
        throw std::invalid_argument(usage);
    }
    return args;
}
}
