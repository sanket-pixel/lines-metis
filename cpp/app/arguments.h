// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

namespace lines::app {
enum class InputType { Image, Sequence, Video, Webcam };

struct Arguments {
    InputType input_type = InputType::Image;
    std::string input_path;
    int camera_index = 0;
    std::filesystem::path config_path;
    std::filesystem::path output_path;
    std::filesystem::path temporal_config_path;
    bool show = false;
    bool profile = false;
    std::size_t max_frames = 0;  // Zero means no limit.
};

std::optional<Arguments> ParseArguments(int argc, char** argv);
}
