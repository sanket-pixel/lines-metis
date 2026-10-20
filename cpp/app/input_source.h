// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "arguments.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <filesystem>
#include <optional>
#include <vector>

namespace lines::app {
class InputSource {
public:
    explicit InputSource(const Arguments& args);
    InputSource(const InputSource&) = delete;
    InputSource& operator=(const InputSource&) = delete;

    bool Read(cv::Mat& frame);
    const std::filesystem::path& frame_path() const;
    std::optional<double> timestamp_seconds() const;

private:
    InputType input_type_;
    cv::Mat image_;
    cv::VideoCapture capture_;
    std::vector<std::filesystem::path> image_paths_;
    std::size_t image_index_ = 0;
    std::filesystem::path frame_path_;
    std::optional<double> timestamp_seconds_;
    std::chrono::steady_clock::time_point capture_start_ = std::chrono::steady_clock::now();
};
}
