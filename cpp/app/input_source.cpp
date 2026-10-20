// SPDX-License-Identifier: AGPL-3.0-or-later
#include "input_source.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lines::app {
namespace {
std::optional<double> FilenameTimestamp(const std::filesystem::path& path) {
    const auto stem = path.stem().string();
    double timestamp = 0;
    const auto result = std::from_chars(stem.data(), stem.data() + stem.size(), timestamp);
    if (result.ec == std::errc() && result.ptr == stem.data() + stem.size() && std::isfinite(timestamp)) {
        return timestamp;
    }
    return std::nullopt;
}
}

InputSource::InputSource(const Arguments& args) : input_type_(args.input_type) {
    switch (input_type_) {
    case InputType::Image:
        image_ = cv::imread(args.input_path, cv::IMREAD_COLOR);
        if (image_.empty()) throw std::runtime_error("Cannot read image: " + args.input_path);
        frame_path_ = std::filesystem::absolute(args.input_path).lexically_normal();
        timestamp_seconds_ = FilenameTimestamp(frame_path_);
        break;
    case InputType::Sequence:
        for (const auto& entry : std::filesystem::directory_iterator(args.input_path)) {
            const auto extension = entry.path().extension().string();
            if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg")) {
                image_paths_.push_back(std::filesystem::absolute(entry.path()).lexically_normal());
            }
        }
        std::sort(image_paths_.begin(), image_paths_.end());
        if (image_paths_.empty()) throw std::runtime_error("No images in sequence: " + args.input_path);
        break;
    case InputType::Video:
        if (!capture_.open(args.input_path)) throw std::runtime_error("Cannot open video: " + args.input_path);
        break;
    case InputType::Webcam: {
#if defined(__linux__)
        constexpr auto backend = cv::CAP_V4L2;
#else
        constexpr auto backend = cv::CAP_ANY;
#endif
        if (!capture_.open(args.camera_index, backend)) {
            throw std::runtime_error("Cannot open webcam: " + std::to_string(args.camera_index));
        }
        capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        break;
    }
    }
}

bool InputSource::Read(cv::Mat& frame) {
    if (input_type_ == InputType::Image) {
        frame = image_;
        image_.release();
        return !frame.empty();
    }
    if (input_type_ == InputType::Sequence) {
        if (image_index_ == image_paths_.size()) {
            frame.release();
            return false;
        }
        frame_path_ = image_paths_[image_index_++];
        timestamp_seconds_ = FilenameTimestamp(frame_path_);
        frame = cv::imread(frame_path_.string(), cv::IMREAD_COLOR);
        if (frame.empty()) throw std::runtime_error("Cannot read sequence image: " + frame_path_.string());
        return true;
    }
    if (capture_.read(frame) && !frame.empty()) {
        if (input_type_ == InputType::Video) {
            const double position = capture_.get(cv::CAP_PROP_POS_MSEC) / 1000;
            timestamp_seconds_ = std::isfinite(position) && position >= 0 ? std::optional<double>(position) : std::nullopt;
        } else {
            timestamp_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - capture_start_).count();
        }
        return true;
    }
    frame.release();
    if (input_type_ == InputType::Webcam) {
        throw std::runtime_error("Failed to read a frame from the webcam");
    }
    return false;
}

const std::filesystem::path& InputSource::frame_path() const {
    return frame_path_;
}

std::optional<double> InputSource::timestamp_seconds() const {
    return timestamp_seconds_;
}

}
