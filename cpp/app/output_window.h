// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "common/types.h"

#include <opencv2/core/mat.hpp>
#include <vector>

namespace lines::app {
class OutputWindow {
public:
    explicit OutputWindow(bool enabled);
    ~OutputWindow();
    OutputWindow(const OutputWindow&) = delete;
    OutputWindow& operator=(const OutputWindow&) = delete;

    bool Show(const cv::Mat& frame, const std::vector<Segment>& segments);

private:
    bool enabled_;
    double previous_time_ = 0;
    double fps_ = 0;
};
}
