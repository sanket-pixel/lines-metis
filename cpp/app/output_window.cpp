// SPDX-License-Identifier: AGPL-3.0-or-later
#include "output_window.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>

namespace lines::app {
namespace {
constexpr const char* window_name = "lines-metis | Live detections";
}

OutputWindow::OutputWindow(bool enabled) : enabled_(enabled) {
    if (enabled_) cv::namedWindow(window_name, cv::WINDOW_NORMAL);
}

OutputWindow::~OutputWindow() {
    if (enabled_) cv::destroyAllWindows();
}

bool OutputWindow::Show(const cv::Mat& frame, const std::vector<Segment>& segments) {
    if (!enabled_) return true;
    const double now = double(cv::getTickCount()) / cv::getTickFrequency();
    if (previous_time_ > 0 && now > previous_time_) {
        const double current_fps = 1.0 / (now - previous_time_);
        fps_ = fps_ == 0 ? current_fps : 0.9 * fps_ + 0.1 * current_fps;
    }
    previous_time_ = now;
    cv::Mat overlay = frame.clone();
    for (const auto& segment : segments) {
        const cv::Point start(cvRound(segment.x1 * 16), cvRound(segment.y1 * 16));
        const cv::Point end(cvRound(segment.x2 * 16), cvRound(segment.y2 * 16));
        cv::line(overlay, start, end, cv::Scalar(70, 255, 100), 2, cv::LINE_AA, 4);
    }
    const std::string label = cv::format("%.1f FPS | %zu segments | Esc/Q to stop", fps_, segments.size());
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(overlay.cols, 32), cv::Scalar(24, 26, 29), cv::FILLED);
    cv::putText(overlay, label, cv::Point(12, 23), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(235, 238, 240), 1, cv::LINE_AA);
    cv::imshow(window_name, overlay);
    const int key = cv::waitKey(1) & 0xff;
    return key != 27 && key != 'q' && key != 'Q' &&
           cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) >= 1;
}
}
