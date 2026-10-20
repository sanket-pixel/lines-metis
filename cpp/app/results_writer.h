// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "arguments.h"
#include "input_source.h"
#include "pipeline/pipeline.h"

#include <fstream>

namespace lines::app {
class ResultsWriter {
public:
    ResultsWriter(const Arguments& args, const PipelineConfig& config);
    void Write(const InputSource& input, const cv::Mat& frame,
               const std::vector<Segment>& segments, const PipelineTimings& timings);
    void Finish();

private:
    std::ofstream stream_;
    std::size_t frames_ = 0;
};
}
