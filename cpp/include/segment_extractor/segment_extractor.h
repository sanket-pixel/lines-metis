// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include "common/config.h"
#include "common/types.h"
#include <vector>

namespace lines {

class SegmentExtractor {
public:
    explicit SegmentExtractor(SegmentExtractorConfig config);
    // Contiguous row-major network fields: distance in pixels, orientation in radians.
    std::vector<Segment> Process(const float* distance, const float* orientation,
                                 int width, int height) const;
private:
    SegmentExtractorConfig config_;
    std::vector<Segment> JoinSegments(std::vector<Segment> segments, const float* distance,
                                      const float* orientation, int width, int height) const;
    std::vector<Segment> SuppressDuplicates(std::vector<Segment> segments) const;
    bool Fit(const std::vector<int>& region, const float* distance,
             int width, Segment& segment) const;
    bool Validate(const Segment& segment, const float* distance,
                  const float* orientation, int width, int height) const;
};
}
