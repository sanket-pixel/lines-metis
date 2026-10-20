// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

namespace lines {
struct Segment { float x1, y1, x2, y2; };

struct ResizeTransform {
    int original_width = 0;
    int original_height = 0;
    int network_width = 320;
    int network_height = 240;
    // Original/network scale; Pipeline alone applies (coordinate + 0.5)*scale - 0.5.
    double scale_x = 0;
    double scale_y = 0;
};

}
