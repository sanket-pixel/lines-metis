// SPDX-License-Identifier: AGPL-3.0-or-later
#include "segment_extractor/segment_extractor.h"
#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lines {
namespace {
constexpr float pi = 3.14159265358979323846f;

float Length(const Segment& segment) {
    return std::hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);
}

float Angle(const Segment& segment) {
    return std::atan2(segment.y2 - segment.y1, segment.x2 - segment.x1);
}

struct SegmentGeometry {
    float length, angle, ux, uy;
    explicit SegmentGeometry(const Segment& segment)
        : length(Length(segment)), angle(Angle(segment)),
          ux((segment.x2-segment.x1)/length), uy((segment.y2-segment.y1)/length) {}
};

float AngleError(float first, float second) {
    const float difference = std::abs(first - second);
    if (difference <= pi / 2) return difference;
    if (difference <= pi) return pi - difference;
    if (double(difference) <= 1.5 * double(pi)) return difference - pi;
    return std::abs(std::remainder(first - second, pi));
}

class OrientationCache {
public:
    struct Direction { double cosine, sine; };
    Direction Get(float angle) {
        std::uint32_t key;
        std::memcpy(&key, &angle, sizeof(key));
        auto hash = key ^ (key >> 16);
        hash *= 0x7feb352dU;
        hash ^= hash >> 15;
        auto& entry = entries_[hash & (entries_.size()-1)];
        if (entry.key != key) {
            entry.key = key;
            entry.direction = {std::cos(2.0 * angle), std::sin(2.0 * angle)};
        }
        return entry.direction;
    }
private:
    struct Entry { std::uint32_t key = 0x7f800000U; Direction direction{}; };
    std::array<Entry, 512> entries_{};
};

float Sample(const float* field, float x, float y, int width, int height) {
    x = std::clamp(x, 0.0f, float(width - 1));
    y = std::clamp(y, 0.0f, float(height - 1));
    const int x0 = int(x), y0 = int(y);
    const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float dx = x - x0, dy = y - y0;
    return (1 - dy) * ((1 - dx) * field[y0 * width + x0] + dx * field[y0 * width + x1])
           + dy * ((1 - dx) * field[y1 * width + x0] + dx * field[y1 * width + x1]);
}

void Canonicalize(Segment& segment) {
    if (segment.x2 < segment.x1 || (segment.x1 == segment.x2 && segment.y2 < segment.y1)) {
        std::swap(segment.x1, segment.x2);
        std::swap(segment.y1, segment.y2);
    }
}

bool ClipToImage(Segment& segment, int width, int height) {
    const float dx = segment.x2 - segment.x1, dy = segment.y2 - segment.y1;
    float start = 0, end = 1;
    for (const auto& axis : {std::array<float, 3>{segment.x1, dx, float(width - 1)},
                             std::array<float, 3>{segment.y1, dy, float(height - 1)}}) {
        const auto position = axis[0], direction = axis[1], limit = axis[2];
        if (direction == 0) {
            if (position < 0 || position > limit) return false;
            continue;
        }
        const auto first = -position / direction, second = (limit - position) / direction;
        start = std::max(start, std::min(first, second));
        end = std::min(end, std::max(first, second));
        if (start > end) return false;
    }
    segment = {std::clamp(segment.x1 + start * dx, 0.0f, float(width - 1)),
               std::clamp(segment.y1 + start * dy, 0.0f, float(height - 1)),
               std::clamp(segment.x1 + end * dx, 0.0f, float(width - 1)),
               std::clamp(segment.y1 + end * dy, 0.0f, float(height - 1))};
    Canonicalize(segment);
    return true;
}

void SortSeeds(std::vector<int>& seeds, const float* distance) {
    static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
    // Nonnegative finite IEEE floats sort by bits. Normalize signed zero to keep ties stable.
    auto key = [distance](int index) {
        std::uint32_t bits;
        std::memcpy(&bits, distance + index, sizeof(bits));
        return (bits & 0x7fffffffU) == 0 ? 0U : bits;
    };
    std::vector<int> scratch(seeds.size());
    for (int shift : {0, 11, 22}) {
        std::array<std::size_t, 2048> offsets{};
        for (int index : seeds) ++offsets[(key(index) >> shift) & 2047];
        std::size_t offset = 0;
        for (auto& bucket : offsets) { const auto size = bucket; bucket = offset; offset += size; }
        for (int index : seeds) scratch[offsets[(key(index) >> shift) & 2047]++] = index;
        seeds.swap(scratch);
    }
}

void SortByLength(std::vector<Segment>& segments) {
    std::stable_sort(segments.begin(), segments.end(), [](const auto& first, const auto& second) {
        return Length(first) > Length(second);
    });
}
}

SegmentExtractor::SegmentExtractor(SegmentExtractorConfig config) : config_(std::move(config)) {
    ValidateSegmentThresholds(config_.thresholds);
}

bool SegmentExtractor::Fit(const int* region, std::size_t region_size, const float* distance,
                           int width, Segment& segment) const {
    const auto& thresholds = config_.thresholds;
    double weight_sum = 0, mean_x = 0, mean_y = 0;
    for (std::size_t sample = 0; sample < region_size; ++sample) {
        const int index = region[sample];
        const double weight = std::pow(std::max(0.01f, thresholds.max_distance - distance[index]), 2);
        weight_sum += weight;
        mean_x += weight * (index % width);
        mean_y += weight * (index / width);
    }
    if (weight_sum == 0) return false;
    mean_x /= weight_sum;
    mean_y /= weight_sum;
    double xx = 0, xy = 0, yy = 0;
    for (std::size_t sample = 0; sample < region_size; ++sample) {
        const int index = region[sample];
        const double weight = std::pow(std::max(0.01f, thresholds.max_distance - distance[index]), 2);
        const double dx = index % width - mean_x, dy = index / width - mean_y;
        xx += weight * dx * dx;
        xy += weight * dx * dy;
        yy += weight * dy * dy;
    }
    if (std::hypot(xx - yy, 2 * xy) == 0) return false;
    const double angle = 0.5 * std::atan2(2 * xy, xx - yy);
    const double ux = std::cos(angle), uy = std::sin(angle);
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (std::size_t sample = 0; sample < region_size; ++sample) {
        const int index = region[sample];
        const double projection = (index % width - mean_x) * ux + (index / width - mean_y) * uy;
        low = std::min(low, projection);
        high = std::max(high, projection);
    }
    segment = {float(mean_x + low * ux), float(mean_y + low * uy),
               float(mean_x + high * ux), float(mean_y + high * uy)};
    Canonicalize(segment);
    return Length(segment) >= thresholds.min_length;
}

bool SegmentExtractor::Validate(const Segment& segment, const float* distance,
                                const float* orientation, int width, int height) const {
    const auto& thresholds = config_.thresholds;
    const float length = Length(segment);
    if (length < thresholds.min_length) return false;
    const float angle = Angle(segment);
    const auto samples = std::max(std::size_t(50), std::size_t(std::ceil(double(length) * 2)) + 1);
    const float step = length / float(samples - 1);
    std::size_t supported = 0;
    double distance_sum = 0;
    float unsupported_run = 0, longest_gap = 0;
    for (std::size_t index = 0; index < samples; ++index) {
        const float fraction = float(index) / float(samples - 1);
        const float x = segment.x1 + fraction * (segment.x2 - segment.x1);
        const float y = segment.y1 + fraction * (segment.y2 - segment.y1);
        const int ix = std::clamp(int(std::lround(x)), 0, width - 1);
        const int iy = std::clamp(int(std::lround(y)), 0, height - 1);
        const float predicted_distance = Sample(distance, x, y, width, height);
        distance_sum += predicted_distance;
        const bool valid = predicted_distance < thresholds.max_distance &&
                           AngleError(orientation[iy * width + ix], angle) < thresholds.angle_tolerance;
        if (valid) {
            ++supported;
            unsupported_run = 0;
        } else {
            unsupported_run += step;
            longest_gap = std::max(longest_gap, unsupported_run);
        }
        if (longest_gap > thresholds.merge_gap + 1.0f ||
            distance_sum / samples > thresholds.max_mean_distance ||
            double(supported + samples - index - 1) / samples < thresholds.min_support) return false;
    }
    return double(supported) / samples >= thresholds.min_support &&
           distance_sum / samples <= thresholds.max_mean_distance &&
           longest_gap <= thresholds.merge_gap + 1.0f;
}

std::vector<Segment> SegmentExtractor::JoinSegments(std::vector<Segment> segments,
                                                   const float* distance, const float* orientation,
                                                   int width, int height) const {
    const auto& thresholds = config_.thresholds;
    SortByLength(segments);
    std::vector<SegmentGeometry> geometry;
    geometry.reserve(segments.size());
    for (const auto& segment : segments) geometry.emplace_back(segment);
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t first = 0; first < segments.size() && !changed; ++first) {
            const auto a = segments[first];
            const float length_a = geometry[first].length;
            const float ux = geometry[first].ux, uy = geometry[first].uy;
            for (std::size_t second = first + 1; second < segments.size(); ++second) {
                const auto b = segments[second];
                if (AngleError(geometry[first].angle, geometry[second].angle) > thresholds.angle_tolerance / 2) continue;
                const float dx1 = b.x1 - a.x1, dy1 = b.y1 - a.y1;
                const float dx2 = b.x2 - a.x1, dy2 = b.y2 - a.y1;
                if (std::max(std::abs(-dx1 * uy + dy1 * ux), std::abs(-dx2 * uy + dy2 * ux)) >
                    thresholds.merge_distance) continue;
                const float projection1 = dx1 * ux + dy1 * uy, projection2 = dx2 * ux + dy2 * uy;
                const float gap = std::max(0.0f, std::max(-std::max(projection1, projection2),
                                                        std::min(projection1, projection2) - length_a));
                if (gap > thresholds.merge_gap) continue;

                const float length_b = geometry[second].length;
                float vx = b.x2 - b.x1, vy = b.y2 - b.y1;
                if (vx * ux + vy * uy < 0) { vx = -vx; vy = -vy; }
                const float direction_x = (a.x2 - a.x1) + vx;
                const float direction_y = (a.y2 - a.y1) + vy;
                const float norm = std::hypot(direction_x, direction_y);
                if (norm == 0) continue;
                const float tx = direction_x / norm, ty = direction_y / norm;
                const float cx = ((a.x1 + a.x2) * length_a + (b.x1 + b.x2) * length_b) /
                                 (2 * (length_a + length_b));
                const float cy = ((a.y1 + a.y2) * length_a + (b.y1 + b.y2) * length_b) /
                                 (2 * (length_a + length_b));
                const std::array<float, 4> projections{
                    (a.x1 - cx) * tx + (a.y1 - cy) * ty, (a.x2 - cx) * tx + (a.y2 - cy) * ty,
                    (b.x1 - cx) * tx + (b.y1 - cy) * ty, (b.x2 - cx) * tx + (b.y2 - cy) * ty};
                const auto [low, high] = std::minmax_element(projections.begin(), projections.end());
                Segment joined{cx + *low * tx, cy + *low * ty, cx + *high * tx, cy + *high * ty};
                if (!ClipToImage(joined, width, height) || !Validate(joined, distance, orientation, width, height)) continue;
                segments[first] = joined;
                geometry[first] = SegmentGeometry(joined);
                segments.erase(segments.begin() + second);
                geometry.erase(geometry.begin() + second);
                changed = true;
                break;
            }
        }
    }
    return segments;
}

std::vector<Segment> SegmentExtractor::SuppressDuplicates(std::vector<Segment> segments) const {
    const auto& thresholds = config_.thresholds;
    SortByLength(segments);
    std::vector<Segment> result;
    std::vector<SegmentGeometry> geometry;
    for (const auto& segment : segments) {
        const SegmentGeometry candidate(segment);
        bool duplicate = false;
        for (std::size_t index = 0; index < result.size(); ++index) {
            const auto& kept = result[index];
            const auto& info = geometry[index];
            if (AngleError(info.angle, candidate.angle) > thresholds.angle_tolerance / 2) continue;
            const float length = info.length;
            const float ux = info.ux, uy = info.uy;
            const float dx1 = segment.x1 - kept.x1, dy1 = segment.y1 - kept.y1;
            const float dx2 = segment.x2 - kept.x1, dy2 = segment.y2 - kept.y1;
            if (std::max(std::abs(-dx1 * uy + dy1 * ux), std::abs(-dx2 * uy + dy2 * ux)) >
                thresholds.duplicate_distance) continue;
            const float first = dx1 * ux + dy1 * uy, second = dx2 * ux + dy2 * uy;
            if (std::min(first, second) >= -thresholds.duplicate_distance &&
                std::max(first, second) <= length + thresholds.duplicate_distance) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) { result.push_back(segment); geometry.push_back(candidate); }
    }
    return result;
}

std::vector<Segment> SegmentExtractor::Process(const float* distance, const float* orientation,
                                              int width, int height) const {
    if (!distance || !orientation || width <= 0 || height <= 0 ||
        std::size_t(width) * std::size_t(height) > std::size_t(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("SegmentExtractor requires valid fields and positive image dimensions");
    }
    const auto& thresholds = config_.thresholds;
    const int count = width * height;
    std::vector<int> seeds;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(distance[index]) || distance[index] < 0 || !std::isfinite(orientation[index])) {
            throw std::invalid_argument("Segment fields must be finite, with nonnegative distances");
        }
        if (distance[index] < thresholds.max_distance) seeds.push_back(index);
    }
    SortSeeds(seeds, distance);
    std::vector<unsigned char> visited(count, 0);
    std::vector<int> region, region_pixels;
    struct Region { std::size_t begin, size; };
    std::vector<Region> regions;
    OrientationCache directions;
    std::vector<Segment> segments;
    for (int seed : seeds) {
        if (visited[seed]) continue;
        region.clear();
        region.push_back(seed);
        visited[seed] = 1;
        double sum_x = seed % width, sum_y = seed / width;
        const auto seed_direction = directions.Get(orientation[seed]);
        double sum_cos = seed_direction.cosine, sum_sin = seed_direction.sine;
        bool direction_changed = true;
        float angle = 0, sine = 0, cosine = 0;
        double cx = 0, cy = 0;
        for (std::size_t head = 0; head < region.size(); ++head) {
            const int index = region[head], x = index % width, y = index / width;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if ((dx == 0 && dy == 0) || nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const int next = ny * width + nx;
                    if (visited[next] || distance[next] >= thresholds.max_distance) continue;
                    if (AngleError(orientation[next], orientation[seed]) >= thresholds.angle_tolerance) continue;
                    if (direction_changed) {
                        angle = float(0.5 * std::atan2(sum_sin, sum_cos));
                        sine = std::sin(angle); cosine = std::cos(angle);
                        cx = sum_x / region.size(); cy = sum_y / region.size();
                        direction_changed = false;
                    }
                    if (AngleError(orientation[next], angle) >= thresholds.angle_tolerance) continue;
                    const double offset = -(nx - cx) * sine + (ny - cy) * cosine;
                    if (std::abs(offset) > thresholds.region_radius) continue;
                    visited[next] = 1;
                    region.push_back(next);
                    sum_x += nx;
                    sum_y += ny;
                    const auto direction = directions.Get(orientation[next]);
                    sum_cos += direction.cosine;
                    sum_sin += direction.sine;
                    direction_changed = true;
                }
            }
        }
        if (region.size() >= std::size_t(thresholds.min_pixels)) {
            regions.push_back({region_pixels.size(), region.size()});
            region_pixels.insert(region_pixels.end(), region.begin(), region.end());
        }
    }
    std::vector<Segment> fitted(regions.size());
    std::vector<unsigned char> accepted(regions.size());
    auto fit_regions = [&](const cv::Range& range) {
        for (int index = range.start; index < range.end; ++index) {
            auto& segment = fitted[index];
            const auto& region = regions[index];
            accepted[index] = Fit(region_pixels.data()+region.begin, region.size, distance, width, segment) &&
                ClipToImage(segment, width, height) && Validate(segment, distance, orientation, width, height);
        }
    };
    // Seed growth stays ordered. Independent fits write separate slots, then compact in seed order.
    if (regions.size() >= 32) cv::parallel_for_(cv::Range(0, int(regions.size())), fit_regions, 4);
    else fit_regions(cv::Range(0, int(regions.size())));
    for (std::size_t index = 0; index < regions.size(); ++index)
        if (accepted[index]) segments.push_back(fitted[index]);
    if (config_.joining_enabled) segments = JoinSegments(std::move(segments), distance, orientation, width, height);
    return SuppressDuplicates(std::move(segments));
}
}
