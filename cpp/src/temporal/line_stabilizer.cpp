// SPDX-License-Identifier: AGPL-3.0-or-later
#include "temporal/line_stabilizer.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace lines {
namespace {
cv::Point2f Start(const Segment& s) { return {s.x1, s.y1}; }
cv::Point2f End(const Segment& s) { return {s.x2, s.y2}; }
Segment MakeSegment(cv::Point2f a, cv::Point2f b) { return {a.x, a.y, b.x, b.y}; }
float Length(const Segment& s) { return cv::norm(End(s) - Start(s)); }
float Median(std::vector<float> values) {
    auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}
bool Clip(Segment& segment, cv::Size size) {
    const auto a = Start(segment), delta = End(segment) - a;
    const float p[] = {-delta.x, delta.x, -delta.y, delta.y};
    const float q[] = {a.x, size.width-1.f-a.x, a.y, size.height-1.f-a.y};
    float first = 0, last = 1;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(p[i]) < 1e-6f) { if (q[i] < 0) return false; }
        else {
            const float ratio = q[i] / p[i];
            if (p[i] < 0) first = std::max(first, ratio);
            else last = std::min(last, ratio);
        }
    }
    if (first > last) return false;
    segment = MakeSegment(a + delta*first, a + delta*last);
    return Length(segment) >= 2;
}
void Validate(const TemporalConfig& c) {
    if (c.confirmation_frames < 1 || c.confirmation_frames > 10 ||
        c.max_missing_frames < 0 || c.max_missing_frames > 5 ||
        !std::isfinite(c.observation_weight) || c.observation_weight <= 0 || c.observation_weight > 1 ||
        !std::isfinite(c.max_angle_degrees) || c.max_angle_degrees <= 0 || c.max_angle_degrees > 30 ||
        !std::isfinite(c.max_normal_distance) || c.max_normal_distance <= 0 ||
        !std::isfinite(c.max_time_gap) || c.max_time_gap <= 0) {
        throw std::invalid_argument("Invalid temporal settings");
    }
}
}

TemporalConfig LoadTemporalConfig(const std::filesystem::path& path) {
    const auto yaml = YAML::LoadFile(path.string());
    const std::set<std::string> keys = {"confirmation_frames", "max_missing_frames", "observation_weight",
        "max_angle_degrees", "max_normal_distance", "max_time_gap"};
    if (!yaml.IsMap()) throw std::invalid_argument("Temporal config must be a mapping");
    std::set<std::string> seen;
    for (const auto& item : yaml) {
        const auto key = item.first.as<std::string>();
        if (!keys.count(key) || !seen.insert(key).second) throw std::invalid_argument("Unknown/duplicate temporal key: " + key);
    }
    TemporalConfig c;
    c.confirmation_frames = yaml["confirmation_frames"].as<int>();
    c.max_missing_frames = yaml["max_missing_frames"].as<int>();
    c.observation_weight = yaml["observation_weight"].as<float>();
    c.max_angle_degrees = yaml["max_angle_degrees"].as<float>();
    c.max_normal_distance = yaml["max_normal_distance"].as<float>();
    c.max_time_gap = yaml["max_time_gap"].as<double>();
    Validate(c);
    return c;
}

LineStabilizer::LineStabilizer(const TemporalConfig& config) : config_(config) { Validate(config_); }

void LineStabilizer::Reset() {
    tracks_.clear();
    previous_gray_.release();
    previous_timestamp_.reset();
    original_size_ = {};
}

std::vector<bool> LineStabilizer::Predict(const cv::Mat& gray) {
    std::vector<bool> reliable(tracks_.size(), false);
    if (tracks_.empty()) return reliable;
    std::vector<cv::Point2f> points;
    for (const auto& track : tracks_) {
        const auto a = Start(track.segment), direction = End(track.segment) - a;
        const auto normal = cv::Point2f(-direction.y, direction.x) / Length(track.segment);
        for (int i = 0; i < 5; ++i)
            for (int offset : {-3, 0, 3}) points.push_back(a + direction * (i / 4.f) + normal * float(offset));
    }
    std::vector<cv::Point2f> forward, backward;
    std::vector<unsigned char> forward_ok, backward_ok;
    std::vector<float> forward_error, backward_error;
    cv::calcOpticalFlowPyrLK(previous_gray_, gray, points, forward, forward_ok, forward_error,
                            {21, 21}, 3, {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03});
    cv::calcOpticalFlowPyrLK(gray, previous_gray_, forward, backward, backward_ok, backward_error,
                            {21, 21}, 3, {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03});
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        std::vector<cv::Point2f> from, to;
        std::vector<float> dx, dy;
        for (std::size_t k = t * 15; k < (t + 1) * 15; ++k) {
            const auto p = forward[k];
            if (!forward_ok[k] || !backward_ok[k] || forward_error[k] > 25 ||
                cv::norm(backward[k] - points[k]) > 1.2 || p.x < 0 || p.y < 0 ||
                p.x >= gray.cols || p.y >= gray.rows) continue;
            from.push_back(points[k]); to.push_back(p);
            dx.push_back(p.x - points[k].x); dy.push_back(p.y - points[k].y);
        }
        if (from.size() < 5) continue;
        const cv::Point2f translation(Median(dx), Median(dy));
        cv::Point2f center_from, center_to;
        std::vector<std::size_t> inliers;
        for (std::size_t k = 0; k < from.size(); ++k) {
            if (cv::norm(to[k] - from[k] - translation) < 4) {
                inliers.push_back(k); center_from += from[k]; center_to += to[k];
            }
        }
        if (inliers.size() < 5) continue;
        center_from *= 1.f / inliers.size(); center_to *= 1.f / inliers.size();
        float denominator = 0, dot = 0, cross = 0;
        for (auto k : inliers) {
            const auto a = from[k] - center_from, b = to[k] - center_to;
            denominator += a.dot(a); dot += a.dot(b); cross += a.x*b.y - a.y*b.x;
        }
        if (denominator < 1) continue;
        const float c = dot / denominator, s = cross / denominator;
        const float scale = std::hypot(c, s);
        if (scale < .85 || scale > 1.18 || std::abs(std::atan2(s, c)) > .2) continue;
        auto warp = [&](cv::Point2f p) {
            p -= center_from;
            return center_to + cv::Point2f(c*p.x - s*p.y, s*p.x + c*p.y);
        };
        float residual = 0;
        for (auto k : inliers) residual += cv::norm(warp(from[k]) - to[k]);
        if (residual / inliers.size() > 1.5) continue;
        tracks_[t].segment = MakeSegment(warp(Start(tracks_[t].segment)), warp(End(tracks_[t].segment)));
        reliable[t] = true;
    }
    return reliable;
}

float LineStabilizer::MatchCost(const Segment& prediction, const Segment& observation) const {
    const auto u = (End(prediction) - Start(prediction)) / Length(prediction);
    const auto v = (End(observation) - Start(observation)) / Length(observation);
    const float angle = std::acos(std::clamp(std::abs(u.dot(v)), 0.f, 1.f));
    const auto normal = cv::Point2f(-u.y, u.x);
    const auto offset = (Start(observation) + End(observation) - Start(prediction) - End(prediction)) * .5f;
    const float distance = std::abs(offset.dot(normal));
    if (angle > config_.max_angle_degrees * float(CV_PI / 180) || distance > config_.max_normal_distance)
        return std::numeric_limits<float>::infinity();
    float a = (Start(observation) - Start(prediction)).dot(u);
    float b = (End(observation) - Start(prediction)).dot(u);
    if (a > b) std::swap(a, b);
    const float length = Length(prediction);
    const float overlap = std::max(0.f, std::min(length, b) - std::max(0.f, a));
    if (overlap < .35f * std::min(length, b-a))
        return std::numeric_limits<float>::infinity();
    return distance + angle * 15 + (1 - overlap / std::max(length, b-a)) * 2;
}

Segment LineStabilizer::Smooth(const Segment& prediction, Segment observation) const {
    if ((End(prediction)-Start(prediction)).dot(End(observation)-Start(observation)) < 0)
        observation = MakeSegment(End(observation), Start(observation));
    const float weight = config_.observation_weight;
    return MakeSegment(Start(prediction)*(1-weight) + Start(observation)*weight,
                       End(prediction)*(1-weight) + End(observation)*weight);
}

std::vector<Segment> LineStabilizer::Process(const cv::Mat& image, const std::vector<Segment>& segments,
                                            std::optional<double> timestamp) {
    if (image.empty() || image.depth() != CV_8U || (image.channels() != 1 && image.channels() != 3))
        throw std::invalid_argument("Temporal input must be gray or BGR uint8");
    if (timestamp && !std::isfinite(*timestamp)) throw std::invalid_argument("Invalid temporal timestamp");
    if (image.size() != original_size_ || (timestamp && previous_timestamp_ &&
        (*timestamp <= *previous_timestamp_ || *timestamp - *previous_timestamp_ > config_.max_time_gap))) Reset();
    original_size_ = image.size();
    previous_timestamp_ = timestamp;
    const float scale = std::min(1.f, 640.f / image.cols);
    const cv::Size working_size(std::lround(image.cols*scale), std::max(1L, std::lround(image.rows*scale)));
    const float scale_x = float(working_size.width) / image.cols;
    const float scale_y = float(working_size.height) / image.rows;
    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image;
    cv::resize(gray, gray, working_size, 0, 0, cv::INTER_AREA);
    if (!previous_gray_.empty()) {
        cv::Mat difference;
        cv::absdiff(gray, previous_gray_, difference);
        if (cv::mean(difference)[0] > 40) tracks_.clear();
    }
    const auto reliable = previous_gray_.empty() ? std::vector<bool>() : Predict(gray);
    std::vector<Segment> observations;
    for (auto s : segments) {
        if (!(std::isfinite(s.x1) && std::isfinite(s.y1) && std::isfinite(s.x2) && std::isfinite(s.y2))) continue;
        s = {(s.x1+.5f)*scale_x-.5f, (s.y1+.5f)*scale_y-.5f,
             (s.x2+.5f)*scale_x-.5f, (s.y2+.5f)*scale_y-.5f};
        if (Length(s) >= 2) observations.push_back(s);
    }
    struct Candidate { float cost; std::size_t track, observation; };
    std::vector<Candidate> candidates;
    for (std::size_t t = 0; t < tracks_.size(); ++t)
        for (std::size_t o = 0; o < observations.size(); ++o) {
            const float cost = MatchCost(tracks_[t].segment, observations[o]);
            if (std::isfinite(cost)) candidates.push_back({cost, t, o});
        }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.cost != b.cost) return a.cost < b.cost;
        return std::tie(a.track, a.observation) < std::tie(b.track, b.observation);
    });
    std::vector<bool> matched_track(tracks_.size()), matched_observation(observations.size());
    for (const auto& c : candidates) {
        if (matched_track[c.track] || matched_observation[c.observation]) continue;
        auto& track = tracks_[c.track];
        // Without trustworthy motion, use the observation directly to avoid camera lag.
        track.segment = reliable[c.track] ? Smooth(track.segment, observations[c.observation]) : observations[c.observation];
        track.hits = track.missing ? 1 : track.hits + 1;
        track.missing = 0;
        track.confirmed = track.confirmed || track.hits >= config_.confirmation_frames;
        matched_track[c.track] = matched_observation[c.observation] = true;
    }
    cv::Mat edges, nearby;
    cv::Canny(gray, edges, 40, 100);
    cv::dilate(edges, nearby, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5,5}));
    std::vector<Track> next;
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        auto track = tracks_[t];
        if (!matched_track[t]) {
            if (!track.confirmed || ++track.missing > config_.max_missing_frames || !reliable[t]) continue;
            int supported = 0;
            for (int k = 0; k < 16; ++k) {
                const auto p = Start(track.segment) + (End(track.segment)-Start(track.segment))*(k/15.f);
                const int x = std::lround(p.x), y = std::lround(p.y);
                if (x >= 0 && y >= 0 && x < nearby.cols && y < nearby.rows && nearby.at<unsigned char>(y,x)) ++supported;
            }
            if (supported < 10) continue;
            bool duplicate = false;
            for (std::size_t o = 0; o < observations.size(); ++o)
                if (matched_observation[o] && std::isfinite(MatchCost(track.segment, observations[o]))) duplicate = true;
            if (duplicate) continue;
        }
        next.push_back(track);
    }
    for (std::size_t o = 0; o < observations.size(); ++o)
        if (!matched_observation[o]) next.push_back({observations[o], 1, 0, config_.confirmation_frames == 1});
    tracks_ = std::move(next);
    gray.copyTo(previous_gray_);
    std::vector<Segment> output;
    for (const auto& track : tracks_) {
        if (!track.confirmed) continue;
        const auto& s = track.segment;
        Segment result{(s.x1+.5f)/scale_x-.5f, (s.y1+.5f)/scale_y-.5f,
                       (s.x2+.5f)/scale_x-.5f, (s.y2+.5f)/scale_y-.5f};
        if (Clip(result, image.size())) output.push_back(result);
    }
    return output;
}
}
