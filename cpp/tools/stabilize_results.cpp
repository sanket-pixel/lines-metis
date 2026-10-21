// SPDX-License-Identifier: AGPL-3.0-or-later
#include "line_stabilizer.h"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: stabilize_results input.jsonl temporal.yaml output.jsonl");
        const auto config = lines::LoadTemporalConfig(argv[2]);
        lines::LineStabilizer stabilizer(config);
        if (std::filesystem::exists(argv[3])) throw std::runtime_error("Output already exists");
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open input export");
        std::string line;
        if (!std::getline(input, line)) throw std::runtime_error("Empty export");
        auto metadata = nlohmann::json::parse(line);
        if (metadata.at("type") != "metadata" || metadata.at("coordinate_space") != "original_image_pixels" ||
            metadata.contains("temporal")) throw std::runtime_error("Expected raw original-image export");
        const std::filesystem::path image_root = metadata.at("input").at("path").get<std::string>();
        metadata["schema_version"] = 2;
        metadata["temporal"] = {{"method", "motion_compensated_line_stabilizer_v1"},
            {"confirmation_frames", config.confirmation_frames}, {"max_missing_frames", config.max_missing_frames},
            {"observation_weight", config.observation_weight}, {"max_angle_degrees", config.max_angle_degrees},
            {"max_normal_distance", config.max_normal_distance},
            {"max_time_gap", config.max_time_gap}, {"working_max_width", 640},
            {"timing_scope", "temporal_ms includes motion estimation and association; excludes image loading"}};
        metadata["source_results"] = std::filesystem::absolute(argv[1]).string();
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("Cannot create output export");
        output << metadata.dump() << '\n';
        std::size_t count = 0;
        bool complete = false;
        while (std::getline(input, line)) {
            auto record = nlohmann::json::parse(line);
            if (record.at("type") == "summary") {
                if (record.at("frames").get<std::size_t>() != count) throw std::runtime_error("Frame count mismatch");
                complete = true;
                break;
            }
            if (record.at("type") != "frame" || record.at("index").get<std::size_t>() != count)
                throw std::runtime_error("Unexpected frame order");
            const auto file = image_root / record.at("file").get<std::string>();
            auto image = cv::imread(file.string());
            if (image.empty()) throw std::runtime_error("Cannot read image: " + file.string());
            if (image.cols != record.at("width") || image.rows != record.at("height"))
                throw std::runtime_error("Image dimensions do not match export");
            std::vector<lines::Segment> segments;
            for (const auto& s : record.at("segments")) segments.push_back({s[0],s[1],s[2],s[3]});
            std::optional<double> timestamp;
            if (!record.at("timestamp_seconds").is_null()) timestamp = record.at("timestamp_seconds").get<double>();
            const auto start = std::chrono::steady_clock::now();
            const auto stable = stabilizer.Process(image, segments, timestamp);
            record["temporal_ms"] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            record["raw_segments"] = record.at("segments");
            record["segments"] = nlohmann::json::array();
            for (const auto& s : stable) record["segments"].push_back({s.x1,s.y1,s.x2,s.y2});
            output << record.dump() << '\n';
            if (!output) throw std::runtime_error("Cannot write frame");
            if (++count % 100 == 0) std::cout << "Stabilized " << count << " frames" << std::endl;
        }
        if (!complete) throw std::runtime_error("Incomplete source export");
        output << nlohmann::json({{"type","summary"},{"frames",count}}).dump() << '\n';
        output.close();
        if (!output) throw std::runtime_error("Cannot finish output");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
