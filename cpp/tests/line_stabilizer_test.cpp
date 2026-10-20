// SPDX-License-Identifier: AGPL-3.0-or-later
#include "temporal/line_stabilizer.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

void Check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

int main() {
    try {
        cv::Mat image(240,320,CV_8UC3,cv::Scalar(35,35,35));
        cv::RNG rng(42);
        for (int i=0; i<700; ++i) {
            const cv::Point p(rng.uniform(0,320),rng.uniform(0,240));
            cv::rectangle(image,cv::Rect(p.x,p.y,2,2),cv::Scalar::all(rng.uniform(40,220)),cv::FILLED);
        }
        cv::line(image,{40,100},{270,100},{240,240,240},2);
        const lines::Segment line{40,100,270,100};
        lines::LineStabilizer filter;
        Check(filter.Process(image,{line},0).empty(),"One-frame detection must not be confirmed");
        Check(filter.Process(image,{line},.033).size()==1,"Two-frame detection should be confirmed");
        Check(filter.Process(image,{{270,101,40,101}},.066).size()==1,"Reversed endpoints must match");
        auto held=filter.Process(image,{},.099);
        Check(held.size()==1,"Reliable short dropout should be bridged");
        Check(filter.Process(image,{},.132).size()==1,"Second missing frame should be bridged");
        Check(filter.Process(image,{},.165).empty(),"Expired dropout must disappear");
        filter.Reset();
        filter.Process(image,{line},0); filter.Process(image,{line},.033);
        cv::Mat shifted;
        cv::warpAffine(image,shifted,cv::Matx23f(1,0,8,0,1,5),image.size());
        auto moving=filter.Process(shifted,{{48,105,278,105}},.066);
        Check(moving.size()==1 && std::abs(moving[0].y1-105)<.6,"Camera motion must not cause smoothing lag");
        auto blank=cv::Mat(image.size(),CV_8UC3,cv::Scalar::all(0));
        Check(filter.Process(blank,{},.099).empty(),"Occlusion/cut must not retain unsupported lines");
        filter.Reset(); filter.Process(image,{line},0); filter.Process(image,{line},.033);
        Check(filter.Process(image,{line},1).empty(),"Timestamp gap must reset tracks");
        Check(filter.Process(image,{line},.5).empty(),"Backward timestamp must reset tracks");
        cv::Mat bigger;
        cv::resize(image,bigger,{640,480});
        Check(filter.Process(bigger,{{80,200,540,200}},.533).empty(),"Resolution change must reset tracks");
        filter.Reset();
        double raw_error = 0, stable_error = 0;
        for (int i=0; i<60; ++i) {
            const float noise = (i%2 ? 1.5f : -1.5f);
            const auto stable = filter.Process(image,{{40,100+noise,270,100+noise}},i/30.);
            if (i>3) {
                Check(stable.size()==1,"Persistent line must remain visible during jitter");
                raw_error += noise*noise;
                stable_error += std::pow(stable[0].y1-100,2);
            }
        }
        Check(stable_error < raw_error*.5,"Stationary endpoint noise should be reduced");
        filter.Reset();
        filter.Process(image,{line},0); filter.Process(image,{line},.033);
        cv::Mat rotation = cv::getRotationMatrix2D({160,120},3,1.03);
        cv::Mat rotated;
        cv::warpAffine(image,rotated,rotation,image.size());
        auto warp = [&](float x,float y) {
            return cv::Point2f(rotation.at<double>(0,0)*x+rotation.at<double>(0,1)*y+rotation.at<double>(0,2),
                               rotation.at<double>(1,0)*x+rotation.at<double>(1,1)*y+rotation.at<double>(1,2));
        };
        const auto a=warp(40,100),b=warp(270,100);
        const auto turned=filter.Process(rotated,{{a.x,a.y,b.x,b.y}},.066);
        Check(turned.size()==1 && cv::norm(cv::Point2f(turned[0].x1,turned[0].y1)-a)<1,
              "Rotation and zoom should not introduce lag");
        lines::TemporalConfig immediate;
        immediate.confirmation_frames=1;
        lines::LineStabilizer clipper(immediate);
        const auto clipped=clipper.Process(image,{{-20,80,100,140}},0);
        Check(clipped.size()==1 && std::abs(clipped[0].x1)<.01 && std::abs(clipped[0].y1-90)<.01,
              "Boundary clipping must preserve the line direction");
        clipper.Reset();
        Check(clipper.Process(image,{{-20,80,-10,140}},.033).empty(),"Outside lines must not be clamped onto borders");
        cv::Mat large;
        cv::resize(image,large,{1000,751});
        const auto scaled=clipper.Process(large,{{123.25f,200.5f,650.75f,200.5f}},.066);
        Check(scaled.size()==1 && std::abs(scaled[0].x1-123.25f)<.001 && std::abs(scaled[0].y1-200.5f)<.001,
              "Internal pixel-center resizing must round-trip original coordinates");
        bool invalid=false;
        try { lines::TemporalConfig config; config.observation_weight=0; lines::LineStabilizer bad(config); }
        catch(const std::invalid_argument&) { invalid=true; }
        Check(invalid,"Invalid settings must be rejected");
        std::cout << "Temporal confirmation, dropout, motion, occlusion, reset and validation checks passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
