// SPDX-License-Identifier: AGPL-3.0-or-later
#include "segment_extractor.h"
#include "reference/reference_segment_extractor.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

void Compare(const std::vector<float>& distance, const std::vector<float>& orientation,
             int width, int height, const lines::SegmentExtractorConfig& config) {
    const auto expected=lines::test::ReferenceSegmentExtractor(config).Process(distance.data(),orientation.data(),width,height);
    for(int threads : {1,4}) {
        cv::setNumThreads(threads);
        const auto actual=lines::SegmentExtractor(config).Process(distance.data(),orientation.data(),width,height);
        if(expected.size()!=actual.size() || (!expected.empty() && std::memcmp(expected.data(),actual.data(),expected.size()*sizeof(lines::Segment))))
            throw std::runtime_error("Segment endpoint/order parity failure");
    }
}
int main(int argc,char**argv) {
    try {
        if(argc!=4) throw std::runtime_error("Three preset paths required");
        constexpr int width=128,height=96;
        cv::RNG rng(61023);
        for(int preset=1;preset<4;++preset) {
            auto config=lines::LoadConfig(argv[preset]).segment_extractor;
            for(int trial=0;trial<90;++trial) {
                std::vector<float> distance(width*height,5),orientation(width*height,0);
                for(int i=0;i<width*height;++i) {
                    if(trial%3==0) {
                        distance[i]=rng.uniform(0.f,6.f);
                        orientation[i]=rng.uniform(-25.f,25.f);
                    } else if(trial%3==1) {
                        distance[i]=float(rng.uniform(0,16))*.2f;
                        orientation[i]=float(rng.uniform(0,8))*float(CV_PI/8);
                    }
                }
                if(trial%3==2) {
                    for(int line=0;line<18;++line) {
                        const int y=rng.uniform(0,height), start=rng.uniform(0,width/2);
                        const int end=rng.uniform(width/2,width);
                        for(int x=start;x<end;++x) {
                            distance[y*width+x]=(x%2 ? -0.f : 0.f);
                            orientation[y*width+x]=trial%2 ? float(CV_PI) : 0.f;
                        }
                    }
                }
                config.joining_enabled=trial%2==0;
                Compare(distance,orientation,width,height,config);
            }
            // Dense disjoint short lines exercise parallel fits and stable equal-distance ties.
            std::vector<float> distance(320*240,5),orientation(320*240,0);
            for(int y=2;y<240;y+=5) for(int x=1;x<320;++x) if(x%40<30) distance[y*320+x]=0;
            Compare(distance,orientation,320,240,config);
            // Values near orientation gates, periodic wrapping and signed zero.
            for(float angle : {0.f,config.thresholds.angle_tolerance,std::nextafter(config.thresholds.angle_tolerance,0.f),
                               float(CV_PI),float(-CV_PI),float(CV_PI/2),1e20f,-1e20f}) {
                std::fill(distance.begin(),distance.end(),5);
                std::fill(orientation.begin(),orientation.end(),angle);
                for(int x=0;x<320;++x) distance[100*320+x]=x%2 ? -0.f : 0.f;
                Compare(distance,orientation,320,240,config);
            }
            lines::SegmentExtractor extractor(config);
            for(float invalid : {-1.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
                distance[0]=invalid;
                bool rejected=false;
                try {extractor.Process(distance.data(),orientation.data(),320,240);} catch(const std::invalid_argument&) {rejected=true;}
                if(!rejected) throw std::runtime_error("Invalid field accepted");
            }
        }
        std::cout<<"Bit-exact parity: 297 synthetic cases, three presets, 1/4 threads; invalid fields rejected\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
