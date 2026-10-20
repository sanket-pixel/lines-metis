// SPDX-License-Identifier: AGPL-3.0-or-later
#include "segment_extractor/segment_extractor.h"
#include "reference/reference_segment_extractor.h"
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace {
using Clock = std::chrono::steady_clock;
nlohmann::json Stats(std::vector<double> samples) {
    const double mean=std::accumulate(samples.begin(),samples.end(),0.)/samples.size();
    std::sort(samples.begin(),samples.end());
    return {{"mean_ms",mean},{"median_ms",samples[samples.size()/2]},
            {"p95_ms",samples[std::size_t(.95*(samples.size()-1))]},{"max_ms",samples.back()}};
}
}
int main(int argc,char**argv) {
    try {
        if(argc<4 || argc>7) throw std::runtime_error("Usage: benchmark_segment_extractor config.yaml fields.bin report.json [stride=1] [repeats=3] [threads=4]");
        const int stride=argc>4?std::stoi(argv[4]):1, repeats=argc>5?std::stoi(argv[5]):3, threads=argc>6?std::stoi(argv[6]):4;
        if(stride<1 || repeats<1 || threads<1) throw std::runtime_error("Positive stride/repeats/threads required");
        if(std::filesystem::exists(argv[3])) throw std::runtime_error("Report already exists");
        auto config=lines::LoadConfig(argv[1]);
        lines::SegmentExtractor optimized(config.segment_extractor);
        lines::test::ReferenceSegmentExtractor reference(config.segment_extractor);
        cv::setNumThreads(threads);
        std::ifstream input(argv[2],std::ios::binary);
        std::uint32_t header[3]{};
        input.read(reinterpret_cast<char*>(header),sizeof(header));
        const auto [width,height,count]=std::tuple(header[0],header[1],header[2]);
        if(!input || width!=320 || height!=240 || count==0) throw std::runtime_error("Expected 320x240 cached fields");
        std::vector<float> distance(width*height),orientation(width*height);
        std::vector<double> old_ms,new_ms;
        std::size_t checked=0,total_segments=0;
        auto run=[&](const auto& extractor,std::vector<double>& times) {
            const auto start=Clock::now();
            auto result=extractor.Process(distance.data(),orientation.data(),width,height);
            times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
            return result;
        };
        for(std::uint32_t frame=0;frame<count;frame+=stride) {
            input.seekg(sizeof(header)+std::uint64_t(frame)*width*height*8);
            input.read(reinterpret_cast<char*>(distance.data()),distance.size()*sizeof(float));
            input.read(reinterpret_cast<char*>(orientation.data()),orientation.size()*sizeof(float));
            if(!input) throw std::runtime_error("Truncated field cache");
            if(frame==0) for(int warmup=0;warmup<4;++warmup) {
                reference.Process(distance.data(),orientation.data(),width,height);
                optimized.Process(distance.data(),orientation.data(),width,height);
            }
            for(int repeat=0;repeat<repeats;++repeat) {
                std::vector<lines::Segment> a,b;
                if((frame+repeat)%2) {b=run(optimized,new_ms);a=run(reference,old_ms);}
                else {a=run(reference,old_ms);b=run(optimized,new_ms);}
                if(a.size()!=b.size() || (!a.empty() && std::memcmp(a.data(),b.data(),a.size()*sizeof(lines::Segment))))
                    throw std::runtime_error("Endpoint/order parity failure on frame "+std::to_string(frame));
                if(repeat==0) total_segments+=a.size();
            }
            if(++checked%200==0) std::cout<<"Checked "<<checked<<" frames"<<std::endl;
        }
        const auto old_stats=Stats(old_ms),new_stats=Stats(new_ms);
        nlohmann::json report={{"config",argv[1]},{"frames",checked},{"source_frames",count},{"stride",stride},
            {"repeats",repeats},{"opencv_threads",threads},{"segments",total_segments},{"bit_exact_endpoints_and_order",true},
            {"reference",old_stats},{"optimized",new_stats},
            {"speedup",old_stats["mean_ms"].get<double>()/new_stats["mean_ms"].get<double>()},
            {"scope","Extraction only; cached real fields, alternating order, four warmups. Excludes inference, I/O and temporal layer."}};
        std::ofstream output(argv[3]);output<<report.dump(2)<<'\n';
        if(!output) throw std::runtime_error("Cannot write report");
        std::cout<<report.dump(2)<<'\n';
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
