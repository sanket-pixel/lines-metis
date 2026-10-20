// SPDX-License-Identifier: AGPL-3.0-or-later
#include "common/config.h"
#include "inference/aipu_inference.h"
#include "preprocessing/preprocessing.h"
#include "postprocessing/postprocessing.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char**argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: cache_extraction_fields config.yaml sequence_directory output.bin");
        if(std::filesystem::exists(argv[3])) throw std::runtime_error("Output already exists");
        std::vector<std::filesystem::path> paths;
        for(const auto& entry : std::filesystem::directory_iterator(argv[2]))
            if(entry.is_regular_file() && entry.path().extension()==".png") paths.push_back(entry.path());
        std::sort(paths.begin(),paths.end());
        if(paths.empty() || paths.size()>std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("Invalid sequence size");
        const auto config=lines::LoadConfig(argv[1]);
        lines::AipuContext context;
        lines::AipuInference inference(context,config.inference);
        lines::Preprocessing preprocessing;
        lines::Postprocessing postprocessing(inference.output_info(0),inference.output_info(1));
        std::ofstream output(argv[3],std::ios::binary);
        const std::uint32_t header[]={320,240,std::uint32_t(paths.size())};
        output.write(reinterpret_cast<const char*>(header),sizeof(header));
        std::vector<float> distance,orientation;
        for(std::size_t index=0;index<paths.size();++index) {
            const auto image=cv::imread(paths[index].string());
            if(image.empty()) throw std::runtime_error("Cannot read "+paths[index].string());
            const auto& info=inference.input_info();
            preprocessing.Process(image,info,inference.input_buffer(),info.size_bytes);
            inference.Process();
            postprocessing.Process(inference.output_buffer(0),inference.output_buffer(1),distance,orientation);
            output.write(reinterpret_cast<const char*>(distance.data()),distance.size()*sizeof(float));
            output.write(reinterpret_cast<const char*>(orientation.data()),orientation.size()*sizeof(float));
            if(!output) throw std::runtime_error("Cannot write fields");
            if((index+1)%200==0) std::cout<<"Cached "<<index+1<<" frames"<<std::endl;
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
