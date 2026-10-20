cmake_minimum_required(VERSION 3.18)
project(lines_metis LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

if(DEFINED ENV{AXRUNTIME_ROOT})
    set(AXRUNTIME_ROOT "$ENV{AXRUNTIME_ROOT}" CACHE PATH "AxRuntime installation root")
else()
    set(AXRUNTIME_ROOT "/opt/axelera" CACHE PATH "AxRuntime installation root")
endif()

# Accept either /opt/axelera or a specific runtime installation.
file(GLOB AXRUNTIME_INSTALLATIONS LIST_DIRECTORIES TRUE "${AXRUNTIME_ROOT}/runtime-*")
list(SORT AXRUNTIME_INSTALLATIONS COMPARE NATURAL ORDER DESCENDING)
find_path(AXRUNTIME_INCLUDE_DIR
    NAMES axruntime/axruntime.h
    HINTS ${AXRUNTIME_ROOT} ${AXRUNTIME_INSTALLATIONS}
    PATH_SUFFIXES include
)
find_library(AXRUNTIME_LIBRARY
    NAMES axruntime
    HINTS "${AXRUNTIME_INCLUDE_DIR}/.." ${AXRUNTIME_ROOT} ${AXRUNTIME_INSTALLATIONS}
    PATH_SUFFIXES lib lib64
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AxRuntime
    REQUIRED_VARS AXRUNTIME_INCLUDE_DIR AXRUNTIME_LIBRARY
    FAIL_MESSAGE "AxRuntime not found. Set AXRUNTIME_ROOT to its installation directory."
)

add_library(axruntime::axruntime UNKNOWN IMPORTED)
set_target_properties(axruntime::axruntime PROPERTIES
    IMPORTED_LOCATION "${AXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AXRUNTIME_INCLUDE_DIR}"
)

add_library(lines_cpp STATIC

    src/common/config.cpp
    src/common/types.cpp

    src/preprocessing/preprocessing.cpp
    src/inference/aipu_context.cpp
    src/inference/aipu_inference.cpp
    src/postprocessing/postprocessing.cpp

    src/segment_extractor/segment_extractor.cpp

    src/pipeline/pipeline.cpp
)

target_compile_features(lines_cpp PUBLIC cxx_std_17)
set_target_properties(lines_cpp PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(lines_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lines_cpp SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lines_cpp PUBLIC axruntime::axruntime opencv_core opencv_imgproc
    PRIVATE yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json OpenSSL::Crypto)
target_compile_options(lines_cpp PRIVATE -Wall -Wextra -Wpedantic)

add_executable(lines_metis
    app/main.cpp
    app/arguments.cpp
    app/input_source.cpp
    app/output_window.cpp
    app/results_writer.cpp
)
target_link_libraries(lines_metis PRIVATE lines_cpp opencv_imgcodecs opencv_videoio opencv_highgui
    nlohmann_json::nlohmann_json yaml-cpp::yaml-cpp)
set_target_properties(lines_metis PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(lines_metis PRIVATE -Wall -Wextra -Wpedantic)
