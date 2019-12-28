cmake_minimum_required(VERSION 2.8)

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS})


add_library(showPointCloud SHARED showPointCloud.cpp)
target_link_libraries(showPointCloud ${Pangolin_LIBRARIES})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision showPointCloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES})


