cmake_minimum_required(VERSION 2.8)
project(undistortImage)

# 添加C++11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11")  

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(undistortImage undistortImage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})