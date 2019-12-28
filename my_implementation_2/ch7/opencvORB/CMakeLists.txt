cmake_minimum_required(VERSION 2.8)
project(opencvORB)

# 添加C++11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11")  

# 设置Debug模式或Release模式
set(CMAKE_BUILD_TYPE "Debug")


# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})



add_executable(opencvORB opencvORB.cpp)
target_link_libraries(opencvORB ${OpenCV_LIBS})