cmake_minimum_required(VERSION 2.8)
project(gaussNewton)

# 添加C++11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11")  

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})