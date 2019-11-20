cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

project(g2oCurveFitting)


include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(orb_self orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})