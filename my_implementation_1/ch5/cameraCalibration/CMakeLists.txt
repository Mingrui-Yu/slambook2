cmake_minimum_required(VERSION 2.8)

project(cameraCalibration)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(calibration calibration.cpp)
# 链接OpenCV库
target_link_libraries(calibration ${OpenCV_LIBS})