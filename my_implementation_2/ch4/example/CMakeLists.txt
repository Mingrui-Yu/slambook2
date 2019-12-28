cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE "Debug")

project(trajectoryError)

include_directories("/usr/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})


add_executable(trajectoryError trajectoryError.cpp)