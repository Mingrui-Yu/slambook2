cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE "Debug")

project(coordinateTransform)

include_directories("/usr/include/eigen3")

add_executable(coordinateTransform coordinateTransform.cpp)


