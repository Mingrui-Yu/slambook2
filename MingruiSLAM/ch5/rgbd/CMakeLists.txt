set(CMAKE_CXX_STANDARD 11)

cmake_minimum_required(VERSION 2.8)

project(rgbd)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# include_directories("/home/ymr/Mingrui/SLAMlearning/slambook2/3rdparty/Sophus")

include_directories("/usr/include/eigen3")


add_executable(joinMap joinMap.cpp)
target_link_libraries(joinMap ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
