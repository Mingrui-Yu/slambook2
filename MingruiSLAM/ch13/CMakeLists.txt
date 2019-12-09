cmake_minimum_required( VERSION 2.8 )
project(myslam)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -03 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
#Eigen
include_directories("/usr/include/eigen3")

#OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_Include_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog: 实现应用级别的logging
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})






enable_testing()



############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
# add_subdirectory(test)
# add_subdirectory(app)