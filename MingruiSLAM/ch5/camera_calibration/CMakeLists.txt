project(Camera_Calibration)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIR})
add_executable(Camera_Calibration camera_calibration.cpp)
target_link_libraries(Camera_Calibration ${OpenCV_LIBS})
