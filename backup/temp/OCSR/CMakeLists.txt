cmake_minimum_required(VERSION 3.17)
project(OCSR)

#opencv package
find_package(OpenCV
        PATHS 3Party/opencv
        )
include_directories(3Party/opencv/include)
message(${OpenCV_LIB_PATH})
file(GLOB temp
        RELATIVE_PATH ${_OpenCV_LIB_PATH}
        lib*.dll)
message(${temp})
#file(COPY 3Party/opencv/x64/)

#Eigen package
include_directories(3Party/eigen/include/eigen3)

set(CMAKE_CXX_STANDARD 14)

add_executable(OCSR main.cpp)