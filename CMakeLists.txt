#cmake_minimum_required(VERSION 2.8)
#project( test_of )
#find_package( OpenCV REQUIRED )

#add_executable( test_of test_of.cpp )
#include_directories( ${OpenCV_INCLUDE_DIRS} )  
#include_directories( include )
#target_link_libraries( test_of ${OpenCV_LIBS} )

cmake_minimum_required(VERSION 2.8)
project( test_cuda )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

file( GLOB *.cpp)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDA_INCLUDE_DIRS} )
include_directories( include )
CUDA_ADD_EXECUTABLE(test_of test_of.cpp lkpyramidal.cu)
target_link_libraries( test_of ${CUDA_LIBRARIES} ${OpenCV_LIBS} )

