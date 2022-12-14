cmake_minimum_required(VERSION 3.12)
project(sobel)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_compile_options(-O3 -Wall)
set(USE_OMP ON)

if(OPENMP_FOUND AND USE_OMP)
add_compile_options(-fopenmp)
message(STATUS "Use OPENMP")
else()
message(STATUS "Not use OPENMP")
endif()

add_executable(sobel sobel.cpp)
target_link_libraries(sobel ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})