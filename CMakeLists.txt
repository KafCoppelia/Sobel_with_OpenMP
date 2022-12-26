cmake_minimum_required(VERSION 3.12)
project(cuda_sobel)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

include_directories(${OpenCV_INCLUDE_DIRS})
add_compile_options(-fopenmp)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} sobel.cu)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})