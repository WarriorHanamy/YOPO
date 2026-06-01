cmake_minimum_required(VERSION 3.16)
project(yopo_data_gen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CONFIG_FILE   "/app/config/config.yaml"  CACHE STRING "Runtime config file path")
set(CUDA_ARCH     "86"                        CACHE STRING "CUDA SM architecture (75, 86, 89, 120, ...)")
set(DATASET_GENERATOR_TARGET "dataset_generator" CACHE STRING "Dataset generator executable target")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(yaml-cpp REQUIRED)

cuda_select_nvcc_arch_flags(ARCH_FLAGS)
if("${ARCH_FLAGS}" STREQUAL "")
    message(WARNING "GPU not detected at build time. Using arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
    set(ARCH_FLAGS "-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
endif()
message(STATUS "CUDA Architecture: ${ARCH_FLAGS}")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-O3")

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

cuda_add_library(raycast_cuda src/sensor_simulator.cu)
target_link_libraries(raycast_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(${DATASET_GENERATOR_TARGET}
    src/dataset_generator.cpp
    src/maps.cpp
    src/perlinnoise.cpp
)
target_link_libraries(${DATASET_GENERATOR_TARGET}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX
    raycast_cuda
    yaml-cpp
)
target_compile_definitions(${DATASET_GENERATOR_TARGET} PRIVATE
    CONFIG_FILE_PATH="${CONFIG_FILE}"
)
