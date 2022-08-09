cmake_minimum_required(VERSION 3.10)
project(bb_segsort LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

add_library(bb_segsort ${CMAKE_CURRENT_LIST_DIR}/bb_segsort.cu)
set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/bb_segsort.cuh PROPERTIES LANGUAGE CUDA)

set_target_properties(bb_segsort PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(bb_segsort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bb_segsort PROPERTIES COMPILE_FLAGS "-w")  # disable all warnings
set_property(TARGET bb_segsort PROPERTY CUDA_STANDARD 11)
set_property(TARGET bb_segsort PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(bb_segsort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
