cmake_minimum_required(VERSION 3.10)
project(brkga_cuda LANGUAGES CUDA CXX)

set(BB_SEGSORT_ROOT "${CMAKE_CURRENT_LIST_DIR}/bb_segsort")
include("${BB_SEGSORT_ROOT}/CMakeLists.txt")

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

add_library(brkga-cuda
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/brkga/PathRelinking.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/brkga/Filter.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Brkga.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/BrkgaConfiguration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/BrkgaFilter.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Chromosome.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/CudaUtils.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Decoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/DecodeType.cpp
)

set_target_properties(brkga-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-cuda PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda PRIVATE bb-segsort)
target_link_libraries(brkga-cuda PRIVATE OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
