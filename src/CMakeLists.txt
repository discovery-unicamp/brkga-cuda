cmake_minimum_required(VERSION 3.10)
project(brkga_cuda LANGUAGES CUDA CXX)

set(BB_SEGSORT_ROOT "${CMAKE_CURRENT_LIST_DIR}/gpu-sort")
include("${BB_SEGSORT_ROOT}/CMakeLists.txt")

find_package(OpenMP)
find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

add_library(brkga-cuda
  # ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/brkga/PathRelink.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/brkga/PathRelinkCEAndrade.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/brkga/Filter.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/except/InvalidArgument.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/except/NotImplemented.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/utils/GpuUtils.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/utils/GpuUtils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/utils/StringUtils.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Bias.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Bias.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Brkga.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Brkga.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/BrkgaConfiguration.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/BrkgaConfiguration.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Chromosome.cu
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Chromosome.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Comparator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Comparator.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Decoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Decoder.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/DecodeType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/DecodeType.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/Logger.hpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/PathRelinkPair.cpp
  ${CMAKE_CURRENT_LIST_DIR}/brkga-cuda/PathRelinkPair.hpp
)

set_target_properties(brkga-cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(brkga-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET brkga-cuda PROPERTY CUDA_STANDARD 11)
set_property(TARGET brkga-cuda PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(brkga-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(brkga-cuda PRIVATE bb-segsort)
target_link_libraries(brkga-cuda PRIVATE OpenMP::OpenMP_CXX PUBLIC -lcurand -fopenmp)
