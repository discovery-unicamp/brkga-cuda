cmake_minimum_required(VERSION 3.10)
project(bb_segsort LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
enable_language(CUDA)
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}")

include(GenerateExportHeader)
add_library(bb-segsort SHARED ${CMAKE_CURRENT_LIST_DIR}/bb_segsort.cu)
generate_export_header(bb-segsort)

set_target_properties(bb-segsort PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(bb-segsort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(bb-segsort PROPERTIES COMPILE_FLAGS "-w --expt-relaxed-constexpr")
set_property(TARGET bb-segsort PROPERTY CUDA_STANDARD 11)
set_property(TARGET bb-segsort PROPERTY CUDA_ARCHITECTURES 61)

target_compile_options(bb-segsort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)

target_include_directories(bb-segsort PUBLIC ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
