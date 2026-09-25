cmake_minimum_required(VERSION 3.16)

project(
  mycroft_nccl_reproduction
  VERSION 0.1.0
  DESCRIPTION "Independent Mycroft-style NCCL diagnosis reproduction"
  LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_executable(mycroft_cpp_smoke tests/cpp/smoke_test.cpp)
target_compile_options(
  mycroft_cpp_smoke
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
add_test(NAME cpp_smoke COMMAND mycroft_cpp_smoke)

add_subdirectory(experiments/e01_ring_dependency)
