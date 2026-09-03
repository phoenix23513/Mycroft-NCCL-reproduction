cmake_minimum_required(VERSION 3.16)

project(
  mycroft_nccl_reproduction
  VERSION 0.1.0
  DESCRIPTION "Independent Mycroft-style NCCL diagnosis reproduction"
  LANGUAGES CXX
)

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
