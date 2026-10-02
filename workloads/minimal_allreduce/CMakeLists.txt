cmake_minimum_required(VERSION 3.17)
project(day15_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NCCL_ROOT "" CACHE PATH "Day14 构建产物目录，包含 include/ 和 lib/")
if(NOT NCCL_ROOT)
  message(FATAL_ERROR "请指定 -DNCCL_ROOT=<Day14产物目录>")
endif()
find_path(DAY15_NCCL_INCLUDE nccl.h PATHS "${NCCL_ROOT}/include" NO_DEFAULT_PATH)
find_library(DAY15_NCCL_LIBRARY nccl PATHS "${NCCL_ROOT}/lib" NO_DEFAULT_PATH)
if(NOT DAY15_NCCL_INCLUDE OR NOT DAY15_NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL_ROOT 必须包含项目构建的 NCCL 头文件和共享库")
endif()
get_filename_component(DAY15_NCCL_REAL_LIBRARY "${DAY15_NCCL_LIBRARY}" REALPATH)
get_filename_component(DAY15_NCCL_LIBRARY_DIR "${DAY15_NCCL_LIBRARY}" DIRECTORY)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(day15_allreduce src/main.cpp src/bootstrap.cpp)
target_include_directories(day15_allreduce PRIVATE include "${DAY15_NCCL_INCLUDE}")
target_link_libraries(day15_allreduce PRIVATE
  "${DAY15_NCCL_LIBRARY}" CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(day15_allreduce PRIVATE
  DAY15_EXPECTED_NCCL_LIBRARY="${DAY15_NCCL_REAL_LIBRARY}")
target_compile_options(day15_allreduce PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(day15_allreduce PROPERTIES
  BUILD_RPATH "${DAY15_NCCL_LIBRARY_DIR}")
