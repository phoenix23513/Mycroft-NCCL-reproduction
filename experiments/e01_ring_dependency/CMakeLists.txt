option(MYCROFT_ENABLE_ASAN "Enable AddressSanitizer for supported C targets" OFF)

add_library(e01_ring_sim_lib STATIC src/ring_sim.c)
target_include_directories(e01_ring_sim_lib PUBLIC include)
target_compile_options(e01_ring_sim_lib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(e01_ring_sim src/main.c)
target_link_libraries(e01_ring_sim PRIVATE e01_ring_sim_lib)
target_compile_options(e01_ring_sim PRIVATE -Wall -Wextra -Wpedantic)

add_executable(test_e01_reduce_scatter tests/test_reduce_scatter.c)
target_link_libraries(test_e01_reduce_scatter PRIVATE e01_ring_sim_lib)
target_compile_options(
  test_e01_reduce_scatter
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_test(NAME e01_reduce_scatter COMMAND test_e01_reduce_scatter)

if(MYCROFT_ENABLE_ASAN AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(e01_ring_sim_lib PRIVATE -fsanitize=address -fno-omit-frame-pointer)

  foreach(target e01_ring_sim test_e01_reduce_scatter)
    target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address)
  endforeach()
endif()
