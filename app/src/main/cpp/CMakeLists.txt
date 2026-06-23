cmake_minimum_required(VERSION 3.22.1)
project(vkcoopmat)

add_library(native-bench SHARED native-bench.cpp)

find_library(log-lib log)
find_library(vulkan-lib vulkan)

target_compile_options(native-bench PRIVATE -Wall -O3)

target_link_libraries(native-bench
    ${vulkan-lib}
    ${log-lib})
