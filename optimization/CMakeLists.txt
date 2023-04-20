cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(optimization LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_FLAGS "$ENV{CMAKE_CXX_FLAGS} -fPIC")

file(GLOB_RECURSE SRC src/*.h src/*.cc src/*.cpp)
add_library(optimization ${SRC})

file(GLOB_RECURSE TEST test/*.cpp)
add_executable(test_optimization ${TEST})
target_link_libraries(test_optimization optimization)
