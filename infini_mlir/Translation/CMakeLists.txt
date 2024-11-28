cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(translation VERSION 0.0.0 LANGUAGES CXX)
message(STATUS "Project " ${PROJECT_NAME} " version " ${PROJECT_VERSION})

file(GLOB_RECURSE COMMON_SRC src/*.cc)
add_library(translation SHARED ${COMMON_SRC})
target_include_directories(translation PUBLIC include)
target_link_libraries(translation PUBLIC
  InfiniDialectDirectories
  InfiniOps)

# set_target_properties(translation PROPERTIES POSITION_INDEPENDENT_CODE ON)
