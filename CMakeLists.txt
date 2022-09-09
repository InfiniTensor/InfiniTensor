cmake_minimum_required(VERSION 3.10) # Required by CMAKE_CUDA_HOST_COMPILER
include(CMakeDependentOption)
project(InfiniTensor C CXX)

# Do not change these options in this file. Use cmake.config, cmake -DOPTION=VALUE, or ccmake to specify them.
option(USE_CUDA "Support CUDA GPU" ON)
option(USE_BACKTRACE "Print backtrace on exception and segmentation fault" ON)
option(BUILD_TEST "Build tests" ON)

cmake_dependent_option(BUILD_TEST_CORE "Build tests for core components" ON BUILD_TEST OFF)
cmake_dependent_option(BUILD_TEST_PET "Build tests for PET" OFF BUILD_TEST OFF)
cmake_dependent_option(BUILD_TEST_EINNET "Build tests for EINNET" OFF BUILD_TEST OFF)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Werror -Wno-error=deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG") # Enable assertion
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG") # Enable assertion

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED)
# OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
#Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "protobuf include: " ${PROTOBUF_INCLUDE_DIRS})
message(STATUS "protobuf libraries: " ${PROTOBUF_LIBRARIES})
message(STATUS "protoc executable: " ${PROTOBUF_PROTOC_EXECUTABLE})
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/proto")
file(GLOB PROTO_FILES "${PROTO_PATH}/data.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
message(${PROTO_SRCS} "-----------" ${PROTO_FILES})
message(${PROTO_HDRS} "-----------" ${PROTO_FILES})
add_library(tensor_proto SHARED ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(tensor_proto PUBLIC ${PROTOBUF_LIBRARIES})


include_directories(include)

# # Pybind11
# add_subdirectory(3rd-party/pybind11)
# include_directories(3rd-party/pybind11/include)

# nlohmann_json
add_subdirectory(3rd-party/nlohmann_json_cmake_fetchcontent)
include_directories(3rd-party/nlohmann_json_cmake_fetchcontent/single_include)

if(BUILD_TEST)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "Do not build gmock" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "Do not install gtest" FORCE)
  add_subdirectory(3rd-party/googletest)
  include_directories(3rd-party/googletest/googletest/include)
endif()

file(GLOB_RECURSE SRC src/*.cc src/*.cu)

add_library(InfiniTensor SHARED ${SRC})
target_link_libraries(InfiniTensor tensor_proto)

if(USE_BACKTRACE)
  add_definitions(-D BACKWARD_TRACE)
  add_subdirectory(3rd-party/backward-cpp)
  include_directories(3rd-party/backward-cpp)
  add_backward(InfiniTensor)
  target_link_libraries(InfiniTensor dw)
endif()

if(USE_CUDA)
  # Since enable_language only executes once, rerun cmake is required if CMAKE_CUDA_HOST_COMPILER is wrong
  set(CMAKE_CUDA_HOST_COMPILER
      ${CMAKE_CXX_COMPILER}
      CACHE STRING "Set cuda host compiler path")
  enable_language(CUDA)
  # TODO: find_package seems unnecessary for CMake >= 3.8
  find_package(CUDA REQUIRED)
  # message("CUBLAS_LIBRARIES: ${CUDA_LIBRARIES}")
  target_link_libraries(InfiniTensor cudnn curand cublas ${CUDA_LIBRARIES})
endif()

# # Python bindings
# pybind11_add_module(infini MODULE ${FFI})
# target_link_libraries(infini PRIVATE infini_cpp)

function(build_test files)
  # Non-recursive glob for skip failed tests
  file(GLOB TEST_SOURCES ${files})
  foreach(testsourcefile ${TEST_SOURCES})
    get_filename_component(testname ${testsourcefile} NAME_WE)
    add_executable(${testname} ${testsourcefile})
    target_link_libraries(${testname} InfiniTensor GTest::gtest_main)
    add_test(NAME ${testname} COMMAND ${testname})
  endforeach(testsourcefile ${TEST_SOURCES})
endfunction()

if(BUILD_TEST)
  enable_testing()
  if(USE_TRACE)
    build_test(test/trace/*.cc)
  endif()
  if(BUILD_TEST_CORE)
    build_test(test/core/*.cc)
    build_test(test/operators/*.cc)
  endif()
  if(BUILD_TEST_PET)
    build_test(test/pet/*.cc)
  endif()
  if(BUILD_TEST_EINNET)
    build_test(test/nnet/*.cc)
  endif()
endif()
