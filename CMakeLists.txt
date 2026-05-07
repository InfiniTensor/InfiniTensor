# Do not change these options in this file. Use cmake.config, cmake -DOPTION=VALUE, or ccmake to specify them.
option(USE_BACKTRACE "Print backtrace on exception and segmentation fault" ON)
option(USE_PROTOBUF "Serialize and deserialize tensors" OFF)
option(BUILD_NNET "Build nnet" OFF)
option(BUILD_DIST "Build project for distributed running" OFF)
option(BUILD_TEST "Build tests" OFF)
option(USE_CUDA "Use CUDA" OFF)
option(USE_ASCEND "Use Ascend" OFF)
option(USE_METAX "Use Metax" OFF)
option(USE_MOORE "Use Moore" OFF)
option(USE_BANG "Support BANG MLU" OFF)
option(USE_ILUVATAR "Use Iluvatar" OFF)
option(USE_KUNLUN "Use Kunlun" OFF)
option(USE_TORCH "Use Torch" OFF)

# InfiniOps requires CMake 3.18+ for CUDAToolkit support
cmake_minimum_required(VERSION 3.18)

include(CMakeDependentOption)
project(InfiniTensor C CXX)

cmake_dependent_option(BUILD_TEST_CORE "Build tests for core components" ON BUILD_TEST OFF)
cmake_dependent_option(BUILD_TEST_PET "Build tests for PET" OFF BUILD_TEST OFF)

set(DEFAULT_BUILD_TYPE "RelWithDebInfo")
# Build Type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Configuring for Debug build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    add_compile_definitions(DEBUG_MODE)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Configuring for Release build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    add_compile_definitions(NDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message("Configuring for RelWithDebInfo build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
else()
    message("Build type not specified. Configuring for RelWithDebInfo build.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2")
endif()


if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  message(STATUS "Using config.cmake in CMAKE_CURRENT_BINARY_DIR directory")
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    message(STATUS "Using config.cmake in CMAKE_CURRENT_SOURCE_DIR directory")
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=gnu++11 when on, -std=c++11 when off
add_compile_options(-Wno-error=unused-variable)

# REF: 
# 1. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107852
# 2. https://github.com/assimp/assimp/pull/5577
if(((CMAKE_C_COMPILER_ID MATCHES "GNU") AND NOT MINGW AND NOT HAIKU) AND CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 13)
  message(STATUS "GCC13 or later detected, disabling \"-Warray-bounds and -Wstringop-overflow\" for all source files as it appears to be a false positive")
  add_compile_options(-Wno-array-bounds -Wno-stringop-overflow)
endif()

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
if(USE_PROTOBUF)
  add_definitions(-D TENSOR_PROTOBUF)
  find_package(Protobuf REQUIRED)
  message(STATUS "protobuf include: " ${PROTOBUF_INCLUDE_DIRS})
  message(STATUS "protobuf libraries: " ${PROTOBUF_LIBRARIES})
  message(STATUS "protoc executable: " ${PROTOBUF_PROTOC_EXECUTABLE})
  include_directories(${PROTOBUF_INCLUDE_DIR})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/proto")
  file(GLOB PROTO_FILES "${PROTO_PATH}/data.proto")
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
  set_source_files_properties (${PROTO_SRCS} PROPERTIES COMPILE_FLAGS -Wno-unused-variable)
  add_library(tensor_proto SHARED ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(tensor_proto PUBLIC ${PROTOBUF_LIBRARIES})
endif()

include_directories(include)
# Pybind11
add_subdirectory(3rd-party/pybind11)
include_directories(3rd-party/pybind11/include)

# nlohmann_json
add_subdirectory(3rd-party/nlohmann_json_cmake_fetchcontent)
include_directories(3rd-party/nlohmann_json_cmake_fetchcontent/single_include)

set(INFINIOPS_DIR "${CMAKE_SOURCE_DIR}/3rd-party/infiniops"
    CACHE PATH "Path to InfiniOps source directory (default: git submodule)")

set(GENERATE_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(WITH_CPU ON CACHE BOOL "" FORCE)

if(USE_CUDA)
    set(WITH_NVIDIA ON CACHE BOOL "" FORCE)
endif()
if(USE_BANG)
    set(WITH_CAMBRICON ON CACHE BOOL "" FORCE)
endif()
if(USE_KUNLUN)
    set(WITH_KUNLUN ON CACHE BOOL "" FORCE)
endif()
if(USE_ASCEND)
    set(WITH_NVIDIA ON CACHE BOOL "" FORCE)
endif()
if(USE_METAX)
    set(WITH_METAX ON CACHE BOOL "" FORCE)
endif()
if(USE_MOORE)
    set(WITH_MOORE ON CACHE BOOL "" FORCE)
endif()
if(USE_ILUVATAR)
    set(WITH_ILUVATAR ON CACHE BOOL "" FORCE)
endif()
if(USE_TORCH)
    set(WITH_TORCH ON CACHE BOOL "" FORCE)
endif()


add_subdirectory(${INFINIOPS_DIR} ${CMAKE_BINARY_DIR}/infiniops EXCLUDE_FROM_ALL)

# TVM backend
if(BUILD_NNET AND BUILD_TEST)
  # TVM and DMLC for invoking TVM packed functions
  include_directories(${TVM_INCLUDE_DIR})
  include_directories(${DMLC_INCLUDE_DIR})
  include_directories(${DLPACK_INCLUDE_DIR})
  if (TVM_INCLUDE_DIR AND DMLC_INCLUDE_DIR AND DLPACK_INCLUDE_DIR AND DLPACK_INCLUDE_DIR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_LOGGING_LIBRARY=\\\<${TVM_INCLUDE_DIR}/tvm/runtime/logging.h\\\> ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DINFINI_USE_TVM=1") # Enable TVM codegen kernels
  else()
    # message(FATAL_ERROR "TVM_INCLUDE_DIR, DMLC_INCLUDE_DIR, and DLPACK_INCLUDE_DIR must be set when BUILD_NNET AND BUILD_TEST is ON")
  endif()
endif()

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=pointer-arith")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UNDEBUG") # Enable assertion
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -UNDEBUG") # Enable assertion


# Source files
file(GLOB_RECURSE SRC src/ffi/*.cc src/core/*.cc src/operators/*.cc src/utils/*.cc)

if(BUILD_NNET)
  add_compile_definitions(BUILD_NNET=1)
  file(GLOB_RECURSE SRC_NNET src/nnet/*.cc)
  list (APPEND SRC ${SRC_NNET})
  # For locating resource files
  set_source_files_properties(src/nnet/test.cc PROPERTIES COMPILE_OPTIONS "-DINFINI_PROJECT_HOME=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# InfiniOps adapter kernels
file(GLOB_RECURSE SRC_INFINIOPS_ADAPTER src/kernels/*.cc)
if(WITH_NVIDIA OR WITH_ILUVATAR)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set_source_files_properties(${SRC_INFINIOPS_ADAPTER}
            PROPERTIES LANGUAGE CUDA)
    endif()
endif()
list(APPEND SRC ${SRC_INFINIOPS_ADAPTER})

# Libraries
add_library(InfiniTensor SHARED ${SRC})
if(USE_PROTOBUF)
  target_link_libraries(InfiniTensor PRIVATE tensor_proto)
endif()

# Link InfiniOps operator library (PUBLIC so InfiniOps headers are available
# to all downstream consumers of InfiniTensor, e.g. the backend pybind11 module)
target_link_libraries(InfiniTensor PUBLIC infiniops)

target_link_options(InfiniTensor PRIVATE -Wl,--allow-multiple-definition)

target_link_libraries(InfiniTensor PRIVATE pybind11::embed)

# TVM backend
if(BUILD_NNET AND BUILD_TEST AND TVM_LIB_DIR)
  target_link_libraries(InfiniTensor PRIVATE ${TVM_LIB_DIR}/libtvm.so)
endif()

# Python bindings
file(GLOB_RECURSE FFIS src/ffi/ffi_infinitensor.cc)
pybind11_add_module(backend MODULE ${FFIS})
target_link_libraries(backend PRIVATE InfiniTensor)

if(USE_BACKTRACE)
  add_definitions(-D BACKWARD_TRACE)
  add_subdirectory(3rd-party/backward-cpp)
  include_directories(3rd-party/backward-cpp)
  add_backward(InfiniTensor)
  target_link_libraries(InfiniTensor PRIVATE dw)
endif()

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
  add_compile_definitions(BUILD_TEST=1)
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
  include_directories(test/kernels)
  build_test(test/kernels/*.cpp)
  if(BUILD_NNET AND BUILD_TEST)
    build_test(test/nnet/test_*.cc)

    # Build expression reader
    add_executable(nnet_reader test/nnet/readlog.cc)
    target_link_libraries(nnet_reader InfiniTensor)
  endif()
endif()
