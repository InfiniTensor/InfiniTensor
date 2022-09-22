cmake_minimum_required(VERSION 3.10) # Required by CMAKE_CUDA_HOST_COMPILER
include(CMakeDependentOption)
project(InfiniTensor C CXX)

# Do not change these options in this file. Use cmake.config, cmake -DOPTION=VALUE, or ccmake to specify them.
option(USE_CUDA "Support CUDA GPU" OFF)
option(USE_BANG "Support BANG MLU" ON)
option(USE_BACKTRACE "Print backtrace on exception and segmentation fault" ON)
option(USE_PROTOBUF "Serialize and deserialize tensors" ON)
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
  message(${PROTO_SRCS} "-----------" ${PROTO_FILES})
  message(${PROTO_HDRS} "-----------" ${PROTO_FILES})
  add_library(tensor_proto SHARED ${PROTO_SRCS} ${PROTO_HDRS})
  target_link_libraries(tensor_proto PUBLIC ${PROTOBUF_LIBRARIES})
endif()


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

# Source files
file(GLOB_RECURSE SRC src/core/*.cc src/kernels/cpu/*.cc src/nnet/*.cc src/operators/*.cc src/utils/*.cc)

if(USE_CUDA)
  file(GLOB_RECURSE SRC_CUDA src/cuda/*.cc src/cuda/*.cu src/kernels/cuda/*.cc src/kernels/cuda/*.cu)
  list (APPEND SRC ${SRC_CUDA})
endif()

if(USE_BANG)
  file(GLOB_RECURSE SRC_BANG src/bang/*.cc src/bang/*.mlu src/kernels/bang/*.cc src/kernels/bang/*.mlu)
  list (APPEND SRC ${SRC_BANG})
endif()

# Libraries
add_library(InfiniTensor SHARED ${SRC})
if(USE_PROTOBUF)
  target_link_libraries(InfiniTensor tensor_proto)
endif()

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
  # CMP0104 requires CUDA_ARCHITECTURES
  set_target_properties(InfiniTensor PROPERTIES CUDA_ARCHITECTURES "70;80")
  enable_language(CUDA)
  # TODO: find_package seems unnecessary for CMake >= 3.8
  find_package(CUDA REQUIRED)
  # message("CUBLAS_LIBRARIES: ${CUDA_LIBRARIES}")
  target_link_libraries(InfiniTensor cudnn curand cublas ${CUDA_LIBRARIES})
endif()

if(USE_BANG)
  ################################################################################
  # Neuware Evironment
  ################################################################################
  # cnrt cndrv cnnl
  if ((NOT DEFINED NEUWARE_HOME) AND (NOT DEFINED ENV{NEUWARE_HOME}))
    message(FATAL_ERROR "NEUWARE_HOME is not defined from cmake or env")
  elseif (DEFINED NEUWARE_HOME)
    set(NEUWARE_HOME ${NEUWARE_HOME} CACHE STRING "NEUWARE_HOME directory for Cambricon Neuware development")
  else()
    set(NEUWARE_HOME $ENV{NEUWARE_HOME} CACHE STRING "NEUWARE_HOME directory for Cambricon Neuware development")
  endif()
  message(STATUS "NEUWARE_HOME: ${NEUWARE_HOME}")
  
  include_directories("${NEUWARE_HOME}/include")
  find_library(CAMBRICON_CNNL libcnnl.so "${NEUWARE_HOME}/lib64")
  find_library(CAMBRICON_CNRT libcnrt.so "${NEUWARE_HOME}/lib64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++ -Wall -Werror")
    
  if ((NOT DEFINED TARGET_CPU_ARCH) AND (NOT DEFINED ENV{TARGET_CPU_ARCH}))
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _uname_m OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(TARGET_CPU_ARCH "${_uname_m}" CACHE STRING "Target CPU ARCH")
  elseif(DEFINED TARGET_CPU_ARCH)
    set(TARGET_CPU_ARCH ${TARGET_CPU_ARCH} CACHE STRING "Target CPU ARCH")
  else()
    set(TARGET_CPU_ARCH $ENV{TARGET_CPU_ARCH} CACHE STRING "Target CPU ARCH")
  endif()
  message(STATUS "TARGET_CPU_ARCH: ${TARGET_CPU_ARCH}")

  ################################################################################
  # Sample Kernels
  ################################################################################
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{NEUWARE_HOME}/cmake" "$ENV{NEUWARE_HOME}/cmake/modules")
  find_package(BANG)
  if(NOT BANG_FOUND)
    message(FATAL_ERROR "BANG cannot be found.")
  elseif(NOT BANG_CNCC_EXECUTABLE)
    message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should check path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
  endif()
  set(BANG_CNCC_FLAGS "-Wall -Werror -fPIC -std=c++11 --target=${TARGET_CPU_ARCH} -O3")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}"
                                             "--bang-arch=compute_20"
                                             "--bang-arch=compute_30"
                                             "--bang-mlu-arch=mtp_322"
                                             "--bang-wram-align64"
  )
  
  if(${TARGET_CPU_ARCH} MATCHES "aarch64-linux-gnu")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    execute_process(
      COMMAND uname -m
      OUTPUT_VARIABLE _uname_m
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT ("${TARGET_CPU_ARCH}" MATCHES ".*${_uname_m}.*" AND "${_uname_m}" MATCHES "aarch64"))
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" "-v" "-c" "-x" "c++" "/dev/null" "-M"
        ERROR_VARIABLE _cxx_verbose
      )
      execute_process(
        COMMAND "echo" "${_cxx_verbose}"
        COMMAND "sed" "-n" "/include.*search starts here/,/End of search list/{s/^ //p}"
        COMMAND "tr" "'\n'" ";"
        OUTPUT_VARIABLE _cxx_includes
      )
      list(REMOVE_ITEM _cxx_includes "/usr/include")
      foreach(_include ${_cxx_includes})
        message(STATUS "add include path: ${_include}")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -idirafter ${_include}")
      endforeach()
    endif()
  endif()

  target_link_libraries(InfiniTensor ${CAMBRICON_CNNL} ${CAMBRICON_CNRT} stdc++)

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
    if (USE_CUDA)
      build_test(test/kernels/cuda/*.cc)
    endif()
    if (USE_BANG)
      #build_test(test/kernels/bang/*.cc)
    endif()
  endif()
  if(BUILD_TEST_PET)
    build_test(test/pet/*.cc)
  endif()
  if(BUILD_TEST_EINNET)
    build_test(test/nnet/test_*.cc)
  endif()
endif()
