# Do not change these options in this file. Use cmake.config, cmake -DOPTION=VALUE, or ccmake to specify them.
option(USE_CUDA "Support CUDA GPU" OFF)
option(USE_BANG "Support BANG MLU" OFF)
option(USE_KUNLUN "Support KUNLUN XPU" OFF)
option(USE_INTELCPU "Support INTELCPU" OFF)
option(USE_BACKTRACE "Print backtrace on exception and segmentation fault" ON)
option(USE_PROTOBUF "Serialize and deserialize tensors" OFF)
option(BUILD_NNET "Build nnet" OFF)
option(BUILD_DIST "Build project for distributed running" OFF)
option(BUILD_TEST "Build tests" OFF)
option(SUP_CODE_GEN "Support code generate" OFF)

if(USE_CUDA OR SUP_CODE_GEN)
    message("CMake 3.18 or higher is required for setting CUDAToolkit")
    cmake_minimum_required(VERSION 3.18) # FindCUDAToolkit
else()
    cmake_minimum_required(VERSION 3.17)
endif()

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
add_subdirectory(3rd-party/fmt)
include_directories(3rd-party/fmt/include)

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
file(GLOB_RECURSE SRC src/ffi/*.cc src/core/*.cc src/kernels/cpu/*.cc src/operators/*.cc src/utils/*.cc)

if(BUILD_NNET)
  add_compile_definitions(BUILD_NNET=1)
  file(GLOB_RECURSE SRC_NNET src/nnet/*.cc)
  list (APPEND SRC ${SRC_NNET})
  # For locating resource files
  set_source_files_properties(src/nnet/test.cc PROPERTIES COMPILE_OPTIONS "-DINFINI_PROJECT_HOME=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(USE_CUDA)
  file(GLOB_RECURSE SRC_CUDA src/cuda/*.cc src/cuda/*.cu src/kernels/cuda/*.cc src/kernels/cuda/*.cu)
  list (APPEND SRC ${SRC_CUDA})
endif()

if(USE_BANG)
  file(GLOB_RECURSE SRC_BANG src/bang/*.cc src/kernels/bang/*.cc )
  list (APPEND SRC ${SRC_BANG})
endif()

if(USE_KUNLUN)
  file(GLOB_RECURSE SRC_KUNLUN src/kunlun/*.cc src/kernels/kunlun/*.cc )
  list (APPEND SRC ${SRC_KUNLUN})
endif()

if(USE_INTELCPU)
  file(GLOB_RECURSE SRC_INTELCPU src/intelcpu/*.cc src/kernels/intelcpu/*.cc )
  list (APPEND SRC ${SRC_INTELCPU})
endif()

if(SUP_CODE_GEN)
file(GLOB_RECURSE CUS_OPS_SRC src/code_gen/*.cc src/code_gen/custom_ops.cu)
list(APPEND SRC ${CUS_OPS_SRC})
endif()

# Libraries
add_library(InfiniTensor SHARED ${SRC})
if(USE_PROTOBUF)
  target_link_libraries(InfiniTensor tensor_proto)
endif()

target_link_libraries(InfiniTensor pybind11::embed)
target_link_libraries(InfiniTensor fmt::fmt)

if(SUP_CODE_GEN)
set_target_properties(InfiniTensor PROPERTIES CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)
find_package(CUDAToolkit)
target_link_libraries(InfiniTensor cudnn CUDA::curand CUDA::cublas CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
endif()

# TVM backend
if(BUILD_NNET AND BUILD_TEST AND TVM_LIB_DIR)
  target_link_libraries(InfiniTensor ${TVM_LIB_DIR}/libtvm.so)
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
  target_link_libraries(InfiniTensor dw)
endif()

if(USE_INTELCPU)
  add_compile_definitions(USE_INTELCPU=1)
  find_package(MKL CONFIG REQUIRED)

  # Refer to https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
  target_link_libraries(InfiniTensor sycl OpenCL)

  set(DNNL_CONFIGURATION "cpu_gomp")
  find_package(dnnl CONFIG REQUIRED)
  if(dnnl_FOUND)
      add_compile_definitions(USE_MKL=1)
      include_directories(BEFORE ${dnnl_DIR}/../../../cpu_gomp/include/)
      link_directories(${dnnl_DIR}/../../../cpu_gomp/lib)
      target_link_libraries(InfiniTensor   dnnl)
  else()
      message(FATAL_ERROR "dnnl library not found")
  endif()
  set(WNO_ERRORS "-Wno-error=unused-parameter -Wno-error=unused-function -Wno-error=unused-private-field -Wno-error=ignored-attributes -Wno-error=unused-const-variable -Wno-error=inconsistent-missing-override -Wno-error=unused-variable -Wno-error=tautological-constant-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -qmkl=parallel -Werror  ${WNO_ERRORS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMKL_ILP64 -qmkl=parallel ${WNO_ERRORS}") # Enable assertion
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DMKL_ILP64 -qmkl=parallel ${WNO_ERRORS}") # Enable assertion

  find_package(IntelDPCPP REQUIRED)
endif()

if(USE_CUDA)
  add_compile_definitions(USE_CUDA=1)
  # Since enable_language only executes once, rerun cmake is required if CMAKE_CUDA_HOST_COMPILER is wrong
  set(CMAKE_CUDA_HOST_COMPILER
      ${CMAKE_CXX_COMPILER}
      CACHE STRING "Set cuda host compiler path")
  # CMP0104 requires CUDA_ARCHITECTURES
  set_target_properties(InfiniTensor PROPERTIES CUDA_ARCHITECTURES "70;80")
  enable_language(CUDA)
  find_package(CUDAToolkit) # For nvrtc and cuda driver
  target_link_libraries(InfiniTensor cudnn CUDA::curand CUDA::cublas CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
  if (BUILD_DIST)
    message(STATUS "Add BUILD_DIST, use NCCL with CUDA")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(NCCL REQUIRED)
    add_compile_definitions(INFINI_USE_NCCL=1)
    target_link_libraries(InfiniTensor nccl)
  endif()
endif()

if(USE_BANG)
  add_compile_definitions(USE_BANG=1)
  include_directories(src/kernels/mlu/include)
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
  find_library(CAMBRICON_CNDRV libcndrv.so "${NEUWARE_HOME}/lib64")
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
  # BangC Kernels
  ################################################################################

  if (BUILD_DIST)
    find_library(CAMBRICON_CNCL libcncl.so "${NEUWARE_HOME}/lib64")
    target_link_libraries(InfiniTensor ${CAMBRICON_CNCL} ${CAMBRICON_CNNL} ${CAMBRICON_CNRT} ${CAMBRICON_CNDRV} stdc++)
    message(STATUS "Add BUILD_DIST, use CNCL with BANG")
    add_compile_definitions(INFINI_USE_CNCL=1)
  else()
    target_link_libraries(InfiniTensor ${CAMBRICON_CNNL} ${CAMBRICON_CNRT} ${CAMBRICON_CNDRV} stdc++)
  endif()
endif()

if(USE_KUNLUN)
  add_compile_definitions(USE_KUNLUN=1)
  if ((NOT DEFINED KUNLUN_HOME) AND (NOT DEFINED ENV{KUNLUN_HOME}))
    message(FATAL_ERROR "KUNLUN_HOME is not defined from cmake or env")
  elseif (DEFINED KUNLUN_HOME)
          set(KUNLUN_HOME ${KUNLUN_HOME} CACHE STRING "KUNLUN_HOME directory for Kunlun development")
  else()
          set(KUNLUN_HOME $ENV{KUNLUN_HOME} CACHE STRING "KUNLUN_HOME directory for Kunlun development")
  endif()
  message(STATUS "KUNLUN_HOME: ${KUNLUN_HOME}")

  include_directories("${KUNLUN_HOME}/include/")
  find_library(KUNLUN_RT libxpurt.so "${KUNLUN_HOME}/lib64/")
  find_library(KUNLUN_DNN libxpuapi.so "${KUNLUN_HOME}/lib64/")
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

  if (BUILD_DIST)
    message(STATUS "Add BUILD_DIST, use XCCL with KUNLUN XPU")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(XCCL REQUIRED)
    add_compile_definitions(INFINI_USE_XCCL=1)
    target_link_libraries(InfiniTensor ${XCCL_LIBRARIES})
  endif()
  target_link_libraries(InfiniTensor ${KUNLUN_RT} ${KUNLUN_DNN} stdc++)
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
  add_compile_definitions(BUILD_TEST=1)
  enable_testing()
  if(USE_TRACE)
    build_test(test/trace/*.cc)
  endif()
  if(BUILD_TEST_CORE)
    build_test(test/core/*.cc)
    build_test(test/operators/*.cc)
    build_test(test/kernels/nativecpu/*.cc)
    if (USE_CUDA)
      build_test(test/kernels/cuda/*.cc)
      build_test(test/cuda/*.cc)
    endif()
    if (USE_BANG)
      build_test(test/kernels/bang/*.cc)
      build_test(test/bang/*.cc)
    endif()
    if (USE_KUNLUN)
      build_test(test/kernels/kunlun/*.cc)
      build_test(test/kunlun/*.cc)
    endif()
    if (USE_INTELCPU)
      build_test(test/kernels/intelcpu/*.cc)
    endif()
  endif()

  if(BUILD_NNET AND BUILD_TEST)
    build_test(test/nnet/test_*.cc)

    # Build expression reader
    add_executable(nnet_reader test/nnet/readlog.cc)
    target_link_libraries(nnet_reader InfiniTensor)
  endif()

  if(SUP_CODE_GEN)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/code_gen")
    add_executable(conv1 test/code_generate/conv_test_1.cc)
    target_link_libraries(conv1 InfiniTensor)

    build_test(test/code_generate/conv_test_2.cc)
    #target_link_libraries(conv2 InfiniTensor)

    build_test(test/code_generate/conv_test_3.cc)
    #target_link_libraries(conv3 InfiniTensor)

    add_executable(conv4 test/code_generate/conv_test_4_perf.cc)
    target_link_libraries(conv4 InfiniTensor)

    build_test(test/code_generate/conv_test_5_trans_kernel.cc)
    #target_link_libraries(conv5 InfiniTensor)

    build_test(test/code_generate/convtrans_test_1.cc)
    #target_link_libraries(convtrans1 InfiniTensor)

    build_test(test/code_generate/convtrans_test_2.cc)
    #target_link_libraries(convtrans2 InfiniTensor)

    add_executable(dilation test/code_generate/dilation_test.cpp)
    target_link_libraries(dilation InfiniTensor)

    add_executable(graph1 test/code_generate/SampleGraph1.cpp)
    target_link_libraries(graph1 InfiniTensor)

    add_executable(graph2 test/code_generate/SampleGraph2.cpp)
    target_link_libraries(graph2 InfiniTensor)

    add_executable(graph3 test/code_generate/SampleGraph3.cpp)
    target_link_libraries(graph3 InfiniTensor)

    add_executable(graph4 test/code_generate/SampleGraph4.cpp)
    target_link_libraries(graph4 InfiniTensor)

    add_executable(mutant1 test/code_generate/mutant_test_1.cc)
    target_link_libraries(mutant1 InfiniTensor)

    add_executable(mutant2 test/code_generate/mutant_test_2.cc)
    target_link_libraries(mutant2 InfiniTensor)

    add_executable(mutant3 test/code_generate/mutant_test_3.cc)
    target_link_libraries(mutant3 InfiniTensor)

    add_executable(trans1 test/code_generate/transpose_test_1.cc)
    target_link_libraries(trans1 InfiniTensor)

    add_executable(trans2 test/code_generate/transpose_test_2.cc)
    target_link_libraries(trans2 InfiniTensor)

    add_executable(trans3 test/code_generate/transpose_test_3.cc)
    target_link_libraries(trans3 InfiniTensor)

    add_executable(trans4 test/code_generate/transpose_test_4.cc)
    target_link_libraries(trans4 InfiniTensor)

    add_executable(trans5 test/code_generate/transpose_test_5.cc)
    target_link_libraries(trans5 InfiniTensor)

    add_executable(tensor test/code_generate/tensor_test.cc)
    target_link_libraries(tensor InfiniTensor)

    add_executable(rule1 test/code_generate/single_rule_test_1_n2h.cc)
    target_link_libraries(rule1 InfiniTensor)

    add_executable(rule2 test/code_generate/single_rule_test_2_n2w.cc)
    target_link_libraries(rule2 InfiniTensor)

    add_executable(rule3 test/code_generate/single_rule_test_3_c2h.cc)
    target_link_libraries(rule3 InfiniTensor)

    add_executable(rule4 test/code_generate/single_rule_test_4_c2w.cc)
    target_link_libraries(rule4 InfiniTensor)

    add_executable(rule5 test/code_generate/single_rule_test_5_dilated.cc)
    target_link_libraries(rule5 InfiniTensor)

    add_executable(rule6 test/code_generate/single_rule_test_6_bgemm.cc)
    target_link_libraries(rule6 InfiniTensor)

    add_executable(rule7 test/code_generate/single_rule_test_7_trans_gemm.cc)
    target_link_libraries(rule7 InfiniTensor)

    add_executable(rule8 test/code_generate/single_rule_test_8_conv2group.cc)
    target_link_libraries(rule8 InfiniTensor)

    add_executable(rule9 test/code_generate/single_rule_test_9_group_conv_a.cc)
    target_link_libraries(rule9 InfiniTensor)

    add_executable(rule10 test/code_generate/single_rule_test_10_group_conv_b.cc)
    target_link_libraries(rule10 InfiniTensor)

    add_executable(rule11 test/code_generate/single_rule_test_11_group_conv_c.cc)
    target_link_libraries(rule11 InfiniTensor)

    add_executable(rule12 test/code_generate/single_rule_test_12_1xk_trans.cc)
    target_link_libraries(rule12 InfiniTensor)

    add_executable(rule13 test/code_generate/single_rule_test_13_nx1xk_trans.cc)
    target_link_libraries(rule13 InfiniTensor)

    add_executable(rule14 test/code_generate/single_rule_test_14_split_group_conv.cc)
    target_link_libraries(rule14 InfiniTensor)

    add_executable(rule15 test/code_generate/single_rule_test_15_conv_1x1.cc)
    target_link_libraries(rule15 InfiniTensor)

    add_executable(rule16 test/code_generate/single_rule_test_16_group_conv_d.cc)
    target_link_libraries(rule16 InfiniTensor)

    add_executable(op1 test/code_generate/single_op_test_1.cc)
    target_link_libraries(op1 InfiniTensor)

    add_executable(op2 test/code_generate/single_op_test_2.cc)
    target_link_libraries(op2 InfiniTensor)

    add_executable(op3 test/code_generate/single_op_test_3.cc)
    target_link_libraries(op3 InfiniTensor)

    add_executable(op4 test/code_generate/single_op_test_4.cc)
    target_link_libraries(op4 InfiniTensor)

    add_executable(op5 test/code_generate/single_op_test_5.cc)
    target_link_libraries(op5 InfiniTensor)

    add_executable(op6 test/code_generate/single_op_test_6.cc)
    target_link_libraries(op6 InfiniTensor)

    add_executable(op7 test/code_generate/single_op_test_7_batch_gemm.cc)
    target_link_libraries(op7 InfiniTensor)

    add_executable(te1 test/code_generate/eliminator_test_1.cc)
    target_link_libraries(te1 InfiniTensor)

    add_executable(split1 test/code_generate/split_test_1.cc)
    target_link_libraries(split1 InfiniTensor)

    add_executable(split2 test/code_generate/split_test_2.cc)
    target_link_libraries(split2 InfiniTensor)

    add_executable(split3 test/code_generate/split_test_3.cc)
    target_link_libraries(split3 InfiniTensor)

    add_executable(concat1 test/code_generate/concat_test_1.cc)
    target_link_libraries(concat1 InfiniTensor)

    add_executable(extend1 test/code_generate/extend_test_1.cc)
    target_link_libraries(extend1 InfiniTensor)

    add_executable(group1 test/code_generate/group_conv_test_1.cc)
    target_link_libraries(group1 InfiniTensor)

    add_executable(group2 test/code_generate/group_conv_test_2.cc)
    target_link_libraries(group2 InfiniTensor)

    add_executable(fuse_activation test/code_generate/fuse_activation.cpp)
    target_link_libraries(fuse_activation InfiniTensor)

    add_executable(gen_transpose test/code_generate/gen_transpose.cpp)
    target_link_libraries(gen_transpose InfiniTensor)

    add_executable(stat_graph1 test/code_generate/graph_stat_test_1_transpose_group.cc)
    target_link_libraries(stat_graph1 InfiniTensor)

    add_executable(stat_graph2 test/code_generate/graph_stat_test_2_batchmm.cc)
    target_link_libraries(stat_graph2 InfiniTensor)

    add_executable(pad_slice test/code_generate/pad_slice_test.cc)
    target_link_libraries(pad_slice InfiniTensor)

    add_executable(bgemm test/code_generate/batch_gemm_test.cc)
    target_link_libraries(bgemm InfiniTensor)

    add_executable(gemm1 test/code_generate/gemm_test_1_perf.cc)
    target_link_libraries(gemm1 InfiniTensor)

    add_executable(perf_pool test/code_generate/perf_pool.cc)
    target_link_libraries(perf_pool InfiniTensor)

    add_executable(cache1 test/code_generate/cache_test_1.cc)
    target_link_libraries(cache1 InfiniTensor)

    add_executable(cache2 test/code_generate/cache_test_2.cc)
    target_link_libraries(cache2 InfiniTensor)

    add_executable(cache3 test/code_generate/cache_test_3.cc)
    target_link_libraries(cache3 InfiniTensor)
  endif()
endif()
