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
    add_compile_definitions(WITH_CAMBRICON=1)
endif()
if(USE_KUNLUN)
    set(WITH_KUNLUN ON CACHE BOOL "" FORCE)
endif()
if(USE_ASCEND)
    set(WITH_ASCEND ON CACHE BOOL "" FORCE)
endif()
if(USE_METAX)
    set(WITH_METAX ON CACHE BOOL "" FORCE)
    add_compile_definitions(WITH_METAX=1)
endif()
if(USE_MOORE)
    set(WITH_MOORE ON CACHE BOOL "" FORCE)
    add_compile_definitions(WITH_MOORE=1)
endif()
if(USE_ILUVATAR)
    set(WITH_ILUVATAR ON CACHE BOOL "" FORCE)
    add_compile_definitions(WITH_ILUVATAR=1)
endif()
if(USE_TORCH)
    set(WITH_TORCH ON CACHE BOOL "" FORCE)
endif()


add_subdirectory(${INFINIOPS_DIR} ${CMAKE_BINARY_DIR}/infiniops EXCLUDE_FROM_ALL)

# Auto-generate InfiniTensor kernel bridges for InfiniOps torch ops.
# Runs at configure time so the generated .cc files are visible to GLOB.
if(WITH_TORCH)
    set(GENERATED_KERNELS_DIR "${CMAKE_BINARY_DIR}/generated_kernels")
    file(MAKE_DIRECTORY ${GENERATED_KERNELS_DIR})
    execute_process(
        COMMAND ${Python_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/generate_infiniops_kernels.py
            --mapping ${CMAKE_SOURCE_DIR}/scripts/infiniops_op_mapping.yaml
            --infiniops-dir ${INFINIOPS_DIR}
            --output-dir ${GENERATED_KERNELS_DIR}
        RESULT_VARIABLE _gen_result
    )
    if(NOT _gen_result EQUAL 0)
        message(FATAL_ERROR "Failed to generate InfiniTensor kernel bridges")
    endif()
endif()

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
file(GLOB_RECURSE SRC src/core/*.cc src/operators/*.cc src/utils/*.cc)

if(BUILD_NNET)
  add_compile_definitions(BUILD_NNET=1)
  file(GLOB_RECURSE SRC_NNET src/nnet/*.cc)
  list (APPEND SRC ${SRC_NNET})
  # For locating resource files
  set_source_files_properties(src/nnet/test.cc PROPERTIES COMPILE_OPTIONS "-DINFINI_PROJECT_HOME=${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# InfiniOps adapter kernels
file(GLOB_RECURSE SRC_INFINIOPS_ADAPTER src/kernels/*.cc)
if(WITH_ILUVATAR)
    # Iluvatar: use clang++ with -x ivcore (similar to Moore/Metax approach)
    # because adapter files include CUDA-specific headers that g++ cannot handle
    set(ILUVATAR_CLANGXX "clang++")
    set(ILUVATAR_ROOT "")
    foreach(_env COREX_ROOT ILUVATAR_ROOT)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            set(ILUVATAR_ROOT "$ENV{${_env}}")
            set(ILUVATAR_CLANGXX "${ILUVATAR_ROOT}/bin/clang++")
            break()
        endif()
    endforeach()
    if(NOT ILUVATAR_ROOT AND EXISTS "/usr/local/corex")
        set(ILUVATAR_ROOT "/usr/local/corex")
        set(ILUVATAR_CLANGXX "${ILUVATAR_ROOT}/bin/clang++")
    endif()
    set(_iluvatar_adapter_objs "")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_adapter")
    foreach(_src ${SRC_INFINIOPS_ADAPTER})
        get_filename_component(_name ${_src} NAME_WE)
        set(_obj "${CMAKE_CURRENT_BINARY_DIR}/iluvatar_adapter/${_name}.o")
        add_custom_command(
            OUTPUT ${_obj}
            COMMAND ${ILUVATAR_CLANGXX} -c
                --cuda-gpu-arch=${ILUVATAR_ARCH}
                --cuda-path=${ILUVATAR_ROOT}
                -x ivcore -std=c++17 -O2 -fPIC
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/nlohmann_json_cmake_fetchcontent/single_include
                -I${CMAKE_CURRENT_SOURCE_DIR}/test/kernels
                -I${INFINIOPS_DIR}/src
                -I${ILUVATAR_ROOT}/include
                -Wno-error=unused-variable -Wno-error=unused-private-field -Wno-unused-variable
                -DNDEBUG -DWITH_CPU=1 -DWITH_ILUVATAR=1
                -o ${_obj} ${_src}
            DEPENDS ${_src}
            COMMENT "Compiling ${_src} with clang++ (Iluvatar ivcore compiler)"
        )
        list(APPEND _iluvatar_adapter_objs ${_obj})
    endforeach()
    add_custom_target(iluvatar_adapter DEPENDS ${_iluvatar_adapter_objs})
    # Object files will be added to InfiniTensor after target creation
elseif(WITH_NVIDIA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set_source_files_properties(${SRC_INFINIOPS_ADAPTER}
            PROPERTIES LANGUAGE CUDA)
    endif()
    list(APPEND SRC ${SRC_INFINIOPS_ADAPTER})
elseif(WITH_MOORE)
    # Adapter kernel files must be compiled with mcc (MUSA compiler)
    # because they include MUSA-specific headers (__device__, musa_fp16, etc.)
    # that g++ cannot handle. Compile them via custom commands and link .o files.
    set(MCC_WRAPPER "${INFINIOPS_DIR}/scripts/mcc_wrapper.sh")
    set(MUSA_ROOT "")
    foreach(_env MUSA_ROOT MUSA_HOME MUSA_PATH)
        if(NOT MUSA_ROOT AND DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            set(MUSA_ROOT "$ENV{${_env}}")
        endif()
    endforeach()
    if(NOT MUSA_ROOT AND EXISTS "/usr/local/musa")
        set(MUSA_ROOT "/usr/local/musa")
    endif()
    set(_moore_adapter_objs "")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/moore_adapter")
    foreach(_src ${SRC_INFINIOPS_ADAPTER})
        get_filename_component(_name ${_src} NAME_WE)
        set(_obj "${CMAKE_CURRENT_BINARY_DIR}/moore_adapter/${_name}.o")
        add_custom_command(
            OUTPUT ${_obj}
            COMMAND ${MCC_WRAPPER} -c -x musa -std=c++17 -O2 -fPIC
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/nlohmann_json_cmake_fetchcontent/single_include
                -I${CMAKE_CURRENT_SOURCE_DIR}/test/kernels
                -I${INFINIOPS_DIR}/src
                -I${MUSA_ROOT}/include
                -DNDEBUG -DWITH_CPU=1 -DWITH_MOORE=1 -DBUILD_TEST=1
                -o ${_obj} ${_src}
            DEPENDS ${_src}
            COMMENT "Compiling ${_src} with mcc (MUSA compiler)"
        )
        list(APPEND _moore_adapter_objs ${_obj})
    endforeach()
    add_custom_target(moore_adapter DEPENDS ${_moore_adapter_objs})
    # Object files will be added to InfiniTensor after target creation
elseif(WITH_METAX)
    # Adapter kernel files must be compiled with mxcc (METAX compiler)
    # because they include METAX-specific headers that g++ cannot handle.
    # Compile them via custom commands and link .o files.
    set(METAX_COMPILER "mxcc")
    foreach(_env MXCC_PATH MACA_HOME METAX_ROOT)
        if(DEFINED ENV{${_env}} AND NOT "$ENV{${_env}}" STREQUAL "")
            set(METAX_COMPILER "$ENV{${_env}}/bin/mxcc")
            break()
        endif()
    endforeach()
    if(NOT EXISTS "${METAX_COMPILER}" AND EXISTS "/opt/maca/mxgpu_llvm/bin/mxcc")
        set(METAX_COMPILER "/opt/maca/mxgpu_llvm/bin/mxcc")
    endif()
    set(_metax_adapter_objs "")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metax_adapter")
    foreach(_src ${SRC_INFINIOPS_ADAPTER})
        get_filename_component(_name ${_src} NAME_WE)
        set(_obj "${CMAKE_CURRENT_BINARY_DIR}/metax_adapter/${_name}.o")
        add_custom_command(
            OUTPUT ${_obj}
            COMMAND ${METAX_COMPILER} -c -x maca -std=c++17 -O2 -fPIC
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/nlohmann_json_cmake_fetchcontent/single_include
                -I${CMAKE_CURRENT_SOURCE_DIR}/test/kernels
                -I${INFINIOPS_DIR}/src
                -DNDEBUG -DWITH_CPU=1 -DWITH_METAX=1 -DBUILD_TEST=1
                -o ${_obj} ${_src}
            DEPENDS ${_src}
            COMMENT "Compiling ${_src} with mxcc (METAX compiler)"
        )
        list(APPEND _metax_adapter_objs ${_obj})
    endforeach()
    add_custom_target(metax_adapter DEPENDS ${_metax_adapter_objs})
    # Object files will be added to InfiniTensor after target creation
else()
    list(APPEND SRC ${SRC_INFINIOPS_ADAPTER})
endif()

# Generated torch kernel bridges (only when WITH_TORCH is enabled).
if(WITH_TORCH)
    file(GLOB GENERATED_KERNEL_SRCS "${GENERATED_KERNELS_DIR}/*.cc")
    list(APPEND SRC ${GENERATED_KERNEL_SRCS})
endif()

# Libraries
add_library(InfiniTensor SHARED ${SRC})
if(WITH_MOORE)
    target_sources(InfiniTensor PRIVATE ${_moore_adapter_objs})
    add_dependencies(InfiniTensor moore_adapter)
endif()
if(WITH_METAX)
    target_sources(InfiniTensor PRIVATE ${_metax_adapter_objs})
    add_dependencies(InfiniTensor metax_adapter)
endif()
if(WITH_ILUVATAR)
    target_sources(InfiniTensor PRIVATE ${_iluvatar_adapter_objs})
    add_dependencies(InfiniTensor iluvatar_adapter)
endif()
if(USE_PROTOBUF)
  target_link_libraries(InfiniTensor PRIVATE tensor_proto)
endif()

# Link InfiniOps operator library (PUBLIC so InfiniOps headers are available
# to all downstream consumers of InfiniTensor, e.g. the backend pybind11 module)
target_link_libraries(InfiniTensor PUBLIC infiniops)

# Remove inherited -x musa/-x maca from InfiniTensor's global compile options;
# these flags should only apply to the adapter kernel files, not all sources.
# infiniops stores "-x;musa" or "-x;maca" as separate list entries in
# INTERFACE_COMPILE_OPTIONS — remove both the "-x" and the language name.
get_target_property(_infiniops_interface_opts infiniops INTERFACE_COMPILE_OPTIONS)
if(_infiniops_interface_opts)
    list(FILTER _infiniops_interface_opts EXCLUDE REGEX "^-x$")
    list(FILTER _infiniops_interface_opts EXCLUDE REGEX "^(musa|maca)$")
    set_target_properties(infiniops PROPERTIES INTERFACE_COMPILE_OPTIONS "${_infiniops_interface_opts}")
endif()

target_link_options(InfiniTensor PRIVATE -Wl,--allow-multiple-definition)

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
  build_test(test/kernels/*.cc)
  if(BUILD_NNET AND BUILD_TEST)
    build_test(test/nnet/test_*.cc)

    # Build expression reader
    add_executable(nnet_reader test/nnet/readlog.cc)
    target_link_libraries(nnet_reader InfiniTensor)
  endif()
endif()
