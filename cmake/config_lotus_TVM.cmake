set(TVM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/tvm")
set(TVM_INCLUDE_DIR "${TVM_HOME}/include")
set(TVM_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")
set(DMLC_INCLUDE_DIR "${TVM_HOME}/3rdparty/dmlc-core/include")
set(DLPACK_INCLUDE_DIR "${TVM_HOME}/3rdparty/dlpack/include")

set(USE_CUDA ON)
set(USE_BANG OFF)

set(BUILD_TEST ON)
set(BUILD_TEST_CORE ON)
set(BUILD_TEST_PET OFF)
set(BUILD_TEST_EINNET ON)
