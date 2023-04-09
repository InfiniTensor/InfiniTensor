set(TVM_HOME "/home/zly/Apps/tvm-v0.10.0")

# CMake_BUILD_TYPE is not set
set(USE_CUDA ON)
set(USE_BANG OFF)
set(TVM_INCLUDE_DIR "${TVM_HOME}/include")
set(DMLC_INCLUDE_DIR "${TVM_HOME}/3rdparty/dmlc-core/include")
set(DLPACK_INCLUDE_DIR "${TVM_HOME}/3rdparty/dlpack/include")

set(BUILD_TEST ON)
set(BUILD_TEST_CORE ON)
set(BUILD_TEST_PET OFF)
set(BUILD_TEST_EINNET ON)