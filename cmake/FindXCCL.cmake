# Find the xccl libraries
set(XCCL_INCLUDE_DIR $ENV{KUNLUN_HOME}/include CACHE PATH "Folder contains KUNLUN XCCL headers")
set(XCCL_LIB_DIR $ENV{KUNLUN_HOME}  CACHE PATH "Folder contains KUNLUN XCCL libraries")

list(APPEND CMAKE_PREFIX_PATH $ENV{KUNLUN_HOME})

find_path(XCCL_INCLUDE_DIRS # ${XCCL_INCLUDE_DIR}
  NAMES xpu/bkcl.h
  HINTS XCCL_INCLUDE_DIR)

find_library(XCCL_LIBRARIES # ${XCCL_LIB_DIR}
  NAMES so/libbkcl.so
  HINTS XCCL_LIB_DIR)

message(STATUS "XCCL_INCLUDE_DIRS: ${XCCL_INCLUDE_DIRS}")
message(STATUS "XCCL_LIBRARIES: ${XCCL_LIBRARIES}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XCCL DEFAULT_MSG XCCL_INCLUDE_DIRS XCCL_LIBRARIES)

if (XCCL_FOUND)
  set (XCCL_HEADER_FILE "${XCCL_INCLUDE_DIRS}/xpu/bkcl.h")
  message (STATUS "Determing XCCL version from ${XCCL_HEADER_FILE}...")
  list (APPEND CMAKE_REQUIRED_INCLUDES ${XCCL_INCLUDE_DIRS})
  message(STATUS "Found XCCL (include: ${XCCL_INCLUDE_DIRS}, library: ${XCCL_LIBRARIES})")
  mark_as_advanced(XCCL_INCLUDE_DIRS XCCL_LIBRARIES)
endif()


