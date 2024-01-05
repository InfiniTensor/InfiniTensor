SET(CNCL_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)
SET(CNCL_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)

set(CNCL_INCLUDE_DIR $ENV{NEUWARE_HOME}/include)
set(CNCL_LIB_DIR $ENV{NEUWARE_HOME}/lib64)
set(CNCL_VERSION $ENV{CNCL_VERSION} CACHE STRING "Version of CNCL to build with")

if ($ENV{CNCL_ROOT_DIR})
  message(WARNING "CNCL_ROOT_DIR is deprecated. Please set CNCL_ROOT instead.")
endif()
list(APPEND CNCL_ROOT $ENV{CNCL_ROOT_DIR} ${MLU_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. CNCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${CNCL_ROOT})

find_path(CNCL_INCLUDE_DIRS
  NAMES cncl.h
  HINTS ${CNCL_INCLUDE_DIR})

if (USE_STATIC_CNCL)
  MESSAGE(STATUS "USE_STATIC_CNCL is set. Linking with static CNCL library.")
  SET(CNCL_LIBNAME "CNCL_static")
  if (CNCL_VERSION)  # Prefer the versioned library if a specific CNCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${CNCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(CNCL_LIBNAME "cncl")
  if (CNCL_VERSION)  # Prefer the versioned library if a specific CNCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${CNCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(CNCL_LIBRARIES
  NAMES ${CNCL_LIBNAME}
  HINTS ${CNCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CNCL DEFAULT_MSG CNCL_INCLUDE_DIRS CNCL_LIBRARIES)

if(CNCL_FOUND)  # obtaining CNCL version and some sanity checks
  set (CNCL_HEADER_FILE "${CNCL_INCLUDE_DIRS}/cncl.h")
  message (STATUS "Determining CNCL version from ${CNCL_HEADER_FILE}...")
  set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list (APPEND CMAKE_REQUIRED_INCLUDES ${CNCL_INCLUDE_DIRS})
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(CNCL_VERSION_CODE CNCL.h CNCL_VERSION_DEFINED)

  if (CNCL_VERSION_DEFINED)
    set(file "${PROJECT_BINARY_DIR}/detect_cncl_version.cc")
    file(WRITE ${file} "
      #include <iostream>
      #include <cncl.h>
      int main()
      {
        std::cout << CNCL_MAJOR << '.' << CNCL_MINOR << '.' << CNCL_PATCH << std::endl;
        int x;
        CNCLGetVersion(&x);
        return x == CNCL_VERSION_CODE;
      }
")
    try_run(CNCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
          RUN_OUTPUT_VARIABLE CNCL_VERSION_FROM_HEADER
          CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${CNCL_INCLUDE_DIRS}"
          LINK_LIBRARIES ${CNCL_LIBRARIES})
    if (NOT CNCL_VERSION_MATCHED)
      message(FATAL_ERROR "Found CNCL header version and library version do not match! \
(include: ${CNCL_INCLUDE_DIRS}, library: ${CNCL_LIBRARIES}) Please set CNCL_INCLUDE_DIR and CNCL_LIB_DIR manually.")
    endif()
    message(STATUS "CNCL version: ${CNCL_VERSION_FROM_HEADER}")
  else()
    # message(STATUS "CNCL version < 2.3.5-5")
  endif ()
  set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

  message(STATUS "Found CNCL (include: ${CNCL_INCLUDE_DIRS}, library: ${CNCL_LIBRARIES})")
  mark_as_advanced(CNCL_ROOT_DIR CNCL_INCLUDE_DIRS CNCL_LIBRARIES)
endif()
