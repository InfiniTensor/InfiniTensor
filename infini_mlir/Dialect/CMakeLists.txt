set(Dialect_Dir ${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${Dialect_Dir})
add_subdirectory(include)
add_subdirectory(src)

add_library(InfiniDialectDirectories INTERFACE)
target_include_directories(InfiniDialectDirectories INTERFACE ${Dialect_Dir})
