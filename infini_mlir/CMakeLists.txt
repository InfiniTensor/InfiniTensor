
include_directories(Dialect/include)
add_subdirectory(Dialect)
add_subdirectory(Translation)

add_library(infini_mlir INTERFACE)
target_include_directories(infini_mlir INTERFACE Dialect/include)
target_link_libraries(infini_mlir INTERFACE translation)

