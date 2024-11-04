# set(tablegen_exe ${PROJECT_SOURCE_DIR}/3rd-party/llvm-project/build/bin/mlir-tblgen)
# set(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Dialect")
# set(MLIR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd-party/llvm-project/mlir/include)

# function(mlir_tablegen)  
#     # 获取所有传递给函数的参数  
#     set(params)  
#     foreach(arg IN LISTS ARGV)  
#         list(APPEND params "${arg}")  
#     endforeach() 
#     list(GET params 0 input_file) 
#     list(GET params 1 output_file)
#     list(SUBLIST params 2 -1 remaining_args)      
#     add_custom_command(
#         OUTPUT ${CURRENT_DIR}/${output_file}
#         COMMAND ${tablegen_exe} ${remaining_args} -I ${MLIR_INCLUDE_DIR} ${CURRENT_DIR}/${input_file} -o ${CURRENT_DIR}/${output_file}
#         DEPENDS ${CURRENT_DIR}/${input_file}
#     )
#     set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CURRENT_DIR}/${output_file} PARENT_SCOPE)
#     set_source_files_properties(${CURRENT_DIR}/${output_file} PROPERTIES GENERATED 1)
#     message("OUTPUT: ${CURRENT_DIR} ${output_file}")
#     message("DEPENDS: ${CURRENT_DIR}/${input_file}")
#     message("COMMAND: ${tablegen_exe} ${remaining_args} -I ${MLIR_INCLUDE_DIR} ${CURRENT_DIR}/${input_file} -o ${CURRENT_DIR}/${output_file}")
# endfunction()  

# function(add_public_tablegen_target target)
#     message("TABLEGEN_OUTPUT: ${TABLEGEN_OUTPUT}")
#     add_custom_target(${target}
#         DEPENDS ${TABLEGEN_OUTPUT}
#     )
# endfunction()

# function(add_mlir_dialect dialect dialect_namespace)
# #   set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
#   mlir_tablegen(${dialect}.td ${dialect}.h.inc -gen-op-decls)
#   mlir_tablegen(${dialect}.td ${dialect}.cpp.inc -gen-op-defs)
# #   mlir_tablegen(${dialect}.td ${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
# #   mlir_tablegen(${dialect}.td ${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
# #   mlir_tablegen(${dialect}.td ${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
# #   mlir_tablegen(${dialect}.td ${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
#   add_public_tablegen_target(MLIR${dialect}IncGen)
#   add_dependencies(mlir-headers MLIR${dialect}IncGen)
# endfunction()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include/Dialect)
add_subdirectory(src)